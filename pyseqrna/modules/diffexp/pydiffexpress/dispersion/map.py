"""
MAP (Maximum A Posteriori) dispersion estimation.

This module implements MAP dispersion estimation using the exact inmoose functions.

Classes:
    - MAPDispersionEstimator: Maximum A Posteriori (MAP) dispersion estimator

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from anndata import AnnData
from scipy.stats import median_abs_deviation

from .base import BaseDispersionEstimator
from .utils import DispersionEstimationUtils

logger = logging.getLogger(__name__)


class MAPDispersionEstimator(BaseDispersionEstimator):
    """
    Maximum A Posteriori (MAP) dispersion estimator.

    This implementation uses the exact inmoose functions for optimization.
    """

    def __init__(
        self,
        outlier_sd: float = 2.0,
        outlier_log_boundary_tol: float = 0.035,
        disp_prior_var: Optional[float] = None,
        min_disp: float = 1e-8,
        kappa_0: float = 1.0,
        disp_tol: float = 1e-6,
        max_iter: int = 100,
        use_cox_reid_adjustment: bool = True,
        weight_threshold: float = 1e-2,
        quiet: bool = False,
    ):
        """
        Initialize MAP dispersion estimator.

        Parameters
        ----------
        outlier_sd : float
            Number of standard deviations above which genes are considered outliers
        outlier_log_boundary_tol : float
            Small tolerance subtracted from the log-scale outlier boundary to
            account for cross-implementation numeric drift in the fitted trend.
        disp_prior_var : float, optional
            Prior variance for dispersion estimates
        min_disp : float
            Minimum dispersion value
        kappa_0 : float
            Parameter for initial proposal in backtracking search
        disp_tol : float
            Tolerance for convergence
        max_iter : int
            Maximum number of iterations
        use_cox_reid_adjustment : bool
            Whether to use Cox-Reid adjustment
        weight_threshold : float
            Weight threshold for subsetting design matrix
        quiet : bool
            Whether to suppress messages
        """
        super().__init__()

        self.parameters = {
            "outlier_sd": outlier_sd,
            "outlier_log_boundary_tol": outlier_log_boundary_tol,
            "disp_prior_var": disp_prior_var,
            "min_disp": min_disp,
            "kappa_0": kappa_0,
            "disp_tol": disp_tol,
            "max_iter": max_iter,
            "use_cox_reid_adjustment": use_cox_reid_adjustment,
            "weight_threshold": weight_threshold,
            "quiet": quiet,
        }

        self._fitted = False
        self._results = {}

    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "MAPDispersionEstimator":
        """
        Fit MAP dispersion estimates using exact inmoose functions.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix
        design_matrix : np.ndarray, optional
            Design matrix. If None, uses data.obsm['design']

        Returns
        -------
        MAPDispersionEstimator
            Self
        """
        # Extract parameters
        outlier_sd = self.parameters["outlier_sd"]
        outlier_log_boundary_tol = self.parameters["outlier_log_boundary_tol"]
        disp_prior_var = self.parameters["disp_prior_var"]
        min_disp = self.parameters["min_disp"]
        kappa_0 = self.parameters["kappa_0"]
        disp_tol = self.parameters["disp_tol"]
        max_iter = self.parameters["max_iter"]
        use_cox_reid_adjustment = self.parameters["use_cox_reid_adjustment"]
        weight_threshold = self.parameters["weight_threshold"]
        quiet = self.parameters["quiet"]

        if not quiet:
            logger.info("Fitting MAP dispersions using exact inmoose functions...")

        # Check for required data
        if "disp_gene_est" not in data.var:
            raise ValueError("Gene-wise dispersion estimates not found. Run GeneWiseDispersionEstimator first.")

        if "disp_fit" not in data.var:
            raise ValueError("Fitted dispersion estimates not found. Run TrendDispersionEstimator first.")

        # Get design matrix
        if design_matrix is None:
            design_matrix = data.obsm.get("design", None)
            if design_matrix is None:
                raise ValueError("Design matrix not found in data.obsm['design']")

        # Get non-zero genes
        all_zero = data.var.get("all_zero", data.var["base_mean"] == 0)
        data_nz = data[:, ~all_zero]

        # Get dispersion estimates
        disp_gene_est = data_nz.var["disp_gene_est"]
        disp_fitted = data_nz.var["disp_fit"]

        # Get counts and size factors
        counts = data.X.toarray() if hasattr(data.X, "toarray") else data.X
        counts_nz = counts[:, ~all_zero]
        size_factors = data.obs.get("size_factors", np.ones(data.n_obs))

        # Get normalized counts
        if "normalized_counts" in data.layers:
            norm_counts = (
                data.layers["normalized_counts"].toarray()
                if hasattr(data.layers["normalized_counts"], "toarray")
                else data.layers["normalized_counts"]
            )
            norm_counts_nz = norm_counts[:, ~all_zero]
        else:
            size_factors_array = np.asarray(size_factors)
            norm_counts_nz = counts_nz / size_factors_array[:, None]

        # Get base means
        base_means = data_nz.var["base_mean"]

        # Estimate prior variance if not provided
        if disp_prior_var is None:
            disp_prior_var = self._estimate_prior_variance(disp_gene_est, disp_fitted, base_means, min_disp, design_matrix)

        # Detect outliers
        outliers = self._detect_outliers(
            disp_gene_est,
            disp_fitted,
            outlier_sd,
            outlier_log_boundary_tol,
            min_disp,
        )

        # Calculate prior means (fitted values)
        log_alpha_prior_mean = np.log(disp_fitted)

        # Get weights (if available)
        weights, use_weights = self._get_weights(data_nz, design_matrix, weight_threshold)

        # Get previously calculated mu
        if "mu" in data_nz.layers:
            mu = data_nz.layers["mu"]
        else:
            # Estimate means using linear model
            try:
                beta = np.linalg.lstsq(design_matrix, norm_counts_nz, rcond=None)[0]
                mu = design_matrix @ beta
            except np.linalg.LinAlgError:
                mu = np.mean(norm_counts_nz, axis=0, keepdims=True)
            mu = np.clip(mu, 0.5, None)

        # Fit MAP estimates using exact inmoose functions
        map_results = self._fit_map_dispersions_exact_inmoose(
            counts_nz,
            design_matrix,
            mu,
            disp_gene_est,
            disp_fitted,
            log_alpha_prior_mean,
            disp_prior_var,
            min_disp,
            kappa_0,
            disp_tol,
            max_iter,
            use_cox_reid_adjustment,
            weights,
            use_weights,
            weight_threshold,
            quiet,
        )

        # Store results
        self._results = {
            "dispersions": self._build_vector_with_na_cols(map_results["dispersions"], all_zero),
            "disp_gene_est": self._build_vector_with_na_cols(disp_gene_est, all_zero),
            "disp_fitted": self._build_vector_with_na_cols(disp_fitted, all_zero),
            "outliers": self._build_vector_with_na_cols(outliers, all_zero),
            "disp_prior_var": disp_prior_var,
            "disp_iter": self._build_vector_with_na_cols(map_results["iterations"], all_zero),
            "disp_map": self._build_vector_with_na_cols(map_results["dispersions"], all_zero),
            "disp_conv": self._build_vector_with_na_cols(map_results["converged"], all_zero),
        }

        # Apply outlier replacement (like inmoose)
        dispersion_final = self._results["dispersions"].copy()
        outlier_mask = self._results["outliers"].astype(bool)
        dispersion_final[outlier_mask] = self._results["disp_gene_est"][outlier_mask]

        # Update the dispersions result with outlier replacement
        self._results["dispersions"] = dispersion_final

        # Debug output
        if not quiet:
            logger.debug(
                "MAP dispersions range: [%.6f, %.6f]",
                np.min(map_results["dispersions"]),
                np.max(map_results["dispersions"]),
            )
            logger.debug("MAP mean dispersion: %.6f", np.mean(map_results["dispersions"]))
            logger.debug(
                "Number of unique MAP values: %s",
                len(np.unique(map_results["dispersions"])),
            )
            logger.debug(
                "Genes that fell back to grid search: %s/%s",
                map_results["grid_fallback_count"],
                len(map_results["dispersions"]),
            )

        self._fitted = True
        return self

    def _fit_map_dispersions_exact_inmoose(
        self,
        counts: np.ndarray,
        design_matrix: np.ndarray,
        mu: np.ndarray,
        disp_gene_est: np.ndarray,
        disp_fitted: np.ndarray,
        log_alpha_prior_mean: np.ndarray,
        disp_prior_var: float,
        min_disp: float,
        kappa_0: float,
        disp_tol: float,
        max_iter: int,
        use_cox_reid_adjustment: bool,
        weights: np.ndarray,
        use_weights: bool,
        weight_threshold: float,
        quiet: bool,
    ) -> Dict[str, Any]:
        """
        Fit MAP dispersion estimates using exact inmoose functions.
        """
        n_genes = counts.shape[1]
        map_disps = np.zeros(n_genes)
        iterations = np.zeros(n_genes)
        converged = np.zeros(n_genes, dtype=bool)
        grid_fallback_count = 0

        # Inmoose-style initialization
        # Treat zero gene-wise estimates as "failed estimates" (like inmoose's 1e-08)
        # Use fitted value for genes with zero gene-wise estimates
        disp_init = np.where(
            (disp_gene_est > 0) & (disp_gene_est > 0.1 * disp_fitted),
            disp_gene_est,
            disp_fitted,
        )

        # Fill any missing values with fitted values
        disp_init[np.isnan(disp_init)] = disp_fitted[np.isnan(disp_init)]

        # Prepare data for inmoose function
        y = counts
        x = design_matrix
        mu_hat = mu
        log_alpha = np.log(disp_init)
        log_alpha_prior_mean_array = (
            log_alpha_prior_mean.values if hasattr(log_alpha_prior_mean, "values") else log_alpha_prior_mean
        )
        min_log_alpha = np.log(min_disp / 10)

        # Use exact inmoose line search function
        try:
            result = DispersionEstimationUtils.fit_dispersion_line_search(
                y=y,
                x=x,
                mu_hat=mu_hat,
                log_alpha=log_alpha,
                log_alpha_prior_mean=log_alpha_prior_mean_array,
                log_alpha_prior_sigmasq=disp_prior_var,
                min_log_alpha=min_log_alpha,
                kappa_0=kappa_0,
                tol=disp_tol,
                maxit=max_iter,
                usePrior=True,
                weights=weights,
                useWeights=use_weights,
                weightThreshold=weight_threshold,
                useCR=use_cox_reid_adjustment,
            )

            map_disps = np.exp(result["log_alpha"])
            iterations = result["iter"]
            converged = result["iter"] < max_iter

            # If any genes didn't converge, use grid search fallback
            if not np.all(converged):
                grid_fallback_count = np.sum(~converged)
                if not quiet:
                    logger.info(
                        "%s genes did not converge, using grid search fallback",
                        grid_fallback_count,
                    )

                # For non-converged genes, use grid search
                for i in range(n_genes):
                    if not converged[i]:
                        grid_result = self._fit_disp_grid(
                            counts[:, i],
                            design_matrix,
                            mu[:, i],
                            log_alpha_prior_mean.iloc[i],
                            disp_prior_var,
                            use_weights,
                            weights[:, i] if use_weights else None,
                            weight_threshold,
                            use_cox_reid_adjustment,
                        )
                        map_disps[i] = np.exp(grid_result)

        except Exception as e:
            if not quiet:
                logger.warning("Error in inmoose line search: %s", e)
                logger.info("Falling back to individual gene optimization")

            # Fallback to individual gene optimization
            for i in range(n_genes):
                if not quiet and (i % 1000 == 0 or i < 5):
                    logger.debug("Fitting MAP dispersion for gene %s/%s", i + 1, n_genes)

                try:
                    # Use exact inmoose line search for individual gene
                    result = DispersionEstimationUtils.fit_dispersion_line_search(
                        y=counts[:, i : i + 1],
                        x=design_matrix,
                        mu_hat=mu[:, i : i + 1],
                        log_alpha=log_alpha[i : i + 1],
                        log_alpha_prior_mean=log_alpha_prior_mean.iloc[i : i + 1],
                        log_alpha_prior_sigmasq=disp_prior_var,
                        min_log_alpha=min_log_alpha,
                        kappa_0=kappa_0,
                        tol=disp_tol,
                        maxit=max_iter,
                        usePrior=True,
                        weights=weights[:, i : i + 1] if use_weights else None,
                        useWeights=use_weights,
                        weightThreshold=weight_threshold,
                        useCR=use_cox_reid_adjustment,
                    )

                    map_disps[i] = np.exp(result["log_alpha"][0])
                    iterations[i] = result["iter"][0]
                    converged[i] = result["iter"][0] < max_iter

                    # If optimization failed, use grid search fallback
                    if not converged[i]:
                        grid_result = self._fit_disp_grid(
                            counts[:, i],
                            design_matrix,
                            mu[:, i],
                            log_alpha_prior_mean.iloc[i],
                            disp_prior_var,
                            use_weights,
                            weights[:, i] if use_weights else None,
                            weight_threshold,
                            use_cox_reid_adjustment,
                        )
                        map_disps[i] = np.exp(grid_result)
                        grid_fallback_count += 1

                except Exception as e:
                    # Fallback to gene-wise estimate
                    map_disps[i] = disp_gene_est.iloc[i]
                    iterations[i] = max_iter
                    converged[i] = False
                    grid_fallback_count += 1
                    if i < 3:
                        logger.debug("Gene %s optimization failed: %s", i + 1, e)

        # Bound the dispersion estimates (inmoose-style)
        max_disp = np.maximum(10, counts.shape[0])
        map_disps = np.clip(map_disps, min_disp, max_disp)

        return {
            "dispersions": map_disps,
            "iterations": iterations,
            "converged": converged,
            "grid_fallback_count": grid_fallback_count,
        }

    def _fit_disp_grid(
        self,
        y: np.ndarray,
        x: np.ndarray,
        mu: np.ndarray,
        log_alpha_prior_mean: float,
        log_alpha_prior_sigmasq: float,
        use_weights: bool,
        weights: Optional[np.ndarray],
        weight_threshold: float,
        use_cr: bool,
    ) -> float:
        """
        Fit dispersion using grid search (inmoose's fallback method).
        """
        # Create grid
        min_log_alpha = np.log(self.parameters["min_disp"] / 10)
        max_log_alpha = np.log(np.maximum(10, len(y)))
        grid_length = 100
        disp_grid = np.linspace(min_log_alpha, max_log_alpha, grid_length)

        # Evaluate log posterior over grid
        logpost_vec = np.zeros(grid_length)
        for i, log_alpha in enumerate(disp_grid):
            logpost_vec[i] = DispersionEstimationUtils.calculate_dispersion_log_posterior(
                log_alpha,
                y,
                mu,
                x,
                log_alpha_prior_mean,
                log_alpha_prior_sigmasq,
                use_prior_regularization=True,
                observation_weights=weights,
                use_observation_weights=use_weights,
                minimum_weight_threshold=weight_threshold,
                use_cox_reid_adjustment=use_cr,
            )

        # Find maximum
        max_idx = np.argmax(logpost_vec)
        a_hat = disp_grid[max_idx]

        # Fine grid around maximum
        delta = disp_grid[1] - disp_grid[0]
        fine_grid = np.linspace(a_hat - delta, a_hat + delta, grid_length)

        # Evaluate on fine grid
        fine_logpost_vec = np.zeros(grid_length)
        for i, log_alpha in enumerate(fine_grid):
            fine_logpost_vec[i] = DispersionEstimationUtils.calculate_dispersion_log_posterior(
                log_alpha,
                y,
                mu,
                x,
                log_alpha_prior_mean,
                log_alpha_prior_sigmasq,
                use_prior_regularization=True,
                observation_weights=weights,
                use_observation_weights=use_weights,
                minimum_weight_threshold=weight_threshold,
                use_cox_reid_adjustment=use_cr,
            )

        # Return maximum from fine grid
        fine_max_idx = np.argmax(fine_logpost_vec)
        return fine_grid[fine_max_idx]

    def _get_weights(self, data: AnnData, design_matrix: np.ndarray, weight_threshold: float) -> tuple[np.ndarray, bool]:
        """
        Get weights for observations (if available).
        """
        if "weights" in data.layers:
            weights = data.layers["weights"]
            use_weights = True
        else:
            weights = np.ones((data.n_obs, data.n_vars))
            use_weights = False

        return weights, use_weights

    def _estimate_prior_variance(
        self,
        disp_gene_est: np.ndarray,
        disp_fitted: np.ndarray,
        base_means: np.ndarray,
        min_disp: float,
        design_matrix: np.ndarray,
    ) -> float:
        """
        Estimate prior variance for dispersion estimates using inmoose's exact approach.

        This matches inmoose's estimateDispersionsPriorVar function:
        1. Calculate log dispersion residuals around fitted trend
        2. Use MAD to estimate variance of residuals
        3. Calculate expected sampling variance using polygamma
        4. Set prior variance as max(observed_variance - expected_variance, 0.25)
        """
        # Filter genes for variance estimation (same as inmoose)
        above_min_disp = disp_gene_est >= 100 * min_disp

        if np.sum(above_min_disp) == 0:
            # No genes above minimum dispersion threshold
            return 0.25  # Default value

        # Calculate log dispersion residuals (same as inmoose)
        disp_residuals = np.log(disp_gene_est[above_min_disp]) - np.log(disp_fitted[above_min_disp])

        # Calculate variance using MAD (same as inmoose)
        from scipy.stats import median_abs_deviation

        mad_scale = median_abs_deviation(disp_residuals, nan_policy="omit", scale="normal")
        var_log_disp_ests = mad_scale**2

        # Calculate expected sampling variance (same as inmoose)
        m, p = design_matrix.shape

        if m > p:
            # Calculate expected variance using polygamma function
            from scipy.special import polygamma

            exp_var_log_disp = polygamma(1, (m - p) / 2)

            # Set prior variance as max(observed - expected, 0.25)
            disp_prior_var = max(var_log_disp_ests - exp_var_log_disp, 0.25)
        else:
            # We have m = p, so do not try to subtract sampling variance
            disp_prior_var = var_log_disp_ests

        return disp_prior_var

    def _detect_outliers(
        self,
        disp_gene_est: np.ndarray,
        disp_fitted: np.ndarray,
        outlier_sd: float,
        outlier_log_boundary_tol: float,
        min_disp: float,
    ) -> np.ndarray:
        """
        Detect outlier genes.
        """
        # Use a robust spread estimate on log residuals around the fitted trend.
        # This tracks reference robust-dispersion behavior better than plain variance, which
        # gets inflated by the very high-dispersion genes we are trying to flag.
        use_for_var = disp_gene_est > 100 * min_disp
        if np.sum(use_for_var) > 0:
            log_ratios = np.log(disp_gene_est[use_for_var] / disp_fitted[use_for_var])
            mad_scale = median_abs_deviation(log_ratios, nan_policy="omit", scale="normal")
            if np.isnan(mad_scale) or mad_scale == 0:
                var_log_disp = np.var(log_ratios)
            else:
                var_log_disp = mad_scale**2
        else:
            var_log_disp = 0.25

        # Detect outliers. A small tolerance helps recover borderline genes that
        # the reference method classifies as outliers, but that can miss the cutoff here due to
        # tiny numerical differences in the global trend fit.
        boundary = np.log(disp_fitted) + outlier_sd * np.sqrt(var_log_disp) - outlier_log_boundary_tol
        outliers = np.log(disp_gene_est) > boundary
        outliers[np.isnan(outliers)] = False

        return outliers.astype(bool)

    def _build_vector_with_na_cols(self, values: np.ndarray, all_zero: np.ndarray) -> np.ndarray:
        """Build vector with NA values for zero genes."""
        result = np.full(len(all_zero), np.nan)
        result[~all_zero] = values
        return result

    def estimate(self, data: AnnData) -> np.ndarray:
        """
        Estimate MAP dispersions for the given data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            MAP dispersion estimates
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before estimating")

        return self._results["dispersions"]

    def get_outliers(self) -> np.ndarray:
        """
        Get outlier genes detected during MAP estimation.

        Returns
        -------
        np.ndarray
            Boolean array indicating outliers
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting outliers")

        return self._results["outliers"]

    def get_prior_variance(self) -> float:
        """
        Get the prior variance used for MAP estimation.

        Returns
        -------
        float
            Prior variance
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting prior variance")

        return self._results["disp_prior_var"]
