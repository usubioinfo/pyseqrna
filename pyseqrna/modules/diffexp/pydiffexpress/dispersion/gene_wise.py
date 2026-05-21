"""
Gene-wise dispersion estimation.

This module implements gene-wise dispersion estimation by maximizing the
Cox-Reid adjusted profile likelihood for each gene.

Classes:
    - Factor: Simple factor class for checking model matrix groups
    - GeneWiseDispersionEstimator: Gene-wise dispersion estimator

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import logging
from typing import Optional, Dict, Any

from anndata import AnnData

from .base import BaseDispersionEstimator
from .utils import DispersionEstimationUtils

logger = logging.getLogger(__name__)


# Our own Factor class for model matrix group checking
class Factor:
    """Simple factor class for checking model matrix groups."""

    def __init__(self, values):
        self.values = values
        self.levels = list(set(values))

    @property
    def nlevels(self):
        return len(self.levels)


class GeneWiseDispersionEstimator(BaseDispersionEstimator):
    """
    Gene-wise dispersion estimator.

    This estimator calculates gene-specific dispersion parameters by maximizing
    the Cox-Reid adjusted profile likelihood for each gene independently.
    """

    def __init__(
        self,
        min_disp: float = 1e-8,
        kappa_0: float = 1.0,
        disp_tol: float = 1e-6,
        max_iter: int = 100,
        use_cox_reid_adjustment: bool = True,
        weight_threshold: float = 1e-2,
        quiet: bool = False,
        n_iter: int = 1,
        linear_mu: Optional[bool] = None,
        min_mu: Optional[float] = None,
        alpha_init: Optional[np.ndarray] = None,
    ):
        """
        Initialize the gene-wise dispersion estimator.

        Parameters
        ----------
        min_disp : float
            Minimum dispersion value for numerical stability
        kappa_0 : float
            Parameter for backtracking search (larger values = larger steps)
        disp_tol : float
            Tolerance for convergence of log dispersion
        max_iter : int
            Maximum number of iterations for optimization
        use_cox_reid_adjustment : bool
            Whether to use Cox-Reid adjustment
        weight_threshold : float
            Threshold for subsetting design matrix and weights
        quiet : bool
            Whether to suppress progress messages
        n_iter : int
            Number of iterations between mean and dispersion estimation
        linear_mu : bool, optional
            Whether to use linear model for mean estimation
        min_mu : float, optional
            Lower bound on estimated counts for fitting
        alpha_init : np.ndarray, optional
            Initial guess for dispersion estimates
        """
        super().__init__(
            min_disp=min_disp,
            kappa_0=kappa_0,
            disp_tol=disp_tol,
            max_iter=max_iter,
            use_cox_reid_adjustment=use_cox_reid_adjustment,
            weight_threshold=weight_threshold,
            quiet=quiet,
            n_iter=n_iter,
            linear_mu=linear_mu,
            min_mu=min_mu,
            alpha_init=alpha_init,
        )

        # Validate parameters
        if np.log(min_disp / 10) <= -30:
            raise ValueError("For computational stability, log(min_disp/10) should be above -30")

        if n_iter <= 0:
            raise ValueError("n_iter should be strictly positive")

    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "GeneWiseDispersionEstimator":
        """
        Fit the gene-wise dispersion estimator to the data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix with counts in .X
        design_matrix : np.ndarray, optional
            Design matrix for experimental design

        Returns
        -------
        self : GeneWiseDispersionEstimator
            Fitted estimator
        """
        self._validate_data(data)

        # Store design matrix in obsm and add categorical columns to obs
        if design_matrix is not None:
            data.obsm["design"] = design_matrix
        else:
            from ..utils import create_design_matrix

            design_matrix = create_design_matrix(data)
            data.obsm["design"] = design_matrix

        self._validate_design_matrix(design_matrix, data)

        # Add categorical columns to obs if they don't exist
        # This mimics inmoose behavior where C(condition) columns are added
        if "condition" in data.obs.columns and "C(condition)" not in data.obs.columns:
            data.obs["C(condition)"] = data.obs["condition"]

        # Ensure size_factors column exists with correct name
        if "sizeFactors" in data.obs.columns and "size_factors" not in data.obs.columns:
            data.obs["size_factors"] = data.obs["sizeFactors"]

        # Check for experimental replicates
        if not DispersionEstimationUtils.has_sufficient_experimental_replicates(data, design_matrix):
            raise ValueError("Insufficient experimental replicates for dispersion estimation")

        # Get parameters
        min_disp = self.parameters["min_disp"]
        kappa_0 = self.parameters["kappa_0"]
        disp_tol = self.parameters["disp_tol"]
        max_iter = self.parameters["max_iter"]
        use_cox_reid_adjustment = self.parameters["use_cox_reid_adjustment"]
        weight_threshold = self.parameters["weight_threshold"]
        quiet = self.parameters["quiet"]
        n_iter = self.parameters["n_iter"]
        linear_mu = self.parameters["linear_mu"]
        min_mu = self.parameters["min_mu"]
        alpha_init = self.parameters["alpha_init"]

        # Set default min_mu (like inmoose)
        if min_mu is None:
            min_mu = 0.5

        # Get raw counts from X (data.X now contains raw counts)
        counts = data.X.toarray() if hasattr(data.X, "toarray") else data.X

        size_factors = data.obs.get("size_factors", np.ones(data.n_obs))

        # Get normalized counts from layers (data.layers['normalized_counts'] contains normalized counts)
        if "normalized_counts" in data.layers:
            norm_counts = (
                data.layers["normalized_counts"].toarray()
                if hasattr(data.layers["normalized_counts"], "toarray")
                else data.layers["normalized_counts"]
            )
        else:
            # Fallback: calculate normalized counts on the fly
            norm_counts = counts / size_factors.reshape(-1, 1)

        # Get weights and check if they should be used
        data, weights, use_weights = DispersionEstimationUtils.get_and_check_weights(data, design_matrix, weight_threshold)

        # Calculate base means and variances like inmoose (with weights if present)
        if "base_mean" not in data.var:
            # Apply weights to normalized counts if weights exist
            if use_weights:
                weighted_norm_counts = weights * norm_counts
            else:
                weighted_norm_counts = norm_counts

            # Calculate base means and variances
            base_means = np.mean(weighted_norm_counts, axis=0)
            base_vars = np.var(weighted_norm_counts, axis=0, ddof=1)  # Use ddof=1 like inmoose
            data.var["base_mean"] = base_means
            data.var["base_variance"] = base_vars

        # Identify non-zero genes (like inmoose: all raw counts for a gene are zero)
        all_zero = np.sum(counts, axis=0) == 0
        data.var["all_zero"] = all_zero

        # Work with non-zero genes only
        data_nz = data[:, ~all_zero]
        counts_nz = counts[:, ~all_zero]
        norm_counts_nz = norm_counts[:, ~all_zero]
        weights_nz = weights[:, ~all_zero] if use_weights else np.ones((data.n_obs, np.sum(~all_zero)))

        # Initialize dispersion estimates
        if alpha_init is None:
            # Rough dispersion estimate
            rough_dispersion_estimates = DispersionEstimationUtils.estimate_rough_dispersions(norm_counts_nz, design_matrix)
            moments_dispersion_estimates = DispersionEstimationUtils.estimate_dispersions_by_moments(data_nz, size_factors)
            alpha_hat = np.minimum(rough_dispersion_estimates, moments_dispersion_estimates)
        else:
            if np.isscalar(alpha_init):
                alpha_hat = np.full(data_nz.n_vars, alpha_init)
            else:
                alpha_hat = alpha_init[~all_zero]

        # Bound initial estimates (like inmoose)
        max_disp = max(10.0, data.n_obs)
        alpha_init_bounded = np.clip(alpha_hat, min_disp, max_disp)
        alpha_hat = alpha_init_bounded.copy()
        alpha_hat_new = alpha_init_bounded.copy()

        # Determine if linear mu fitting should be used
        if linear_mu is None:
            # Check if number of groups equals number of coefficients (like inmoose)
            model_matrix_groups = Factor([tuple(design_matrix[i]) for i in range(design_matrix.shape[0])])
            linear_mu = model_matrix_groups.nlevels == design_matrix.shape[1]
            # also check for weights (then can't do linear mu)
            if use_weights:
                linear_mu = False

        # Iterate between mean and dispersion estimation (like inmoose)
        fitidx = np.repeat(True, data_nz.n_vars)
        mu = np.zeros(data_nz.shape)
        disp_iter = np.zeros(data_nz.n_vars)

        for iter_num in range(n_iter):
            if not quiet:
                logger.info("Iteration %s/%s", iter_num + 1, n_iter)

            # Estimate means for genes that are still being fitted
            if linear_mu:
                mu_fit = self._linear_model_mu(data_nz[:, fitidx], design_matrix)
            else:
                mu_fit = self._glm_mu(data_nz[:, fitidx], design_matrix, alpha_hat[fitidx])

            # Bound means
            mu_fit = np.clip(mu_fit, min_mu, None)
            mu[:, fitidx] = mu_fit

            # Estimate dispersions for genes that are still being fitted
            disp_results = self._fit_dispersions(
                counts_nz[:, fitidx],
                design_matrix,
                mu_fit,
                np.log(alpha_hat[fitidx]),  # log_alpha (like inmoose)
                np.log(alpha_hat[fitidx]),  # prior mean (like inmoose)
                1.0,  # prior variance
                np.log(min_disp / 10),  # min_log_alpha (like inmoose)
                kappa_0,
                disp_tol,
                max_iter,
                use_prior_regularization=False,
                use_cox_reid_adjustment=use_cox_reid_adjustment,
                weight_threshold=weight_threshold,
                weights=weights_nz[:, fitidx] if use_weights else np.ones((data_nz.n_obs, np.sum(fitidx))),
                use_weights=use_weights,
            )

            # Update results for genes that are still being fitted
            disp_iter[fitidx] = disp_results["iter"]
            alpha_hat_new = np.minimum(np.exp(disp_results["log_alpha"]), max_disp)
            alpha_hat[fitidx] = alpha_hat_new

            # Check convergence (like inmoose)
            fitidx = np.abs(np.log(alpha_hat_new) - np.log(alpha_hat[fitidx])) > 0.5

            if np.sum(fitidx) == 0:
                if not quiet:
                    logger.info("All genes converged after %s iterations", iter_num + 1)
                break

        # Handle convergence issues (like inmoose)
        disp_gene_est = alpha_hat
        if n_iter == 1:
            # Check if log posterior increased
            no_increase = disp_results["last_lp"] < disp_results["initial_lp"] + np.abs(disp_results["initial_lp"]) / 1e6
            disp_gene_est[no_increase] = alpha_init_bounded[no_increase]

        # Refit genes that didn't converge (like inmoose)
        conv_mask = (disp_iter < max_iter) & (disp_iter > 1)
        refit_mask = ~conv_mask & (disp_gene_est > min_disp * 10)

        if np.sum(refit_mask) > 0:
            if not quiet:
                logger.info("Refitting %s genes that did not converge", np.sum(refit_mask))

            disp_grid = self._fit_dispersion_grid(
                counts_nz[:, refit_mask],
                design_matrix,
                mu[:, refit_mask],
                np.zeros(np.sum(refit_mask)),  # prior mean
                1.0,  # prior variance
                use_prior_regularization=False,
                use_cox_reid_adjustment=use_cox_reid_adjustment,
                weight_threshold=weight_threshold,
            )
            disp_gene_est[refit_mask] = disp_grid

        # Final bounds
        disp_gene_est = np.clip(disp_gene_est, min_disp, max_disp)

        # Store results
        self._results = {
            "disp_gene_est": self._build_vector_with_na_cols(disp_gene_est, all_zero),
            "disp_gene_iter": self._build_vector_with_na_cols(disp_iter, all_zero),
            "mu": self._build_matrix_with_na_cols(mu, all_zero),
            "alpha_init": alpha_init_bounded,
        }

        self._fitted = True
        return self

    def estimate(self, data: AnnData) -> np.ndarray:
        """
        Estimate dispersions for the given data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            Gene-wise dispersion estimates
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before estimating")

        return self._results["disp_gene_est"]

    def _linear_model_mu(self, data: AnnData, design_matrix: np.ndarray) -> np.ndarray:
        """Estimate means using QR decomposition for numerical stability."""
        # Get raw counts and size factors
        raw_counts = data.X.toarray() if hasattr(data.X, "toarray") else data.X

        # Get size factors - handle both AnnData and ExpressionDataset
        if hasattr(data, "size_factors") and data.size_factors is not None:
            # ExpressionDataset with size_factors attribute
            size_factors = data.size_factors
        else:
            # AnnData or ExpressionDataset without size_factors attribute
            size_factors = data.obs.get("size_factors", np.ones(data.n_obs))

        # Normalize counts using size factors (like inmoose)
        size_factors_array = np.asarray(size_factors)
        normalized_counts = raw_counts / size_factors_array[:, None]

        # Use QR decomposition for robust linear model fitting
        mu_normalized = self._fit_linear_model_qr(normalized_counts, design_matrix)

        # Scale back to original scale using size factors
        mu = mu_normalized * size_factors_array[:, None]

        return mu

    def _fit_linear_model_qr(self, response_matrix: np.ndarray, design_matrix: np.ndarray) -> np.ndarray:
        """Fit linear model using QR decomposition for numerical stability."""
        # QR decomposition assumes p <= nobs (number of parameters <= number of observations)
        # This is guaranteed by checking that the design matrix is full rank
        #
        # Linear mean estimate using the QR decomposition.
        # We use: (x Rinv) (Q.T y) to minimize intermediate matrix sizes
        Q, R = np.linalg.qr(design_matrix)
        R_inverse = np.linalg.solve(R, np.identity(R.shape[0]))
        return (design_matrix @ R_inverse) @ (Q.T @ response_matrix)

    def _glm_mu(self, data: AnnData, design_matrix: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Estimate mean expression using negative binomial GLM.

        This uses a proper negative binomial GLM with IRLS algorithm.
        """
        from .glm import fit_negative_binomial_glm

        # Get raw counts and size factors
        counts = data.layers["counts"]
        size_factors = data.obs.get("size_factors", np.ones(data.n_obs))
        size_factors = np.asarray(size_factors)

        # Prepare data for GLM fitting
        # GLM expects (samples, genes) orientation
        y = counts.T  # (genes, samples)
        x = design_matrix  # (samples, coefficients)
        nf = size_factors[None, :]  # (1, samples) - same for all genes
        nf = np.broadcast_to(nf, (y.shape[0], y.shape[1]))  # (genes, samples)

        # Fit negative binomial GLM
        result = fit_negative_binomial_glm(y=y, x=x, nf=nf, alpha_hat=alpha, minmu=0.5, tol=1e-8, maxit=100)

        # Extract fitted means and transpose back to (samples, genes)
        mu_fit = result["mu"].T  # (samples, genes)

        return mu_fit

    def _fit_dispersions(
        self,
        y: np.ndarray,
        x: np.ndarray,
        mu_hat: np.ndarray,
        log_alpha: np.ndarray,
        log_alpha_prior_mean: np.ndarray,
        log_alpha_prior_sigmasq: float,
        min_log_alpha: float,
        kappa_0: float,
        tol: float,
        max_iter: int,
        use_prior_regularization: bool = False,
        use_cox_reid_adjustment: bool = True,
        weight_threshold: float = 1e-2,
        weights: Optional[np.ndarray] = None,
        use_weights: bool = False,
    ) -> Dict[str, Any]:
        """Fit dispersions using exact inmoose line search algorithm."""
        # Use exact inmoose implementation
        results = DispersionEstimationUtils.fit_dispersion_line_search(
            y=y,
            x=x,
            mu_hat=mu_hat,
            log_alpha=log_alpha,
            log_alpha_prior_mean=log_alpha_prior_mean,
            log_alpha_prior_sigmasq=log_alpha_prior_sigmasq,
            min_log_alpha=min_log_alpha,
            kappa_0=kappa_0,
            tol=tol,
            maxit=max_iter,
            usePrior=use_prior_regularization,
            weights=weights if weights is not None else np.ones_like(y),
            useWeights=use_weights,
            weightThreshold=weight_threshold,
            useCR=use_cox_reid_adjustment,
        )

        return {
            "log_alpha": results["log_alpha"],
            "iter": results["iter"],
            "last_lp": results["last_lp"],
            "initial_lp": results["initial_lp"],
        }

    def _fit_dispersion_grid(
        self,
        y: np.ndarray,
        x: np.ndarray,
        mu: np.ndarray,
        log_alpha_prior_mean: np.ndarray,
        log_alpha_prior_sigmasq: float,
        use_prior_regularization: bool = False,
        use_cox_reid_adjustment: bool = True,
        weight_threshold: float = 1e-2,
    ) -> np.ndarray:
        """Fit dispersions using grid search."""
        # Create grid of log alpha values
        log_alpha_grid = np.linspace(-10, 2, 100)
        n_genes = y.shape[1]

        disp_est = np.zeros(n_genes)

        for i in range(n_genes):
            # Calculate log posterior for each grid point
            lp_values = DispersionEstimationUtils.calculate_dispersion_log_posterior(
                log_alpha_grid,
                y[:, i],
                mu[:, i],
                x,
                log_alpha_prior_mean[i],
                log_alpha_prior_sigmasq,
                use_prior_regularization=use_prior_regularization,
                use_cox_reid_adjustment=use_cox_reid_adjustment,
                minimum_weight_threshold=weight_threshold,
            )

            # Find maximum
            max_idx = np.argmax(lp_values)
            disp_est[i] = np.exp(log_alpha_grid[max_idx])

        return disp_est

    def _build_vector_with_na_cols(self, values: np.ndarray, all_zero: np.ndarray) -> np.ndarray:
        """Build vector with NA values for zero genes."""
        result = np.full(len(all_zero), np.nan)
        result[~all_zero] = values
        return result

    def _build_matrix_with_na_cols(self, values: np.ndarray, all_zero: np.ndarray) -> np.ndarray:
        """Build matrix with NA values for zero genes."""
        result = np.full((values.shape[0], len(all_zero)), np.nan)
        # values has shape (n_samples, n_non_zero_genes)
        # all_zero has shape (n_total_genes,)
        # We need to place values in the correct positions
        non_zero_indices = np.where(~all_zero)[0]
        result[:, non_zero_indices] = values
        return result
