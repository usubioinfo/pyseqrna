"""
Utility class for dispersion estimation.

This module provides core mathematical functions for dispersion estimation,
including negative binomial likelihood calculations and optimization algorithms.

Classes:
    - DispersionEstimationUtils: Utility class for dispersion estimation calculations

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from scipy.special import digamma, xlog1py, xlogy
from scipy.special import loggamma as lgamma
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class DispersionEstimationUtils:
    """Utility class for dispersion estimation calculations."""

    @staticmethod
    def negative_binomial_probability_mass_function(
        x: np.ndarray, mu: np.ndarray, alpha: np.ndarray, log: bool = False
    ) -> np.ndarray:
        """
        Negative binomial probability mass function with mean parameterization.

        Parameters
        ----------
        x : np.ndarray
            Count values
        mu : np.ndarray
            Mean values
        alpha : np.ndarray
            Dispersion values (alpha = 1/size)
        log : bool
            Whether to return log probabilities

        Returns
        -------
        np.ndarray
            Probability mass function values
        """
        # Ensure all inputs are arrays
        x = np.asarray(x)
        mu = np.asarray(mu)
        alpha = np.asarray(alpha)

        # Broadcast arrays to same shape
        x, mu, alpha = np.broadcast_arrays(x, mu, alpha)

        # Calculate size parameter from alpha
        size = 1.0 / alpha

        # Calculate probability
        if log:
            # Log probability
            log_prob = (
                lgamma(x + size)
                - lgamma(size)
                - lgamma(x + 1)
                + size * np.log(size)
                + x * np.log(mu)
                - (size + x) * np.log(size + mu)
            )
            return log_prob
        else:
            # Regular probability
            prob = (
                np.exp(lgamma(x + size) - lgamma(size) - lgamma(x + 1))
                * (size / (size + mu)) ** size
                * (mu / (size + mu)) ** x
            )
            return prob

    @staticmethod
    def calculate_dispersion_log_posterior(
        log_dispersion_parameters: np.ndarray,
        gene_expression_counts: np.ndarray,
        expected_expression_means: np.ndarray,
        experimental_design_matrix: np.ndarray,
        prior_mean_log_dispersion: np.ndarray,
        prior_variance_log_dispersion: float,
        use_prior_regularization: bool = True,
        observation_weights: Optional[np.ndarray] = None,
        use_observation_weights: bool = False,
        minimum_weight_threshold: float = 1e-2,
        use_cox_reid_adjustment: bool = True,
    ) -> np.ndarray:
        """
        Calculate the log posterior of dispersion parameters for negative binomial gene expression data.

        This is the exact implementation from inmoose with descriptive variable names.

        Parameters
        ----------
        log_dispersion_parameters : np.ndarray
            Log of dispersion parameters for each gene
        gene_expression_counts : np.ndarray
            Gene expression count data
        expected_expression_means : np.ndarray
            Expected mean expression values
        experimental_design_matrix : np.ndarray
            Experimental design matrix for Cox-Reid adjustment
        prior_mean_log_dispersion : np.ndarray
            Prior mean for log dispersion parameters
        prior_variance_log_dispersion : float
            Prior variance for log dispersion parameters
        use_prior_regularization : bool
            Whether to use prior regularization
        observation_weights : np.ndarray, optional
            Weights for individual observations
        use_observation_weights : bool
            Whether to incorporate observation weights
        minimum_weight_threshold : float
            Minimum weight threshold for Cox-Reid adjustment
        use_cox_reid_adjustment : bool
            Whether to use Cox-Reid adjustment

        Returns
        -------
        np.ndarray
            Log posterior values for each gene
        """
        # make sure that all arrays are broadcastable to the appropriate shapes
        if not isinstance(log_dispersion_parameters, np.ndarray):
            log_dispersion_parameters = np.repeat(log_dispersion_parameters, 1)
        if len(gene_expression_counts.shape) == 1:
            gene_expression_counts = gene_expression_counts[:, None]
        if len(expected_expression_means.shape) == 1:
            expected_expression_means = expected_expression_means[:, None]
        if observation_weights is not None and len(observation_weights.shape) == 1:
            observation_weights = observation_weights[:, None]
        # helper variables to control the shapes
        n_genes = np.maximum(gene_expression_counts.shape[-1], log_dispersion_parameters.shape[-1])
        n_samples, n_coefficients = experimental_design_matrix.shape

        dispersion_values = np.exp(log_dispersion_parameters)
        if use_cox_reid_adjustment:
            experimental_design_matrix = experimental_design_matrix[None]
            # NB: now, experimental_design_matrix.shape == (1, n_samples, n_coefficients)
            mu_neg1 = 1.0 / expected_expression_means
            w_diag = 1.0 / (mu_neg1 + np.expand_dims(dispersion_values, axis=-2))
            if use_observation_weights:
                # cancel out all weights below the threshold
                idx = observation_weights <= minimum_weight_threshold
                w_diag[np.broadcast_to(idx, w_diag.shape)] = 0.0

            assert w_diag.shape[-2:] == (n_samples, n_genes)
            assert w_diag.shape[:-2] == log_dispersion_parameters.shape[:-1]

            # use `np.swapaxes` to transpose the matrices stored in the last 2 dims
            w_diag = np.swapaxes(w_diag, -1, -2)
            # insert a new axis in last position
            w_diag = np.expand_dims(w_diag, axis=-1)
            assert w_diag.shape[-3:] == (n_genes, n_samples, 1)
            assert w_diag.shape[:-3] == log_dispersion_parameters.shape[:-1]
            b = np.swapaxes(experimental_design_matrix * w_diag, -1, -2) @ experimental_design_matrix
            assert b.shape[-3:] == (n_genes, n_coefficients, n_coefficients)
            assert b.shape[:-3] == log_dispersion_parameters.shape[:-1]

            cr_term = -0.5 * np.linalg.slogdet(b)[1]
            assert cr_term.shape[:-1] == log_dispersion_parameters.shape[:-1]
            assert cr_term.shape[-1] == n_genes
        else:
            cr_term = 0.0

        # insert a new axis before the last one, to broadcast on the n_samples dimension
        # of gene_expression_counts, expected_expression_means, observation_weights
        dispersion_values = np.expand_dims(dispersion_values, axis=-2)
        dispersion_neg1 = 1.0 / dispersion_values
        if use_observation_weights:
            ll_part = np.sum(
                observation_weights
                * (
                    lgamma(gene_expression_counts + dispersion_neg1)
                    - lgamma(dispersion_neg1)
                    - xlogy(
                        gene_expression_counts,
                        expected_expression_means + dispersion_neg1,
                    )
                    - xlog1py(dispersion_neg1, dispersion_values * expected_expression_means)
                ),
                axis=-2,
            )
        else:
            ll_part = np.sum(
                lgamma(gene_expression_counts + dispersion_neg1)
                - lgamma(dispersion_neg1)
                - xlogy(gene_expression_counts, expected_expression_means + dispersion_neg1)
                - xlog1py(dispersion_neg1, dispersion_values * expected_expression_means),
                axis=-2,
            )

        assert (
            ll_part.shape[:-1] == log_dispersion_parameters.shape[:-1]
        ), f"{ll_part.shape} vs {log_dispersion_parameters.shape}"
        assert ll_part.shape[-1] == n_genes

        if use_prior_regularization:
            prior_part = -0.5 * (log_dispersion_parameters - prior_mean_log_dispersion) ** 2 / prior_variance_log_dispersion
            assert (
                prior_part.shape[:-1] == log_dispersion_parameters.shape[:-1]
            ), f"{prior_part.shape} vs {log_dispersion_parameters.shape}"
        else:
            prior_part = 0.0

        return ll_part + prior_part + cr_term

    @staticmethod
    def calculate_dispersion_log_posterior_derivative(
        log_dispersion_parameters: np.ndarray,
        gene_expression_counts: np.ndarray,
        expected_expression_means: np.ndarray,
        experimental_design_matrix: np.ndarray,
        prior_mean_log_dispersion: np.ndarray,
        prior_variance_log_dispersion: float,
        use_prior_regularization: bool = True,
        observation_weights: Optional[np.ndarray] = None,
        use_observation_weights: bool = False,
        minimum_weight_threshold: float = 1e-2,
        use_cox_reid_adjustment: bool = True,
    ) -> np.ndarray:
        """
        Calculate the derivative of log posterior with respect to log dispersion parameters.

        This is the exact implementation from inmoose with descriptive variable names.

        Parameters
        ----------
        log_dispersion_parameters : np.ndarray
            Log of dispersion parameters for each gene
        gene_expression_counts : np.ndarray
            Gene expression count data
        expected_expression_means : np.ndarray
            Expected mean expression values
        experimental_design_matrix : np.ndarray
            Experimental design matrix for Cox-Reid adjustment
        prior_mean_log_dispersion : np.ndarray
            Prior mean for log dispersion parameters
        prior_variance_log_dispersion : float
            Prior variance for log dispersion parameters
        use_prior_regularization : bool
            Whether to use prior regularization
        observation_weights : np.ndarray, optional
            Weights for individual observations
        use_observation_weights : bool
            Whether to incorporate observation weights
        minimum_weight_threshold : float
            Minimum weight threshold for Cox-Reid adjustment
        use_cox_reid_adjustment : bool
            Whether to use Cox-Reid adjustment

        Returns
        -------
        np.ndarray
            Derivative of log posterior with respect to log dispersion parameters
        """
        # make sure that all arrays are broadcastable to the appropriate shapes
        if not isinstance(log_dispersion_parameters, np.ndarray):
            log_dispersion_parameters = np.repeat(log_dispersion_parameters, 1)
        if len(gene_expression_counts.shape) == 1:
            gene_expression_counts = gene_expression_counts[:, None]
        if len(expected_expression_means.shape) == 1:
            expected_expression_means = expected_expression_means[:, None]
        if observation_weights is not None and len(observation_weights.shape) == 1:
            observation_weights = observation_weights[:, None]

        # helper variables to control the shapes
        n_genes = log_dispersion_parameters.shape[0]
        n_samples, n_coefficients = experimental_design_matrix.shape

        dispersion_values = np.exp(log_dispersion_parameters)
        if use_cox_reid_adjustment:
            experimental_design_matrix = experimental_design_matrix[None]
            # NB: now, experimental_design_matrix.shape == (1, n_samples, n_coefficients)
            mu_neg1 = 1.0 / expected_expression_means
            w_diag = 1.0 / (mu_neg1 + dispersion_values[None])
            dw_diag = -np.power(mu_neg1 + dispersion_values[None], -2)
            assert w_diag.shape == (n_samples, n_genes)
            assert dw_diag.shape == (n_samples, n_genes)
            # NB: w_diag.shape == dw_diag.shape == expected_expression_means.shape == (n_samples, n_genes)
            if use_observation_weights:
                # cancel out all weights below the threshold
                idx = observation_weights <= minimum_weight_threshold
                w_diag[np.broadcast_to(idx, w_diag.shape)] = 0.0
                dw_diag[np.broadcast_to(idx, dw_diag.shape)] = 0.0

            # use `np.swapaxes` to transpose the matrices stored in the last 2 dims
            w_diag = np.swapaxes(w_diag, -1, -2)
            dw_diag = np.swapaxes(dw_diag, -1, -2)
            # insert a new axis in last position
            w_diag = np.expand_dims(w_diag, axis=-1)
            dw_diag = np.expand_dims(dw_diag, axis=-1)
            b = np.swapaxes(experimental_design_matrix * w_diag, -1, -2) @ experimental_design_matrix
            db = np.swapaxes(experimental_design_matrix * dw_diag, -1, -2) @ experimental_design_matrix
            assert b.shape == (n_genes, n_coefficients, n_coefficients)
            assert db.shape == (n_genes, n_coefficients, n_coefficients)

            # Handle singular matrices in Cox-Reid adjustment
            try:
                cr_term = -0.5 * np.trace(np.linalg.inv(b) @ db, axis1=-2, axis2=-1)
            except np.linalg.LinAlgError:
                # If matrix is singular, skip Cox-Reid adjustment
                cr_term = np.zeros_like(dispersion_values)
            # NB original code computes
            #   ddetb = det(b) * trace(b.i() * db)
            # then
            #   cr_term = -0.5 * ddetb / det(b)
            # not sure why they multiply/divide by det(b)...
            assert cr_term.shape == dispersion_values.shape, f"{cr_term.shape} vs {dispersion_values.shape}"
        else:
            cr_term = 0.0

        dispersion_values = dispersion_values[None]
        dispersion_neg1 = 1.0 / dispersion_values
        dispersion_neg2 = np.power(dispersion_values, -2)
        dispersion_times_mu = dispersion_values * expected_expression_means
        if use_observation_weights:
            ll_part = dispersion_neg2.squeeze() * np.sum(
                observation_weights
                * (
                    digamma(dispersion_neg1)
                    + np.log(1 + dispersion_times_mu)
                    - dispersion_times_mu / (1.0 + dispersion_times_mu)
                    - digamma(gene_expression_counts + dispersion_neg1)
                    + gene_expression_counts / (expected_expression_means + dispersion_neg1)
                ),
                axis=0,
            )
        else:
            ll_part = dispersion_neg2.squeeze() * np.sum(
                digamma(dispersion_neg1)
                + np.log(1 + dispersion_times_mu)
                - dispersion_times_mu / (1.0 + dispersion_times_mu)
                - digamma(gene_expression_counts + dispersion_neg1)
                + gene_expression_counts / (expected_expression_means + dispersion_neg1),
                axis=0,
            )

        # only the prior part is wrt log dispersion_parameters
        if use_prior_regularization:
            prior_part = -1.0 * (log_dispersion_parameters - prior_mean_log_dispersion) / prior_variance_log_dispersion
        else:
            prior_part = 0.0

        # note: return dlog_post / ddispersion_values * dispersion_values because we take derivatives wrt log dispersion_parameters
        return (ll_part + cr_term) * dispersion_values.squeeze() + prior_part
        """
        Calculate the derivative of log posterior with respect to log dispersion parameters.

        Parameters
        ----------
        log_dispersion_parameters : np.ndarray
            Log of dispersion parameters for each gene
        gene_expression_counts : np.ndarray
            Gene expression count data
        expected_expression_means : np.ndarray
            Expected mean expression values
        experimental_design_matrix : np.ndarray
            Experimental design matrix
        prior_mean_log_dispersion : np.ndarray
            Prior mean for log dispersion parameters
        prior_variance_log_dispersion : float
            Prior variance for log dispersion parameters
        use_prior_regularization : bool
            Whether to use prior regularization
        observation_weights : np.ndarray, optional
            Weights for individual observations
        use_observation_weights : bool
            Whether to incorporate observation weights
        minimum_weight_threshold : float
            Weight threshold for calculations
        use_cox_reid_adjustment : bool
            Whether to use Cox-Reid adjustment

        Returns
        -------
        np.ndarray
            Derivative values for each gene
        """
        # Ensure arrays are properly shaped
        if not isinstance(log_dispersion_parameters, np.ndarray):
            log_dispersion_parameters = np.array([log_dispersion_parameters])
        if len(gene_expression_counts.shape) == 1:
            gene_expression_counts = gene_expression_counts[:, None]
        if len(expected_expression_means.shape) == 1:
            expected_expression_means = expected_expression_means[:, None]
        if observation_weights is not None and len(observation_weights.shape) == 1:
            observation_weights = observation_weights[:, None]

        # Broadcast arrays
        log_dispersion_parameters, gene_expression_counts, expected_expression_means = np.broadcast_arrays(
            log_dispersion_parameters,
            gene_expression_counts,
            expected_expression_means,
        )

        # Convert log dispersion parameters to dispersion values
        dispersion_values = np.exp(log_dispersion_parameters)

        # Calculate derivative of log likelihood
        derivative_log_likelihood = np.zeros_like(log_dispersion_parameters)

        for gene_idx in range(len(log_dispersion_parameters)):
            # Calculate size parameter
            size_parameter = 1.0 / dispersion_values[gene_idx]

            # Calculate derivative components
            digamma_term = digamma(gene_expression_counts[:, gene_idx] + size_parameter) - digamma(size_parameter)
            log_term = (
                np.log(size_parameter)
                + 1
                - np.log(size_parameter + expected_expression_means[:, gene_idx])
                - (size_parameter + gene_expression_counts[:, gene_idx])
                / (size_parameter + expected_expression_means[:, gene_idx])
            )

            derivative_log_likelihood[gene_idx] = np.sum(digamma_term + log_term)

        # Add prior derivative if requested
        if use_prior_regularization:
            prior_derivative = -(log_dispersion_parameters - prior_mean_log_dispersion) / prior_variance_log_dispersion
            derivative_log_likelihood += prior_derivative

        return derivative_log_likelihood

    @staticmethod
    def estimate_rough_dispersions(normalized_counts: np.ndarray, design_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate rough dispersion estimates using QR decomposition for numerical stability.

        Parameters
        ----------
        normalized_counts : np.ndarray
            Normalized gene expression count data
        design_matrix : np.ndarray
            Experimental design matrix

        Returns
        -------
        np.ndarray
            Rough dispersion estimates for each gene
        """
        n_samples, n_genes = normalized_counts.shape
        n_params = design_matrix.shape[1]

        # Calculate fitted values using QR decomposition for numerical stability
        try:
            # Use QR decomposition for numerical stability
            Q, R = np.linalg.qr(design_matrix)
            R_inverse = np.linalg.solve(R, np.identity(R.shape[0]))
            fitted_values = (design_matrix @ R_inverse) @ (Q.T @ normalized_counts)
        except np.linalg.LinAlgError:
            # Fallback to mean
            fitted_values = np.mean(normalized_counts, axis=0, keepdims=True)

        # Clip fitted values to be positive
        fitted_values = np.clip(fitted_values, 1, None)

        # Calculate dispersion estimates using method of moments with degrees of freedom correction
        # Formula: sum(((y - mu)^2 - mu) / mu^2) / (n_samples - n_params)
        dispersion_estimates = np.sum(
            ((normalized_counts - fitted_values) ** 2 - fitted_values) / fitted_values**2,
            0,
        ) / (n_samples - n_params)

        return np.clip(dispersion_estimates, 0, None)

    @staticmethod
    def estimate_dispersions_by_moments(expression_data: "AnnData", size_factors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate dispersion estimates using method of moments (inmoose-style).

        Parameters
        ----------
        expression_data : AnnData
            Annotated gene expression data matrix
        size_factors : np.ndarray, optional
            Size factors for normalization. If None, will use ones.

        Returns
        -------
        np.ndarray
            Dispersion estimates for each gene
        """
        gene_expression_counts = expression_data.X.toarray() if hasattr(expression_data.X, "toarray") else expression_data.X

        # Use size factors if provided, otherwise get from expression data
        if size_factors is None:
            if hasattr(expression_data, "size_factors") and expression_data.size_factors is not None:
                size_factors = expression_data.size_factors
            else:
                size_factors = np.ones(expression_data.n_obs)

        # Convert to numpy array if it's a pandas Series
        if hasattr(size_factors, "to_numpy"):
            size_factors = size_factors.to_numpy()

        # Normalize counts using size factors (like inmoose)
        # Reshape size_factors to (n_samples, 1) for broadcasting
        sf_matrix = size_factors.reshape(-1, 1)
        gene_expression_counts / sf_matrix

        # Calculate baseMean and baseVar (like inmoose)
        base_mean = expression_data.var["base_mean"]
        base_var = expression_data.var["base_variance"]
        # base_mean = np.mean(normalized_counts, axis=0)
        # base_var = np.var(normalized_counts, axis=0, ddof=1)  # ddof=1 like inmoose

        # Calculate xim (like inmoose)
        xim = np.mean(1 / size_factors)

        # Inmoose formula: (baseVar - xim * baseMean) / baseMean^2
        dispersion_estimates = np.zeros(len(base_mean))
        for gene_idx in range(len(base_mean)):
            if base_mean.iloc[gene_idx] > 0:
                dispersion_estimates[gene_idx] = (base_var.iloc[gene_idx] - xim * base_mean.iloc[gene_idx]) / (
                    base_mean.iloc[gene_idx] ** 2
                )
            else:
                # For genes with zero mean, use a small positive value to avoid division by zero
                # This matches inmoose's behavior where they can return negative values for other genes
                dispersion_estimates[gene_idx] = 0.1  # Default value

        return dispersion_estimates

    @staticmethod
    def fit_parametric_dispersion_trend(gene_expression_means: np.ndarray, gene_dispersion_estimates: np.ndarray) -> callable:
        """
        Fit parametric dispersion-mean relationship using inmoose's exact approach.

        This implements the same algorithm as inmoose's parametricDispersionFit:
        1. Uses Gamma GLM with identity link
        2. Iterative fitting with outlier removal
        3. Model: dispersion = asymptDisp + extraPois / mean

        Parameters
        ----------
        gene_expression_means : np.ndarray
            Base mean expression values for each gene
        gene_dispersion_estimates : np.ndarray
            Gene-wise dispersion estimates

        Returns
        -------
        callable
            Function that predicts dispersion from mean expression
        """
        # Filter out invalid values (same as inmoose)
        valid_mask = (
            (gene_dispersion_estimates > 1e-8)
            & (gene_expression_means > 0)
            & np.isfinite(gene_dispersion_estimates)
            & np.isfinite(gene_expression_means)
        )
        means = gene_expression_means[valid_mask]
        disps = gene_dispersion_estimates[valid_mask]

        if len(means) == 0:
            # Fallback to constant dispersion
            average_dispersion = np.mean(gene_dispersion_estimates) if len(gene_dispersion_estimates) > 0 else 0.1

            def dispersion_prediction_function(expression_means):
                return np.full_like(expression_means, average_dispersion)

            dispersion_prediction_function.asymptotic_dispersion = average_dispersion
            dispersion_prediction_function.extra_poisson_variance = 0.0
            return dispersion_prediction_function

        # Inmoose's exact implementation
        try:
            import statsmodels.api as sm
            from statsmodels.tools.sm_exceptions import DomainWarning
            import warnings

            # Suppress domain warnings (same as inmoose)
            warnings.simplefilter("ignore", DomainWarning)

            # Initial coefficients (same as inmoose)
            coefs = pd.Series([0.1, 1.0])
            iter_ = 0

            while True:
                # Calculate residuals
                residuals = disps / (coefs.iloc[0] + coefs.iloc[1] / means)

                # Outlier removal (same as inmoose)
                good = (residuals > 1e-4) & (residuals < 15)

                if np.sum(good) < 10:  # Need minimum number of points
                    break

                # Gamma GLM with identity link (same as inmoose)
                glm_gamma = sm.GLM(
                    disps[good],
                    sm.add_constant(1 / means[good]),
                    family=sm.families.Gamma(link=sm.families.links.Identity()),
                )

                fit = glm_gamma.fit(start_params=coefs)
                oldcoefs = coefs.copy()
                coefs = fit.params

                # Check for valid coefficients (same as inmoose)
                if not np.all(coefs > 0):
                    raise RuntimeError("parametric dispersion fit failed")

                # Check convergence (same as inmoose)
                if np.sum(np.log(coefs / oldcoefs.values) ** 2) < 1e-6 and fit.converged:
                    break

                iter_ += 1
                if iter_ > 10:
                    raise RuntimeError("dispersion fit did not converge")

            # Set coefficient names (same as inmoose)
            coefs.index = ["asymptDisp", "extraPois"]
            asympt_disp = coefs.iloc[0]
            extra_pois = coefs.iloc[1]

        except Exception:
            # Fallback to simple mean if GLM fails
            asympt_disp = np.mean(disps)
            extra_pois = 0.0

        def dispersion_prediction_function(expression_means):
            """Predict dispersion from mean expression (same as inmoose)."""
            return asympt_disp + extra_pois / np.maximum(expression_means, 1e-8)

        # Store parameters (same as inmoose)
        dispersion_prediction_function.asymptotic_dispersion = asympt_disp
        dispersion_prediction_function.extra_poisson_variance = extra_pois
        dispersion_prediction_function.coefficients = pd.Series([asympt_disp, extra_pois], index=["asymptDisp", "extraPois"])

        return dispersion_prediction_function

    @staticmethod
    def has_sufficient_experimental_replicates(expression_data: "AnnData", experimental_design_matrix: np.ndarray) -> bool:
        """
        Check if there are sufficient experimental replicates for dispersion estimation.

        Parameters
        ----------
        expression_data : AnnData
            Annotated gene expression data matrix
        experimental_design_matrix : np.ndarray
            Experimental design matrix

        Returns
        -------
        bool
            True if sufficient experimental replicates exist
        """
        number_of_samples = expression_data.n_obs
        number_of_design_coefficients = experimental_design_matrix.shape[1]

        return number_of_samples > number_of_design_coefficients

    @staticmethod
    def get_and_check_weights(data: "AnnData", design_matrix: np.ndarray, weight_threshold: float = 1e-2) -> tuple:
        """
        Check and retrieve weights for dispersion estimation.

        If weights exist in data.layers, they are validated and normalized.
        Otherwise, returns all ones.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix
        design_matrix : np.ndarray
            Design matrix for the analysis
        weight_threshold : float
            Threshold below which weights may be considered zero

        Returns
        -------
        tuple
            (data, weights, use_weights) where:
            - data: Updated AnnData object
            - weights: Weight matrix (same shape as data)
            - use_weights: Boolean indicating if weights are used
        """
        if "weights" in data.layers:
            use_weights = True
            weights = data.layers["weights"]

            # Validate weights are positive
            if not (weights >= 0).all():
                raise ValueError("weights must be positive")

            # Normalize weights by column maximum
            weights = weights / np.max(weights, 0)

            # Check if design matrix remains full rank with weights
            m = design_matrix.shape[1]
            full_rank = np.linalg.matrix_rank(design_matrix) == m

            if full_rank:
                # For full rank designs, check each gene's weights
                weights_ok = np.repeat(False, weights.shape[1])
                for i in range(weights.shape[1]):
                    # Test if weighted design matrix remains full rank
                    test1 = np.linalg.matrix_rank(weights[:, i][:, None] * design_matrix) == m

                    # Test if subsetting based on weight threshold maintains rank
                    mm_sub = design_matrix[weights[:, i] > weight_threshold]
                    mm_sub = mm_sub[:, np.sum(np.abs(mm_sub), 0) > 0]
                    test2 = np.linalg.matrix_rank(mm_sub) == mm_sub.shape[1]
                    weights_ok[i] = test1 and test2

                # Mark problematic genes as all_zero
                if not weights_ok.all():
                    data.var.loc[~weights_ok, "all_zero"] = True
                    data.var["weights_fail"] = ~weights_ok
                    logger.warning(
                        "%s genes have weights that will not allow parameter estimation",
                        np.sum(~weights_ok),
                    )

            # Clip weights to prevent numerical issues
            weights = np.clip(weights, 1e-6, None)

        else:
            use_weights = False
            weights = np.ones(data.shape)

        return data, weights, use_weights

    @staticmethod
    def fit_dispersion_line_search(
        y,
        x,
        mu_hat,
        log_alpha,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        min_log_alpha,
        kappa_0,
        tol,
        maxit,
        usePrior,
        weights,
        useWeights,
        weightThreshold,
        useCR,
    ):
        """
        Exact inmoose fitDisp implementation.
        """
        if isinstance(log_alpha, (int, float)):
            log_alpha = np.repeat(float(log_alpha), y.shape[1])
        if isinstance(log_alpha_prior_mean, (int, float)):
            log_alpha_prior_mean = np.repeat(float(log_alpha_prior_mean), y.shape[1])
        assert y.shape[1] == mu_hat.shape[1]
        assert y.shape[1] == log_alpha.shape[0]
        assert y.shape[1] == log_alpha_prior_mean.shape[0]
        assert y.shape == weights.shape

        y_n = y.shape[1]
        epsilon = 1.0e-4
        iter_ = np.zeros(y_n)
        iter_accept = np.zeros(y_n)

        # maximize the log likelihood over the variable a, the log of alpha, the
        # dispersion parameter.
        # in order to express the optimization in a typical manner, for calculating
        # theta(kappa) we multiply the log likelihood by -1 and seek a minimum
        # we use a line search based on the Armijo rule.
        # define a function theta(kappa) = f(a + kappa * d) where d is the search
        # direction
        # in this case the search direction is given by the first derivative of the
        # log likelihood
        a = log_alpha.copy()
        lp = DispersionEstimationUtils.calculate_dispersion_log_posterior(
            a,
            y,
            mu_hat,
            x,
            log_alpha_prior_mean,
            log_alpha_prior_sigmasq,
            usePrior,
            weights,
            useWeights,
            weightThreshold,
            useCR,
        )
        dlp = DispersionEstimationUtils.calculate_dispersion_log_posterior_derivative(
            a,
            y,
            mu_hat,
            x,
            log_alpha_prior_mean,
            log_alpha_prior_sigmasq,
            usePrior,
            weights,
            useWeights,
            weightThreshold,
            useCR,
        )

        kappa = np.repeat(kappa_0, y_n)
        initial_lp = lp.copy()
        initial_dlp = dlp.copy()
        change = np.repeat(-1.0, y_n)
        last_change = np.repeat(-1.0, y_n)

        idx = np.repeat(True, y_n)
        for t in range(maxit):
            # iter_ counts the number of steps taken out of maxit
            iter_[idx] += 1
            a_propose = a + kappa * dlp
            # note: lgamma is unstable for values around 1e17, where there is a
            # switch in lgamma.c
            # we limit log alpha from going lower than -30 (like inmoose)
            kappa = np.where(a_propose < -30.0, (-30.0 - a) / dlp, kappa)
            # we limit log alpha from going higher than 10
            kappa = np.where(a_propose > 10.0, (10.0 - a) / dlp, kappa)

            lpost = DispersionEstimationUtils.calculate_dispersion_log_posterior(
                a[idx] + kappa[idx] * dlp[idx],
                y[:, idx],
                mu_hat[:, idx],
                x,
                log_alpha_prior_mean[idx],
                log_alpha_prior_sigmasq,
                usePrior,
                weights[:, idx],
                useWeights,
                weightThreshold,
                useCR,
            )
            theta_kappa = np.zeros(y_n)
            theta_kappa[idx] = -lpost
            theta_hat_kappa = -lp - kappa * epsilon * np.power(dlp, 2)

            # if this inequality is true, we have satisfied the Armijo rule and
            # accept the step size kappa, otherwise we halve kappa
            armijo_idx = idx & (theta_kappa <= theta_hat_kappa)
            # iter_accept counts the number of accepted proposals
            iter_accept[armijo_idx] += 1
            a[armijo_idx] = (a + kappa * dlp)[armijo_idx]
            lpnew = np.zeros(y_n)
            lpnew[idx] = lpost
            # look for change in log likelihood
            change[armijo_idx] = lpnew[armijo_idx] - lp[armijo_idx]
            lp = np.where(armijo_idx & (change < tol), lpnew, lp)
            idx[armijo_idx & (change < tol)] = False
            # if log(alpha) is going to -infinity, break the loop
            idx[armijo_idx & (a < min_log_alpha)] = False

            idx2 = idx & armijo_idx
            lp[idx2] = lpnew[idx2]
            dlp[idx2] = DispersionEstimationUtils.calculate_dispersion_log_posterior_derivative(
                a[idx2],
                y[:, idx2],
                mu_hat[:, idx2],
                x,
                log_alpha_prior_mean[idx2],
                log_alpha_prior_sigmasq,
                usePrior,
                weights[:, idx2],
                useWeights,
                weightThreshold,
                useCR,
            )
            # instead of resetting kappa to kappa_0
            # multiply kappa by 1.1
            kappa[idx2] = np.minimum(kappa[idx2] * 1.1, kappa_0)
            # every 5 accepts, halve kappa
            # to prevent slow convergence due to overshooting
            kappa[idx2 & (iter_accept % 5 == 0)] /= 2.0

            kappa[~armijo_idx] /= 2.0

        last_lp = lp
        last_dlp = dlp
        log_alpha = a
        # last change indicates the change for the final iteration
        last_change = change

        return {
            "log_alpha": log_alpha,
            "iter": iter_,
            "iter_accept": iter_accept,
            "last_change": last_change,
            "initial_lp": initial_lp,
            "initial_dlp": initial_dlp,
            "last_lp": last_lp,
            "last_dlp": last_dlp,
        }
