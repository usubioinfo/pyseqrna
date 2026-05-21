"""
Trend dispersion estimation.

This module implements trend fitting for dispersion estimates, which models
the relationship between dispersion and mean expression levels.

Classes:
    - TrendDispersionEstimator: Trend dispersion estimator that fits dispersion-mean relationships

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Callable
from anndata import AnnData

from .base import BaseDispersionEstimator
from .utils import DispersionEstimationUtils

logger = logging.getLogger(__name__)


class TrendDispersionEstimator(BaseDispersionEstimator):
    """
    Trend dispersion estimator that fits dispersion-mean relationships.

    This estimator takes gene-wise dispersion estimates and fits a trend
    to model the relationship between dispersion and mean expression levels.
    """

    def __init__(
        self,
        fit_type: str = "parametric",
        min_disp: float = 1e-8,
        quiet: bool = False,
    ):
        """
        Initialize the trend dispersion estimator.

        Parameters
        ----------
        fit_type : str
            Type of trend fitting: "parametric", "local", "mean", or "glmGamPoi"
        min_disp : float
            Minimum dispersion value for numerical stability
        quiet : bool
            Whether to suppress progress messages
        """
        super().__init__(
            fit_type=fit_type,
            min_disp=min_disp,
            quiet=quiet,
        )

        # Validate fit_type
        valid_types = ["parametric", "local", "mean", "glmGamPoi"]
        if fit_type not in valid_types:
            raise ValueError(f"fit_type must be one of {valid_types}")

        self.fit_type = fit_type

    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "TrendDispersionEstimator":
        """
        Fit the trend dispersion estimator to the data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix with gene-wise dispersion estimates in .var['disp_gene_est']
        design_matrix : np.ndarray, optional
            Design matrix (not used for trend fitting)

        Returns
        -------
        self : TrendDispersionEstimator
            Fitted estimator
        """
        self._validate_data(data)

        # Get parameters
        fit_type = self.parameters["fit_type"]
        min_disp = self.parameters["min_disp"]
        quiet = self.parameters["quiet"]

        # Check for gene-wise dispersion estimates
        if "disp_gene_est" not in data.var:
            raise ValueError("Gene-wise dispersion estimates not found. Run GeneWiseDispersionEstimator first.")

        # Check for base means
        if "base_mean" not in data.var:
            raise ValueError("Base means not found. Run normalization or gene-wise dispersion estimation first.")

        # Get non-zero genes
        all_zero = data.var.get("all_zero", data.var["base_mean"] == 0)
        data_nz = data[:, ~all_zero]

        # Get gene-wise estimates and base means
        disp_gene_est = data_nz.var["disp_gene_est"]
        base_means = data_nz.var["base_mean"]

        # Filter genes for fitting
        use_for_fit = disp_gene_est > 100 * min_disp

        if np.sum(use_for_fit) == 0:
            raise ValueError(
                "All gene-wise dispersion estimates are within 2 orders of magnitude "
                "from the minimum value. Standard curve fitting techniques will not work. "
                "Consider using gene-wise estimates directly or adjusting min_disp."
            )

        # Fit trend based on type
        if fit_type == "parametric":
            try:
                disp_function = DispersionEstimationUtils.fit_parametric_dispersion_trend(
                    base_means[use_for_fit], disp_gene_est[use_for_fit]
                )
            except Exception as e:
                if not quiet:
                    logger.warning("Parametric fit failed: %s", e)
                    logger.info("Falling back to mean dispersion")
                fit_type = "mean"
                disp_function = self._fit_mean_dispersion(disp_gene_est[use_for_fit])

        elif fit_type == "local":
            if not quiet:
                logger.info("Local regression not implemented, using parametric fit")
            fit_type = "parametric"
            disp_function = DispersionEstimationUtils.fit_parametric_dispersion_trend(
                base_means[use_for_fit], disp_gene_est[use_for_fit]
            )

        elif fit_type == "mean":
            disp_function = self._fit_mean_dispersion(disp_gene_est[use_for_fit])

        elif fit_type == "glmGamPoi":
            if not quiet:
                logger.info("glmGamPoi not implemented, using parametric fit")
            fit_type = "parametric"
            disp_function = DispersionEstimationUtils.fit_parametric_dispersion_trend(
                base_means[use_for_fit], disp_gene_est[use_for_fit]
            )

        # Store dispersion function attributes
        disp_function.fit_type = fit_type

        # Calculate fitted dispersions for all genes
        fitted_disps = disp_function(base_means)
        fitted_disps = np.clip(fitted_disps, min_disp, None)

        # Store results
        self._results = {
            "disp_function": disp_function,
            "disp_fitted": self._build_vector_with_na_cols(fitted_disps, all_zero),
            "fit_type": fit_type,
            "use_for_fit": self._build_vector_with_na_cols(use_for_fit, all_zero),
        }

        self._fitted = True
        return self

    def estimate(self, data: AnnData) -> np.ndarray:
        """
        Estimate fitted dispersions for the given data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            Fitted dispersion estimates
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before estimating")

        return self._results["disp_fitted"]

    def predict(self, means: np.ndarray) -> np.ndarray:
        """
        Predict dispersions for given mean values.

        Parameters
        ----------
        means : np.ndarray
            Mean expression values

        Returns
        -------
        np.ndarray
            Predicted dispersion values
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before predicting")

        disp_function = self._results["disp_function"]
        return disp_function(means)

    def _fit_mean_dispersion(self, disp_est: np.ndarray) -> Callable:
        """
        Fit constant dispersion (mean of gene-wise estimates).

        Parameters
        ----------
        disp_est : np.ndarray
            Gene-wise dispersion estimates

        Returns
        -------
        callable
            Function that returns constant dispersion
        """
        # Use trimmed mean for robustness
        from scipy.stats import trim_mean

        mean_disp = trim_mean(disp_est, 0.001)

        def disp_function(means):
            """Return constant dispersion."""
            return np.full_like(means, mean_disp)

        # Store parameters
        disp_function.mean = mean_disp
        disp_function.asympt_disp = mean_disp
        disp_function.extra_pois = 0.0

        return disp_function

    def _build_vector_with_na_cols(self, values: np.ndarray, all_zero: np.ndarray) -> np.ndarray:
        """Build vector with NA values for zero genes."""
        result = np.full(len(all_zero), np.nan)
        result[~all_zero] = values
        return result

    def get_dispersion_function(self) -> Callable:
        """
        Get the fitted dispersion function.

        Returns
        -------
        callable
            Function that predicts dispersion from mean
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting dispersion function")

        return self._results["disp_function"]

    def get_fit_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the fitted trend.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing fit parameters
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting parameters")

        disp_function = self._results["disp_function"]
        fit_type = self._results["fit_type"]

        params = {"fit_type": fit_type}

        if fit_type == "parametric":
            params.update(
                {
                    "asympt_disp": getattr(disp_function, "asympt_disp", None),
                    "extra_pois": getattr(disp_function, "extra_pois", None),
                }
            )
        elif fit_type == "mean":
            params.update(
                {
                    "mean_disp": getattr(disp_function, "mean", None),
                }
            )

        return params
