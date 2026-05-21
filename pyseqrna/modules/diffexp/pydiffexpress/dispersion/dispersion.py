"""
Pipeline dispersion estimation.

This module implements a complete dispersion estimation pipeline that combines
gene-wise estimation, trend fitting, and MAP estimation.

Classes:
    - DispersionEstimator: Complete dispersion estimation pipeline

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Optional
import logging
import numpy as np
from anndata import AnnData

from .base import BaseDispersionEstimator
from .gene_wise import GeneWiseDispersionEstimator
from .trend import TrendDispersionEstimator
from .map import MAPDispersionEstimator

logger = logging.getLogger(__name__)


class DispersionEstimator(BaseDispersionEstimator):
    """
    Complete dispersion estimation pipeline.

    This estimator implements the full dispersion estimation workflow:
    1. Gene-wise dispersion estimation
    2. Trend fitting
    3. Maximum a posteriori (MAP) estimation
    """

    def __init__(
        self,
        fit_type: str = "parametric",
        min_disp: float = 1e-8,
        kappa_0: float = 1.0,
        disp_tol: float = 1e-6,
        max_iter: int = 100,
        use_cox_reid_adjustment: bool = True,
        weight_threshold: float = 1e-2,
        quiet: bool = False,
        n_iter: int = 3,
        linear_mu: Optional[bool] = None,
        min_mu: Optional[float] = None,
        outlier_sd: float = 2.0,
        disp_prior_var: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the pipeline dispersion estimator.

        Parameters
        ----------
        fit_type : str
            Type of trend fitting: "parametric", "local", "mean", or "glmGamPoi"
        min_disp : float
            Minimum dispersion value for numerical stability
        kappa_0 : float
            Parameter for backtracking search
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
        outlier_sd : float
            Standard deviations for outlier detection
        disp_prior_var : float, optional
            Prior variance for dispersion estimates
        **kwargs
            Additional parameters passed to individual estimators
        """
        super().__init__(
            fit_type=fit_type,
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
            outlier_sd=outlier_sd,
            disp_prior_var=disp_prior_var,
            **kwargs,
        )

        # Create individual estimators
        self.gene_wise_estimator = GeneWiseDispersionEstimator(
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
            **kwargs,
        )

        self.trend_estimator = TrendDispersionEstimator(fit_type=fit_type, min_disp=min_disp, quiet=quiet, **kwargs)

        self.map_estimator = MAPDispersionEstimator(
            outlier_sd=outlier_sd,
            disp_prior_var=disp_prior_var,
            min_disp=min_disp,
            kappa_0=kappa_0,
            disp_tol=disp_tol,
            max_iter=max_iter,
            use_cox_reid_adjustment=use_cox_reid_adjustment,
            weight_threshold=weight_threshold,
            quiet=quiet,
            **kwargs,
        )

    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "DispersionEstimator":
        """
        Fit the complete dispersion estimation pipeline.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix with counts in .X
        design_matrix : np.ndarray, optional
            Design matrix for experimental design

        Returns
        -------
        self : DispersionEstimator
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

        # Add categorical columns to obs if they don't exist
        # This mimics inmoose behavior where C(condition) columns are added
        if "condition" in data.obs.columns and "C(condition)" not in data.obs.columns:
            data.obs["C(condition)"] = data.obs["condition"]

        # Step 1: Gene-wise dispersion estimation
        if not self.parameters["quiet"]:
            logger.info("Step 1: Estimating gene-wise dispersions...")

        self.gene_wise_estimator.fit(data, design_matrix)
        self.gene_wise_estimator.update_data(data)

        # Step 2: Trend fitting
        if not self.parameters["quiet"]:
            logger.info("Step 2: Fitting dispersion trend...")

        self.trend_estimator.fit(data, design_matrix)
        self.trend_estimator.update_data(data)

        # Step 3: MAP estimation
        if not self.parameters["quiet"]:
            logger.info("Step 3: Computing MAP dispersions...")

        self.map_estimator.fit(data, design_matrix)
        self.map_estimator.update_data(data)

        # Store final results
        self._results = {
            "dispersions": data.var["dispersion"],
            "disp_gene_est": data.var["disp_gene_est"],
            "disp_fitted": data.var["disp_fit"],
            "outliers": self.map_estimator.get_outliers(),
            "disp_prior_var": self.map_estimator.get_prior_variance(),
            "disp_function": data.uns.get("disp_function"),
        }

        self._fitted = True
        return self

    def estimate(self, data: AnnData) -> np.ndarray:
        """
        Estimate final MAP dispersions for the given data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            Final MAP dispersion estimates
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before estimating")

        return self._results["dispersions"]

    def get_gene_wise_estimates(self) -> np.ndarray:
        """Get gene-wise dispersion estimates."""
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting estimates")
        return self._results["disp_gene_est"]

    def get_fitted_estimates(self) -> np.ndarray:
        """Get fitted dispersion estimates."""
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting estimates")
        return self._results["disp_fitted"]

    def get_outliers(self) -> np.ndarray:
        """Get outlier genes."""
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting outliers")
        return self._results["outliers"]

    def get_dispersion_function(self):
        """Get the fitted dispersion function."""
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting function")
        return self._results["disp_function"]
