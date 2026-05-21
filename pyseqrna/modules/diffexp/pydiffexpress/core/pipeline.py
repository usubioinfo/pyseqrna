#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline analyzer for complete differential expression analysis.

This module provides a pipeline analyzer that orchestrates the complete
differential expression analysis workflow.

This is part of the native pydiffexpress differential-expression implementation.

Classes:
    - DiffExpressAnalyzer: Differential expression analyzer for complete analysis pipeline

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import logging
import numpy as np
from anndata import AnnData

from .base import BaseAnalyzer

if TYPE_CHECKING:
    from ..datasets.dataset import ExpressionDataset

logger = logging.getLogger(__name__)


class DiffExpressAnalyzer(BaseAnalyzer):
    """
    Differential expression analyzer for complete analysis pipeline.

    This class orchestrates the complete differential expression analysis
    pipeline, including size factor estimation, dispersion estimation,
    and statistical testing.
    """

    def __init__(
        self,
        test: str = "Wald",
        fit_type: str = "parametric",
        sf_type: str = "ratio",
        quiet: bool = False,
        min_mu: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the differential expression analyzer.

        Parameters
        ----------
        test : str, optional
            Statistical test to use ("Wald" or "LRT")
        fit_type : str, optional
            Type of dispersion fitting ("parametric", "local", "mean", "glmGamPoi")
        sf_type : str, optional
            Type of size factor estimation ("ratio", "poscounts", "iterate")
        quiet : bool, optional
            Whether to suppress progress messages
        min_mu : float, optional
            Minimum value for fitted means
        **kwargs
            Additional parameters passed to individual analyzers
        """
        super().__init__(name="DiffExpressAnalyzer", **kwargs)

        # Initialize parameters dictionary
        self.parameters = {}

        # Validate parameters
        if test not in ["Wald", "LRT"]:
            raise ValueError("test must be either 'Wald' or 'LRT'")

        if fit_type not in ["parametric", "local", "mean", "glmGamPoi"]:
            raise ValueError(f"invalid fit_type: {fit_type}")

        if sf_type not in ["ratio", "poscounts", "iterate"]:
            raise ValueError(f"invalid sf_type: {sf_type}")

        # Set default min_mu based on fit_type
        if min_mu is None:
            if fit_type == "glmGamPoi":
                min_mu = 1e-6
            else:
                min_mu = 0.5

        self.parameters.update(
            {
                "test": test,
                "fit_type": fit_type,
                "sf_type": sf_type,
                "quiet": quiet,
                "min_mu": min_mu,
            }
        )

        # Initialize sub-analyzers
        self._wald_analyzer = None
        self._lrt_analyzer = None

    def fit(
        self,
        data: Union[AnnData, "ExpressionDataset"],
        design_matrix: Optional[np.ndarray] = None,
    ) -> "DiffExpressAnalyzer":
        """
        Run the complete differential expression analysis pipeline.

        Parameters
        ----------
        data : Union[AnnData, ExpressionDataset]
            Annotated data matrix with counts in .X or ExpressionDataset
        design_matrix : np.ndarray, optional
            Design matrix for experimental design

        Returns
        -------
        self : DiffExpressAnalyzer
            Fitted analyzer
        """
        # Extract parameters
        test = self.parameters["test"]
        fit_type = self.parameters["fit_type"]
        sf_type = self.parameters["sf_type"]
        quiet = self.parameters["quiet"]
        min_mu = self.parameters["min_mu"]

        # Handle both AnnData and ExpressionDataset
        if hasattr(data, "counts"):
            # ExpressionDataset object
            dataset = data
            adata = data._adata
        else:
            # AnnData object
            dataset = None
            adata = data

        # Step 1: Estimate size factors (if not already done)
        if "size_factors" not in adata.obs:
            if not quiet:
                logger.info("Estimating size factors...")

            # Import here to avoid circular imports
            from ..normalization import MedianRatioNormalizer

            normalizer = MedianRatioNormalizer()
            if dataset is not None:
                normalizer.fit(dataset)
            else:
                # Create a temporary ExpressionDataset for normalization
                from ..datasets.dataset import ExpressionDataset

                temp_dataset = ExpressionDataset(counts=adata.X, sample_metadata=adata.obs, gene_metadata=adata.var)
                normalizer.fit(temp_dataset)
                # Copy results back to AnnData
                adata.obs["size_factors"] = temp_dataset.size_factors
                if "base_mean" in temp_dataset.var.columns:
                    adata.var["base_mean"] = temp_dataset.var["base_mean"]
                if "base_variance" in temp_dataset.var.columns:
                    adata.var["base_variance"] = temp_dataset.var["base_variance"]
                if "normalized_counts" in temp_dataset.layers:
                    adata.layers["normalized_counts"] = temp_dataset.layers["normalized_counts"]

        # Step 2: Estimate dispersions (if not already done)
        if "dispersion" not in adata.var:
            if not quiet:
                logger.info("Estimating dispersions...")

            # Import here to avoid circular imports
            from ..dispersion import DispersionEstimator

            dispersion_estimator = DispersionEstimator(
                fit_type=fit_type,
                quiet=quiet,
                min_mu=min_mu,
                **{
                    k: v
                    for k, v in self.parameters.items()
                    if k
                    in [
                        "min_disp",
                        "kappa_0",
                        "disp_tol",
                        "max_iter",
                        "use_cox_reid_adjustment",
                        "weight_threshold",
                    ]
                },
            )
            dispersion_estimator.fit(adata, design_matrix)
            dispersion_estimator.update_data(adata)

        # Step 3: Ensure design matrix exists
        if "design" not in adata.obsm:
            if design_matrix is not None:
                adata.obsm["design"] = design_matrix
            else:
                # Create default design matrix
                from ..utils import create_design_matrix

                design_matrix = create_design_matrix(adata)
                adata.obsm["design"] = design_matrix

        # Step 4: Run statistical test
        if not quiet:
            logger.info("Running %s test...", test)

        if test == "Wald":
            # Run Wald test
            from ..hypothesis_testing import WaldTestAnalyzer

            self._wald_analyzer = WaldTestAnalyzer(
                quiet=quiet,
                min_mu=min_mu,
                **{k: v for k, v in self.parameters.items() if k in ["beta_tol", "max_iter"]},
            )
            self._wald_analyzer.fit(adata, design_matrix)
            self._wald_analyzer.update_data(adata)

            # Store results
            self.results = self._wald_analyzer.get_results()

        elif test == "LRT":
            # Run LRT test
            if not quiet:
                logger.info("Running Likelihood Ratio Test (LRT)...")

            # Initialize LRT analyzer
            from ..hypothesis_testing import LRTAnalyzer

            self._lrt_analyzer = LRTAnalyzer(
                reduced_formula="~1",  # Default to intercept-only model
                quiet=quiet,
                **{k: v for k, v in self.parameters.items() if k in ["beta_tol", "max_iter"]},
            )
            self._lrt_analyzer.fit(adata, design_matrix)
            self._lrt_analyzer.update_data(adata)

            # Store results
            self.results = self._lrt_analyzer.get_results()

        # Add metadata
        if "metadata" in self.results:
            self.results["metadata"].update(
                {
                    "test": test,
                    "fit_type": fit_type,
                    "sf_type": sf_type,
                    "min_mu": min_mu,
                }
            )

        self._fitted = True
        return self

    def get_wald_results(self) -> Optional[Dict[str, Any]]:
        """
        Get Wald test results if available.

        Returns
        -------
        Optional[Dict[str, Any]]
            Wald test results or None if not available
        """
        if self._wald_analyzer is not None:
            return self._wald_analyzer.get_results()
        return None

    def get_lrt_results(self) -> Optional[Dict[str, Any]]:
        """
        Get LRT test results if available.

        Returns
        -------
        Optional[Dict[str, Any]]
            LRT test results or None if not available
        """
        if self._lrt_analyzer is not None:
            return self._lrt_analyzer.get_results()
        return None

    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get coefficient estimates.

        Returns
        -------
        Optional[np.ndarray]
            Coefficient estimates or None if not available
        """
        if not self._fitted:
            raise ValueError("Analyzer must be fitted before getting coefficients")

        if self.parameters["test"] == "Wald" and self._wald_analyzer is not None:
            return self.results["beta_coefficients"]
        return None

    def get_p_values(self) -> Optional[np.ndarray]:
        """
        Get p-values.

        Returns
        -------
        Optional[np.ndarray]
            P-values or None if not available
        """
        if not self._fitted:
            raise ValueError("Analyzer must be fitted before getting p-values")

        if self.parameters["test"] == "Wald" and self._wald_analyzer is not None:
            return self.results["p_values"]
        return None

    def get_statistics(self) -> Optional[np.ndarray]:
        """
        Get test statistics.

        Returns
        -------
        Optional[np.ndarray]
            Test statistics or None if not available
        """
        if not self._fitted:
            raise ValueError("Analyzer must be fitted before getting statistics")

        if self.parameters["test"] == "Wald" and self._wald_analyzer is not None:
            return self.results["wald_statistics"]
        return None
