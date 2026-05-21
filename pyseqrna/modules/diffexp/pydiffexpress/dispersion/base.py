"""
Base dispersion estimator class.

This module provides the base class for dispersion estimation algorithms.

Classes:
    - BaseDispersionEstimator: Base class for dispersion estimation algorithms

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from anndata import AnnData


class BaseDispersionEstimator(ABC):
    """
    Base class for dispersion estimation algorithms.

    This class defines the interface that all dispersion estimators must implement.
    Dispersion estimation is a key step in differential expression analysis that
    models the variance-mean relationship in count data.

    Similar to BaseNormalizer, this provides a consistent interface for
    different dispersion estimation methods.
    """

    def __init__(self, **kwargs):
        """Initialize the dispersion estimator with parameters."""
        self.parameters = kwargs
        self._fitted = False
        self._results = {}

    @abstractmethod
    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "BaseDispersionEstimator":
        """
        Fit the dispersion estimator to the data.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix with counts in .X and metadata in .obs/.var
        design_matrix : np.ndarray, optional
            Design matrix for the experimental design. If None, will be created
            from data.obs metadata.

        Returns
        -------
        self : BaseDispersionEstimator
            Fitted estimator instance
        """
        pass

    @abstractmethod
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
            Array of dispersion estimates for each gene
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results from the dispersion estimation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing dispersion estimates and metadata
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before getting results")
        return self._results.copy()

    def update_data(self, data: AnnData) -> None:
        """
        Update the data with dispersion estimates.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix to update with dispersion estimates
        """
        if not self._fitted:
            raise ValueError("Estimator must be fitted before updating data")

        results = self.get_results()

        # Store dispersion estimates in var
        if "dispersions" in results:
            # `dispersions` should be the final post-processing result used by
            # downstream testing. Keep raw MAP estimates separately in
            # `disp_map`, but do not overwrite the final column with them.
            data.var["dispersion"] = results["dispersions"]

        # Store gene-wise estimates if available
        if "disp_gene_est" in results:
            data.var["disp_gene_est"] = results["disp_gene_est"]

        # Store gene-wise iterations if available
        if "disp_gene_iter" in results:
            data.var["disp_gene_iter"] = results["disp_gene_iter"]

        # Store fitted dispersions if available
        if "disp_fitted" in results:
            data.var["disp_fit"] = results["disp_fitted"]

        # Store outliers if available
        if "outliers" in results:
            data.var["disp_outlier"] = results["outliers"]

        # Store dispersion iterations if available
        if "disp_iter" in results:
            data.var["disp_iter"] = results["disp_iter"]

        # Store MAP dispersions if available
        if "disp_map" in results:
            data.var["disp_map"] = results["disp_map"]

        # Store convergence information if available
        if "disp_conv" in results:
            data.var["disp_conv"] = results["disp_conv"]

        # Store dispersion function if available
        if "disp_function" in results:
            data.uns["disp_function"] = results["disp_function"]

        # Store mean estimates if available
        if "mu" in results:
            data.layers["mu"] = results["mu"]

        # Ensure size_factors column exists with correct name
        if "sizeFactors" in data.obs.columns and "size_factors" not in data.obs.columns:
            data.obs["size_factors"] = data.obs["sizeFactors"]

    @property
    def is_fitted(self) -> bool:
        """Check if the estimator has been fitted."""
        return self._fitted

    def _validate_data(self, data: AnnData) -> None:
        """
        Validate input data.

        Parameters
        ----------
        data : AnnData
            Data to validate

        Raises
        ------
        ValueError
            If data is invalid
        """
        if not isinstance(data, AnnData):
            raise ValueError("Data must be an AnnData object")

        if data.X is None or data.X.size == 0:
            raise ValueError("Data must contain expression counts")

        if data.n_vars == 0:
            raise ValueError("Data must contain at least one gene")

    def _validate_design_matrix(self, design_matrix: np.ndarray, data: AnnData) -> None:
        """
        Validate design matrix.

        Parameters
        ----------
        design_matrix : np.ndarray
            Design matrix to validate
        data : AnnData
            Corresponding data

        Raises
        ------
        ValueError
            If design matrix is invalid
        """
        if design_matrix is None:
            return

        if not isinstance(design_matrix, np.ndarray):
            raise ValueError("Design matrix must be a numpy array")

        if design_matrix.shape[0] != data.n_obs:
            raise ValueError(f"Design matrix must have {data.n_obs} rows (samples)")

        if design_matrix.shape[1] == 0:
            raise ValueError("Design matrix must have at least one column")

        # Check for full rank
        if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
            raise ValueError("Design matrix must be full rank")

        # Check for sufficient replicates
        if design_matrix.shape[0] == design_matrix.shape[1]:
            raise ValueError(
                "Number of samples equals number of model coefficients. No replicates available for dispersion estimation."
            )
