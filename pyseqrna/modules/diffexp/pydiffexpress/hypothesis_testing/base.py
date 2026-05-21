"""
Base hypothesis testing analyzer module.

This module provides the base class for hypothesis testing methods.

Classes:
    - BaseHypothesisTestAnalyzer: Base class for hypothesis testing analyzers

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from abc import abstractmethod
from typing import Optional
import numpy as np
from anndata import AnnData

from ..core.base import BaseAnalyzer


class BaseHypothesisTestAnalyzer(BaseAnalyzer):
    """
    Base class for hypothesis testing analyzers.

    This class provides the interface for performing hypothesis tests
    on gene expression data stored in AnnData objects.
    """

    def __init__(self, **kwargs):
        """
        Initialize the hypothesis test analyzer.

        Parameters
        ----------
        **kwargs
            Additional parameters specific to the analyzer
        """
        super().__init__(**kwargs)

    @abstractmethod
    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "BaseHypothesisTestAnalyzer":
        """
        Fit the hypothesis test model.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix with counts in .X and dispersion estimates
        design_matrix : np.ndarray, optional
            Design matrix for experimental design

        Returns
        -------
        self : BaseHypothesisTestAnalyzer
            Fitted analyzer
        """
        pass

    def update_data(self, data: AnnData) -> None:
        """
        Update the data with hypothesis test results.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix to update with results
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before updating data")

        results = self.get_results()

        # Store results in var
        design_cols = data.uns.get("design_columns", [])

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 1 and len(value) == data.n_vars:
                    # 1D array - store directly
                    data.var[key] = value
                elif len(value.shape) == 2 and value.shape[0] == data.n_vars:
                    # 2D array - store each column separately with proper names
                    for i in range(value.shape[1]):
                        if key in [
                            "beta",
                            "beta_se",
                            "p_values",
                            "wald_statistics",
                        ] and i < len(design_cols):
                            # Use design matrix column names for these results
                            col_name = design_cols[i]
                            if key == "beta":
                                data.var[col_name] = value[:, i]
                            elif key == "beta_se":
                                data.var[f"SE_{col_name}"] = value[:, i]
                            elif key == "p_values":
                                data.var[f"WaldPvalue_{col_name}"] = value[:, i]
                            elif key == "wald_statistics":
                                data.var[f"WaldStatistic_{col_name}"] = value[:, i]
                        elif key == "beta_coefficients" and i < len(design_cols):
                            # Handle beta_coefficients specifically
                            col_name = design_cols[i]
                            data.var[col_name] = value[:, i]
                        else:
                            # Use generic names for other results
                            col_name = f"{key}_{i}" if value.shape[1] > 1 else key
                            data.var[col_name] = value[:, i]

        # Store additional results in uns
        if "metadata" in results:
            data.uns["hypothesis_test"] = results["metadata"]

        # Transfer variance matrix and beta matrix from results to data.uns
        if "beta_var_mat" in results:
            data.uns["beta_var_mat"] = results["beta_var_mat"]
        if "beta_mat" in results:
            data.uns["beta_mat"] = results["beta_mat"]

    def _validate_data(self, data: AnnData) -> None:
        """
        Validate that the data has required attributes for hypothesis testing.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix to validate

        Raises
        ------
        ValueError
            If required data is missing
        """
        # Check for required dispersion estimates
        if "dispersion" not in data.var:
            raise ValueError("Dispersion estimates not found. Run dispersion estimation first.")

        # Check for size factors
        if "size_factors" not in data.obs:
            raise ValueError("Size factors not found. Run size factor estimation first.")

        # Check for design matrix
        if "design" not in data.obsm:
            raise ValueError("Design matrix not found. Create design matrix first.")

    def _get_normalized_counts(self, data: AnnData) -> np.ndarray:
        """
        Get normalized counts for analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            Normalized counts
        """
        # Get raw counts
        counts = data.X.toarray() if hasattr(data.X, "toarray") else data.X

        # Get size factors
        size_factors = data.obs["size_factors"].values

        # Normalize counts
        normalized_counts = counts / size_factors.reshape(-1, 1)

        return normalized_counts

    def _get_dispersions(self, data: AnnData) -> np.ndarray:
        """
        Get dispersion estimates for analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            Dispersion estimates
        """
        return data.var["dispersion"].values

    def _get_design_matrix(self, data: AnnData) -> np.ndarray:
        """
        Get design matrix for analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix

        Returns
        -------
        np.ndarray
            Design matrix
        """
        return data.obsm["design"]
