"""
Base normalization class for gene expression data.

This module provides the base class for all normalization strategies
in PyDiffExpress.

Classes:
    - BaseNormalizer: Base class for all normalization strategies

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from ..core.base import BaseAnalyzer
from ..datasets.dataset import ExpressionDataset


class BaseNormalizer(BaseAnalyzer):
    """
    Base class for all normalization strategies.

    This class provides a common interface for all normalization
    strategies while allowing each strategy to implement its own
    specific logic.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the base normalizer.

        Parameters
        ----------
        name : Optional[str]
            Name identifier for this normalizer.
        """
        super().__init__(name=name or self.__class__.__name__)
        self.size_factors = None
        self.normalization_factors = None

    @abstractmethod
    def _estimate_factors(self, dataset: ExpressionDataset, **kwargs) -> np.ndarray:
        """
        Estimate normalization factors for the dataset.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to normalize.
        **kwargs
            Strategy-specific parameters.

        Returns
        -------
        np.ndarray
            Estimated size factors for each sample.
        """
        pass

    def fit(self, dataset: ExpressionDataset, **kwargs) -> "BaseNormalizer":
        """
        Fit the normalizer to the dataset.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to normalize.
        **kwargs
            Strategy-specific parameters.

        Returns
        -------
        BaseNormalizer
            Self for method chaining.
        """
        self.size_factors = self._estimate_factors(dataset, **kwargs)

        # Store results
        self.results = {"size_factors": self.size_factors.copy(), "parameters": kwargs}

        # Add additional results if available (from median ratio)
        if hasattr(self, "normalized_counts"):
            self.results["normalized_counts"] = self.normalized_counts.copy()
        if hasattr(self, "base_means"):
            self.results["base_means"] = self.base_means.copy()
        if hasattr(self, "base_variances"):
            self.results["base_variances"] = self.base_variances.copy()

        self.fitted = True
        return self

    def _transform(self, dataset: ExpressionDataset, **kwargs) -> ExpressionDataset:
        """
        Transform the dataset by applying normalization factors.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to transform.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        ExpressionDataset
            Dataset with size factors added and stored in AnnData.
        """
        if self.size_factors is None:
            raise ValueError("Normalizer must be fitted before transform")

        # Create a copy of the dataset
        transformed_dataset = dataset.copy()

        # Add size factors to the dataset
        transformed_dataset.size_factors = self.size_factors

        # Copy stored results to the new dataset if available
        if hasattr(self, "normalized_counts") and self.normalized_counts is not None:
            transformed_dataset._adata.layers["normalized_counts"] = self.normalized_counts.copy()

        if hasattr(self, "base_means") and self.base_means is not None:
            transformed_dataset._adata.var["base_mean"] = self.base_means.copy()

        if hasattr(self, "base_variances") and self.base_variances is not None:
            transformed_dataset._adata.var["base_variance"] = self.base_variances.copy()

        return transformed_dataset

    def get_normalized_counts(self, dataset: ExpressionDataset) -> np.ndarray:
        """
        Get normalized counts for the dataset.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to normalize.

        Returns
        -------
        np.ndarray
            Normalized count matrix.
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before getting normalized counts")

        return dataset.get_normalized_counts()

    def _get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the normalization.

        Returns
        -------
        Dict[str, Any]
            Summary statistics and metadata.
        """
        summary = super()._get_summary()

        if self.size_factors is not None:
            summary.update(
                {
                    "n_samples": len(self.size_factors),
                    "size_factor_stats": {
                        "mean": np.mean(self.size_factors),
                        "median": np.median(self.size_factors),
                        "std": np.std(self.size_factors),
                        "min": np.min(self.size_factors),
                        "max": np.max(self.size_factors),
                    },
                }
            )

        return summary
