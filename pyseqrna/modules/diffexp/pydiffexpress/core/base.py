"""
Base analyzer module providing the foundation for all analysis components.

This module defines the BaseAnalyzer class which serves as the parent class
for all analysis components in PyDiffExpress, providing common functionality
and interface consistency.

Classes:
    - BaseAnalyzer: Base class for all analysis components in PyDiffExpress

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets.dataset import ExpressionDataset


class BaseAnalyzer(ABC):
    """
    Base class for all analysis components in PyDiffExpress.

    This abstract base class provides a consistent interface for all analysis
    components, including fitting, transforming, and accessing results.

    Attributes
    ----------
    fitted : bool
        Whether the analyzer has been fitted to data.
    results : Dict[str, Any]
        Dictionary containing analysis results and metadata.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the base analyzer.

        Parameters
        ----------
        name : Optional[str]
            Name identifier for this analyzer instance.
        """
        self.name = name or self.__class__.__name__
        self.fitted = False
        self.results = {}

    @abstractmethod
    def fit(self, dataset: "ExpressionDataset", **kwargs) -> "BaseAnalyzer":
        """
        Fit the analyzer to the dataset.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to fit the analyzer to.
        **kwargs
            Additional keyword arguments specific to the analyzer.

        Returns
        -------
        BaseAnalyzer
            Self for method chaining.
        """
        pass

    def transform(self, dataset: "ExpressionDataset", **kwargs) -> "ExpressionDataset":
        """
        Transform the dataset using the fitted analyzer.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to transform.
        **kwargs
            Additional keyword arguments specific to the analyzer.

        Returns
        -------
        ExpressionDataset
            The transformed dataset.
        """
        if not self.fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        return self._transform(dataset, **kwargs)

    def get_results(self) -> Dict[str, Any]:
        """
        Get the analysis results.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing analysis results and metadata.
        """
        if not self.fitted:
            raise ValueError(f"{self.name} must be fitted before accessing results")
        return self.results.copy()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.

        Returns
        -------
        Dict[str, Any]
            Summary statistics and metadata.
        """
        if not self.fitted:
            raise ValueError(f"{self.name} must be fitted before accessing summary")
        return self._get_summary()

    def _get_summary(self) -> Dict[str, Any]:
        """
        Internal method to generate summary statistics.

        Returns
        -------
        Dict[str, Any]
            Summary statistics and metadata.
        """
        return {
            "analyzer_name": self.name,
            "fitted": self.fitted,
            "results_keys": list(self.results.keys()),
        }

    def reset(self):
        """Reset the analyzer to its initial state."""
        self.fitted = False
        self.results = {}

    def __repr__(self) -> str:
        """String representation of the analyzer."""
        status = "fitted" if self.fitted else "unfitted"
        return f"{self.name}({status})"

    def __str__(self) -> str:
        """String representation of the analyzer."""
        return self.__repr__()
