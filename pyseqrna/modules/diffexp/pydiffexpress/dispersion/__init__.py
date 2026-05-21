"""
Dispersion estimation module for PyDiffExpress.

This module provides dispersion estimation algorithms for differential expression analysis,
including gene-wise dispersion estimation, trend fitting, and maximum a posteriori (MAP) estimation.

Functions:
    - create_dispersion_estimator: Create a dispersion estimator with the specified method

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .base import BaseDispersionEstimator
from .gene_wise import GeneWiseDispersionEstimator
from .trend import TrendDispersionEstimator
from .map import MAPDispersionEstimator
from .dispersion import DispersionEstimator
from .glm import NegativeBinomialGLM


def create_dispersion_estimator(method: str = "pipeline", **kwargs) -> BaseDispersionEstimator:
    """
    Create a dispersion estimator with the specified method.

    Parameters
    ----------
    method : str
        Method to use for dispersion estimation:
        - "gene_wise": Gene-wise dispersion estimation only
        - "trend": Trend fitting only (requires gene-wise estimates)
        - "map": MAP estimation only (requires gene-wise and trend estimates)
        - "pipeline": Complete pipeline (gene-wise + trend + MAP)
    **kwargs
        Additional parameters to pass to the estimator

    Returns
    -------
    BaseDispersionEstimator
        Configured dispersion estimator

    Raises
    ------
    ValueError
        If method is not recognized
    """
    method = method.lower()

    if method == "gene_wise":
        return GeneWiseDispersionEstimator(**kwargs)

    elif method == "trend":
        return TrendDispersionEstimator(**kwargs)

    elif method == "map":
        return MAPDispersionEstimator(**kwargs)

    elif method in ["pipeline", "complete", "tagwise"]:
        # For complete pipeline, we'll create a composite estimator
        return DispersionEstimator(**kwargs)

    else:
        raise ValueError(f"Unknown method '{method}'. Available methods: 'gene_wise', 'trend', 'map', 'tagwise', 'pipeline'")


__all__ = [
    "BaseDispersionEstimator",
    "GeneWiseDispersionEstimator",
    "TrendDispersionEstimator",
    "MAPDispersionEstimator",
    "TagwiseDispersionEstimator",
    "DispersionEstimator",
    "NegativeBinomialGLM",
    "create_dispersion_estimator",
]
