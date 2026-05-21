"""
Normalization module for gene expression data.

This module provides various normalization strategies for gene expression
data through a factory pattern for easy selection and use.

Functions:
    - create_normalizer: Create a normalization strategy using the factory pattern

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .median_ratio import MedianRatioNormalizer
from .poscounts import PosCountsNormalizer
from .iterative import IterativeNormalizer
from .tmm import TMMNormalizer
from .base import BaseNormalizer


# Factory function for creating normalizers
def create_normalizer(strategy: str = "median_ratio", **kwargs) -> "BaseNormalizer":
    """
    Create a normalization strategy using the factory pattern.

    Parameters
    ----------
    strategy : str
        Normalization strategy to use. Options:
        - "median_ratio": Median ratio normalization (default)
        - "poscounts": Positive counts normalization
        - "iterative": Iterative normalization
        - "tmm": TMM normalization
    **kwargs
        Additional keyword arguments specific to the strategy.

    Returns
    -------
    pyseqrna.modules.diffexp.pydiffexpress.normalization.base.BaseNormalizer
        Configured normalizer instance.

    Raises
    ------
    ValueError
        If strategy is not supported.
    """
    strategies = {
        "median_ratio": MedianRatioNormalizer,
        "poscounts": PosCountsNormalizer,
        "iterative": IterativeNormalizer,
        "tmm": TMMNormalizer,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown normalization strategy: {strategy}. Available strategies: {list(strategies.keys())}")

    return strategies[strategy](**kwargs)


__all__ = [
    "create_normalizer",
    "BaseNormalizer",
    "MedianRatioNormalizer",
    "PosCountsNormalizer",
    "IterativeNormalizer",
    "TMMNormalizer",
]
