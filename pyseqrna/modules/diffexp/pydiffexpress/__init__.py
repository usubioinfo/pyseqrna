#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone modular differential-expression sandbox package.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

# Core imports
from .core.base import BaseAnalyzer
from .core.pipeline import DiffExpressAnalyzer

# Normalization modules
from .normalization import create_normalizer, BaseNormalizer, MedianRatioNormalizer

# Hypothesis testing modules
from .hypothesis_testing import WaldTestAnalyzer

# Main analyzer
from .core.pipeline import DiffExpressAnalyzer as PyDiffExpressAnalyzer
from .backends import BaseMeanWaldBackend, TMMLRTBackend, get_backend
from .api import (
    available_abundance,
    available_backends,
    available_dispersions,
    available_normalizations,
    available_tests,
    run_analysis,
    run_backend,
)

__all__ = [
    "BaseAnalyzer",
    "DiffExpressAnalyzer",
    "PyDiffExpressAnalyzer",
    "create_normalizer",
    "BaseNormalizer",
    "MedianRatioNormalizer",
    "WaldTestAnalyzer",
    "BaseMeanWaldBackend",
    "TMMLRTBackend",
    "get_backend",
    "available_normalizations",
    "available_abundance",
    "available_dispersions",
    "available_tests",
    "run_analysis",
    "available_backends",
    "run_backend",
]
