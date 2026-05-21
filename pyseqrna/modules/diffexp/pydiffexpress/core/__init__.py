"""
Core module providing fundamental classes for PyDiffExpress.

This module contains the base classes and data structures that form the
foundation of the PyDiffExpress package.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .base import BaseAnalyzer
from .pipeline import DiffExpressAnalyzer

__all__ = [
    "BaseAnalyzer",
    "DiffExpressAnalyzer",
]
