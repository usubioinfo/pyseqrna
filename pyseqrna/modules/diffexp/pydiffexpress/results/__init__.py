"""
Results module for differential expression analysis.

This module provides tools for extracting and analyzing results from
differential expression analysis, including contrast handling and
result filtering.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .contrasts import ContrastAnalyzer
from .extraction import ResultsExtractor

__all__ = ["ContrastAnalyzer", "ResultsExtractor"]
