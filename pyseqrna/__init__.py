"""
PySeqRNA Package

A Python-based RNA-seq data analysis package with comprehensive
quality control, trimming, alignment, quantification, and analysis tools.

Features:
    - Pipeline orchestration and execution
    - Command-line interface utilities
    - Quality control, trimming, alignment, and quantification modules
    - Expression count normalization and differential expression analysis

:Created: May 20, 2021
:Updated: May 20, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .__version__ import __author__, __email__, __version__

from .utils import LogManager, CommandExecutor, FileManager, InputProcessor
from .cli import ArgumentManager
from .pipeline import Pipeline

# Module imports
from .modules import quality, normalization, diffexp

__all__ = [
    "LogManager",
    "CommandExecutor",
    "FileManager",
    "InputProcessor",
    "ArgumentManager",
    "Pipeline",
    "quality",
    "normalization",
    "diffexp",
]
