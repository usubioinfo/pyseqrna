#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Clustering Utilities

This module provides the abstract base class and custom exception for sample similarity
clustering in the pySeqRNA package. It sets up output directories, controls dry-run modes,
and manages logging channels for concrete clustering implementations.

Features:
    - Abstract base class (BaseClustering) for sample-level similarity clustering algorithms
    - Automatic output directory creation and management
    - Dry-run verification and execution reporting integration
    - Standardized error raising interface through ClusteringError

Configuration:
    Configured via parameters passed to the constructor (such as matrix_file, out_dir,
    logger, dryrun, dry_run_manager).

Dependencies:
    - Python standard library (pathlib, typing)

Classes / Functions / Exceptions:
    - ClusteringError: Custom exception for clustering errors.
    - BaseClustering: Shared base class for sample similarity clustering implementations.

:Created: May 20, 2021
:Updated: April 5, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from typing import Any, Optional


class ClusteringError(Exception):
    """Custom exception for clustering errors."""


class BaseClustering:
    """Shared base class for sample similarity clustering implementations."""

    def __init__(
        self,
        matrix_file: str,
        out_dir: str = ".",
        logger: Optional[Any] = None,
        dryrun: bool = False,
        dry_run_manager: Optional[Any] = None,
    ):
        self.matrix_file = matrix_file
        self.out_dir = Path(out_dir)
        self.logger = logger
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager

        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, level: str, message: str, *args) -> None:
        """Log when a logger exists, otherwise stay quiet."""
        if self.logger:
            getattr(self.logger, level)(message, *args)
