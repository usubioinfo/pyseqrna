#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Co-expression Module

This module provides the abstract base class and custom exception for gene co-expression
analysis in the pySeqRNA pipeline. It defines a common constructor, output directory
creation, dry-run configuration, and helper logging methods for concrete co-expression
implementations.

Features:
    - Base class (BaseCoexpression) defining the template for co-expression analysis modules
    - Automated output directory creation and management
    - Robust support for dry-run simulation mode integration
    - Helper logging function that checks for logger existence

Configuration:
    Configured via parameters passed to the constructor (such as matrix_file, out_dir,
    logger, dryrun, dry_run_manager).

Dependencies:
    - Python standard library (pathlib, typing)

Classes / Functions / Exceptions:
    - CoexpressionError: Custom exception for co-expression-related errors.
    - BaseCoexpression: Shared base class for co-expression analysis implementations.

:Created: May 21, 2026
:Updated: May 21, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from typing import Any, Optional


class CoexpressionError(Exception):
    """Custom exception for co-expression errors."""


class BaseCoexpression:
    """Shared base class for co-expression analysis implementations."""

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
