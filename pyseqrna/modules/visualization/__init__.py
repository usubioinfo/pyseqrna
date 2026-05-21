#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module for PySeqRNA
==================================

This module provides visualization functionality for RNA-seq analysis results
including PCA, t-SNE, volcano plots, MA plots, heatmaps, Venn diagrams,
and UpSet-style intersection summaries.

Functions:
    create_visualization - Factory function to create Visualization instances.

:Created: May 20, 2021
:Updated: May 5, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Optional, Any
from .visualization import Visualization, VisualizationError
from pyseqrna.__version__ import __version__


def create_visualization(
    outdir: str = ".",
    logger: Optional[Any] = None,
    dryrun: bool = False,
    dry_run_manager: Optional[Any] = None,
    **kwargs,
) -> Visualization:
    """
    Factory function to create Visualization instances.

    Args:
        outdir: Output directory for saving plots
        logger: Logger instance
        dryrun: Whether to perform a dry run
        dry_run_manager: Dry run manager instance
        **kwargs: Additional keyword arguments

    Returns:
        Visualization: Configured Visualization instance
    """
    return Visualization(
        outdir=outdir,
        logger=logger,
        dryrun=dryrun,
        dry_run_manager=dry_run_manager,
        **kwargs,
    )


__all__ = [
    "Visualization",
    "VisualizationError",
    "create_visualization",
]

__author__ = "Naveen Duhan"
