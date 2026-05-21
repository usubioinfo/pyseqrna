#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alignment Module for PySeqRNA

This module provides RNA-seq read alignment functionality using various aligners:
- STAR (default)
- HISAT2
- Bowtie2
- BWA
- Minimap2

Functions:
    create_aligner - Factory function to create aligner instances.
    get_available_aligners - Get list of available aligner names.
    get_default_aligner - Get the default aligner name.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from .base import BaseAligner, AlignmentError
from .star import StarAligner
from .hisat2 import Hisat2Aligner
from .bowtie2 import Bowtie2Aligner
from .bwa import BwaAligner
from .minimap2 import Minimap2Aligner
from .stats import AlignmentStats, AlignmentStatsError, SampleStats

# Available aligners mapping
AVAILABLE_ALIGNERS = {
    "star": StarAligner,
    "hisat2": Hisat2Aligner,
    "bowtie2": Bowtie2Aligner,
    "bwa": BwaAligner,
    "minimap2": Minimap2Aligner,
}

# Default aligner
DEFAULT_ALIGNER = "star"


def create_aligner(
    aligner_name: str,
    genome: str,
    out_dir: str,
    param_dir: Optional[str] = None,
    logger=None,
    dryrun: bool = False,
    cpu_threads: Optional[int] = None,
    slurm: bool = False,
    dep: str = "",
    dry_run_manager=None,
    slurm_config: Optional[Dict[str, str]] = None,
) -> BaseAligner:
    """
    Factory function to create aligner instances.

    Args:
        aligner_name: Name of the aligner to create (star, hisat2, bowtie2, bwa, minimap2)
        genome: Path to reference genome file
        out_dir: Output directory for results
        param_dir: Directory containing parameter files
        logger: Logger instance
        dryrun: Whether to perform dry run
        cpu_threads: Number of CPU threads
        slurm: Whether to use SLURM
        dep: SLURM job dependency

    Returns:
        BaseAligner: Instance of the requested aligner

    Raises:
        ValueError: If aligner_name is not supported
    """
    aligner_name = aligner_name.lower()

    if aligner_name not in AVAILABLE_ALIGNERS:
        available = ", ".join(AVAILABLE_ALIGNERS.keys())
        raise ValueError(f"Unsupported aligner: {aligner_name}. Available aligners: {available}")

    aligner_class = AVAILABLE_ALIGNERS[aligner_name]

    aligner_instance = aligner_class(
        genome=genome,
        out_dir=out_dir,
        param_dir=param_dir,
        logger=logger,
        dryrun=dryrun,
        cpu_threads=cpu_threads,
        slurm=slurm,
        dep=dep,
        dry_run_manager=dry_run_manager,
        slurm_config=slurm_config,
    )

    return aligner_instance


def get_available_aligners() -> List[str]:
    """
    Get list of available aligner names.

    Returns:
        List[str]: List of available aligner names
    """
    return list(AVAILABLE_ALIGNERS.keys())


def get_default_aligner() -> str:
    """
    Get the default aligner name.

    Returns:
        str: Default aligner name
    """
    return DEFAULT_ALIGNER


# Export all classes and functions
__all__ = [
    "BaseAligner",
    "AlignmentError",
    "StarAligner",
    "Hisat2Aligner",
    "Bowtie2Aligner",
    "BwaAligner",
    "Minimap2Aligner",
    "AlignmentStats",
    "AlignmentStatsError",
    "SampleStats",
    "create_aligner",
    "get_available_aligners",
    "get_default_aligner",
    "AVAILABLE_ALIGNERS",
    "DEFAULT_ALIGNER",
]
