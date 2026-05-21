#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantification module for RNA-seq data.

This module provides various quantification methods for RNA-seq data including:
- featureCounts: Fast and accurate read counting
- HTSeq-count: Flexible read counting with various overlap modes
- GenomicOverlaps: Novel method implementing advanced overlap counting algorithms

Usage:
    from pyseqrna.modules.quantification import create_quantifier, get_available_quantifiers

Functions:
    get_available_quantifiers - Get list of available quantification tools.
    get_default_quantifier - Get the default quantification tool.
    create_quantifier - Factory function to create quantifier instances.

:Created: May 20, 2021
:Updated: February 25, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from .base import BaseQuantifier, QuantificationError
from .featurecounts import FeatureCountsQuantifier
from .htseq import HTSeqQuantifier
from .genomic_overlaps import GenomicOverlapsQuantifier, OverlapMode

# Available quantifier implementations
AVAILABLE_QUANTIFIERS = {
    "featurecounts": FeatureCountsQuantifier,
    "htseq": HTSeqQuantifier,
    "genomic_overlaps": GenomicOverlapsQuantifier,
}


def get_available_quantifiers() -> List[str]:
    """
    Get list of available quantification tools.

    Returns:
        List of available quantifier names
    """
    return list(AVAILABLE_QUANTIFIERS.keys())


def get_default_quantifier() -> str:
    """
    Get the default quantification tool.

    Returns:
        Default quantifier name
    """
    return "genomic_overlaps"


def create_quantifier(
    quantifier_name: str,
    bam_dict: Dict[str, List[str]],
    annotation_file: str,
    out_dir: str,
    param_dir: Optional[str] = None,
    paired: bool = False,
    slurm: bool = False,
    dryrun: bool = False,
    job_id: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    memory: Optional[int] = None,
    logger: Optional[Any] = None,
    dry_run_manager=None,
    **kwargs,
) -> BaseQuantifier:
    """
    Factory function to create quantifier instances.

    Args:
        quantifier_name: Name of the quantifier to create (featurecounts, htseq, genomic_overlaps)
        bam_dict: Dictionary mapping sample names to BAM file paths
        annotation_file: Path to annotation file (GFF/GTF)
        out_dir: Output directory for results
        param_dir: Directory containing parameter files
        paired: Whether the data is paired-end
        slurm: Whether to use SLURM for job scheduling
        dryrun: Whether to perform a dry run
        job_id: SLURM job dependency ID
        cpu_threads: Number of CPU cores to use
        memory: Memory limit in GB
        logger: Logger instance
        dry_run_manager: Dry run manager instance
        **kwargs: Additional keyword arguments for specific quantifiers

    Returns:
        BaseQuantifier: Instance of the requested quantifier

    Raises:
        ValueError: If quantifier_name is not supported
    """
    quantifier_name = quantifier_name.lower()

    if quantifier_name not in AVAILABLE_QUANTIFIERS:
        available = ", ".join(AVAILABLE_QUANTIFIERS.keys())
        raise ValueError(f"Unsupported quantifier: {quantifier_name}. Available quantifiers: {available}")

    quantifier_class = AVAILABLE_QUANTIFIERS[quantifier_name]

    return quantifier_class(
        bam_dict=bam_dict,
        annotation_file=annotation_file,
        out_dir=out_dir,
        param_dir=param_dir,
        paired=paired,
        slurm=slurm,
        dryrun=dryrun,
        job_id=job_id,
        cpu_threads=cpu_threads,
        memory=memory,
        logger=logger,
        dry_run_manager=dry_run_manager,
        **kwargs,
    )


__all__ = [
    "BaseQuantifier",
    "QuantificationError",
    "FeatureCountsQuantifier",
    "HTSeqQuantifier",
    "GenomicOverlapsQuantifier",
    "OverlapMode",
    "create_quantifier",
    "get_available_quantifiers",
    "get_default_quantifier",
    "AVAILABLE_QUANTIFIERS",
]

from pyseqrna.__version__ import __version__

__author__ = "Naveen Duhan"
