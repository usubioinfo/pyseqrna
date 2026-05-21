#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimapped Groups Module for PySeqRNA

This module provides functionality to count multimapped read groups in aligned BAM files.
It identifies groups of genes that share multimapped reads and provides count matrices
for downstream analysis.

Functions:
    create_multimapped_groups_analyzer - Factory function to create multimapped groups analyzer instances.

:Created: May 20, 2021
:Updated: April 15, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from .analyzer import MultimappedGroupsAnalyzer, MultimappedGroupsError


def create_multimapped_groups_analyzer(
    bam_files: Dict[str, str],
    gff_file: str,
    out_dir: str = ".",
    feature: str = "gene",
    min_count: int = 100,
    percent_sample: float = 0.5,
    logger=None,
    dryrun: bool = False,
    dry_run_manager=None,
    **kwargs: Any,
) -> MultimappedGroupsAnalyzer:
    """
    Factory function to create multimapped groups analyzer instances.

    Args:
        bam_files: Dictionary mapping sample names to BAM file paths
        gff_file: Path to GFF/GTF annotation file
        out_dir: Output directory for results
        feature: Feature type to extract from GFF (default: 'gene')
        min_count: Minimum number of reads per sample for filtering
        percent_sample: Minimum percentage of samples that must meet min_count
        logger: Logger instance
        dryrun: Whether to perform dry run
        dry_run_manager: Dry run manager instance for operation tracking
        **kwargs: Additional keyword arguments

    Returns:
        MultimappedGroupsAnalyzer: Instance of the multimapped groups analyzer
    """
    return MultimappedGroupsAnalyzer(
        bam_files=bam_files,
        gff_file=gff_file,
        out_dir=out_dir,
        feature=feature,
        min_count=min_count,
        percent_sample=percent_sample,
        logger=logger,
        dryrun=dryrun,
        dry_run_manager=dry_run_manager,
        **kwargs,
    )


__all__ = [
    "MultimappedGroupsAnalyzer",
    "MultimappedGroupsError",
    "create_multimapped_groups_analyzer",
]
