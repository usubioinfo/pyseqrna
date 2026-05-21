#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalization Module
====================

This module provides count normalization methods for RNA-seq data.

Available normalizers:
- CPM (Counts Per Million)
- RPKM (Reads Per Kilobase Million)
- TPM (Transcripts Per Million)
- MedianRatio (Median ratio normalization)
- TMM (Trimmed Mean of M-values)

Functions:
    create_normalizer - Factory function to create normalizer instances.
    get_available_normalizers - Get list of available normalizer names.
    get_default_normalizer - Get the default normalizer name.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from .base import BaseNormalizer, NormalizationError
from .cpm import CPMNormalizer
from .rpkm import RPKMNormalizer
from .tpm import TPMNormalizer
from .fpkm import FPKMNormalizer
from .median_ratio import MedianRatioNormalizer
from .tmm import TMMNormalizer

# Available normalizers mapping
AVAILABLE_NORMALIZERS = {
    "cpm": CPMNormalizer,
    "rpkm": RPKMNormalizer,
    "tpm": TPMNormalizer,
    "fpkm": FPKMNormalizer,
    "median_ratio": MedianRatioNormalizer,
    "tmm": TMMNormalizer,
}

# Default normalizer
DEFAULT_NORMALIZER = "rpkm"


def create_normalizer(
    normalizer_name: str,
    count_matrix_file: str,
    annotation_file: Optional[str] = None,
    out_dir: str = ".",
    logger=None,
    dryrun: bool = False,
    dry_run_manager=None,
    **kwargs: Any,
) -> "BaseNormalizer":
    """
    Factory function to create normalizer instances.

    Args:
        normalizer_name: Name of the normalizer to create
        count_matrix_file: Path to count matrix file (Excel/CSV)
        annotation_file: Path to annotation file (GTF/GFF) - required for length-based methods
        out_dir: Output directory for results
        logger: Logger instance
        dryrun: Whether to perform dry run
        dry_run_manager: Dry run manager instance for operation tracking
        **kwargs: Additional keyword arguments

    Returns:
        pyseqrna.modules.normalization.base.BaseNormalizer: Instance of the requested normalizer

    Raises:
        ValueError: If normalizer_name is not supported
    """
    normalizer_name = normalizer_name.lower()

    if normalizer_name not in AVAILABLE_NORMALIZERS:
        available = ", ".join(AVAILABLE_NORMALIZERS.keys())
        raise ValueError(f"Unsupported normalizer: {normalizer_name}. Available normalizers: {available}")

    normalizer_class = AVAILABLE_NORMALIZERS[normalizer_name]

    return normalizer_class(
        count_matrix_file=count_matrix_file,
        annotation_file=annotation_file,
        out_dir=out_dir,
        logger=logger,
        dryrun=dryrun,
        dry_run_manager=dry_run_manager,
        **kwargs,
    )


def get_available_normalizers() -> List[str]:
    """
    Get list of available normalizer names.

    Returns:
        List of available normalizer names
    """
    return list(AVAILABLE_NORMALIZERS.keys())


def get_default_normalizer() -> str:
    """
    Get the default normalizer name.

    Returns:
        Default normalizer name
    """
    return DEFAULT_NORMALIZER


__all__ = [
    "BaseNormalizer",
    "NormalizationError",
    "CPMNormalizer",
    "RPKMNormalizer",
    "TPMNormalizer",
    "FPKMNormalizer",
    "MedianRatioNormalizer",
    "TMMNormalizer",
    "create_normalizer",
    "get_available_normalizers",
    "get_default_normalizer",
    "AVAILABLE_NORMALIZERS",
    "DEFAULT_NORMALIZER",
]

from pyseqrna.__version__ import __version__

__author__ = "Naveen Duhan"
