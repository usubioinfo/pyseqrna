"""
Trimming Module Package

This package contains modular implementations of various read trimming tools
for RNA-seq analysis.

Available Tools:
    - TrimGalore: Wrapper around Cutadapt and FastQC
    - Trimmomatic: Java-based read trimming tool
    - Flexbar: Flexible barcode and adapter removal

Usage:
    from pyseqrna.modules.trimming import create_trimmer, get_available_trimmers
"""

from typing import Dict, List, Optional
from .base import ReadTrimmer
from .trim_galore import TrimGaloreTrimmer
from .trimmomatic import TrimmomaticTrimmer
from .flexbar import FlexbarTrimmer
from .stats import TrimmingStats, TrimmingStatsError, TrimmingSampleStats

# Available trimmer implementations
AVAILABLE_TRIMMERS = {
    "trim_galore": TrimGaloreTrimmer,
    "trimmomatic": TrimmomaticTrimmer,
    "flexbar": FlexbarTrimmer,
}


def get_available_trimmers() -> List[str]:
    """
    Get list of available trimming tools.

    Returns:
        List of available trimmer names
    """
    return list(AVAILABLE_TRIMMERS.keys())


def get_default_trimmer() -> str:
    """
    Get the default trimming tool.

    Returns:
        Default trimmer name
    """
    return "trim_galore"


def create_trimmer(
    trimmer_name: str,
    sample_dict: Dict[str, List[str]],
    out_dir: str,
    param_dir: Optional[str] = None,
    paired: bool = False,
    slurm: bool = False,
    dryrun: bool = False,
    job_id: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    logger=None,
    dry_run_manager=None,
    slurm_config: Optional[Dict[str, str]] = None,
    **kwargs,
) -> ReadTrimmer:
    """
    Factory function to create trimmer instances.

    Args:
        trimmer_name: Name of the trimmer to create (trim_galore, trimmomatic, flexbar)
        sample_dict: Dictionary mapping sample names to input files
        out_dir: Output directory for results
        param_dir: Directory containing parameter files
        paired: Whether using paired-end reads
        slurm: Whether to use SLURM for job scheduling
        dryrun: Whether to perform a dry run
        job_id: SLURM job dependency ID
        cpu_threads: Number of CPU threads to use
        logger: Logger instance
        dry_run_manager: Dry run manager instance
        **kwargs: Additional keyword arguments for specific trimmers

    Returns:
        ReadTrimmer: Instance of the requested trimmer

    Raises:
        ValueError: If trimmer_name is not supported
    """
    trimmer_name = trimmer_name.lower()

    if trimmer_name not in AVAILABLE_TRIMMERS:
        available = ", ".join(AVAILABLE_TRIMMERS.keys())
        raise ValueError(f"Unsupported trimmer: {trimmer_name}. Available trimmers: {available}")

    trimmer_class = AVAILABLE_TRIMMERS[trimmer_name]

    return trimmer_class(
        sample_dict=sample_dict,
        out_dir=out_dir,
        param_dir=param_dir,
        paired=paired,
        slurm=slurm,
        dryrun=dryrun,
        job_id=job_id,
        cpu_threads=cpu_threads,
        logger=logger,
        dry_run_manager=dry_run_manager,
        slurm_config=slurm_config,
        **kwargs,
    )


__all__ = [
    "ReadTrimmer",
    "TrimGaloreTrimmer",
    "TrimmomaticTrimmer",
    "FlexbarTrimmer",
    "TrimmingStats",
    "TrimmingStatsError",
    "TrimmingSampleStats",
    "create_trimmer",
    "get_available_trimmers",
    "get_default_trimmer",
    "AVAILABLE_TRIMMERS",
]
