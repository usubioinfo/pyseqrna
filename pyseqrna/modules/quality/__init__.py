#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quality Control Modules Package

===============================

=

This package contains quality control modules for PySeqRNA.

Available Tools:
    - FastQC: Fast quality control for NGS data

Usage:
    from pyseqrna.modules.quality import create_quality_control, get_available_quality_tools

Functions:
    get_available_quality_tools - Get list of available quality control tools.
    get_default_quality_tool - Get the default quality control tool.
    create_quality_control - Factory function to create quality control instances.

:Created: May 20, 2021
:Updated: January 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from .base import QualityControl
from .fastqc import FastQCQualityControl

# Available quality control implementations
AVAILABLE_QUALITY_TOOLS = {"fastqc": FastQCQualityControl}


def get_available_quality_tools() -> List[str]:
    """
    Get list of available quality control tools.

    Returns:
        List of available quality tool names
    """
    return list(AVAILABLE_QUALITY_TOOLS.keys())


def get_default_quality_tool() -> str:
    """
    Get the default quality control tool.

    Returns:
        Default quality tool name
    """
    return "fastqc"


def create_quality_control(
    tool_name: str,
    sample_dict: Dict[str, List[str]],
    out_dir: str,
    param_dir: Optional[str] = None,
    paired: bool = False,
    slurm: bool = False,
    dryrun: bool = False,
    job_id: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    logger: Optional[Any] = None,
    dry_run_manager=None,
    slurm_config: Optional[Dict[str, str]] = None,
    **kwargs,
) -> QualityControl:
    """
    Factory function to create quality control instances.

    Args:
        tool_name: Name of the quality tool to create (fastqc)
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
        **kwargs: Additional keyword arguments for specific quality tools

    Returns:
        QualityControl: Instance of the requested quality control tool

    Raises:
        ValueError: If tool_name is not supported
    """
    tool_name = tool_name.lower()

    if tool_name not in AVAILABLE_QUALITY_TOOLS:
        available = ", ".join(AVAILABLE_QUALITY_TOOLS.keys())
        raise ValueError(f"Unsupported quality tool: {tool_name}. Available quality tools: {available}")

    quality_class = AVAILABLE_QUALITY_TOOLS[tool_name]

    return quality_class(
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
    "QualityControl",
    "FastQCQualityControl",
    "create_quality_control",
    "get_available_quality_tools",
    "get_default_quality_tool",
    "AVAILABLE_QUALITY_TOOLS",
]
