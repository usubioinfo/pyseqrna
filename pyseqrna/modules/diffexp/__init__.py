#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Differential Expression Module for PySeqRNA

This module provides various differential expression analysis methods for RNA-seq data:
- DESeq2: Popular R-based differential expression analysis
- edgeR: Another R-based differential expression analysis
- limma: Linear modeling approach for differential expression

Functions:
    - create_diffexp_analyzer: Factory function to create differential expression analyzer instances
    - create_deg_filter: Factory function to create DEGFilter instances
    - get_available_tools: Get list of available differential expression tools
    - get_default_tool: Get the default differential expression tool

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from .base import BaseDiffExp, DifferentialExpressionError
from .deseq2 import DESeq2DiffExp
from .edger import EdgeRDiffExp
from .deg_filter import DEGFilter
from .pydiffexpress_wrapper import PyDiffExpressWrapper

# Available differential expression tools mapping
AVAILABLE_TOOLS = {
    "deseq2": DESeq2DiffExp,
    "edger": EdgeRDiffExp,
    "pydiffexpress": PyDiffExpressWrapper,
}

# Default tool
DEFAULT_TOOL = "pydiffexpress"


def create_diffexp_analyzer(
    tool_name: str,
    count_matrix_file: str,
    sample_info_file: str,
    comparisons: List[str],
    out_dir: str = ".",
    species: Optional[str] = None,
    organism_type: str = "plants",
    add_gene_names: bool = True,
    gene_column: str = "Gene",  # Add gene_column parameter
    design_formula: str = "~ condition",  # Add design_formula parameter
    fdr_threshold: float = 0.05,  # Add fdr_threshold parameter
    log2fc_threshold: float = 1.0,  # Add log2fc_threshold parameter
    subset: bool = False,  # Add subset parameter
    logger=None,
    dryrun: bool = False,
    dry_run_manager=None,
    **kwargs: Any,
) -> BaseDiffExp:
    """
    Factory function to create differential expression analyzer instances.

    Args:
        tool_name: Name of the tool to create ('deseq2', 'edger', 'expression_analyzer')
        count_matrix_file: Path to count matrix file (Excel/CSV)
        sample_info_file: Path to sample information file
        comparisons: List of comparisons to perform (e.g., ['Treatment-Control'])
        out_dir: Output directory for results
        species: Species identifier (e.g., 'athaliana') for gene annotation
        organism_type: Type of organism - 'plants' or 'animals'
        add_gene_names: Whether to add gene names and descriptions to results
        logger: Logger instance
        dryrun: Whether to perform dry run
        dry_run_manager: Dry run manager instance for operation tracking
        **kwargs: Additional keyword arguments

    Returns:
        BaseDiffExp: Instance of the requested differential expression analyzer

    Raises:
        ValueError: If tool_name is not supported
    """
    tool_name = tool_name.lower()

    if tool_name not in AVAILABLE_TOOLS:
        available = ", ".join(AVAILABLE_TOOLS.keys())
        raise ValueError(f"Unsupported differential expression tool: {tool_name}. Available tools: {available}")

    tool_class = AVAILABLE_TOOLS[tool_name]

    # Create tool instance with standard signature
    return tool_class(
        count_matrix_file=count_matrix_file,
        sample_info_file=sample_info_file,
        comparisons=comparisons,
        out_dir=out_dir,
        gene_column=gene_column,
        design_formula=design_formula,
        fdr_threshold=fdr_threshold,
        log2fc_threshold=log2fc_threshold,
        species=species,
        organism_type=organism_type,
        add_gene_names=add_gene_names,
        logger=logger,
        subset=subset,
        dryrun=dryrun,
        dry_run_manager=dry_run_manager,
        **kwargs,
    )


def create_deg_filter(
    fdr_threshold: float = 0.05,
    fold_threshold: float = 2.0,
    has_replicates: bool = True,
    mmg: bool = False,
    extra_columns: bool = False,
    logger=None,
) -> DEGFilter:
    """
    Factory function to create DEGFilter instances.

    Args:
        fdr_threshold: False Discovery Rate threshold for filtering
        fold_threshold: Fold change threshold (will be log2 transformed)
        has_replicates: Whether the data has biological replicates
        mmg: Whether data is from multimapped gene groups
        extra_columns: Whether to expect extra annotation columns
        logger: Logger instance

    Returns:
        DEGFilter: Instance of the DEG filter
    """
    return DEGFilter(
        fdr_threshold=fdr_threshold,
        fold_threshold=fold_threshold,
        has_replicates=has_replicates,
        mmg=mmg,
        extra_columns=extra_columns,
        logger=logger,
    )


def get_available_tools() -> List[str]:
    """
    Get list of available differential expression tools.

    Returns:
        List of available tool names
    """
    return list(AVAILABLE_TOOLS.keys())


def get_default_tool() -> str:
    """
    Get the default differential expression tool.

    Returns:
        Default tool name
    """
    return DEFAULT_TOOL


__all__ = [
    "BaseDiffExp",
    "DifferentialExpressionError",
    "DESeq2DiffExp",
    "EdgeRDiffExp",
    "PyDiffExpressWrapper",
    "DEGFilter",
    "create_diffexp_analyzer",
    "create_deg_filter",
    "get_available_tools",
    "get_default_tool",
    "AVAILABLE_TOOLS",
    "DEFAULT_TOOL",
]

from pyseqrna.__version__ import __version__

__author__ = "Naveen Duhan"
