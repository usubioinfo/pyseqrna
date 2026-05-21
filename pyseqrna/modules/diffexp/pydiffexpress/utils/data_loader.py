"""
Data loading utilities for PyDiffExpress.

This module provides functions for loading gene expression data from
various file formats and handling different data orientations.

Functions:
    - load_expression_data: Load expression data from various input formats
    - detect_data_orientation: Detect the orientation of gene expression data using sample metadata
    - validate_file_path: Validate if a file path exists and has a supported extension

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_expression_data(
    data_input: Union[str, pd.DataFrame, np.ndarray],
    sample_metadata: Optional[Union[str, pd.DataFrame]] = None,
    gene_metadata: Optional[Union[str, pd.DataFrame]] = None,
    auto_detect_orientation: bool = True,
    sample_id_column: Optional[str] = None,
    gene_column: str = "Gene",
    design_column: str = "condition",
    **kwargs,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load expression data from various input formats.

    This function provides a flexible interface for loading gene expression
    data from files or existing data structures. It automatically detects
    file formats and can handle different data orientations.

    Parameters
    ----------
    data_input : Union[str, pd.DataFrame, np.ndarray]
        Input data. Can be:
        - File path (string): Will be loaded based on file extension
        - pandas DataFrame: Used directly
        - numpy array: Converted to DataFrame
    sample_metadata : Optional[Union[str, pd.DataFrame]]
        Sample metadata. Can be file path or DataFrame.
    gene_metadata : Optional[Union[str, pd.DataFrame]]
        Gene metadata. Can be file path or DataFrame.
    auto_detect_orientation : bool
        Whether to automatically detect if data needs to be transposed.
    sample_id_column : Optional[str]
        Name of the column containing sample IDs in sample_metadata.
        If None, will use the index of sample_metadata.
    gene_column : str, default='Gene'
        Name of the column containing gene names in counts data.
        If this column exists, it will be set as the index.
    design_column : str, default='condition'
        Name of the column containing design/condition information in sample metadata.
    **kwargs
        Additional arguments passed to pandas read functions.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]
        Tuple of (counts_data, sample_metadata, gene_metadata)

    Raises
    ------
    ValueError
        If file format is not supported or data is invalid.
    FileNotFoundError
        If file path does not exist.
    """

    # Load counts data
    counts_data = _load_data(data_input, **kwargs)
    # Handle gene column if it exists in counts data
    if isinstance(counts_data, pd.DataFrame) and gene_column in counts_data.columns:
        # Set gene column as index
        counts_data = counts_data.set_index(gene_column, drop=True)

    # Load metadata if provided
    sample_meta = _load_data(sample_metadata, **kwargs) if sample_metadata is not None else None
    gene_meta = _load_data(gene_metadata, **kwargs) if gene_metadata is not None else None

    # Handle sample metadata indexing properly
    if sample_meta is not None and isinstance(sample_meta, pd.DataFrame):
        if "sample" in sample_meta.columns:
            # If sample column exists, use it as index
            sample_meta = sample_meta.set_index("sample")

    # Auto-detect and fix orientation if requested
    if auto_detect_orientation:
        orientation = detect_data_orientation(counts_data, sample_meta, gene_meta, sample_id_column)

        if orientation == "genes_x_samples":
            # Transpose to get samples x genes
            counts_data = counts_data.T
            logger.info("Detected genes x samples orientation. Transposing to samples x genes.")

    return counts_data, sample_meta, gene_meta


def _load_data(data_input: Union[str, pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
    """
    Load data from various input types.

    Parameters
    ----------
    data_input : Union[str, pd.DataFrame, np.ndarray]
        Input data or file path.
    **kwargs
        Additional arguments for pandas read functions.

    Returns
    -------
    pd.DataFrame
        Loaded data as DataFrame.
    """
    if isinstance(data_input, pd.DataFrame):
        return data_input.copy()

    elif isinstance(data_input, np.ndarray):
        return pd.DataFrame(data_input)

    elif isinstance(data_input, str):
        return _load_from_file(data_input, **kwargs)

    else:
        raise ValueError(f"Unsupported data input type: {type(data_input)}")


def _load_from_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from file based on file extension.

    Parameters
    ----------
    file_path : str
        Path to the file.
    **kwargs
        Additional arguments for pandas read functions.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file format and load accordingly
    file_extension = file_path.suffix.lower()

    try:
        if file_extension in [".csv"]:
            return pd.read_csv(file_path, **kwargs)

        elif file_extension in [".tsv", ".txt"]:
            # Try tab-separated first, then space-separated
            try:
                return pd.read_csv(file_path, sep="\t", **kwargs)
            except:
                return pd.read_csv(file_path, sep=r"\s+", **kwargs)

        elif file_extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_path, **kwargs)

        elif file_extension in [".h5", ".hdf5"]:
            return pd.read_hdf(file_path, **kwargs)

        elif file_extension in [".parquet"]:
            return pd.read_parquet(file_path, **kwargs)

        else:
            # Try to guess the format
            try:
                return pd.read_csv(file_path, **kwargs)
            except:
                raise ValueError(f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {str(e)}")


def detect_data_orientation(
    counts_data: pd.DataFrame,
    sample_metadata: Optional[pd.DataFrame] = None,
    gene_metadata: Optional[pd.DataFrame] = None,
    sample_id_column: Optional[str] = None,
) -> str:
    """
    Detect the orientation of gene expression data using sample metadata.

    This function uses sample metadata to determine if the data is in
    genes x samples or samples x genes orientation by checking which
    dimension matches the sample names in the metadata.

    Parameters
    ----------
    counts_data : pd.DataFrame
        Count matrix.
    sample_metadata : Optional[pd.DataFrame]
        Sample metadata with sample names as index or in a column.
    gene_metadata : Optional[pd.DataFrame]
        Gene metadata with gene names as index or in a column.
    sample_id_column : Optional[str]
        Name of the column containing sample IDs in sample_metadata.
        If None, will use the index of sample_metadata.

    Returns
    -------
    str
        Either "genes_x_samples" or "samples_x_genes".
    """
    n_rows, n_cols = counts_data.shape
    row_names = set(counts_data.index.astype(str))
    col_names = set(counts_data.columns.astype(str))

    # Get sample names from metadata
    sample_names = set()
    if sample_metadata is not None:
        if sample_id_column is not None and sample_id_column in sample_metadata.columns:
            # Use specified column for sample IDs
            sample_names = set(sample_metadata[sample_id_column].astype(str))
        else:
            # Use index as sample names
            sample_names = set(sample_metadata.index.astype(str))

    # Determine orientation based on sample name matching
    if sample_names:
        # Check which dimension matches the sample names
        row_sample_overlap = len(row_names.intersection(sample_names))
        col_sample_overlap = len(col_names.intersection(sample_names))

        if row_sample_overlap > col_sample_overlap:
            # More sample names match rows -> samples x genes
            return "samples_x_genes"
        elif col_sample_overlap > row_sample_overlap:
            # More sample names match columns -> genes x samples
            return "genes_x_samples"
        elif row_sample_overlap == col_sample_overlap and row_sample_overlap > 0:
            # Equal overlap, use dimension size as tiebreaker
            if n_rows > n_cols:
                return "genes_x_samples"  # More rows likely means genes
            else:
                return "samples_x_genes"  # More columns likely means samples

    # Fallback: Use metadata dimensions if no sample name matching
    if sample_metadata is not None:
        if len(sample_metadata) == n_rows:
            return "samples_x_genes"
        elif len(sample_metadata) == n_cols:
            return "genes_x_samples"

    if gene_metadata is not None:
        if len(gene_metadata) == n_cols:
            return "samples_x_genes"
        elif len(gene_metadata) == n_rows:
            return "genes_x_samples"

    # Final fallback: Use dimension size (typical RNA-seq has more genes than samples)
    if n_rows > n_cols:
        return "genes_x_samples"
    else:
        return "samples_x_genes"


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and has a supported extension.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    bool
        True if file exists and has supported extension.
    """
    if not os.path.exists(file_path):
        return False

    supported_extensions = [
        ".csv",
        ".tsv",
        ".txt",
        ".xlsx",
        ".xls",
        ".h5",
        ".hdf5",
        ".parquet",
    ]
    file_extension = Path(file_path).suffix.lower()

    return file_extension in supported_extensions
