"""
Design matrix utilities for PyDiffExpress.

This module provides functions for creating design matrices from sample
metadata using patsy (like inmoose).

Functions:
    - create_design_matrix: Create a design matrix from sample metadata using patsy
    - detect_design_columns: Automatically detect potential design columns from sample metadata
    - validate_design_matrix: Validate that the design matrix is properly formed

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import patsy


def _rename_model_matrix_columns(design_matrix, sample_metadata):
    """Convert patsy column names to inmoose-style names."""
    # Get categorical factors from design matrix
    factors = [info for f, info in design_matrix.design_info.factor_infos.items() if info.type == "categorical"]

    # Create mapping from patsy names to inmoose names
    name_mapping = {}
    for f in factors:
        # Extract factor name (remove C() wrapper)
        factor_name = f.factor.name()
        if factor_name.startswith("C(") and factor_name.endswith(")"):
            factor_name = factor_name[2:-1]  # Remove C() wrapper

        for lvl in f.categories[1:]:  # Skip reference level
            patsy_name = f"{f.factor.name()}[T.{lvl}]"
            inmoose_name = f"{factor_name}_{lvl}_vs_{f.categories[0]}"
            name_mapping[patsy_name] = inmoose_name

    # Get original column names
    original_names = list(design_matrix.design_info.column_name_indexes.keys())

    # Create new column names
    new_names = []
    for name in original_names:
        if name in name_mapping:
            new_names.append(name_mapping[name])
        else:
            new_names.append(name)

    return new_names


def create_design_matrix(
    sample_metadata: pd.DataFrame,
    design_formula: Optional[str] = None,
    design_column: Optional[str] = None,
    group_column: Optional[str] = None,
    additional_columns: Optional[List[str]] = None,
    intercept: bool = True,
) -> tuple[np.ndarray, List[str]]:
    """
    Create a design matrix from sample metadata using patsy.

    Parameters
    ----------
    sample_metadata : pd.DataFrame
        Sample metadata with samples as rows and covariates as columns.
    design_formula : Optional[str]
        Patsy formula for creating the design matrix (e.g., "~ C(condition)").
        If None, will construct formula from columns.
    design_column : Optional[str]
        Name of the column containing the main design factor (e.g., "condition").
        Used when design_formula is None.
    group_column : Optional[str]
        Name of the column containing grouping information (e.g., "batch").
        Used when design_formula is None.
    additional_columns : Optional[List[str]]
        Additional columns to include in the design matrix.
        Used when design_formula is None.
    intercept : bool
        Whether to include an intercept term in the design matrix.

    Returns
    -------
    tuple[np.ndarray, List[str]]
        Design matrix with samples as rows and factors as columns, and column names.

    Raises
    ------
    ValueError
        If required columns are not found or formula is invalid.
    """

    # If no formula provided, construct one from columns
    if design_formula is None:
        formula_terms = []

        if intercept:
            formula_terms.append("1")

        if design_column is not None:
            if design_column not in sample_metadata.columns:
                raise ValueError(
                    f"Design column '{design_column}' not found in sample metadata. "
                    f"Available columns: {list(sample_metadata.columns)}"
                )
            formula_terms.append(f"C({design_column})")

        if group_column is not None:
            if group_column not in sample_metadata.columns:
                raise ValueError(
                    f"Group column '{group_column}' not found in sample metadata. "
                    f"Available columns: {list(sample_metadata.columns)}"
                )
            formula_terms.append(f"C({group_column})")

        if additional_columns is not None:
            for col in additional_columns:
                if col not in sample_metadata.columns:
                    raise ValueError(
                        f"Additional column '{col}' not found in sample metadata. "
                        f"Available columns: {list(sample_metadata.columns)}"
                    )
                formula_terms.append(f"C({col})")

        if not formula_terms:
            raise ValueError(
                "No columns specified for design matrix. Provide design_column, group_column, or additional_columns."
            )

        design_formula = "~ " + " + ".join(formula_terms)

    # Create design matrix using patsy (like inmoose)
    try:
        design_matrix = patsy.dmatrix(design_formula, data=sample_metadata, NA_action="raise")

        # Convert to inmoose-style column names
        column_names = _rename_model_matrix_columns(design_matrix, sample_metadata)

        return design_matrix, column_names
    except Exception as e:
        raise ValueError(f"Error creating design matrix from formula '{design_formula}': {str(e)}")


def detect_design_columns(sample_metadata: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect potential design columns from sample metadata.

    Parameters
    ----------
    sample_metadata : pd.DataFrame
        Sample metadata.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with detected column types.
    """
    categorical_cols = []
    numeric_cols = []

    for col in sample_metadata.columns:
        values = sample_metadata[col]
        if values.dtype == "object" or values.dtype.name == "category":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return {
        "categorical": categorical_cols,
        "numeric": numeric_cols,
        "all": list(sample_metadata.columns),
    }


def validate_design_matrix(design_matrix: np.ndarray, sample_metadata: pd.DataFrame) -> bool:
    """
    Validate that the design matrix is properly formed.

    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix to validate.
    sample_metadata : pd.DataFrame
        Sample metadata for validation.

    Returns
    -------
    bool
        True if design matrix is valid.
    """
    if design_matrix.shape[0] != len(sample_metadata):
        return False

    if design_matrix.shape[1] == 0:
        return False

    # Check for NaN or infinite values
    if np.any(np.isnan(design_matrix)) or np.any(np.isinf(design_matrix)):
        return False

    return True
