"""
Result export helpers for the TMM/LRT component path.

Functions:
    - bh_adjust: Benjamini-Hochberg adjustment with NaN preservation
    - export_lrt_contrast: Build a standalone LRT contrast table

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment with NaN preservation."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if np.any(valid):
        adjusted[valid] = multipletests(p_values[valid], method="fdr_bh")[1]
    return adjusted


def export_lrt_contrast(
    genes: np.ndarray,
    logfc: np.ndarray,
    logcpm: np.ndarray,
    lr: np.ndarray,
    pvalue: np.ndarray,
) -> pd.DataFrame:
    """Build a standalone LRT contrast table."""
    logfc = np.asarray(logfc, dtype=float)
    logcpm = np.asarray(logcpm, dtype=float)
    lr = np.asarray(lr, dtype=float)
    pvalue = np.asarray(pvalue, dtype=float)

    # Keep fully populated result tables for genes with failed fits.
    lr = np.where(np.isfinite(lr), lr, 0.0)
    pvalue = np.where(np.isfinite(pvalue), pvalue, 1.0)
    logfc = np.where(np.isfinite(logfc), logfc, 0.0)
    logcpm = np.where(np.isfinite(logcpm), logcpm, 0.0)

    fdr = bh_adjust(pvalue)
    return pd.DataFrame(
        {
            "Gene": np.asarray(genes, dtype=str),
            "logFC": logfc,
            "logCPM": logcpm,
            "LR": lr,
            "pvalue": pvalue,
            "FDR": fdr,
        }
    )
