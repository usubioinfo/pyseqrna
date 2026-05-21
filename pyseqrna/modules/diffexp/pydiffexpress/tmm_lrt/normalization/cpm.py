"""
CPM/logCPM helpers for effective-library-size based abundance summaries.

Functions:
    - cpm: Compute counts-per-million using effective library sizes
    - ave_log_cpm: Approximate average log-CPM using a one-group NB fit after prior-count

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def cpm(
    counts: np.ndarray,
    lib_size: np.ndarray,
    log: bool = False,
    prior_count: float = 2.0,
) -> np.ndarray:
    """
    Compute counts-per-million using effective library sizes.

    Parameters
    ----------
    counts
        Matrix of raw counts with shape (samples, genes).
    lib_size
        Effective library sizes for each sample.
    log
        Whether to return log2-CPM values.
    prior_count
        Prior count used for log-CPM shrinkage.
    """
    counts = np.asarray(counts, dtype=float)
    lib_size = np.asarray(lib_size, dtype=float)
    lib_mat = lib_size[:, None]

    if log:
        # Scale the prior to the library size before taking log2 CPM.
        mean_lib = np.exp(np.mean(np.log(lib_size)))
        scaled_prior = prior_count * lib_mat / mean_lib
        numer = counts + scaled_prior
        denom = lib_mat + 2.0 * scaled_prior
        return np.log2(numer / denom * 1e6)

    return counts / lib_mat * 1e6


def ave_log_cpm(
    counts: np.ndarray,
    lib_size: np.ndarray,
    prior_count: float = 2.0,
    dispersion: float | np.ndarray = 0.05,
) -> np.ndarray:
    """
    Approximate average log-CPM using a one-group NB fit after prior-count
    adjustment, then convert the fitted abundance to log2 CPM.
    """
    counts = np.asarray(counts, dtype=float)
    lib_size = np.asarray(lib_size, dtype=float)
    n_samples, n_genes = counts.shape
    mean_lib = np.exp(np.mean(np.log(lib_size)))

    scaled_prior = prior_count * (lib_size[:, None] / mean_lib)
    adjusted_counts = counts + scaled_prior
    adjusted_offset = np.log(lib_size[:, None] + 2.0 * scaled_prior)[:, 0]

    dispersions = np.asarray(dispersion, dtype=float)
    if dispersions.ndim == 0:
        dispersions = np.full(n_genes, float(dispersions))
    elif dispersions.size != n_genes:
        raise ValueError("dispersion must be scalar or length equal to number of genes")

    abundance = np.full(n_genes, np.nan, dtype=float)
    for gene_idx in range(n_genes):
        yy = adjusted_counts[:, gene_idx]
        alpha = float(dispersions[gene_idx])

        def score(eta: float) -> float:
            mu = np.exp(eta + adjusted_offset)
            return np.sum((yy - mu) / (1.0 + alpha * mu))

        try:
            abundance[gene_idx] = brentq(score, -50.0, 50.0)
        except ValueError:
            # Fall back to the logCPM average if the score does not bracket a root.
            abundance[gene_idx] = np.nan

    ave = (abundance + np.log(1e6)) / np.log(2)

    if np.any(~np.isfinite(ave)):
        fallback = np.mean(cpm(counts, lib_size, log=True, prior_count=prior_count), axis=0)
        ave = np.where(np.isfinite(ave), ave, fallback)

    return ave
