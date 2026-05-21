"""
Positive counts normalization strategy for gene expression data.

This module implements the positive counts normalization method, which
computes size factors using only genes with positive counts across samples.

Classes:
    - PosCountsNormalizer: Positive counts normalization strategy

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np

from .base import BaseNormalizer
from ..datasets.dataset import ExpressionDataset


class PosCountsNormalizer(BaseNormalizer):
    """
    Positive counts normalization strategy.

    This normalizer computes size factors using only genes that have
    positive counts across samples, which can be more robust for
    sparse count data.
    """

    def __init__(self, min_geometric_mean: float = 0.0, min_dispersion: float = 0.0):
        """
        Initialize the positive counts normalizer.

        Parameters
        ----------
        min_geometric_mean : float
            Minimum geometric mean for gene filtering.
        min_dispersion : float
            Minimum dispersion for gene filtering.
        """
        super().__init__(name="PosCountsNormalizer")
        self.min_geometric_mean = min_geometric_mean
        self.min_dispersion = min_dispersion
        self.geometric_means = None

    def _estimate_factors(self, dataset: ExpressionDataset, **kwargs) -> np.ndarray:
        """
        Estimate size factors using positive counts method.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to normalize.
        **kwargs
            Additional parameters (can override min_geometric_mean, min_dispersion).

        Returns
        -------
        np.ndarray
            Size factors for each sample.
        """
        # Override parameters if provided in kwargs
        min_geometric_mean = kwargs.get("min_geometric_mean", self.min_geometric_mean)
        min_dispersion = kwargs.get("min_dispersion", self.min_dispersion)

        counts = dataset.counts

        # Calculate log geometric means over positive counts for each gene.
        lc = np.log(counts)
        lc[~np.isfinite(lc)] = 0  # Replace -inf with 0
        log_geometric_means = np.mean(lc, axis=0)

        # Handle all-zero genes
        all_zero = np.sum(counts, axis=0) == 0
        log_geometric_means[all_zero] = -np.inf

        # Filter genes with infinite log geometric means
        valid_genes = np.isfinite(log_geometric_means)

        # Additional filtering based on geometric mean threshold
        if min_geometric_mean > 0:
            geometric_means = np.exp(log_geometric_means)
            valid_genes &= geometric_means > min_geometric_mean

        if min_dispersion > 0:
            # Calculate dispersion for filtering
            normalized_counts = counts / np.exp(log_geometric_means).reshape(1, -1)
            dispersion = np.var(normalized_counts, axis=0) / np.mean(normalized_counts, axis=0)
            valid_genes &= dispersion > min_dispersion

        if np.sum(valid_genes) == 0:
            raise ValueError("No genes passed filtering criteria")

        # Check if all genes have zeros
        if np.isinf(log_geometric_means).all():
            raise ValueError("Every gene contains at least one zero, cannot compute log geometric means")

        # Compute size factors from positive-count ratios.
        def sf_compute(cnts):
            with np.errstate(invalid="ignore", divide="ignore"):
                ratios = (np.log(cnts) - log_geometric_means)[np.isfinite(log_geometric_means) & (cnts > 0)]
                return np.exp(np.median(ratios))

        size_factors = np.apply_along_axis(sf_compute, 1, counts)

        # The poscounts estimator computes pseudo
        # geometric means first, then calls estimateSizeFactorsForMatrix with
        # those geoMeans. In that code path, size factors are centered to have
        # geometric mean 1.
        size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))

        # Store geometric means for reference
        self.geometric_means = np.exp(log_geometric_means)

        # Calculate normalized counts, base means, and variances
        # Ensure we have numpy arrays
        counts_array = np.asarray(counts)
        size_factors_array = np.asarray(size_factors)

        # Calculate normalized counts (samples x genes)
        self.normalized_counts = counts_array / size_factors_array.reshape(-1, 1)

        # Calculate base means (mean across samples for each gene)
        self.base_means = np.mean(self.normalized_counts, axis=0)

        # Calculate base variances (variance across samples for each gene)
        self.base_variances = np.var(self.normalized_counts, axis=0, ddof=1)

        # Store results in dataset and AnnData
        dataset.size_factors = size_factors  # Use the property setter
        dataset._adata.obs["size_factors"] = size_factors  # Store in obs (per-sample)
        dataset._adata.var["base_mean"] = self.base_means
        dataset._adata.var["base_variance"] = self.base_variances
        dataset._adata.layers["normalized_counts"] = self.normalized_counts  # Store in layers (matrix)

        return size_factors

    def _get_summary(self) -> dict:
        """Get summary including geometric means and normalization results."""
        summary = super()._get_summary()

        if self.geometric_means is not None:
            summary.update(
                {
                    "geometric_means_stats": {
                        "mean": np.mean(self.geometric_means),
                        "median": np.median(self.geometric_means),
                        "std": np.std(self.geometric_means),
                        "min": np.min(self.geometric_means),
                        "max": np.max(self.geometric_means),
                    },
                    "filtering_params": {
                        "min_geometric_mean": self.min_geometric_mean,
                        "min_dispersion": self.min_dispersion,
                        "positive_counts_only": True,
                    },
                }
            )

        # Add normalization results
        if hasattr(self, "base_means") and self.base_means is not None:
            summary.update(
                {
                    "base_means_stats": {
                        "mean": np.mean(self.base_means),
                        "median": np.median(self.base_means),
                        "std": np.std(self.base_means),
                        "min": np.min(self.base_means),
                        "max": np.max(self.base_means),
                    }
                }
            )

        if hasattr(self, "base_variances") and self.base_variances is not None:
            summary.update(
                {
                    "base_variances_stats": {
                        "mean": np.mean(self.base_variances),
                        "median": np.median(self.base_variances),
                        "std": np.std(self.base_variances),
                        "min": np.min(self.base_variances),
                        "max": np.max(self.base_variances),
                    }
                }
            )

        return summary
