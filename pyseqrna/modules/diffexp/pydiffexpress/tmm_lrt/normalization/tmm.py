"""
TMM normalization for the TMM/LRT component path.

Classes:
    - TMMFactorNormalizer: Trimmed Mean of M-values normalizer

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

import numpy as np

from ...normalization.base import BaseNormalizer
from ...datasets.dataset import ExpressionDataset


class TMMFactorNormalizer(BaseNormalizer):
    """Trimmed Mean of M-values normalizer."""

    def __init__(
        self,
        logratio_trim: float = 0.3,
        sum_trim: float = 0.05,
        do_weighting: bool = True,
        acutoff: float = -1e10,
        p: float = 0.75,
    ):
        super().__init__(name="TMMFactorNormalizer")
        self.logratio_trim = logratio_trim
        self.sum_trim = sum_trim
        self.do_weighting = do_weighting
        self.acutoff = acutoff
        self.p = p

    def _estimate_factors(self, dataset: ExpressionDataset, **kwargs) -> np.ndarray:
        counts = np.asarray(dataset.counts, dtype=float).T  # genes x samples
        nsamples = counts.shape[1]

        if counts.size == 0 or nsamples == 1:
            factors = np.ones(nsamples, dtype=float)
            self._store_dataset_stats(dataset, factors)
            return factors

        lib_size = counts.sum(axis=0)
        all_zero = np.sum(counts > 0, axis=1) == 0
        if np.any(all_zero):
            counts = counts[~all_zero, :]

        if counts.shape[0] == 0:
            factors = np.ones(nsamples, dtype=float)
            self._store_dataset_stats(dataset, factors)
            return factors

        ref_column = self._select_reference_column(counts, lib_size)

        factors = np.full(nsamples, np.nan, dtype=float)
        ref = counts[:, ref_column]
        ref_lib = lib_size[ref_column]
        for i in range(nsamples):
            factors[i] = self._calc_factor_tmm(
                obs=counts[:, i],
                ref=ref,
                libsize_obs=lib_size[i],
                libsize_ref=ref_lib,
            )

        factors = factors / np.exp(np.mean(np.log(factors)))
        self._store_dataset_stats(dataset, factors)
        return factors

    def _select_reference_column(self, counts: np.ndarray, lib_size: np.ndarray) -> int:
        f75 = np.array(
            [np.quantile(counts[:, j], self.p) for j in range(counts.shape[1])],
            dtype=float,
        )
        f75 = f75 / lib_size
        if np.median(f75) < 1e-20:
            return int(np.argmax(np.sum(np.sqrt(counts), axis=0)))
        return int(np.argmin(np.abs(f75 - np.mean(f75))))

    def _calc_factor_tmm(
        self,
        obs: np.ndarray,
        ref: np.ndarray,
        libsize_obs: float,
        libsize_ref: float,
    ) -> float:
        obs = np.asarray(obs, dtype=float)
        ref = np.asarray(ref, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.log2((obs / libsize_obs) / (ref / libsize_ref))
            abs_e = (np.log2(obs / libsize_obs) + np.log2(ref / libsize_ref)) / 2.0
            v = (libsize_obs - obs) / libsize_obs / obs + (libsize_ref - ref) / libsize_ref / ref

        finite = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > self.acutoff)
        log_r = log_r[finite]
        abs_e = abs_e[finite]
        v = v[finite]

        if log_r.size == 0:
            return 1.0
        if np.max(np.abs(log_r)) < 1e-6:
            return 1.0

        n = log_r.size
        lo_l = int(np.floor(n * self.logratio_trim) + 1)
        hi_l = int(n + 1 - lo_l)
        lo_s = int(np.floor(n * self.sum_trim) + 1)
        hi_s = int(n + 1 - lo_s)

        keep = (
            (self._r_rank(log_r) >= lo_l)
            & (self._r_rank(log_r) <= hi_l)
            & (self._r_rank(abs_e) >= lo_s)
            & (self._r_rank(abs_e) <= hi_s)
        )

        if self.do_weighting:
            numerator = np.nansum(log_r[keep] / v[keep])
            denominator = np.nansum(1.0 / v[keep])
            f = numerator / denominator if denominator > 0 else np.nan
        else:
            kept = log_r[keep]
            f = np.nanmean(kept) if kept.size else np.nan

        if np.isnan(f):
            f = 0.0
        return float(2.0**f)

    @staticmethod
    def _r_rank(values: np.ndarray) -> np.ndarray:
        """Replicate R's default rank() behavior with average ties."""
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)

        i = 0
        while i < values.size:
            j = i + 1
            while j < values.size and values[order[j]] == values[order[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    def _store_dataset_stats(self, dataset: ExpressionDataset, factors: np.ndarray) -> None:
        normalized_counts = np.asarray(dataset.counts, dtype=float) / factors[:, None]
        self.size_factors = factors
        self.normalization_factors = factors.copy()
        self.normalized_counts = normalized_counts
        self.base_means = normalized_counts.mean(axis=0)
        self.base_variances = np.var(normalized_counts, axis=0, ddof=1)

        dataset.size_factors = factors
        dataset._adata.obs["size_factors"] = factors
        dataset._adata.obs["normalization_factors"] = factors
        dataset._adata.var["base_mean"] = self.base_means
        dataset._adata.var["base_variance"] = self.base_variances
        dataset._adata.layers["normalized_counts"] = normalized_counts
