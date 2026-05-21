#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TMM (Trimmed Mean of M-values) Normalizer Module

This module implements edgeR-style Trimmed Mean of M-values (TMM) normalization for RNA-seq
read count data. TMM estimates relative library sizes by trimming extreme log-fold changes
(M-values) and overall intensities (A-values) to compute robust normalization factors,
which adjust the library size to output composition-corrected CPM values.

Features:
    - edgeR-style TMM (Trimmed Mean of M-values) normalization method
    - Reference sample selection based on upper-quartile CPM behavior
    - Weighted log-ratio (M-value) and absolute expression (A-value) calculation
    - Rank-average tying logic compatible with R standard library ranking
    - Outputs adjusted normalization factors and effective library sizes

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    logratio_trim, sum_trim, do_weighting, acutoff, ref_quantile, out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - numpy
    - pyseqrna.modules.normalization.base (BaseNormalizer, NormalizationError)

Classes / Functions / Exceptions:
    - TMMNormalizer: Concrete class providing TMM normalization functionality.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import pandas as pd

from .base import BaseNormalizer, NormalizationError


class TMMNormalizer(BaseNormalizer):
    """
    TMM normalizer implementation.

    TMM estimates sample-specific composition factors and uses them to compute
    normalized CPM values: counts / (library_size * norm_factor) * 1e6.
    """

    def __init__(
        self,
        *args,
        logratio_trim: float = 0.3,
        sum_trim: float = 0.05,
        do_weighting: bool = True,
        acutoff: float = -1e10,
        ref_quantile: float = 0.75,
        **kwargs,
    ):
        """Initialize TMMNormalizer."""
        super().__init__(*args, **kwargs)
        self.logratio_trim = logratio_trim
        self.sum_trim = sum_trim
        self.do_weighting = do_weighting
        self.acutoff = acutoff
        self.ref_quantile = ref_quantile
        self.normalization_factors = None
        self.effective_library_sizes = None
        self.logger.info("TMM normalizer initialized")

    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Perform TMM normalization and return TMM-normalized CPM.

        Args:
            plot: Whether to create comparison plots
            save_results: Whether to save results to file
            count_df: Pre-loaded count matrix DataFrame
            gene_lengths: Unused for TMM

        Returns:
            DataFrame containing TMM-normalized CPM values
        """
        try:
            self.logger.info("Starting TMM normalization")
            self._record_internal_operation("normalization_started", "Starting TMM normalization")

            df = count_df if count_df is not None else self.load_count_matrix()
            gene_names = df[self.gene_column]
            count_df_indexed = df.set_index(self.gene_column)
            counts = np.asarray(count_df_indexed, dtype=float)

            if np.any(counts < 0):
                raise NormalizationError("TMM normalization requires non-negative counts")

            lib_sizes = counts.sum(axis=0)
            if np.any(lib_sizes == 0):
                zero_samples = count_df_indexed.columns[lib_sizes == 0].tolist()
                raise NormalizationError(f"Samples with zero total counts found: {zero_samples}")

            self._record_internal_operation(
                "calculation_started",
                f"Calculating TMM factors for {counts.shape[0]} genes across {counts.shape[1]} samples",
            )

            factors = self._estimate_tmm_factors(counts, lib_sizes)
            effective_lib_sizes = lib_sizes * factors
            tmm_cpm = (counts / effective_lib_sizes) * 1e6

            tmm_df = pd.DataFrame(tmm_cpm, index=count_df_indexed.index, columns=count_df_indexed.columns)
            tmm_df.insert(0, self.gene_column, gene_names.values)

            self.normalized_data = tmm_cpm
            self.normalization_factors = pd.DataFrame(
                {
                    "Sample": count_df_indexed.columns,
                    "Library_Size": lib_sizes,
                    "TMM_Factor": factors,
                    "Effective_Library_Size": effective_lib_sizes,
                }
            )
            self.effective_library_sizes = effective_lib_sizes

            if plot:
                fig, ax = self.create_boxplot(
                    raw_data=counts,
                    normalized_data=tmm_cpm,
                    sample_names=count_df_indexed.columns.tolist(),
                    save_plot=True,
                )
                if fig is not None:
                    self.logger.info("TMM comparison plot created successfully")
                    self._record_internal_operation("plot_created", "TMM comparison boxplot created and saved")

            if save_results:
                output_file = self.save_results(tmm_df)
                self._save_factors()
                self.logger.info(f"TMM normalized CPM saved to: {output_file}")
                self._record_internal_operation("results_saved", f"TMM normalized CPM saved to: {output_file}")

            total_genes = len(tmm_df)
            total_samples = len(count_df_indexed.columns)
            mean_tmm_cpm = float(np.nanmean(tmm_cpm))
            self.logger.info("TMM normalization completed successfully")
            self.logger.info(f"Processed {total_genes} genes across {total_samples} samples")
            self.logger.info(f"Mean TMM CPM per sample: {mean_tmm_cpm:.2f}")
            self._record_internal_operation(
                "normalization_completed",
                f"TMM normalization completed: {total_genes} genes, {total_samples} samples, mean TMM CPM: {mean_tmm_cpm:.2f}",
            )

            return tmm_df

        except Exception as e:
            raise NormalizationError(f"TMM normalization failed: {str(e)}")

    def _estimate_tmm_factors(self, counts: np.ndarray, lib_sizes: np.ndarray) -> np.ndarray:
        """Estimate TMM normalization factors for all samples."""
        counts = np.asarray(counts, dtype=float)
        lib_sizes = np.asarray(lib_sizes, dtype=float)
        nonzero_gene_mask = np.sum(counts > 0, axis=1) > 0
        counts = counts[nonzero_gene_mask, :]

        if counts.shape[0] == 0 or counts.shape[1] <= 1:
            return np.ones(len(lib_sizes), dtype=float)

        ref_column = self._select_reference_column(counts, lib_sizes)
        ref_counts = counts[:, ref_column]
        ref_lib_size = lib_sizes[ref_column]

        factors = np.ones(counts.shape[1], dtype=float)
        for index in range(counts.shape[1]):
            factors[index] = self._calc_factor_tmm(
                obs=counts[:, index],
                ref=ref_counts,
                libsize_obs=lib_sizes[index],
                libsize_ref=ref_lib_size,
            )

        geometric_mean = np.exp(np.mean(np.log(factors)))
        if np.isfinite(geometric_mean) and geometric_mean > 0:
            factors = factors / geometric_mean

        return factors

    def _select_reference_column(self, counts: np.ndarray, lib_sizes: np.ndarray) -> int:
        """Select reference sample using upper-quartile CPM behavior."""
        with np.errstate(divide="ignore", invalid="ignore"):
            upper_quartiles = np.array(
                [np.quantile(counts[:, sample], self.ref_quantile) for sample in range(counts.shape[1])],
                dtype=float,
            )
            scaled_upper_quartiles = upper_quartiles / lib_sizes

        if np.nanmedian(scaled_upper_quartiles) < 1e-20:
            return int(np.argmax(np.sum(np.sqrt(counts), axis=0)))

        mean_upper_quartile = np.nanmean(scaled_upper_quartiles)
        return int(np.nanargmin(np.abs(scaled_upper_quartiles - mean_upper_quartile)))

    def _calc_factor_tmm(
        self,
        obs: np.ndarray,
        ref: np.ndarray,
        libsize_obs: float,
        libsize_ref: float,
    ) -> float:
        """Calculate one TMM factor against a reference sample."""
        obs = np.asarray(obs, dtype=float)
        ref = np.asarray(ref, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log2((obs / libsize_obs) / (ref / libsize_ref))
            abs_expr = (np.log2(obs / libsize_obs) + np.log2(ref / libsize_ref)) / 2.0
            variance = ((libsize_obs - obs) / libsize_obs / obs) + ((libsize_ref - ref) / libsize_ref / ref)

        finite = np.isfinite(log_ratio) & np.isfinite(abs_expr) & np.isfinite(variance)
        finite &= abs_expr > self.acutoff
        log_ratio = log_ratio[finite]
        abs_expr = abs_expr[finite]
        variance = variance[finite]

        if log_ratio.size == 0 or np.max(np.abs(log_ratio)) < 1e-6:
            return 1.0

        n_values = log_ratio.size
        lo_log = int(np.floor(n_values * self.logratio_trim) + 1)
        hi_log = int(n_values + 1 - lo_log)
        lo_sum = int(np.floor(n_values * self.sum_trim) + 1)
        hi_sum = int(n_values + 1 - lo_sum)

        keep = (
            (self._r_rank(log_ratio) >= lo_log)
            & (self._r_rank(log_ratio) <= hi_log)
            & (self._r_rank(abs_expr) >= lo_sum)
            & (self._r_rank(abs_expr) <= hi_sum)
        )

        if not np.any(keep):
            return 1.0

        if self.do_weighting:
            weights = 1.0 / variance[keep]
            finite_weights = np.isfinite(weights) & (weights > 0)
            if np.any(finite_weights):
                weighted_mean = np.sum(log_ratio[keep][finite_weights] * weights[finite_weights])
                weighted_mean /= np.sum(weights[finite_weights])
            else:
                weighted_mean = np.nanmean(log_ratio[keep])
        else:
            weighted_mean = np.nanmean(log_ratio[keep])

        if not np.isfinite(weighted_mean):
            weighted_mean = 0.0

        return float(2.0**weighted_mean)

    @staticmethod
    def _r_rank(values: np.ndarray) -> np.ndarray:
        """Replicate R's default rank behavior with average ties."""
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)

        index = 0
        while index < values.size:
            next_index = index + 1
            while next_index < values.size and values[order[next_index]] == values[order[index]]:
                next_index += 1
            avg_rank = (index + 1 + next_index) / 2.0
            ranks[order[index:next_index]] = avg_rank
            index = next_index

        return ranks

    def _save_factors(self) -> str:
        """Save TMM normalization factors alongside normalized counts."""
        output_file = self.out_dir / "TMM_normalization_factors.xlsx"

        if self.dryrun:
            self.logger.info(f"DRYRUN: Would save TMM normalization factors to: {output_file}")
            return str(output_file)

        self.normalization_factors.to_excel(output_file, index=False)
        self.logger.info(f"TMM normalization factors saved to: {output_file}")
        return str(output_file)

    def get_normalization_factors(self) -> pd.DataFrame:
        """Return TMM normalization factors after normalization."""
        if self.normalization_factors is None:
            raise NormalizationError("TMM factors are not available. Run normalize() first.")
        return self.normalization_factors.copy()
