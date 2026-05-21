#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Median Ratio Normalizer Module

This module implements DESeq-style Median Ratio normalization for RNA-seq read counts.
It works by computing geometric means for each gene across samples, calculating ratios of
counts to these geometric means, and using the median of these ratios per sample as size
factors to scale the raw counts to account for sequencing depth and library composition.

Features:
    - DESeq-style Median Ratio normalization method implementation
    - Calculation of gene-specific geometric means across all samples
    - Automatic derivation of sample size factors based on median ratios
    - Filters out genes with zero counts across all samples during size factor calculation
    - Generation of log-transformed boxplot comparisons (raw vs normalized counts)

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    ref_sample, out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - pyseqrna.modules.normalization.base (BaseNormalizer, NormalizationError)

Classes / Functions / Exceptions:
    - MedianRatioNormalizer: Concrete class providing Median Ratio normalization functionality.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
from .base import BaseNormalizer, NormalizationError


class MedianRatioNormalizer(BaseNormalizer):
    """Median ratio normalizer implementation."""

    def __init__(self, *args, ref_sample: str = None, **kwargs):
        """
        Initialize MedianRatio normalizer.

        Args:
            *args: Positional arguments passed to BaseNormalizer
            ref_sample: Reference sample for normalization
            **kwargs: Keyword arguments passed to BaseNormalizer
        """
        super().__init__(*args, **kwargs)
        self.ref_sample = ref_sample
        self.logger.info("Median ratio normalizer initialized")

    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Perform median ratio normalization."""
        try:
            self.logger.info("Starting median ratio normalization")

            # Record this operation for the execution report
            self._record_internal_operation("normalization_started", "Starting median ratio normalization")

            # Use pre-loaded data if available, otherwise load
            if count_df is None:
                df = self.load_count_matrix()
            else:
                df = count_df
            gene_names = df[self.gene_column]
            count_df = df.set_index(self.gene_column)

            # Record calculation start
            self._record_internal_operation(
                "calculation_started",
                f"Calculating median ratio normalization for {len(count_df)} genes across {len(count_df.columns)} samples",
            )

            # Remove genes with zero counts across all samples
            non_zero_genes = count_df[(count_df != 0).any(axis=1)]

            # Calculate geometric mean for each gene
            geometric_means = non_zero_genes.mean(axis=1)

            # Calculate ratio of each sample to geometric mean
            ratios = non_zero_genes.div(geometric_means, axis=0)

            # Calculate median ratio for each sample (size factor)
            size_factors = ratios.median()

            # Normalize counts by dividing by size factors
            normalized_counts = count_df.div(size_factors, axis=1)

            # Create normalized DataFrame
            normalized_df = normalized_counts.copy()
            normalized_df.insert(0, self.gene_column, gene_names.values)

            self.normalized_data = normalized_counts.values

            # Record calculation completion
            mean_counts_per_sample = normalized_counts.mean(axis=0).mean()
            self._record_internal_operation(
                "calculation_completed",
                f"Median ratio normalization completed. Mean normalized counts per sample: {mean_counts_per_sample:.2f}",
            )

            if plot:
                sample_names = count_df.columns.tolist()
                fig, ax = self.create_boxplot(
                    count_df.values,
                    normalized_counts.values,
                    sample_names,
                    save_plot=True,
                )
                if fig is not None:
                    self._record_internal_operation(
                        "plot_created",
                        "Median ratio comparison boxplot created and saved",
                    )

            if save_results:
                output_file = self.save_results(normalized_df)
                self._record_internal_operation(
                    "results_saved",
                    f"Median ratio normalized counts saved to: {output_file}",
                )

            # Record overall completion
            self._record_internal_operation(
                "normalization_completed",
                f"Median ratio normalization completed: {len(count_df)} genes, {len(count_df.columns)} samples, mean counts: {mean_counts_per_sample:.2f}",
            )

            self.logger.info(f"Median ratio normalization completed: {len(count_df)} genes")
            return normalized_df

        except Exception as e:
            raise NormalizationError(f"Median ratio normalization failed: {str(e)}")
