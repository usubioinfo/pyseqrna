#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TPM (Transcripts Per Million) Normalizer Module

This module provides Transcripts Per Million (TPM) normalization for RNA-seq count data.
TPM normalization adjusts raw read counts for gene length first (giving reads per kilobase),
and then normalizes by the total abundance of these length-corrected counts per sample to
allow comparisons of relative transcript abundance.

Features:
    - TPM normalization algorithm implementation
    - Length-first normalisation (adjusting counts by gene kilobase length) followed by library size scaling
    - Automatically handles intersection between count matrix genes and annotation gene lengths
    - Generates log-transformed boxplot comparisons (raw vs TPM normalized counts)
    - Exports normalized results to Excel spreadsheets

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    annotation_file, out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - numpy
    - pyseqrna.modules.normalization.base (BaseNormalizer, NormalizationError)

Classes / Functions / Exceptions:
    - TPMNormalizer: Concrete class providing TPM normalization functionality.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
import numpy as np

from .base import BaseNormalizer, NormalizationError


class TPMNormalizer(BaseNormalizer):
    """
    TPM (Transcripts Per Million) normalizer implementation.

    TPM normalization accounts for both gene length and library size.
    Formula: TPM = (counts / gene_length) * 1e6 / sum(counts / gene_length)
    """

    def __init__(self, *args, **kwargs):
        """Initialize TPMNormalizer."""
        super().__init__(*args, **kwargs)
        self.logger.info("TPM normalizer initialized")

    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Perform TPM normalization on count data.

        Args:
            plot: Whether to create comparison plots
            save_results: Whether to save results to file
            count_df: Pre-loaded count matrix DataFrame (to avoid duplicate loading)
            gene_lengths: Pre-loaded gene lengths DataFrame (to avoid duplicate loading)

        Returns:
            DataFrame containing TPM normalized counts
        """
        try:
            self.logger.info("Starting TPM normalization")

            # Record this operation for the execution report
            self._record_internal_operation("normalization_started", "Starting TPM normalization")

            # Use pre-loaded data if available, otherwise load
            if count_df is None:
                df = self.load_count_matrix()
            else:
                df = count_df
            if gene_lengths is None:
                gene_lengths = self._extract_gene_lengths()

            # Extract gene names and count data
            df[self.gene_column]
            count_df = df.set_index(self.gene_column)

            # Match genes between count data and length data
            matched_genes = pd.Index.intersection(count_df.index, gene_lengths.index)

            if len(matched_genes) == 0:
                raise NormalizationError("No matching genes found between count matrix and annotation file")

            # Record calculation start
            self._record_internal_operation(
                "calculation_started",
                f"Calculating TPM for {len(matched_genes)} genes across {len(count_df.columns)} samples with gene length normalization",
            )

            counts = np.asarray(count_df.loc[matched_genes], dtype=float)
            lengths = np.asarray(gene_lengths.loc[matched_genes]["Length"], dtype=float)

            # Calculate TPM: normalize by gene length first, then by library size
            counts_per_kb = counts / (lengths[:, np.newaxis] / 1000)
            tpm = (counts_per_kb * 1e6) / counts_per_kb.sum(axis=0)

            # Create TPM DataFrame
            tpm_df = pd.DataFrame(data=tpm, index=matched_genes, columns=count_df.columns)
            tpm_df.insert(0, self.gene_column, matched_genes.values)

            # Store normalized data
            self.normalized_data = tpm

            # Record calculation completion
            mean_tpm_per_sample = tpm.mean(axis=0).mean()
            self._record_internal_operation(
                "calculation_completed",
                f"TPM calculation completed. Mean TPM per sample: {mean_tpm_per_sample:.2f}",
            )

            # Create plots and save results
            if plot:
                sample_names = count_df.columns.tolist()
                fig, ax = self.create_boxplot(counts, tpm, sample_names, save_plot=True)
                if fig is not None:
                    self._record_internal_operation("plot_created", "TPM comparison boxplot created and saved")

            if save_results:
                output_file = self.save_results(tpm_df)
                self._record_internal_operation("results_saved", f"TPM normalized counts saved to: {output_file}")

            # Record overall completion
            self._record_internal_operation(
                "normalization_completed",
                f"TPM normalization completed: {len(matched_genes)} genes, {len(count_df.columns)} samples, mean TPM: {mean_tpm_per_sample:.2f}",
            )

            self.logger.info(f"TPM normalization completed: {len(matched_genes)} genes")
            return tpm_df

        except Exception as e:
            raise NormalizationError(f"TPM normalization failed: {str(e)}")
