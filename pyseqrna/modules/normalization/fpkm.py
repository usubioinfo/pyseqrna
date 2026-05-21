#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FPKM (Fragments Per Kilobase Million) Normalizer Module

This module implements Fragments Per Kilobase Million (FPKM) normalization for RNA-seq
read counts. FPKM is a fragment-based normalization method (typically for paired-end
sequencing data) that adjusts raw counts to account for both gene length and sequencing depth.

Features:
    - FPKM normalization algorithm implementation
    - Fragment-based scaling (paired-end read fragments count, dividing total reads by two)
    - Automatically handles intersection between count matrix genes and annotation gene lengths
    - Generates log-transformed boxplot comparisons (raw vs FPKM normalized counts)
    - Exports normalized results to Excel spreadsheets

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    annotation_file, out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - numpy
    - pyseqrna.modules.normalization.base (BaseNormalizer, NormalizationError)

Classes / Functions / Exceptions:
    - FPKMNormalizer: Concrete class providing FPKM normalization functionality.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
import numpy as np
from .base import BaseNormalizer, NormalizationError


class FPKMNormalizer(BaseNormalizer):
    """FPKM normalizer implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("FPKM normalizer initialized")

    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Perform FPKM normalization."""
        try:
            self.logger.info("Starting FPKM normalization")

            # Record this operation for the execution report
            self._record_internal_operation("normalization_started", "Starting FPKM normalization")

            # Use pre-loaded data if available, otherwise load
            if count_df is None:
                df = self.load_count_matrix()
            else:
                df = count_df
            if gene_lengths is None:
                gene_lengths = self._extract_gene_lengths()

            count_df = df.set_index(self.gene_column)
            matched_genes = pd.Index.intersection(count_df.index, gene_lengths.index)

            if len(matched_genes) == 0:
                raise NormalizationError("No matching genes found between count matrix and annotation file")

            # Record calculation start
            self._record_internal_operation(
                "calculation_started",
                f"Calculating FPKM for {len(matched_genes)} genes across {len(count_df.columns)} samples (fragment-based normalization)",
            )

            counts = np.asarray(count_df.loc[matched_genes], dtype=float)
            lengths = np.asarray(gene_lengths.loc[matched_genes]["Length"], dtype=float)

            # FPKM: fragments = total_reads / 2 for paired-end
            total_fragments = counts.sum(axis=0) / 2
            fpkm = 1e9 * counts / (total_fragments[np.newaxis, :] * lengths[:, np.newaxis])

            fpkm_df = pd.DataFrame(data=fpkm, index=matched_genes, columns=count_df.columns)
            fpkm_df.insert(0, self.gene_column, matched_genes.values)

            self.normalized_data = fpkm

            # Record calculation completion
            mean_fpkm_per_sample = fpkm.mean(axis=0).mean()
            self._record_internal_operation(
                "calculation_completed",
                f"FPKM calculation completed. Mean FPKM per sample: {mean_fpkm_per_sample:.2f}",
            )

            if plot:
                sample_names = count_df.columns.tolist()
                fig, ax = self.create_boxplot(counts, fpkm, sample_names, save_plot=True)
                if fig is not None:
                    self._record_internal_operation("plot_created", "FPKM comparison boxplot created and saved")

            if save_results:
                output_file = self.save_results(fpkm_df)
                self._record_internal_operation("results_saved", f"FPKM normalized counts saved to: {output_file}")

            # Record overall completion
            self._record_internal_operation(
                "normalization_completed",
                f"FPKM normalization completed: {len(matched_genes)} genes, {len(count_df.columns)} samples, mean FPKM: {mean_fpkm_per_sample:.2f}",
            )

            self.logger.info(f"FPKM normalization completed: {len(matched_genes)} genes")
            return fpkm_df

        except Exception as e:
            raise NormalizationError(f"FPKM normalization failed: {str(e)}")
