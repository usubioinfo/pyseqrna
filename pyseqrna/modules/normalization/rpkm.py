#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RPKM (Reads Per Kilobase Million) Normalizer Module

This module implements Reads Per Kilobase Million (RPKM) normalization for RNA-seq
read counts. RPKM adjusts raw counts by correcting for both sequencing depth (total
mapped reads) and gene length (derived from the annotation GFF/GTF file) to allow
expression level comparisons between genes and samples.

Features:
    - RPKM normalization algorithm implementation
    - Reads-based scaling (typically for single-end sequencing datasets)
    - Automatically handles intersection between count matrix genes and annotation gene lengths
    - Generates log-transformed boxplot comparisons (raw vs RPKM normalized counts)
    - Exports normalized results to Excel spreadsheets

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    annotation_file, out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - numpy
    - pyseqrna.modules.normalization.base (BaseNormalizer, NormalizationError)

Classes / Functions / Exceptions:
    - RPKMNormalizer: Concrete class providing RPKM normalization functionality.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
import numpy as np
from .base import BaseNormalizer, NormalizationError


class RPKMNormalizer(BaseNormalizer):
    """RPKM normalizer implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("RPKM normalizer initialized")

    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Perform RPKM normalization."""
        try:
            self.logger.info("Starting RPKM normalization")

            # Record this operation for the execution report
            self._record_internal_operation("normalization_started", "Starting RPKM normalization")

            # Use pre-loaded data if available, otherwise load
            if count_df is None:
                df = self.load_count_matrix()
            else:
                df = count_df

            if gene_lengths is None:
                gene_lengths = self._extract_gene_lengths()

            count_df_indexed = df.set_index(self.gene_column)
            matched_genes = pd.Index.intersection(count_df_indexed.index, gene_lengths.index)

            if len(matched_genes) == 0:
                raise NormalizationError("No matching genes found between count matrix and annotation file")

            # Record calculation start
            self._record_internal_operation(
                "calculation_started",
                f"Calculating RPKM for {len(matched_genes)} genes across {len(count_df_indexed.columns)} samples with gene length normalization",
            )

            counts = np.asarray(count_df_indexed.loc[matched_genes], dtype=float)
            lengths = np.asarray(gene_lengths.loc[matched_genes]["Length"], dtype=float)

            total_reads = counts.sum(axis=0)
            rpkm = 1e9 * counts / (total_reads[np.newaxis, :] * lengths[:, np.newaxis])

            rpkm_df = pd.DataFrame(data=rpkm, index=matched_genes, columns=count_df_indexed.columns)
            rpkm_df.insert(0, self.gene_column, matched_genes.values)

            self.normalized_data = rpkm

            # Record calculation completion
            mean_rpkm_per_sample = rpkm.mean(axis=0).mean()
            self._record_internal_operation(
                "calculation_completed",
                f"RPKM calculation completed. Mean RPKM per sample: {mean_rpkm_per_sample:.2f}",
            )

            if plot:
                sample_names = count_df_indexed.columns.tolist()
                fig, ax = self.create_boxplot(counts, rpkm, sample_names, save_plot=True)
                if fig is not None:
                    self._record_internal_operation("plot_created", "RPKM comparison boxplot created and saved")

            if save_results:
                output_file = self.save_results(rpkm_df)
                self._record_internal_operation("results_saved", f"RPKM normalized counts saved to: {output_file}")

            # Record overall completion
            self._record_internal_operation(
                "normalization_completed",
                f"RPKM normalization completed: {len(matched_genes)} genes, {len(count_df_indexed.columns)} samples, mean RPKM: {mean_rpkm_per_sample:.2f}",
            )

            self.logger.info(f"RPKM normalization completed: {len(matched_genes)} genes")
            return rpkm_df

        except Exception as e:
            raise NormalizationError(f"RPKM normalization failed: {str(e)}")
