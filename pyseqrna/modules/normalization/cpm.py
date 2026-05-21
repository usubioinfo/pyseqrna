#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPM (Counts Per Million) Normalizer Module

This module provides Counts Per Million (CPM) normalization for RNA-seq count data.
CPM normalizes raw counts by the library size (total number of mapped reads) per sample,
scaling counts to a depth of one million reads to account for sequencing depth differences.

Features:
    - Normalization of raw count matrices to Counts Per Million (CPM)
    - Automated detection and protection against samples with zero total reads
    - Methods to retrieve sample-specific scaling factors and library sizes
    - Built-in validation checks to ensure normalized counts sum to 1,000,000 per sample
    - Generation of comparison boxplots (raw vs CPM log-transformed counts)

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - numpy
    - pyseqrna.modules.normalization.base (BaseNormalizer, NormalizationError)

Classes / Functions / Exceptions:
    - CPMNormalizer: Concrete class providing CPM normalization functionality.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Any, Dict

from .base import BaseNormalizer, NormalizationError


class CPMNormalizer(BaseNormalizer):
    """
    CPM (Counts Per Million) normalizer implementation.

    This class provides functionality to normalize raw counts to counts per million (CPM).
    CPM normalization divides each count by the total number of counts in the sample
    and multiplies by 1 million.

    Formula: CPM = (counts * 1,000,000) / total_counts_per_sample
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize CPMNormalizer.

        Args:
            *args: Positional arguments passed to BaseNormalizer
            **kwargs: Keyword arguments passed to BaseNormalizer
        """
        super().__init__(*args, **kwargs)
        self.logger.info("CPM normalizer initialized")

    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Perform CPM normalization on count data.

        Args:
            plot: Whether to create comparison plots
            save_results: Whether to save results to file
            count_df: Pre-loaded count matrix DataFrame (to avoid duplicate loading)
            gene_lengths: Pre-loaded gene lengths DataFrame (unused for CPM)

        Returns:
            DataFrame containing CPM normalized counts

        Raises:
            NormalizationError: If CPM normalization fails
        """
        try:
            self.logger.info("Starting CPM normalization")

            # Record this operation for the execution report
            self._record_internal_operation("normalization_started", "Starting CPM normalization")

            # Use pre-loaded data if available, otherwise load
            if count_df is None:
                df = self.load_count_matrix()
            else:
                df = count_df

            # Extract gene names and count data
            gene_names = df[self.gene_column]
            count_df_indexed = df.set_index(self.gene_column)

            # Convert to numpy array for calculation
            counts = np.asarray(count_df_indexed, dtype=float)

            # Record calculation start
            self._record_internal_operation(
                "calculation_started",
                f"Calculating CPM for {len(gene_names)} genes across {len(count_df_indexed.columns)} samples",
            )

            # Calculate CPM: (counts * 1e6) / total_counts_per_sample
            total_counts_per_sample = counts.sum(axis=0)

            # Check for samples with zero total counts
            if np.any(total_counts_per_sample == 0):
                zero_samples = count_df_indexed.columns[total_counts_per_sample == 0].tolist()
                raise NormalizationError(f"Samples with zero total counts found: {zero_samples}")

            cpm = (counts * 1e6) / total_counts_per_sample

            # Create CPM DataFrame
            cpm_df = pd.DataFrame(data=cpm, index=count_df_indexed.index, columns=count_df_indexed.columns)

            # Add gene names back as first column
            cpm_df.insert(0, self.gene_column, gene_names.values)

            # Store normalized data
            self.normalized_data = cpm

            # Record calculation completion
            mean_cpm_per_sample = cpm.mean(axis=0).mean()
            self._record_internal_operation(
                "calculation_completed",
                f"CPM calculation completed. Mean CPM per sample: {mean_cpm_per_sample:.2f}",
            )

            # Create comparison plot if requested
            if plot:
                sample_names = count_df_indexed.columns.tolist()
                fig, ax = self.create_boxplot(
                    raw_data=counts,
                    normalized_data=cpm,
                    sample_names=sample_names,
                    save_plot=True,
                )

                if fig is not None:
                    self.logger.info("CPM comparison plot created successfully")
                    self._record_internal_operation("plot_created", "CPM comparison boxplot created and saved")

            # Save results if requested
            if save_results:
                output_file = self.save_results(cpm_df)
                # Sanitize file path for logging to prevent log injection
                safe_output_file = str(output_file).replace("\n", "").replace("\r", "")
                self.logger.info(f"CPM normalized counts saved to: {safe_output_file}")
                self._record_internal_operation(
                    "results_saved",
                    f"CPM normalized counts saved to: {safe_output_file}",
                )

            # Log summary statistics
            total_genes = len(cpm_df)
            total_samples = len(count_df_indexed.columns)

            self.logger.info("CPM normalization completed successfully")
            safe_total_genes = str(total_genes).replace("\n", "").replace("\r", "")
            safe_total_samples = str(total_samples).replace("\n", "").replace("\r", "")
            safe_mean_cpm = str(f"{mean_cpm_per_sample:.2f}").replace("\n", "").replace("\r", "")
            self.logger.info(f"Processed {safe_total_genes} genes across {safe_total_samples} samples")
            self.logger.info(f"Mean CPM per sample: {safe_mean_cpm}")

            # Record overall completion
            self._record_internal_operation(
                "normalization_completed",
                f"CPM normalization completed: {total_genes} genes, {total_samples} samples, mean CPM: {mean_cpm_per_sample:.2f}",
            )

            return cpm_df

        except Exception as e:
            raise NormalizationError(f"CPM normalization failed: {str(e)}")

    def get_scaling_factors(self) -> pd.DataFrame:
        """
        Get the scaling factors used for CPM normalization.

        Returns:
            DataFrame containing scaling factors for each sample
        """
        if self.count_data is None:
            raise NormalizationError("Count data not loaded. Run normalize() first.")

        try:
            # Extract count data
            count_df = self.count_data.set_index(self.gene_column)
            counts = np.asarray(count_df, dtype=float)

            # Calculate total counts per sample
            total_counts = counts.sum(axis=0)

            # Calculate scaling factors (total_counts / 1e6)
            scaling_factors = total_counts / 1e6

            # Create DataFrame
            scaling_df = pd.DataFrame(
                {
                    "Sample": count_df.columns,
                    "Total_Counts": total_counts,
                    "Scaling_Factor": scaling_factors,
                }
            )

            return scaling_df

        except Exception as e:
            raise NormalizationError(f"Failed to calculate scaling factors: {str(e)}")

    def get_library_sizes(self) -> pd.DataFrame:
        """
        Get library sizes (total counts) for each sample.

        Returns:
            DataFrame containing library sizes
        """
        if self.count_data is None:
            raise NormalizationError("Count data not loaded. Run normalize() first.")

        try:
            # Extract count data
            count_df = self.count_data.set_index(self.gene_column)
            counts = np.asarray(count_df, dtype=float)

            # Calculate total counts per sample
            total_counts = counts.sum(axis=0)

            # Create DataFrame
            library_sizes = pd.DataFrame({"Sample": count_df.columns, "Library_Size": total_counts})

            return library_sizes

        except Exception as e:
            raise NormalizationError(f"Failed to calculate library sizes: {str(e)}")

    def validate_normalization(self) -> Dict[str, Any]:
        """
        Validate CPM normalization results.

        Returns:
            Dictionary containing validation metrics
        """
        if self.normalized_data is None:
            raise NormalizationError("Normalized data not available. Run normalize() first.")

        try:
            # Check if CPM sums to approximately 1e6 per sample
            cpm_sums = self.normalized_data.sum(axis=0)
            expected_sum = 1e6

            # Calculate relative error
            relative_errors = np.abs(cpm_sums - expected_sum) / expected_sum

            validation_results = {
                "method": "CPM",
                "expected_sum_per_sample": expected_sum,
                "actual_sums": cpm_sums.tolist(),
                "relative_errors": relative_errors.tolist(),
                "max_relative_error": relative_errors.max(),
                "mean_relative_error": relative_errors.mean(),
                "validation_passed": relative_errors.max() < 0.01,  # 1% tolerance
            }

            return validation_results

        except Exception as e:
            raise NormalizationError(f"Validation failed: {str(e)}")
