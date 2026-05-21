#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DESeq2 Differential Expression Module

This module provides DESeq2-based differential expression analysis for RNA-seq data.
It interfaces with the DESeq2 R package through rpy2, supporting pairwise biological
comparisons, subsetting of sample count matrices using regex, and robust statistic extraction.

Features:
    - Pairwise differential expression analysis using DESeq2 Wald test
    - Sample/condition subsetting from the global count matrix
    - Automated handling of R package loading and error checking via rpy2
    - Robust handling of missing value replacements (e.g., NaN FDR to 1, NaN logFC to 0)
    - Consistency in output schema columns (baseMean, logFC, lfcSE, stat, pvalue, FDR)

Configuration:
    - Configured via parameters passed to analyze_differential_expression (count_df, sample_df)
      and class constructor arguments (comparisons, fdr_threshold, log2fc_threshold, subset).

Dependencies:
    - R: DESeq2 package (must be installed in the R environment)
    - Python: rpy2, pandas, numpy

Classes / Functions / Exceptions:
    - DESeq2DiffExp: DESeq2-based differential expression analyzer.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List

from .base import BaseDiffExp, DifferentialExpressionError


class DESeq2DiffExp(BaseDiffExp):
    """
    DESeq2-based differential expression analyzer.

    This class provides differential expression analysis using the DESeq2 R package
    through a Python interface, based on the original working implementation.
    """

    def __init__(self, count_matrix_file=None, sample_info_file=None, *args, **kwargs):
        """Initialize DESeq2DiffExp."""

        # Call parent constructor (handles all data loading)
        super().__init__(
            count_matrix_file=count_matrix_file,
            sample_info_file=sample_info_file,
            *args,
            **kwargs,
        )

        # Set tool name
        self.tool_name = "deseq2"

        self.logger.info("DESeq2 differential expression analyzer initialized")

        # Check R dependencies
        self._check_r_dependencies()

    # Data loading is now handled by the base class
    # Subclasses can access data via self.count_data and self.sample_data

    def _check_r_dependencies(self) -> None:
        """Check if R and required packages are available."""
        try:
            # Try to import rpy2
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr

            # Record dependency check
            self._record_internal_operation("dependency_check", "Checking R and DESeq2 dependencies")

            # Try to import DESeq2
            try:
                importr("DESeq2")
                self.logger.info("DESeq2 R package found and loaded successfully")
                self._record_internal_operation("dependency_found", "DESeq2 R package is available")
            except Exception:
                raise DifferentialExpressionError(
                    "DESeq2 R package not found. Please install DESeq2 in R using: BiocManager::install('DESeq2')"
                )

        except ImportError:
            raise DifferentialExpressionError("rpy2 package not found. Please install rpy2 using: pip install rpy2")

    def _run_deseq2_analysis(self, count_df: pd.DataFrame, sample_df: pd.DataFrame, comparison: str) -> pd.DataFrame:
        """Run DESeq2 analysis using the exact working original approach."""
        try:
            # Import R packages exactly like the original working code
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri, numpy2ri, Formula, default_converter
            from rpy2.robjects.conversion import localconverter
            from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
            import logging

            # Suppress R warnings
            rpy2_logger.setLevel(logging.ERROR)

            converter = default_converter + pandas2ri.converter + numpy2ri.converter

            # Import DESeq2
            deseq = importr("DESeq2")
            to_dataframe = robjects.r("function(x) data.frame(x)")

            # Parse comparison (format: group1-group2)
            c1, c2 = comparison.split("-")

            self._record_internal_operation(
                "deseq2_analysis_start",
                f"Starting DESeq2 analysis for comparison: {comparison}",
                comparison,
            )

            # Use exact variable names from working original code
            countDF = count_df.set_index(self.gene_column)
            targetFile = sample_df.copy()
            design = "condition"  # Use condition column for design formula

            # Use the original subset approach with regex filtering
            if self.subset:
                # Original working code: subDF = countDF.filter(regex='|'.join([c1,c2]))
                subDF = countDF.filter(regex="|".join([c1, c2]))

                # Original working code: subTF = targetFile[targetFile['condition'].str.contains('|'.join([c1,c2]))]
                subTF = targetFile[targetFile["condition"].str.contains("|".join([c1, c2]))]

                self.logger.info(f"Subset for {comparison}: Count matrix {subDF.shape}, Target file {subTF.shape}")

                # Convert to R using exact original approach
                with localconverter(converter):
                    count_matrix = robjects.conversion.py2rpy(subDF)
                    design_matrix = robjects.conversion.py2rpy(subTF)
            else:
                # Use all data
                with localconverter(converter):
                    count_matrix = robjects.conversion.py2rpy(countDF)
                    design_matrix = robjects.conversion.py2rpy(targetFile)

            # Create design formula exactly like original
            designFormula = "~ " + design
            design_formula = Formula(designFormula)

            # Create DESeq dataset
            dds = deseq.DESeqDataSetFromMatrix(countData=count_matrix, colData=design_matrix, design=design_formula)

            # Run DESeq2
            dds1 = deseq.DESeq(dds, quiet=True)

            # Get results using exact original approach
            R_contrast = robjects.vectors.StrVector(np.array([design, c1, c2]))
            result = deseq.results(dds1, contrast=R_contrast)

            # Convert to dataframe using exact original approach
            result = to_dataframe(result)

            with localconverter(converter):
                result = robjects.conversion.rpy2py(result)

            # Process results exactly like original
            result = pd.DataFrame(result)

            result["padj"] = result["padj"].replace(np.nan, 1)
            result["log2FoldChange"] = result["log2FoldChange"].replace(np.nan, 0)

            # Standardize column names for consistency across methods
            result.columns = ["baseMean", "logFC", "lfcSE", "stat", "pvalue", "FDR"]
            result.reset_index(drop=True, inplace=True)

            # Add gene IDs back
            gene_id = count_df[[self.gene_column]].values
            result[self.gene_column] = gene_id.flatten()

            # Reorder columns to put gene column first
            cols = [self.gene_column] + [col for col in result.columns if col != self.gene_column]
            result = result[cols]

            self._record_internal_operation(
                "deseq2_analysis_complete",
                f"DESeq2 analysis completed for {comparison}: {len(result)} genes analyzed",
                comparison,
            )

            return result

        except Exception as e:
            self.logger.error(f"DESeq2 analysis failed for {comparison}: {str(e)}")
            raise DifferentialExpressionError(f"DESeq2 analysis failed for {comparison}: {str(e)}")

    def analyze_differential_expression(
        self, count_df: pd.DataFrame = None, sample_df: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Implement the exact original DESeq2 approach that works.

        Args:
            count_df: Pre-loaded count matrix DataFrame
            sample_df: Pre-loaded sample information DataFrame

        Returns:
            Dictionary containing combined differential expression results
        """
        try:
            # Use the DataFrames from base class
            if count_df is None:
                count_df = self.count_data
            if sample_df is None:
                sample_df = self.sample_data

            # Initialize results DataFrame with gene names exactly like original
            gene_id = count_df[[self.gene_column]].values
            deseq_results = pd.DataFrame(gene_id, columns=[self.gene_column])
            successful_comparisons: List[str] = []
            failed_comparisons: List[str] = []

            # Process each comparison using the exact original approach
            for comparison in self.comparisons:
                self.logger.info(f"Running DESeq2 analysis for comparison: {comparison}")

                try:
                    # Run DESeq2 analysis using the working approach
                    result_df = self._run_deseq2_analysis(count_df, sample_df, comparison)

                    # Prepare result for concatenation exactly like original
                    result_for_concat = result_df[["baseMean", "logFC", "lfcSE", "stat", "pvalue", "FDR"]].copy()
                    result_for_concat.reset_index(drop=True, inplace=True)

                    # Add comparison names to column names exactly like original
                    result_for_concat.columns = [s + "(" + comparison + ")" for s in result_for_concat.columns]

                    # Concatenate with main results exactly like original
                    deseq_results.reset_index(drop=True, inplace=True)
                    deseq_results = pd.concat([deseq_results, result_for_concat], axis=1)
                    successful_comparisons.append(comparison)

                except Exception as e:
                    self.logger.error(f"Failed to analyze {comparison}: {str(e)}")
                    failed_comparisons.append(comparison)
                    continue

            if not successful_comparisons:
                raise DifferentialExpressionError("DESeq2 failed for all comparisons: " + ", ".join(failed_comparisons))

            # Return combined results DataFrame in the expected format
            return {"combined_results": deseq_results}

        except Exception as e:
            raise DifferentialExpressionError(f"DESeq2 differential expression analysis failed: {str(e)}")
