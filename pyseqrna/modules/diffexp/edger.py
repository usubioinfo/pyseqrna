#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
edgeR Differential Expression Module

This module provides edgeR-based differential expression analysis for RNA-seq data.
It interfaces with edgeR and limma R packages through rpy2, handling dispersion estimation,
generalized linear model fitting, and likelihood ratio testing for pairwise comparisons.

Features:
    - Pairwise differential expression analysis using edgeR glmLRT
    - Biological Coefficient of Variation (BCV) parameters for replicate-free studies
    - Empirical Bayes dispersion estimation (common, trended, tagwise)
    - Normalized factor calculations using calcNormFactors (TMM/etc.)
    - Consistent column schema layout mapping logFC, logCPM, LR, pvalue, and FDR

Configuration:
    - Configured via parameters passed to analyze_differential_expression (count_df, sample_df)
      and class constructor arguments (comparisons, fdr_threshold, log2fc_threshold, bcv, has_replicates, subset).

Dependencies:
    - R: edgeR and limma packages (must be installed in the R environment)
    - Python: rpy2, pandas, numpy

Classes / Functions / Exceptions:
    - EdgeRDiffExp: edgeR-based differential expression analyzer.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
from typing import Dict, Any, List

from .base import BaseDiffExp, DifferentialExpressionError


class EdgeRDiffExp(BaseDiffExp):
    """
    edgeR-based differential expression analyzer.

    This class provides differential expression analysis using the edgeR R package
    through a Python interface, based on the original working implementation.
    """

    def __init__(
        self,
        count_matrix_file=None,
        sample_info_file=None,
        bcv: float = 0.4,
        has_replicates: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize EdgeRDiffExp.

        Args:
            count_matrix_file: Path to count matrix file or DataFrame
            sample_info_file: Path to sample info file or DataFrame
            bcv: Biological coefficient of variation (used if no replicates)
            has_replicates: Whether the data has biological replicates
            *args, **kwargs: Additional arguments passed to parent class
        """

        super().__init__(
            count_matrix_file=count_matrix_file,
            sample_info_file=sample_info_file,
            *args,
            **kwargs,
        )
        self.bcv = bcv
        self.has_replicates = has_replicates
        self.logger.info("edgeR differential expression analyzer initialized")

        # Check R dependencies
        self._check_r_dependencies()

    def _check_r_dependencies(self) -> None:
        """Check if R and required packages are available."""
        try:
            # Try to import rpy2
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr

            # Record dependency check
            self._record_internal_operation("dependency_check", "Checking R, edgeR and limma dependencies")

            # Try to import edgeR and limma
            try:
                importr("edgeR")
                importr("limma")
                self.logger.info("edgeR and limma R packages found and loaded successfully")
                self._record_internal_operation("dependency_found", "edgeR and limma R packages are available")
            except Exception:
                raise DifferentialExpressionError(
                    "edgeR or limma R packages not found. Please install them in R using: "
                    "BiocManager::install(c('edgeR', 'limma'))"
                )

        except ImportError:
            raise DifferentialExpressionError(
                "rpy2 package not found. Please install rpy2 using: pip install rpy2\n"
                "Note: You may also need to install R and set R_HOME environment variable"
            )

    def _run_edger_analysis(self, count_df: pd.DataFrame, sample_df: pd.DataFrame, comparison: str) -> pd.DataFrame:
        """
        Run edgeR analysis using the exact working original approach.
        """
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri, numpy2ri, default_converter
            from rpy2.robjects.conversion import localconverter
            from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
            import logging

            # Define to_dataframe function like in original
            to_dataframe = robjects.r("function(x) data.frame(x)")

            # Suppress R warnings
            rpy2_logger.setLevel(logging.ERROR)

            converter = default_converter + pandas2ri.converter + numpy2ri.converter

            # Import packages
            edgeR = importr("edgeR")
            limma = importr("limma")

            # Parse comparison (format: group1-group2)
            c1, c2 = comparison.split("-")

            self._record_internal_operation(
                "edger_analysis_start",
                f"Starting edgeR analysis for comparison: {comparison}",
                comparison,
            )

            # Use exact variable names from working original code
            count_df[[self.gene_column]].values
            countDF = count_df.set_index(self.gene_column)
            targetFile = sample_df.copy()

            # Use the original subset approach with regex filtering
            if self.subset:
                # Original working code approach
                subDF = countDF.filter(regex="|".join([c1, c2]))
                subTF = targetFile[targetFile["condition"].str.contains("|".join([c1, c2]))]

                # Get gene IDs for the subsetted data
                subDF.index.values.reshape(-1, 1)

                self.logger.info(
                    "Subset for %s: Count matrix %s, Target file %s",
                    comparison,
                    subDF.shape,
                    subTF.shape,
                )

                # Convert to R using exact original approach
                with localconverter(converter):
                    count_matrix = robjects.conversion.py2rpy(subDF)
                    group = robjects.conversion.py2rpy(subTF["condition"])  # Use sample column for groups
            else:
                # Use all data
                with localconverter(converter):
                    count_matrix = robjects.conversion.py2rpy(countDF)
                    group = robjects.conversion.py2rpy(targetFile["condition"])  # Use sample column for groups

            # Create DGEList
            dds = edgeR.DGEList(counts=count_matrix, group=group)
            dds = edgeR.calcNormFactors(dds)

            # Build the design matrix directly in R and keep it there; converting
            # it through pandas here can drop dimensions for small subsets.
            robjects.r.assign("group", group)
            robjects.r.assign("dds", dds)

            robjects.r("design <- model.matrix(~0 + dds$samples$group, data=dds$samples)")
            robjects.r("colnames(design) <- levels(dds$samples$group)")
            design = robjects.r("design")

            # Create contrasts (use original comparison format)
            cont = robjects.vectors.StrVector([f"{c1}-{c2}"])  # Ensure proper contrast format
            contrasts = limma.makeContrasts(contrasts=cont, levels=design)

            # Estimate dispersions and fit model
            if self.has_replicates:
                dds = edgeR.estimateGLMCommonDisp(dds, design)
                dds = edgeR.estimateGLMTrendedDisp(dds, design)
                dds = edgeR.estimateGLMTagwiseDisp(dds, design)
                fit = edgeR.glmFit(dds, design)
            else:
                fit = edgeR.glmFit(dds, design, dispersion=float(self.bcv**2))

            # Perform likelihood ratio test
            lrt = edgeR.glmLRT(fit, contrast=contrasts)

            # Get top tags
            deg = edgeR.topTags(lrt, countDF.shape[0], sort_by="none")

            # Convert to pandas using the original working approach
            result = to_dataframe(deg)

            with localconverter(converter):
                result = pandas2ri.rpy2py(result)

            # Process results using the original working approach
            result = pd.DataFrame(result)

            result.columns = ["logFC", "logCPM", "LR", "pvalue", "FDR"]
            result.reset_index(drop=True, inplace=True)

            # Add comparison names to column names exactly like original
            result.columns = [s + "(" + comparison + ")" for s in result.columns]

            self._record_internal_operation(
                "edger_analysis_complete",
                f"edgeR analysis completed for {comparison}: {len(result)} genes analyzed",
                comparison,
            )

            return result

        except Exception as e:
            self.logger.error(f"edgeR analysis failed for {comparison}: {str(e)}")
            raise DifferentialExpressionError(f"edgeR analysis failed for {comparison}: {str(e)}")

    def analyze_differential_expression(
        self, count_df: pd.DataFrame = None, sample_df: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Implement the exact original edgeR approach that works.

        Args:
            count_df: Pre-loaded count matrix DataFrame
            sample_df: Pre-loaded sample information DataFrame

        Returns:
            Dictionary containing differential expression results for all comparisons
        """
        try:
            # Use the DataFrames directly (no file loading!)
            if count_df is None:
                count_df = self.count_data
            if sample_df is None:
                sample_df = self.sample_data

            # Initialize results DataFrame with gene names exactly like original
            gene_id = count_df[[self.gene_column]].values
            edger_results = pd.DataFrame(gene_id, columns=[self.gene_column])
            successful_comparisons: List[str] = []
            failed_comparisons: List[str] = []

            # Process each comparison using the exact original approach
            for comparison in self.comparisons:
                self.logger.info(f"Running edgeR analysis for comparison: {comparison}")

                try:
                    # Run edgeR analysis using the working approach
                    result_df = self._run_edger_analysis(count_df, sample_df, comparison)

                    # The result_df already has comparison names added to columns
                    # Just reset index and concatenate directly
                    result_df.reset_index(drop=True, inplace=True)

                    # Concatenate with main results exactly like original
                    edger_results.reset_index(drop=True, inplace=True)
                    edger_results = pd.concat([edger_results, result_df], axis=1)
                    successful_comparisons.append(comparison)

                except Exception as e:
                    self.logger.error(f"Failed to analyze {comparison}: {str(e)}")
                    failed_comparisons.append(comparison)
                    continue

            if not successful_comparisons:
                raise DifferentialExpressionError("edgeR failed for all comparisons: " + ", ".join(failed_comparisons))

            # Return single DataFrame like DESeq2 in the expected format
            return {"combined_results": edger_results}

        except Exception as e:
            raise DifferentialExpressionError(f"edgeR differential expression analysis failed: {str(e)}")

    # Use base class run method - no custom run method needed

    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for all comparisons.

        Args:
            results_df: Wide DataFrame with all comparison results

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "tool": "edgeR",
            "total_comparisons": len(self.comparisons),
            "comparisons": {},
            "has_replicates": self.has_replicates,
            "bcv": self.bcv if not self.has_replicates else None,
        }

        for comparison in self.comparisons:
            # Find columns for this comparison
            padj_col = f"FDR({comparison})"
            logfc_col = f"logFC({comparison})"

            if padj_col in results_df.columns and logfc_col in results_df.columns:
                significant = results_df[(results_df[padj_col] <= self.fdr_threshold) & (results_df[padj_col].notna())]

                upregulated = significant[significant[logfc_col] >= self.log2fc_threshold]
                downregulated = significant[significant[logfc_col] <= -self.log2fc_threshold]

                summary["comparisons"][comparison] = {
                    "total_genes": len(results_df),
                    "significant_genes": len(significant),
                    "upregulated_genes": len(upregulated),
                    "downregulated_genes": len(downregulated),
                    "mean_log2fc": results_df[logfc_col].mean(),
                }

        return summary
