#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyDiffExpress Wrapper Module

This module provides a wrapper class that implements the BaseDiffExp interface
for the native Python differential-expression engine used by PySeqRNA.

Features:
    - Wrapper implementation for native PyDiffExpress analysis within the PySeqRNA pipeline
    - Support for multiple normalization methods, abundance metrics, dispersion models, and test types
    - Capture and redirection of standard stdout/stderr output from the core engine
    - Horizontal concatenation and formatting of multi-contrast wide tables for downstream report generation
    - Integration with GeneDescriptionService for automatic Ensembl BioMart annotation enrichment

Configuration:
    The wrapper takes parameters including:
    - normalization method (e.g., 'median_ratio')
    - abundance metric (e.g., 'base_mean')
    - dispersion model (e.g., 'map')
    - test type (e.g., 'wald')

Dependencies:
    - numpy
    - pandas
    - pydiffexpress

Classes:
    PyDiffExpressWrapper: PyDiffExpress wrapper for native Python differential expression analysis

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from .base import BaseDiffExp


class PyDiffExpressWrapper(BaseDiffExp):
    """
    PyDiffExpress wrapper for native Python differential expression analysis.

    This wrapper integrates the modular pydiffexpress engine with the PySeqRNA
    pipeline by implementing the BaseDiffExp interface.

    Parameters
    ----------
    count_matrix_file : str or pd.DataFrame
        Path to count matrix file or DataFrame
    sample_info_file : str or pd.DataFrame
        Path to sample info file or DataFrame
    comparisons : List[str]
        List of comparisons to perform
    out_dir : str, default="."
        Output directory for results
    gene_column : str, default='Gene'
        Name of the gene column
    design_formula : str, default='~ sample'
        Design formula for the model
    fdr_threshold : float, default=0.05
        FDR threshold for significance
    log2fc_threshold : float, default=1.0
        Log2 fold change threshold
    species : Optional[str], default=None
        Species for annotation
    organism_type : str, default='plants'
        Type of organism
    add_gene_names : bool, default=True
        Whether to add gene names
    subset : bool, default=False
        Whether to subset the data
    dryrun : bool, default=False
        Whether to run in dry-run mode
    logger : Optional[Any], default=None
        Logger object
    dry_run_manager : Optional[Any], default=None
        Dry run manager
    **kwargs
        Additional arguments passed to DiffExpressAnalyzer

    Author: Naveen Duhan
    """

    def __init__(
        self,
        count_matrix_file,
        sample_info_file,
        comparisons: List[str],
        out_dir: str = ".",
        gene_column: str = "Gene",
        design_formula: str = "~ condition",
        fdr_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        species: Optional[str] = None,
        organism_type: str = "plants",
        add_gene_names: bool = True,
        subset: bool = False,
        dryrun: bool = False,
        logger: Optional[Any] = None,
        dry_run_manager=None,
        **kwargs: Any,
    ):
        """
        Initialize the PyDiffExpress wrapper.
        """
        # Preprocess count matrix only if gene names are stored in the index.
        # A normal PySeqRNA count matrix already has a Gene column and an unnamed
        # RangeIndex; resetting that would create duplicate Gene columns.
        if (
            isinstance(count_matrix_file, pd.DataFrame)
            and gene_column not in count_matrix_file.columns
            and count_matrix_file.index.name is not None
        ):
            count_matrix_file = count_matrix_file.reset_index()
            count_matrix_file.columns = [gene_column] + list(count_matrix_file.columns[1:])

        super().__init__(
            count_matrix_file=count_matrix_file,
            sample_info_file=sample_info_file,
            comparisons=comparisons,
            out_dir=out_dir,
            gene_column=gene_column,
            design_formula=design_formula,
            fdr_threshold=fdr_threshold,
            log2fc_threshold=log2fc_threshold,
            species=species,
            organism_type=organism_type,
            add_gene_names=add_gene_names,
            subset=subset,
            dryrun=dryrun,
            logger=logger,
            dry_run_manager=dry_run_manager,
            **kwargs,
        )

        self.diffexp_normalization = kwargs.get("diffexp_normalization", kwargs.get("normalization", "median_ratio"))
        self.diffexp_abundance = kwargs.get("diffexp_abundance", kwargs.get("abundance", "base_mean"))
        self.diffexp_dispersion = kwargs.get("diffexp_dispersion", kwargs.get("dispersion", "map"))
        self.diffexp_test = kwargs.get("diffexp_test", kwargs.get("test", "wald"))
        self.tool_name = "pydiffexpress"

        self.logger.info(
            "PyDiffExpress wrapper initialized with components: normalization=%s, abundance=%s, dispersion=%s, test=%s",
            self.diffexp_normalization,
            self.diffexp_abundance,
            self.diffexp_dispersion,
            self.diffexp_test,
        )

    def analyze_differential_expression(
        self, count_df: pd.DataFrame = None, sample_df: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform native Python differential expression analysis.

        Parameters
        ----------
        count_df : pd.DataFrame, optional
            Count matrix DataFrame (if not provided, uses loaded data)
        sample_df : pd.DataFrame, optional
            Sample info DataFrame (if not provided, uses loaded data)

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing analysis results for each comparison

        Author: Naveen Duhan
        """
        try:
            # Use provided data or loaded data
            if count_df is not None:
                self.count_data = count_df
            if sample_df is not None:
                self.sample_data = sample_df

            # Validate inputs
            self._validate_inputs()

            from .pydiffexpress.api import run_analysis

            captured_output = io.StringIO()
            processed_results = {}

            with tempfile.TemporaryDirectory(prefix="pyseqrna_pydiffexpress_") as tmpdir:
                component_dir = Path(tmpdir)
                with redirect_stdout(captured_output), redirect_stderr(captured_output):
                    self.logger.info(
                        "Running PyDiffExpress analysis with normalization=%s, abundance=%s, dispersion=%s, test=%s",
                        self.diffexp_normalization,
                        self.diffexp_abundance,
                        self.diffexp_dispersion,
                        self.diffexp_test,
                    )
                    run_analysis(
                        counts=self.count_data,
                        samples=self.sample_data,
                        outdir=component_dir,
                        gene_column=self.gene_column,
                        comparisons=self.comparisons,
                        normalization=self.diffexp_normalization,
                        abundance=self.diffexp_abundance,
                        dispersion=self.diffexp_dispersion,
                        test=self.diffexp_test,
                    )

                internal_output = captured_output.getvalue().strip()
                if internal_output:
                    self.logger.debug("PyDiffExpress internal output suppressed:\n%s", internal_output)

                for comparison in self.comparisons:
                    self.logger.info(f"Processing comparison: {comparison}")

                    try:
                        contrast_file = component_dir / "contrasts" / f"{comparison}.tsv"
                        if not contrast_file.exists():
                            raise FileNotFoundError(f"Expected PyDiffExpress contrast file not found: {contrast_file}")
                        comparison_results = pd.read_csv(contrast_file, sep="\t")

                        if self.gene_column not in comparison_results.columns:
                            if len(comparison_results.index) == len(self.count_data):
                                comparison_results.insert(
                                    0,
                                    self.gene_column,
                                    self.count_data[self.gene_column].astype(str).values,
                                )
                            else:
                                comparison_results.insert(
                                    0,
                                    self.gene_column,
                                    comparison_results.index.astype(str),
                                )

                        # Add gene annotations if requested
                        if self.add_gene_names and self.species:
                            comparison_results = self.add_gene_annotations(comparison_results)

                        processed_results[comparison] = comparison_results

                    except Exception as e:
                        self.logger.warning(f"Failed to process comparison '{comparison}': {str(e)}")
                        # Create empty results for failed comparison
                        empty_results = pd.DataFrame(
                            {
                                self.gene_column: self.count_data[self.gene_column].astype(str).values,
                                "logFC": np.nan,
                                "lfcSE": np.nan,
                                "stat": np.nan,
                                "pvalue": np.nan,
                                "padj": np.nan,
                            }
                        )
                        processed_results[comparison] = empty_results

            # Create combined results in wide format for downstream PySeqRNA output.
            if processed_results:
                # Start with the first comparison as base
                first_comparison = list(processed_results.keys())[0]
                result_columns = [
                    col
                    for col in [
                        "baseMean",
                        "logFC",
                        "lfcSE",
                        "stat",
                        "logCPM",
                        "LR",
                        "pvalue",
                        "FDR",
                    ]
                    if col in processed_results[first_comparison].columns
                ]
                combined_results = processed_results[first_comparison][result_columns].copy()

                # Add comparison names to column names
                combined_results.columns = [col + "(" + first_comparison + ")" for col in combined_results.columns]

                # Add other comparisons
                for comparison in list(processed_results.keys())[1:]:
                    if comparison != "combined_results":  # Skip if already added
                        comparison_columns = [col for col in result_columns if col in processed_results[comparison].columns]
                        comparison_df = processed_results[comparison][comparison_columns].copy()
                        comparison_df.columns = [col + "(" + comparison + ")" for col in comparison_df.columns]

                        # Concatenate horizontally
                        combined_results = pd.concat([combined_results, comparison_df], axis=1)

                # Add gene information columns
                if self.gene_column in processed_results[first_comparison].columns:
                    combined_results.insert(
                        0,
                        self.gene_column,
                        processed_results[first_comparison][self.gene_column],
                    )
                if "Name" in processed_results[first_comparison].columns:
                    combined_results.insert(1, "Name", processed_results[first_comparison]["Name"])
                if "Description" in processed_results[first_comparison].columns:
                    combined_results.insert(
                        2,
                        "Description",
                        processed_results[first_comparison]["Description"],
                    )

                processed_results["combined_results"] = combined_results

            self.logger.info(f"PyDiffExpress analysis completed for {len(self.comparisons)} comparisons")

            return processed_results

        except Exception as e:
            self.logger.error(f"PyDiffExpress differential expression analysis failed: {str(e)}")
            raise

    def save_results(self, results: Dict[str, pd.DataFrame], output_file: str = None) -> str:
        """
        Save differential expression results to files.

        This method saves the results in multiple formats including CSV and Excel,
        following the same interface as other differential expression tools.

        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Dictionary containing analysis results for each comparison
        output_file : str, optional
            Base name for output files (if None, uses default naming)

        Returns
        -------
        str
            Path to the main output file
        """
        if output_file is None:
            output_file = f"pydiffexpress_results_{self.species or 'analysis'}"

        # Use the base class save_results method
        return super().save_results(results, output_file)
