#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Differential Expression Module

This module provides the abstract base class and common infrastructure for all
differential expression analysis methods in the pySeqRNA pipeline. It defines
interfaces and provides utility functions for file loading, data alignment,
result output formatting, DEG filtering, and annotation integration.

Features:
    - Abstract base class framework for differential expression analysis tools
    - Data loading support for multiple formats including CSV, TSV, and Excel
    - Automatic sample metadata alignment with count matrix columns
    - Simulation of differential expression results for dry-run validation
    - Multi-sheet Excel exporting for all/significant/up/down regulated genes
    - Support for gene name annotation and individual gene list extraction for downstream analysis
    - Comprehensive execution logging and statistics reporting

Configuration:
    Configured via constructor arguments specifying:
    - count_matrix_file: Path to counts or DataFrame
    - sample_info_file: Path to sample metadata or DataFrame
    - comparisons: List of comparison pairs to perform
    - design_formula: Statistical design formula
    - fdr_threshold: FDR significance threshold
    - log2fc_threshold: Log2 Fold Change threshold
    - species and organism_type: Annotation settings
    - dryrun: Dry-run simulation flag

Dependencies:
    - pandas
    - numpy
    - openpyxl
    - pyseqrna.utils.file_manager.FileManager
    - pyseqrna.modules.annotation.create_gene_annotator

Classes / Functions / Exceptions:
    - DifferentialExpressionError (Exception): Custom exception for differential expression analysis errors.
    - BaseDiffExp (Class): Abstract base class for all differential expression analysis methods.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ...utils.file_manager import FileManager
from ..annotation import create_gene_annotator


class DifferentialExpressionError(Exception):
    """Custom exception for differential expression analysis errors."""

    pass


class BaseDiffExp(ABC):
    """
    Abstract base class for all differential expression analysis methods.

    This class provides the common interface and shared functionality
    for differential expression analysis tools like DESeq2, edgeR, etc.

    Attributes:
        tool_name (str): Name of the differential expression tool
        logger: Logger instance for tracking progress
        file_manager: FileManager instance for file operations
        resource_manager: ResourceManager instance for resource management
        command_executor: CommandExecutor instance for executing commands
    """

    def __init__(
        self,
        count_matrix_file,
        sample_info_file,
        comparisons: List[str],
        out_dir: str = ".",
        gene_column: str = "Gene",
        design_formula: str = "~ sample",
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
        Initialize base differential expression analyzer.

        Args:
            count_matrix_file: Path to count matrix file or DataFrame object
            sample_info_file: Path to sample info file or DataFrame object
            comparisons: List of comparisons to perform
            out_dir: Output directory
            gene_column: Name of gene column in count matrix
            design_formula: Design formula for analysis
            fdr_threshold: FDR threshold for significance
            log2fc_threshold: Log2 fold change threshold (converted from fold_threshold internally)
            species: Species name for annotation
            organism_type: Organism type ('plants' or 'animals')
            add_gene_names: Whether to add gene name annotations
            subset: Whether to subset data for each comparison
            dryrun: Whether to run in dry-run mode
            logger: Logger instance
            dry_run_manager: Dry run manager instance
            **kwargs: Additional keyword arguments
        """
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Initialize data storage
        self.count_data = None
        self.sample_data = None

        # Store parameters needed during early data loading/alignment.
        self.gene_column = gene_column
        self.design_formula = design_formula
        self.fdr_threshold = fdr_threshold
        self.log2fc_threshold = log2fc_threshold
        self.species = species
        self.organism_type = organism_type
        self.add_gene_names = add_gene_names
        self.subset = subset
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager

        # Store original inputs for reference
        self.count_matrix_file = count_matrix_file
        self.sample_info_file = sample_info_file

        # Load data immediately
        self._load_data()

        # Store remaining parameters
        self.comparisons = comparisons
        self.out_dir = Path(out_dir)

        # Store additional kwargs
        self.kwargs = kwargs

        # Initialize managers
        self.file_manager = FileManager(logger=self.logger)

        # Set tool name (to be overridden by subclasses)
        self.tool_name = "base_diffexp"

        # Initialize gene annotator if gene names should be added
        if self.add_gene_names and self.species:
            try:
                self.gene_annotator = create_gene_annotator(self.species, self.organism_type)
                self.logger.info(f"Gene annotator initialized for {self.species} ({self.organism_type})")
            except Exception as e:
                self.logger.warning(f"Failed to initialize gene annotator: {str(e)}")
                self.add_gene_names = False
                self.gene_annotator = None
        else:
            self.gene_annotator = None

        # Set up output directory
        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results containers
        self.results = {}
        self.filtered_results = {}

        # Validate loaded data
        self._validate_loaded_data()

        self.logger.debug(f"{self.tool_name} differential expression analyzer initialization complete")

    def _load_data(self):
        """Load count matrix and sample data, handling both DataFrame and file inputs."""
        import pandas as pd

        # Load count matrix
        if isinstance(self.count_matrix_file, pd.DataFrame):
            self.logger.info("Using provided count matrix DataFrame")
            self.count_data = self.count_matrix_file.copy()
        else:
            self.logger.info(f"Loading count matrix from: {self.count_matrix_file}")
            self.count_data = self._load_file_as_dataframe(self.count_matrix_file, "count matrix")

        # Load sample info
        if isinstance(self.sample_info_file, pd.DataFrame):
            self.logger.info("Using provided sample info DataFrame")
            self.sample_data = self.sample_info_file.copy()
        else:
            self.logger.info(f"Loading sample info from: {self.sample_info_file}")
            self.sample_data = self._load_file_as_dataframe(self.sample_info_file, "sample info")

        self._align_sample_metadata()

    def _load_file_as_dataframe(self, file_path: str, data_type: str) -> pd.DataFrame:
        """Load file as DataFrame, supporting multiple formats."""
        import pandas as pd

        if not os.path.exists(file_path):
            raise DifferentialExpressionError(f"{data_type} file not found: {file_path}")

        # Determine file format and load accordingly
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif file_ext == ".csv":
                df = pd.read_csv(file_path)
            elif file_ext in [".txt", ".tsv"]:
                df = pd.read_csv(file_path, sep="\t")
            else:
                raise DifferentialExpressionError(f"Unsupported file format for {data_type}: {file_ext}")

            sanitized_data_type = str(data_type).replace("\n", " ").replace("\r", " ")
            self.logger.info(f"Loaded {sanitized_data_type}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            raise DifferentialExpressionError(f"Failed to load {data_type} from {file_path}: {str(e)}")

    def _validate_loaded_data(self):
        """Validate loaded count matrix and sample data."""
        if self.count_data is None:
            raise DifferentialExpressionError("Count matrix data is None")

        if self.sample_data is None:
            raise DifferentialExpressionError("Sample data is None")

        # Validate gene column exists in count matrix
        if self.gene_column not in self.count_data.columns:
            raise DifferentialExpressionError(f"Gene column '{self.gene_column}' not found in count matrix")

        # Validate sample data has required columns
        if "condition" not in self.sample_data.columns:
            raise DifferentialExpressionError("Sample data must contain 'condition' column")

        self.logger.info(f"Data validation passed: {self.count_data.shape[0]} genes, {self.count_data.shape[1] - 1} samples")

    def _align_sample_metadata(self) -> None:
        """
        Align sample metadata rows to count matrix sample columns.

        When sample metadata includes a `sample` column, use it to reorder
        metadata and subset the count matrix so both inputs describe the same
        samples in the same order.
        """
        if self.count_data is None or self.sample_data is None:
            return

        if self.gene_column not in self.count_data.columns:
            return

        count_sample_columns = [str(col) for col in self.count_data.columns if col != self.gene_column]
        sample_identifier_column = None

        if "sample" in self.sample_data.columns:
            sample_identifier_column = "sample"
        elif "Sample" in self.sample_data.columns:
            sample_identifier_column = "Sample"

        if sample_identifier_column is not None:
            sample_ids = self.sample_data[sample_identifier_column].astype(str)
            duplicated = sample_ids[sample_ids.duplicated()].unique().tolist()
            if duplicated:
                raise DifferentialExpressionError(f"Duplicate sample identifiers found in sample metadata: {duplicated}")

            missing_in_counts = [sample for sample in sample_ids if sample not in count_sample_columns]
            if missing_in_counts:
                raise DifferentialExpressionError(f"Samples from metadata not found in count matrix: {missing_in_counts}")

            extra_in_counts = [sample for sample in count_sample_columns if sample not in set(sample_ids)]
            if extra_in_counts:
                self.logger.info(
                    "Subsetting count matrix from %d to %d samples based on sample metadata",
                    len(count_sample_columns),
                    len(sample_ids),
                )
                self.count_data = self.count_data[[self.gene_column] + sample_ids.tolist()].copy()

            self.sample_data = self.sample_data.copy()
            self.sample_data[sample_identifier_column] = sample_ids
            self.sample_data = self.sample_data.set_index(sample_identifier_column, drop=False)
            self.sample_data = self.sample_data.loc[sample_ids.tolist()].copy()
            return

        if len(self.sample_data) == len(count_sample_columns):
            self.sample_data = self.sample_data.copy()
            self.sample_data.index = count_sample_columns

    def _validate_inputs(self) -> None:
        """Validate input files and parameters."""
        # Skip file validation if using direct DataFrames (dummy paths)
        if (isinstance(self.count_matrix_file, str) and self.count_matrix_file == "direct_dataframe") or (
            isinstance(self.sample_info_file, str) and self.sample_info_file == "direct_dataframe"
        ):
            self.logger.debug("Using direct DataFrames, skipping file validation")
            return

        # Validate count matrix file
        if self.count_matrix_file is None:
            raise DifferentialExpressionError("Count matrix file not specified")
        if isinstance(self.count_matrix_file, str) and not self.file_manager.verify_files_exist(self.count_matrix_file):
            raise DifferentialExpressionError(f"Count matrix file not found: {self.count_matrix_file}")

        # Validate sample info file
        if self.sample_info_file is None:
            raise DifferentialExpressionError("Sample information file not specified")
        if isinstance(self.sample_info_file, str) and not self.file_manager.verify_files_exist(self.sample_info_file):
            raise DifferentialExpressionError(f"Sample information file not found: {self.sample_info_file}")

        # Validate thresholds
        if self.fdr_threshold <= 0 or self.fdr_threshold > 1:
            raise DifferentialExpressionError("FDR threshold must be between 0 and 1")
        if self.log2fc_threshold < 0:
            raise DifferentialExpressionError("Log2 fold change threshold must be non-negative")

    def _record_internal_operation(self, operation_type: str, details: str, comparison: str = None) -> None:
        """
        Record internal operations for execution reporting.

        Args:
            operation_type: Type of operation (e.g., 'data_loading', 'analysis')
            details: Details about the operation
            comparison: Comparison name if operation is comparison-specific
        """
        if hasattr(self, "dry_run_manager") and self.dry_run_manager:
            operation_record = {
                "operation": "diffexp_internal",
                "operation_type": operation_type,
                "details": details,
                "stage": f"{self.tool_name}_differential_expression",
                "timestamp": self.dry_run_manager._get_timestamp(),
                "tool": self.tool_name,
                "comparison": comparison,
            }

            # Add to appropriate operations list based on dry run mode
            if self.dryrun:
                self.dry_run_manager.simulated_operations.append(operation_record)
            else:
                self.dry_run_manager.executed_operations.append(operation_record)

    def load_count_matrix(self) -> pd.DataFrame:
        """
        Get count matrix DataFrame (already loaded in constructor).

        Returns:
            DataFrame containing the count matrix
        """
        return self.count_data.copy()

    def load_sample_info(self) -> pd.DataFrame:
        """
        Get sample info DataFrame (already loaded in constructor).

        Returns:
            DataFrame containing sample information
        """
        return self.sample_data.copy()

    def _build_dryrun_results(self, count_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Build simulated differential expression results for dry-run mode.

        Dry-run should validate pipeline wiring, not execute real statistical
        models on synthetic count matrices that may be rank-deficient or flat.
        """
        gene_ids = count_df[self.gene_column].astype(str).reset_index(drop=True)
        total_genes = len(gene_ids)

        if total_genes == 0:
            raise DifferentialExpressionError("Cannot simulate dry-run differential expression with zero genes")

        base_mean = np.linspace(50.0, 500.0, total_genes)
        logfc_pattern = np.array([1.4, -1.2, 0.35, -0.25, 0.0, 0.85, -0.9, 0.15])
        pvalue_pattern = np.array([0.001, 0.004, 0.03, 0.08, 0.5, 0.02, 0.01, 0.2])
        lfcse_pattern = np.array([0.22, 0.28, 0.18, 0.25, 0.30, 0.24, 0.27, 0.20])

        combined_results = pd.DataFrame({self.gene_column: gene_ids})

        for index, comparison in enumerate(self.comparisons):
            shift = index % len(logfc_pattern)
            logfc = np.roll(logfc_pattern, shift)
            pvalues = np.roll(pvalue_pattern, shift)
            lfcse = np.roll(lfcse_pattern, shift)

            tiled_logfc = np.resize(logfc, total_genes)
            tiled_pvalues = np.resize(pvalues, total_genes)
            tiled_lfcse = np.resize(lfcse, total_genes)
            tiled_basemean = base_mean + (index * 15.0)
            stat = np.divide(
                tiled_logfc,
                tiled_lfcse,
                out=np.zeros_like(tiled_logfc, dtype=float),
                where=tiled_lfcse != 0,
            )
            fdr = np.minimum(1.0, tiled_pvalues * 1.5)

            comparison_df = pd.DataFrame(
                {
                    self.gene_column: gene_ids,
                    "baseMean": tiled_basemean,
                    "logFC": tiled_logfc,
                    "lfcSE": tiled_lfcse,
                    "stat": stat,
                    "pvalue": tiled_pvalues,
                    "FDR": fdr,
                }
            )

            for column in ["baseMean", "logFC", "lfcSE", "stat", "pvalue", "FDR"]:
                combined_results[f"{column}({comparison})"] = comparison_df[column].values

        self.logger.info(
            "DRYRUN: Simulating %s differential expression results for %d comparison(s)",
            self.tool_name,
            len(self.comparisons),
        )
        self._record_internal_operation(
            "dryrun_analysis",
            f"Simulated differential expression results for {len(self.comparisons)} comparison(s)",
        )

        return {"combined_results": combined_results}

    def _create_diffexp_log(self, log_file: Path) -> None:
        """Create a detailed log file for differential expression analysis."""
        with open(log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{self.tool_name.upper()} DIFFERENTIAL EXPRESSION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tool: {self.tool_name.upper()}\n")
            if isinstance(self.count_matrix_file, str):
                f.write(f"Count Matrix File: {self.count_matrix_file}\n")
            else:
                f.write(f"Count Matrix File: DataFrame with shape {self.count_matrix_file.shape}\n")
            if isinstance(self.sample_info_file, str):
                f.write(f"Sample Info File: {self.sample_info_file}\n")
            else:
                f.write(f"Sample Info File: DataFrame with shape {self.sample_info_file.shape}\n")
            f.write(f"Output Directory: {self.out_dir}\n")
            f.write(f"Gene Column: {self.gene_column}\n")
            f.write(f"Design Formula: {self.design_formula}\n")
            f.write(f"FDR Threshold: {self.fdr_threshold}\n")
            f.write(f"Log2FC Threshold: {self.log2fc_threshold}\n")
            f.write(f"Comparisons: {', '.join(self.comparisons)}\n")
            f.write("-" * 80 + "\n")
            f.write("PROCESSING LOG:\n")

    def _update_diffexp_log(self, log_file: Path, result_data: dict) -> None:
        """Update the differential expression log file with completion status."""
        with open(log_file, "a") as f:
            f.write("-" * 80 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Status: SUCCESS\n")
            f.write(f"Total Genes Analyzed: {result_data.get('total_genes', 0)}\n")
            f.write(f"Total Samples: {result_data.get('total_samples', 0)}\n")
            f.write(f"Comparisons Completed: {result_data.get('comparisons_completed', 0)}\n")
            f.write(f"Total DEGs Found: {result_data.get('total_degs', 0)}\n")
            f.write(f"Results Files Created: {result_data.get('results_files', 0)}\n")
            f.write("Gene Files for Annotation: Created in 'diff_genes' directory\n")
            f.write("Annotation Files: Individual .txt files for each comparison\n")
            f.write("=" * 80 + "\n")

    # save_results method removed - no longer needed

    def save_excel_with_sheets(self, data_dict: Dict[str, pd.DataFrame], filename: str) -> str:
        """
        Generic method to save DataFrames as Excel file with multiple sheets.

        Args:
            data_dict: Dictionary of {sheet_name: DataFrame}
            filename: Name of the Excel file

        Returns:
            Path to saved file
        """
        try:
            output_file = self.out_dir / filename

            # Check if any DataFrame has data
            has_data = any(not df.empty for df in data_dict.values())

            if not has_data:
                # If all DataFrames are empty, create a placeholder sheet
                self.logger.warning(f"No data found for {filename}, creating placeholder sheet")
                placeholder_df = pd.DataFrame({"Message": ["No significant DEGs found with current thresholds"]})
                placeholder_df.to_excel(output_file, sheet_name="No_DEGs", index=False)
                self.logger.info(f"Created placeholder Excel file: {output_file}")
                return str(output_file)

            # Save DataFrames with data
            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                sheets_added = 0
                for sheet_name, df in data_dict.items():
                    if not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        self.logger.info(f"Added {sheet_name} sheet to {output_file}")
                        sheets_added += 1

                if sheets_added == 0:
                    # Fallback: create a placeholder sheet if somehow no sheets were added
                    placeholder_df = pd.DataFrame({"Message": ["No significant DEGs found with current thresholds"]})
                    placeholder_df.to_excel(writer, sheet_name="No_DEGs", index=False)
                    self.logger.info(f"Added placeholder sheet to {output_file}")

            self.logger.info(f"Saved Excel file with {sheets_added} sheets to: {output_file}")
            return str(output_file)

        except Exception as e:
            self.logger.error(f"Failed to save Excel file {filename}: {str(e)}")
            raise DifferentialExpressionError(f"Failed to save Excel file {filename}: {str(e)}")

    # Filtering functionality moved to DEGFilter class - this method removed

    def create_filtered_deg_files(
        self,
        combined_results: pd.DataFrame,
        individual_results: Dict[str, pd.DataFrame],
    ) -> Tuple[List[str], int, Dict[str, pd.DataFrame]]:
        """
        Create filtered DEG files with the specified naming convention.

        Args:
            combined_results: Combined results DataFrame
            individual_results: Dictionary of individual comparison DataFrames

        Returns:
            List of created file paths, total DEGs, and filtered results dictionary
        """
        try:
            from .deg_filter import DEGFilter

            # Check if we have annotation columns (Name, Description) in combined_results
            has_annotations = all(col in combined_results.columns for col in ["Name", "Description"])

            # Initialize DEGFilter with current settings
            deg_filter = DEGFilter(
                fdr_threshold=self.fdr_threshold,
                fold_threshold=2**self.log2fc_threshold,  # Convert log2 to fold
                has_replicates=True,  # Assuming we have replicates
                extra_columns=has_annotations,  # Set to True if we have Name, Description columns
                logger=self.logger,
            )

            # Filter DEGs using existing functionality
            filtered_results = deg_filter.filter_degs(
                deg_df=combined_results,
                compare_list=self.comparisons,
                create_plot=True,
                save_plot_path=str(self.out_dir / "Filtered_DEG.png"),
            )

            # Save files with the specified naming convention
            output_files = []

            if not self.dryrun:
                # Save all significant DEGs
                output_files.append(self.save_excel_with_sheets(filtered_results["filtered"], "Filtered_DEGs.xlsx"))

                # Save upregulated DEGs
                output_files.append(self.save_excel_with_sheets(filtered_results["filteredup"], "Filtered_upDEGs.xlsx"))

                # Save downregulated DEGs
                output_files.append(self.save_excel_with_sheets(filtered_results["filtereddown"], "Filtered_downDEGs.xlsx"))

                # Save summary
                summary_file = self.out_dir / "Filtered_DEGs_summary.xlsx"
                filtered_results["summary"].to_excel(summary_file, index=False)
                output_files.append(str(summary_file))
                self.logger.info(f"Saved filtered DEGs summary to: {summary_file}")

            # Calculate total DEGs from filtered results
            total_degs = 0
            if "summary" in filtered_results:
                total_degs = filtered_results["summary"]["Total_DEGs"].sum()

            return output_files, total_degs, filtered_results

        except Exception as e:
            self.logger.error(f"Failed to create filtered DEG files: {str(e)}")
            raise DifferentialExpressionError(f"Failed to create filtered DEG files: {str(e)}")

    def create_gene_files_for_annotation(self, filtered_results: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Create individual gene files for each comparison for annotation purposes.

        Args:
            filtered_results: Dictionary of filtered results from DEGFilter
                           {'filtered': dict, 'filteredup': dict, 'filtereddown': dict}

        Returns:
            List of created gene file paths
        """
        try:
            # Create diff_genes directory
            diff_genes_dir = self.out_dir / "diff_genes"
            if not self.dryrun:
                diff_genes_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created gene files directory: {diff_genes_dir}")
            else:
                self.logger.info(f"Would create gene files directory: {diff_genes_dir}")

            output_files = []

            if not self.dryrun:
                # Process each comparison from the filtered results
                for comparison in self.comparisons:
                    # Get filtered data for this comparison
                    if comparison in filtered_results.get("filtered", {}):
                        comparison_df = filtered_results["filtered"][comparison]

                        if comparison_df.empty:
                            self.logger.warning(f"No filtered data for comparison {comparison}, skipping gene file creation")
                            continue

                        # Extract gene names and clean them
                        if self.gene_column in comparison_df.columns:

                            def _clean_genes(df: pd.DataFrame) -> pd.Series:
                                if df is None or df.empty or self.gene_column not in df.columns:
                                    return pd.Series(dtype=str)
                                return df[self.gene_column].astype(str).str.replace("gene:", "", regex=False).str.upper()

                            # Use the DEGFilter's already separated groups so
                            # suffixed result columns (for example logFC(A-B))
                            # do not break up/down gene-list creation.
                            gene_types = {
                                "all": _clean_genes(comparison_df),
                                "up": _clean_genes(filtered_results.get("filteredup", {}).get(comparison)),
                                "down": _clean_genes(filtered_results.get("filtereddown", {}).get(comparison)),
                            }

                            for gene_type, gene_list in gene_types.items():
                                if not gene_list.empty:
                                    # Create filename based on gene type
                                    if gene_type == "all":
                                        filename = f"{comparison}.txt"
                                    else:
                                        filename = f"{comparison}_{gene_type}.txt"

                                    file_path = diff_genes_dir / filename
                                    gene_list.to_csv(file_path, sep="\t", index=False, header=False)
                                    output_files.append(str(file_path))
                                    self.logger.info(f"Created gene file: {file_path} ({len(gene_list)} genes)")
                                else:
                                    self.logger.warning(f"No {gene_type} genes found for comparison {comparison}")
                        else:
                            self.logger.warning(f"Gene column '{self.gene_column}' not found in comparison {comparison}")
                    else:
                        self.logger.warning(f"Comparison '{comparison}' not found in filtered results.")

                if output_files:
                    self.logger.info(f"Created {len(output_files)} gene files for annotation purposes")
                    self.logger.info(f"Gene files directory: {diff_genes_dir}")
                    self.logger.info("Gene files can be used for downstream annotation analysis")
                else:
                    self.logger.warning("No gene files were created")

            return output_files

        except Exception as e:
            self.logger.error(f"Failed to create gene files for annotation: {str(e)}")
            raise DifferentialExpressionError(f"Failed to create gene files for annotation: {str(e)}")

    @abstractmethod
    def analyze_differential_expression(
        self, count_df: pd.DataFrame = None, sample_df: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform differential expression analysis.

        Args:
            count_df: Pre-loaded count matrix DataFrame (to avoid duplicate loading)
            sample_df: Pre-loaded sample information DataFrame

        Returns:
            Dictionary containing differential expression results for each comparison

        Raises:
            DifferentialExpressionError: If analysis fails
        """
        pass

    def run(self, save_results: bool = True, filter_results: bool = True) -> Dict[str, Any]:
        """
        Run the differential expression analysis pipeline.

        Args:
            save_results: Whether to save results to files
            filter_results: Whether to filter results for DEGs

        Returns:
            Dictionary of results and summary statistics
        """
        total_degs = 0  # Ensure this is always defined
        try:
            self._validate_inputs()
            self._load_data()
            self._validate_loaded_data()
            self._record_internal_operation("data_loading", "Loaded count matrix and sample info")

            self.logger.info(f"Starting {self.tool_name} differential expression analysis")

            # Create log file for differential expression
            if not self.dryrun:
                log_file = self.out_dir / f"{self.tool_name}_differential_expression.log"
                self._create_diffexp_log(log_file)

            # Get data (already loaded in constructor)
            count_df = self.count_data
            sample_df = self.sample_data

            # Perform differential expression analysis, or simulate it for dry-run.
            if self.dryrun:
                analysis_results = self._build_dryrun_results(count_df)
            else:
                analysis_results = self.analyze_differential_expression(count_df=count_df, sample_df=sample_df)
            self._record_internal_operation("analysis", "Differential expression analysis complete")

            # Handle different result formats
            if isinstance(analysis_results, pd.DataFrame):
                # Single combined DataFrame (like DESeq2)
                combined_results = analysis_results
                individual_results = {}

                # Extract individual comparison results for sheets
                for comparison in self.comparisons:
                    comparison_cols = [col for col in combined_results.columns if f"({comparison})" in col]
                    if comparison_cols:
                        # Create individual comparison DataFrame
                        comparison_df = combined_results[[self.gene_column] + comparison_cols].copy()
                        # Rename columns to remove comparison suffix
                        new_cols = [self.gene_column] + [col.replace(f"({comparison})", "") for col in comparison_cols]
                        comparison_df.columns = new_cols
                        individual_results[comparison] = comparison_df

                self.results = {"combined_results": combined_results}
                self.results.update(individual_results)

            elif isinstance(analysis_results, dict):
                # Dictionary format (like edgeR)
                if "combined_results" in analysis_results:
                    combined_results = analysis_results["combined_results"]
                    individual_results = {}

                    # Extract individual comparison results for sheets
                    for comparison in self.comparisons:
                        comparison_cols = [col for col in combined_results.columns if f"({comparison})" in col]
                        if comparison_cols:
                            # Check if Name and Description columns exist in combined_results
                            annotation_cols = []
                            if "Name" in combined_results.columns:
                                annotation_cols.append("Name")
                            if "Description" in combined_results.columns:
                                annotation_cols.append("Description")

                            # Include gene column, annotation columns, and comparison columns
                            selected_cols = [self.gene_column] + annotation_cols + comparison_cols
                            comparison_df = combined_results[selected_cols].copy()

                            # Rename comparison columns (remove the comparison suffix)
                            new_cols = (
                                [self.gene_column]
                                + annotation_cols
                                + [col.replace(f"({comparison})", "") for col in comparison_cols]
                            )
                            comparison_df.columns = new_cols
                            individual_results[comparison] = comparison_df

                    self.results = {"combined_results": combined_results}
                    self.results.update(individual_results)
                else:
                    # Traditional dict format
                    self.results = analysis_results
                    combined_results = None
                    individual_results = {}
            else:
                raise DifferentialExpressionError(f"Unexpected result format from {self.tool_name}")

            # Save results if requested
            output_files = []
            total_degs = 0  # Initialize total_degs to 0
            if save_results and not self.dryrun:
                if combined_results is not None:
                    # Save combined results
                    combined_file = self.out_dir / "All_gene_expression.xlsx"
                    combined_results.to_excel(combined_file, index=False)
                    output_files.append(str(combined_file))
                    self.logger.info(f"Saved combined results to: {combined_file}")

                    # Save individual comparison sheets
                    output_files.append(self.save_excel_with_sheets(individual_results, "All_gene_expression_sheet.xlsx"))

                    # Create filtered DEG files
                    if filter_results:
                        filtered_files, total_degs, filtered_results = self.create_filtered_deg_files(
                            combined_results=combined_results,
                            individual_results=individual_results,
                        )
                        output_files.extend(filtered_files)

                        # Create individual gene files for annotation purposes
                        gene_files = self.create_gene_files_for_annotation(filtered_results=filtered_results)
                        output_files.extend(gene_files)
                    else:
                        total_degs = 0
                # All tools now return combined results format - no fallback needed

            # Generate summary statistics
            summary_stats = {
                "tool": self.tool_name,
                "total_genes": len(count_df),
                "total_samples": len(sample_df),
                "comparisons": len(self.comparisons),
                "total_degs": total_degs,  # Calculated from filtering if enabled
                "results_files": len(output_files),
            }

            # Update log file with completion status
            if not self.dryrun:
                self._update_diffexp_log(log_file, summary_stats)

            self.logger.info(f"{self.tool_name} differential expression analysis completed successfully")

            return {
                "results": self.results,
                "summary": summary_stats,
                "output_files": output_files,
            }

        except Exception as e:
            self.logger.error(f"{self.tool_name} differential expression analysis failed: {str(e)}")
            raise DifferentialExpressionError(f"{self.tool_name} differential expression analysis failed: {str(e)}")

    def get_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for differential expression results.

        Args:
            results: Results dictionary from run() method

        Returns:
            Dictionary containing summary statistics
        """
        try:
            # Use the summary that's already calculated in run() method
            if "summary" in results:
                summary = results["summary"].copy()

                # Convert NumPy types to native Python types for JSON serialization
                import numpy as np

                for key, value in summary.items():
                    if isinstance(value, np.integer):
                        summary[key] = int(value)
                    elif isinstance(value, np.floating):
                        summary[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        summary[key] = value.tolist()

                # Add pipeline-expected fields. This is the number of genes in
                # the result matrix, not the number of comparisons.
                summary["total_genes_tested"] = summary.get("total_genes", 0)
                summary["genes_with_significant_results"] = summary.get("total_degs", 0)

                return summary
            else:
                # Fallback if no summary exists
                return {
                    "tool": self.tool_name,
                    "total_genes": 0,
                    "total_samples": 0,
                    "total_genes_tested": 0,
                    "genes_with_significant_results": 0,
                }

        except Exception as e:
            self.logger.warning(f"Failed to generate summary stats: {str(e)}")
            return {
                "tool": self.tool_name,
                "total_genes": 0,
                "total_samples": 0,
                "total_genes_tested": 0,
                "genes_with_significant_results": 0,
            }

    def add_gene_annotations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add gene names and descriptions to results DataFrame.

        Args:
            results_df: Results DataFrame with gene IDs

        Returns:
            DataFrame with added gene names and descriptions
        """
        if not self.add_gene_names or not self.gene_annotator:
            return results_df

        try:
            self.logger.info("Adding gene names and descriptions to results")
            annotated_df = self.gene_annotator.add_descriptions_to_dataframe(
                results_df, gene_column=self.gene_column, insert_position=1
            )
            self.logger.info("Gene names and descriptions added successfully")
            return annotated_df
        except Exception as e:
            self.logger.warning(f"Failed to add gene names: {str(e)}")
            return results_df
