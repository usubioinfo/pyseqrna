#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Normalizer Module

This module provides the abstract base class and custom exception for all count normalization
methods in the pySeqRNA pipeline. It handles loading count matrices, extracting gene lengths
from GFF/GTF annotation files, creating comparison boxplots, calculating summary statistics,
and writing output logs/results.

Features:
    - Abstract base class (BaseNormalizer) defining the template for count normalization
    - Auto-detection of GFF/GTF/GFF3 formats and robust attribute parsing to extract gene lengths
    - Support for loading count matrices from CSV, TSV, TXT, and Excel formats
    - Built-in comparison plotting (boxplots of log-transformed counts comparing raw and normalized states)
    - Structured logging and execution status tracking (compatible with dry-run modes)

Configuration:
    Configured via parameters passed to the constructor (such as count_matrix_file,
    annotation_file, out_dir, gene_column, dryrun).

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - pyseqrna.utils (FileManager, LogManager, ResourceManager)

Classes / Functions / Exceptions:
    - NormalizationError: Custom exception for normalization-related errors.
    - BaseNormalizer: Abstract base class for all count normalization methods.

:Created: May 20, 2021
:Updated: February 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from ...utils.file_manager import FileManager
from ...utils.log_manager import LogManager
from ...utils.resource_manager import ResourceManager


class NormalizationError(Exception):
    """Custom exception for normalization errors."""

    pass


class BaseNormalizer(ABC):
    """
    Abstract base class for all count normalization methods.

    This class provides the common interface and shared functionality
    for count normalization methods like CPM, RPKM, TPM, etc.

    Attributes:
        normalizer_name (str): Name of the normalization method
        logger: Logger instance for tracking progress
        file_manager: FileManager instance for file operations
        resource_manager: ResourceManager instance for resource management
    """

    def __init__(
        self,
        count_matrix_file: Union[str, pd.DataFrame],
        annotation_file: Optional[str] = None,
        out_dir: str = ".",
        gene_column: str = "Gene",
        dryrun: bool = False,
        logger: Optional[Any] = None,
        dry_run_manager=None,
        **kwargs: Any,
    ):
        """
        Initialize the BaseNormalizer.

        Args:
            count_matrix_file: Path to count matrix file (Excel/CSV) or pre-loaded DataFrame
            annotation_file: Path to annotation file (GTF/GFF) - required for length-based methods
            out_dir: Output directory path
            gene_column: Name of gene column in count matrix
            dryrun: Whether to perform a dry run
            logger: Logger instance
            **kwargs: Additional keyword arguments
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.logger
        else:
            self.logger = logger

        # Extract class name and set normalizer name
        self.normalizer_name = self.__class__.__name__.replace("Normalizer", "").lower()
        self.logger.debug(f"Initializing {self.normalizer_name} normalizer")

        # Initialize utilities
        self.file_manager = FileManager(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)

        # Store basic attributes
        self.count_matrix_file = count_matrix_file
        self.annotation_file = annotation_file
        self.gene_column = gene_column
        self.dryrun = dryrun

        # Store dry run manager
        self.dry_run_manager = dry_run_manager

        # Set up output directory
        self.out_dir = Path(out_dir)

        # Create output directory if it doesn't exist
        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.count_data = None
        self.gene_lengths = None
        self.normalized_data = None

        # Validate inputs
        self._validate_inputs()

        safe_normalizer_name = re.sub(r"[\r\n]", "", self.normalizer_name)
        self.logger.debug(f"{safe_normalizer_name} normalizer initialization complete")

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.count_matrix_file is None:
            raise NormalizationError("Count matrix file path is required")

        if isinstance(self.count_matrix_file, pd.DataFrame):
            if self.count_matrix_file.empty:
                raise NormalizationError("Count matrix DataFrame is empty")
            if self.gene_column not in self.count_matrix_file.columns:
                raise NormalizationError(f"Gene column '{self.gene_column}' not found in count matrix")
        else:
            if not self.file_manager.verify_files_exist(self.count_matrix_file):
                raise NormalizationError(f"Count matrix file not found: {self.count_matrix_file}")

        # Check if annotation file is required for this normalizer
        if self._requires_gene_lengths() and not self.annotation_file:
            raise NormalizationError(f"{self.normalizer_name} normalization requires annotation file for gene lengths")

        if self.annotation_file and not self.file_manager.verify_files_exist(self.annotation_file):
            raise NormalizationError(f"Annotation file not found: {self.annotation_file}")

    def _requires_gene_lengths(self) -> bool:
        """
        Check if this normalizer requires gene lengths.

        Returns:
            bool: True if gene lengths are required
        """
        # Length-based normalizers
        length_based_normalizers = ["rpkm", "tpm", "fpkm"]
        return self.normalizer_name in length_based_normalizers

    def _record_internal_operation(self, operation_type: str, details: str, method: str = None) -> None:
        """
        Record internal operations for execution reporting.

        Since normalization methods don't execute external commands, we record
        their internal operations for the execution report.

        Args:
            operation_type: Type of operation (e.g., 'data_loading', 'normalization')
            details: Details about the operation
            method: Normalization method name if operation is method-specific
        """
        if hasattr(self, "dry_run_manager") and self.dry_run_manager:
            operation_record = {
                "operation": "normalization_internal",
                "operation_type": operation_type,
                "details": details,
                "stage": f"{self.normalizer_name}_normalization",
                "timestamp": self.dry_run_manager._get_timestamp(),
                "normalization_method": method or self.normalizer_name,
            }

            # Add to executed operations list
            self.dry_run_manager.executed_operations.append(operation_record)

    def load_count_matrix(self) -> pd.DataFrame:
        """
        Load count matrix from file.

        Returns:
            DataFrame containing the count matrix

        Raises:
            NormalizationError: If count matrix loading fails
        """
        try:
            if isinstance(self.count_matrix_file, pd.DataFrame):
                df = self.count_matrix_file.copy()
                self.logger.info(f"Using provided count matrix DataFrame with shape {df.shape}")
                self._record_internal_operation(
                    "data_loading",
                    f"Using provided count matrix DataFrame: {df.shape[0]} genes, {df.shape[1] - 1} samples",
                )
            else:
                self.logger.info(f"Loading count matrix from: {self.count_matrix_file}")

                # Record this operation for the execution report
                self._record_internal_operation(
                    "data_loading",
                    f"Loading count matrix from: {self.count_matrix_file}",
                )

                # Determine file format and load accordingly
                file_ext = os.path.splitext(self.count_matrix_file)[1].lower()

                if file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(self.count_matrix_file)
                elif file_ext == ".csv":
                    df = pd.read_csv(self.count_matrix_file)
                elif file_ext in [".txt", ".tsv"]:
                    df = pd.read_csv(self.count_matrix_file, sep="\t")
                else:
                    raise NormalizationError(f"Unsupported file format: {file_ext}")

            # Validate gene column exists
            if self.gene_column not in df.columns:
                raise NormalizationError(f"Gene column '{self.gene_column}' not found in count matrix")

            # Store original data
            self.count_data = df.copy()

            # Record completion
            self._record_internal_operation(
                "data_loaded",
                f"Loaded count matrix: {df.shape[0]} genes, {df.shape[1] - 1} samples",
            )

            self.logger.info(f"Loaded count matrix: {df.shape[0]} genes, {df.shape[1] - 1} samples")
            return df

        except Exception as e:
            raise NormalizationError(f"Failed to load count matrix: {str(e)}")

    def _extract_gene_lengths(self) -> pd.DataFrame:
        """
        Extract gene lengths from annotation file.

        Returns:
            DataFrame with gene IDs and their lengths

        Raises:
            NormalizationError: If gene length extraction fails
        """
        try:
            self.logger.info(f"Extracting gene lengths from: {self.annotation_file}")

            # Record this operation for the execution report
            self._record_internal_operation(
                "annotation_parsing",
                f"Extracting gene lengths from: {self.annotation_file}",
            )

            # Read annotation file
            gtf = pd.read_csv(
                self.annotation_file,
                sep="\t",
                header=None,
                comment="#",
                low_memory=False,
            )
            gtf.columns = [
                "seqname",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attributes",
            ]
            annotation_format = self._annotation_format(self.annotation_file)
            self.logger.info(f"Detected annotation format for gene lengths: {annotation_format.upper()}")

            # Filter for gene features
            gtf = gtf[gtf["feature"] == "gene"]

            # Calculate gene lengths
            gtf["Length"] = gtf["end"] - gtf["start"] + 1

            # Keep stable gene identifiers only. GFF3 uses ID= and GTF uses
            # gene_id; display-name aliases such as Name= are intentionally not
            # used as primary identifiers.
            gene_length_map = {}
            ambiguous_aliases = set()
            for _, row in gtf.iterrows():
                attrs = self._parse_annotation_attributes(str(row["attributes"]))
                aliases = self._gene_aliases_from_attributes(attrs, annotation_format)
                for alias in aliases:
                    if not alias:
                        continue
                    if alias in gene_length_map and gene_length_map[alias] != row["Length"]:
                        ambiguous_aliases.add(alias)
                        continue
                    gene_length_map[alias] = row["Length"]

            for alias in ambiguous_aliases:
                gene_length_map.pop(alias, None)

            if not gene_length_map:
                raise NormalizationError("No gene identifiers found in annotation attributes")

            gene_lengths = pd.DataFrame([{"Gene": gene, "Length": length} for gene, length in gene_length_map.items()])
            gene_lengths = gene_lengths.set_index("Gene")

            self.gene_lengths = gene_lengths

            # Record completion
            self._record_internal_operation("annotation_parsed", f"Extracted lengths for {len(gene_lengths)} genes")

            self.logger.info(f"Extracted lengths for {len(gene_lengths)} genes")
            if ambiguous_aliases:
                self.logger.warning(
                    "Skipped %d ambiguous gene identifiers with conflicting lengths",
                    len(ambiguous_aliases),
                )
            return gene_lengths

        except Exception as e:
            raise NormalizationError(f"Failed to extract gene lengths: {str(e)}")

    @staticmethod
    def _annotation_format(annotation_file: str) -> str:
        """Infer GTF/GFF annotation format from the file extension."""
        suffixes = [suffix.lower() for suffix in Path(annotation_file).suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]
        extension = "".join(suffixes[-2:]) if len(suffixes) >= 2 else (suffixes[-1] if suffixes else "")

        if extension in {".gff", ".gff3"} or extension.endswith(".gff3"):
            return "gff"
        if extension == ".gtf" or extension.endswith(".gtf"):
            return "gtf"
        return "auto"

    @staticmethod
    def _parse_annotation_attributes(attributes: str) -> Dict[str, str]:
        """Parse GFF/GFF3/GTF attributes into a dictionary."""
        parsed = {}
        for item in attributes.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, value = item.split("=", 1)
            elif " " in item:
                key, value = item.split(" ", 1)
            else:
                continue
            parsed[key.strip()] = value.strip().strip('"')
        return parsed

    @staticmethod
    def _clean_gene_alias(alias: str) -> str:
        """Return a normalized gene alias while preserving stable IDs."""
        alias = str(alias).strip().strip('"')
        for prefix in ("gene:", "gene-"):
            if alias.startswith(prefix):
                return alias[len(prefix) :]
        return alias

    def _gene_aliases_from_attributes(self, attrs: Dict[str, str], annotation_format: str = "auto") -> List[str]:
        """Collect stable gene IDs according to the annotation format."""
        aliases = []
        if annotation_format == "gff":
            keys = ("ID",)
        elif annotation_format == "gtf":
            keys = ("gene_id",)
        else:
            keys = ("ID", "gene_id")

        for key in keys:
            value = attrs.get(key)
            if value:
                aliases.append(value)
                aliases.append(self._clean_gene_alias(value))

        # Preserve order while removing duplicates and empty values.
        unique_aliases = []
        seen = set()
        for alias in aliases:
            if alias and alias not in seen:
                unique_aliases.append(alias)
                seen.add(alias)
        return unique_aliases

    def _build_dryrun_gene_lengths(self, count_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build synthetic gene lengths for dry-run mode.

        This allows length-based normalizers to exercise their code paths even when
        upstream dry-run stages produced placeholder gene identifiers that do not
        exist in the real annotation file.
        """
        gene_lengths = pd.DataFrame(
            {"Length": np.repeat(1000.0, len(count_df))},
            index=count_df[self.gene_column].astype(str).values,
        )
        self.logger.info(
            "DRYRUN: Using synthetic gene lengths for %d genes in %s normalization",
            len(gene_lengths),
            self.normalizer_name.upper(),
        )
        return gene_lengths

    def create_boxplot(
        self,
        raw_data: np.ndarray,
        normalized_data: np.ndarray,
        sample_names: List[str],
        figsize: Tuple[int, int] = (20, 10),
        save_plot: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create boxplot comparing raw and normalized counts.

        Args:
            raw_data: Raw count data
            normalized_data: Normalized count data
            sample_names: List of sample names
            figsize: Figure size
            save_plot: Whether to save the plot

        Returns:
            Tuple of figure and axes objects
        """
        try:
            # Prepare data for plotting
            log_raw = [np.log(raw_data[:, i] + 1) for i in range(raw_data.shape[1])]
            log_normalized = [np.log(normalized_data[:, i] + 1) for i in range(normalized_data.shape[1])]

            # Create plot data
            plot_data = log_raw + log_normalized
            count_types = ["Raw counts"] * len(sample_names) + [f"{self.normalizer_name.upper()} counts"] * len(sample_names)
            labels = sample_names * 2

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create boxplot with colors
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            type_colors = {
                "Raw counts": colors[0],
                f"{self.normalizer_name.upper()} counts": colors[1],
            }

            # Plot boxplots
            positions = range(1, len(plot_data) + 1)
            bp = ax.boxplot(plot_data, positions=positions, patch_artist=True)

            # Color boxes
            for i, (patch, count_type) in enumerate(zip(bp["boxes"], count_types)):
                patch.set_facecolor(type_colors[count_type])
                patch.set_alpha(0.7)

            # Customize plot
            ax.set_xlabel("Sample Name")
            ax.set_ylabel("Log Counts")
            ax.set_title(f"Raw vs {self.normalizer_name.upper()} Normalized Counts")
            ax.set_xticklabels(labels, rotation=90)

            # Add legend
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7) for color in type_colors.values()]
            ax.legend(handles, type_colors.keys())

            plt.tight_layout()

            # Save plot if requested
            if save_plot and not self.dryrun:
                plot_file = self.out_dir / f"{self.normalizer_name}_comparison_boxplot.png"
                fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                self.logger.info(f"Boxplot saved to: {plot_file}")

            return fig, ax

        except Exception as e:
            self.logger.error(f"Failed to create boxplot: {str(e)}")
            return None, None

    @abstractmethod
    def normalize(
        self,
        plot: bool = True,
        save_results: bool = True,
        count_df: pd.DataFrame = None,
        gene_lengths: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Perform count normalization.

        Args:
            plot: Whether to create comparison plots
            save_results: Whether to save results to file
            count_df: Pre-loaded count matrix DataFrame (to avoid duplicate loading)
            gene_lengths: Pre-loaded gene lengths DataFrame (to avoid duplicate loading)

        Returns:
            DataFrame containing normalized counts

        Raises:
            NormalizationError: If normalization fails
        """
        pass

    def save_results(self, normalized_df: pd.DataFrame, suffix: str = "") -> str:
        """
        Save normalized results to file.

        Args:
            normalized_df: DataFrame with normalized counts
            suffix: Optional suffix for output filename

        Returns:
            Path to saved file
        """
        output_file = self.out_dir / f"{self.normalizer_name.upper()}_normalized_counts{suffix}.xlsx"

        if self.dryrun:
            self.logger.info(f"DRYRUN: Would save normalized counts to: {output_file}")
            return str(output_file)

        try:
            normalized_df.to_excel(output_file, index=False)
            self.logger.info(f"Normalized counts saved to: {output_file}")
            return str(output_file)

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise NormalizationError(f"Failed to save results: {str(e)}")

    def get_summary_statistics(self, normalized_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for normalized data.

        Args:
            normalized_df: DataFrame with normalized counts

        Returns:
            Dictionary with summary statistics
        """
        try:
            # Get numeric columns (exclude gene column)
            numeric_cols = [col for col in normalized_df.columns if col != self.gene_column]
            numeric_data = normalized_df[numeric_cols]

            stats = {
                "method": self.normalizer_name.upper(),
                "total_genes": len(normalized_df),
                "total_samples": len(numeric_cols),
                "mean_counts_per_gene": numeric_data.mean(axis=1).mean(),
                "median_counts_per_gene": numeric_data.median(axis=1).median(),
                "mean_counts_per_sample": numeric_data.mean().mean(),
                "median_counts_per_sample": numeric_data.median().median(),
                "genes_with_zero_counts": (numeric_data == 0).all(axis=1).sum(),
                "samples_with_zero_counts": (numeric_data == 0).all(axis=0).sum(),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to calculate summary statistics: {str(e)}")
            return {}

    def _create_normalization_log(self, log_file: Path) -> None:
        """Create a detailed log file for normalization."""
        from datetime import datetime

        with open(log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{self.normalizer_name.upper()} NORMALIZATION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: {self.normalizer_name.upper()}\n")
            f.write(f"Count Matrix File: {self.count_matrix_file}\n")
            if self.annotation_file:
                f.write(f"Annotation File: {self.annotation_file}\n")
            f.write(f"Output Directory: {self.out_dir}\n")
            f.write(f"Gene Column: {self.gene_column}\n")
            f.write(f"Requires Gene Lengths: {self._requires_gene_lengths()}\n")
            f.write("-" * 80 + "\n")
            f.write("PROCESSING LOG:\n")

    def _update_normalization_log(self, log_file: Path, result_data: dict) -> None:
        """Update the normalization log file with completion status."""
        from datetime import datetime

        with open(log_file, "a") as f:
            f.write("-" * 80 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Status: SUCCESS\n")
            f.write(f"Total Genes: {result_data.get('total_genes', 0)}\n")
            f.write(f"Total Samples: {result_data.get('total_samples', 0)}\n")
            f.write(f"Output File: {result_data.get('output_file', 'N/A')}\n")
            f.write(f"Plots Created: {result_data.get('plots_created', False)}\n")
            if "mean_value" in result_data:
                f.write(f"Mean {self.normalizer_name.upper()} per sample: {result_data['mean_value']:.2f}\n")
            f.write("=" * 80 + "\n")

    def run(self, plot: bool = True, save_results: bool = True) -> pd.DataFrame:
        """
        Run the complete normalization process.

        Args:
            plot: Whether to create comparison plots
            save_results: Whether to save results to file

        Returns:
            DataFrame containing normalized counts

        Raises:
            NormalizationError: If normalization fails
        """
        try:
            self.logger.info(f"Starting {self.normalizer_name} normalization")

            # Create log file for normalization
            if not self.dryrun:
                log_file = self.out_dir / f"{self.normalizer_name}_normalization.log"
                self._create_normalization_log(log_file)

            # Load count matrix (only once)
            count_df = self.load_count_matrix()

            # Extract gene lengths if required (only once)
            gene_lengths = None
            if self._requires_gene_lengths():
                if self.dryrun:
                    gene_lengths = self._build_dryrun_gene_lengths(count_df)
                else:
                    gene_lengths = self._extract_gene_lengths()

            # Perform normalization (pass loaded data to avoid reloading)
            normalized_df = self.normalize(
                plot=plot,
                save_results=save_results,
                count_df=count_df,
                gene_lengths=gene_lengths,
            )

            # Calculate summary statistics
            stats = self.get_summary_statistics(normalized_df)

            # Update log file with completion status
            if not self.dryrun:
                result_data = {
                    "total_genes": stats.get("total_genes", 0),
                    "total_samples": stats.get("total_samples", 0),
                    "output_file": self.save_results(normalized_df) if save_results else "N/A",
                    "plots_created": plot,
                    "mean_value": stats.get("mean_counts_per_sample", 0),
                }
                self._update_normalization_log(log_file, result_data)

            self.logger.info(
                f"Normalization completed: {stats.get('total_genes', 0)} genes, {stats.get('total_samples', 0)} samples"
            )

            return normalized_df

        except Exception as e:
            self.logger.error(f"{self.normalizer_name} normalization failed: {str(e)}")
            raise NormalizationError(f"{self.normalizer_name} normalization failed: {str(e)}")
