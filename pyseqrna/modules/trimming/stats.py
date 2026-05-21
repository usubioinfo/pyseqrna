#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trimming Statistics Module

This module provides functionality for calculating trimming statistics from RNA-seq data.
It processes FASTQ files efficiently to generate comprehensive trimming metrics, Excel/CSV reports,
and publication-ready visualizations without external processing complexities.

Features:
    - Reliable sequential processing of reads for high stability
    - Memory-efficient gzipped or uncompressed FASTQ file parsing and validation
    - Calculations of read count, base count, average read length, and read retention rates
    - Dual single-end and paired-end read library type compatibility
    - Automated cohort metrics summarization (averages, totals, best/worst samples)
    - Detailed Excel/CSV report generation and publication-ready plotting (retention rates, compare plots, heatmaps)

Configuration:
    The module is configured programmatically via constructor parameters, including input directories,
    output directories, raw/trimmed sample dictionary paths, thread allocations, and library type.

Dependencies:
    - pandas: For tabular data manipulation and Excel export
    - pyfastx: Optional dependency for fast FASTQ file parsing
    - matplotlib: For creating quality control figures
    - seaborn: For enhanced plotting visuals

Classes / Functions / Exceptions:
    - TrimmingSampleStats: Data class storing read/base metrics for a single sample.
    - TrimmingStats: Main calculator class to parse raw/trimmed FASTQ files and save reports/figures.
    - TrimmingStatsError: Custom exception class raised for trimming statistics failures.

:Created: January 20, 2025
:Updated: January 15, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Import utility modules
from pyseqrna.utils.file_manager import FileManager
from pyseqrna.utils.command_executor import CommandExecutor
from pyseqrna.utils.log_manager import LogManager
from pyseqrna.utils.resource_manager import ResourceManager
from pyseqrna.utils.dry_run_manager import DryRunManager

# Optional dependencies
try:
    import pyfastx

    HAS_PYFASTX = True
except ImportError:
    HAS_PYFASTX = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class TrimmingStatsError(Exception):
    """Custom exception for trimming statistics-related errors."""

    pass


@dataclass
class TrimmingSampleStats:
    """Data class to store trimming statistics for a single sample."""

    sample_id: str
    raw_reads: int = 0
    raw_bases: int = 0
    trimmed_reads: int = 0
    trimmed_bases: int = 0
    avg_raw_length: float = 0.0
    avg_trimmed_length: float = 0.0
    reads_retained: int = 0
    reads_discarded: int = 0
    retention_rate: float = 0.0
    bases_retained: int = 0
    bases_discarded: int = 0
    base_retention_rate: float = 0.0


class TrimmingStats:
    """
    A class for calculating trimming statistics from RNA-seq data using sequential processing.
    Now uses both the original sample dict (for raw files) and a trimmed dict (for actual trimmed files).
    """

    def __init__(
        self,
        samples_dict: Dict[str, List[str]],
        trimmed_dict: Dict[str, Union[str, List[str]]],
        out_dir: str,
        cpu_threads: int = 1,
        paired: bool = False,
        dryrun: bool = False,
        logger: Optional[Any] = None,
        dry_run_manager: Optional[DryRunManager] = None,
    ):
        """
        Initialize the TrimmingStats calculator.
        Args:
            samples_dict: Dictionary mapping sample names to file info (for raw files)
            trimmed_dict: Dictionary mapping sample names to trimmed file(s)
            out_dir: Output directory for results
            cpu_threads: Number of CPU threads reserved for future parallel statistics work
            paired: Whether the data is paired-end (affects read counting)
            dryrun: Whether to perform a dry run
            logger: Logger instance
            dry_run_manager: Dry run manager instance
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.logger
        else:
            self.logger = logger

        # Store configuration
        self.samples_dict = samples_dict
        self.trimmed_dict = trimmed_dict
        self.out_dir = Path(out_dir)
        self.cpu_threads = cpu_threads
        self.paired = paired
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager

        # Initialize utilities
        self.file_manager = FileManager(logger=self.logger)
        self.command_executor = CommandExecutor(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)

        # Results storage
        self.sample_stats: Dict[str, TrimmingSampleStats] = {}
        self.summary_stats: Optional[pd.DataFrame] = None

        sample_count = str(len(self.samples_dict)).replace("\n", "\\n").replace("\r", "\\r")
        paired_mode = str(self.paired).replace("\n", "\\n").replace("\r", "\\r")
        self.logger.info(f"Initialized TrimmingStats for {sample_count} samples")
        self.logger.info("Using sequential processing for reliability")
        self.logger.info(f"Paired-end mode: {paired_mode}")

    def _analyze_fastq_file(self, file_path: str) -> Tuple[int, int, float]:
        """
        Analyze a single FASTQ file to get read count, total bases, and average length.

        Args:
            file_path: Path to FASTQ file

        Returns:
            Tuple of (read_count, total_bases, avg_length)
        """
        if not os.path.exists(file_path):
            self.logger.error(f"FASTQ file not found: {file_path}")
            return 0, 0, 0.0

        try:
            # Always use pure Python method for portability
            read_count = 0
            total_length = 0
            # Support gzipped and plain FASTQ
            if file_path.endswith(".gz"):
                import gzip

                open_func = gzip.open
                mode = "rt"
            else:
                open_func = open
                mode = "r"
            with open_func(file_path, mode) as f:
                for i, line in enumerate(f):
                    if i % 4 == 1:  # Sequence line
                        read_count += 1
                        total_length += len(line.strip())
            avg_length = total_length / read_count if read_count > 0 else 0.0
            return read_count, total_length, avg_length
        except Exception as e:
            self.logger.error(f"Error analyzing FASTQ file {file_path}: {str(e)}")
            return 0, 0, 0.0

    def _process_sample(self, sample_id: str, sample_info: List[str]) -> TrimmingSampleStats:
        """
        Process a single sample to calculate trimming statistics.
        Args:
            sample_id: Sample identifier
            sample_info: List containing sample information [replication, identifier, file_path, ...]
        Returns:
            TrimmingSampleStats object
        """
        self.logger.info(f"Processing sample: {sample_id}")
        stats = TrimmingSampleStats(sample_id=sample_id)
        try:
            if self.paired:
                if len(sample_info) < 4:
                    self.logger.error(f"Insufficient sample info for paired-end sample {sample_id}: {sample_info}")
                    return stats
                raw_r1 = sample_info[2]
                raw_r2 = sample_info[3]
                trimmed_files = self.trimmed_dict.get(sample_id, [])
                if not trimmed_files or not isinstance(trimmed_files, (list, tuple)) or len(trimmed_files) < 2:
                    self.logger.warning(f"Trimmed files not found for {sample_id} in trimmed_dict: {trimmed_files}")
                    return stats
                trimmed_r1, trimmed_r2 = trimmed_files[0], trimmed_files[1]
                # Analyze raw files
                if os.path.exists(raw_r1) and os.path.exists(raw_r2):
                    raw_r1_reads, raw_r1_bases, avg_r1_length = self._analyze_fastq_file(raw_r1)
                    raw_r2_reads, raw_r2_bases, avg_r2_length = self._analyze_fastq_file(raw_r2)
                    stats.raw_reads = raw_r1_reads  # Number of read pairs
                    stats.raw_bases = raw_r1_bases + raw_r2_bases
                    stats.avg_raw_length = (avg_r1_length + avg_r2_length) / 2
                    self.logger.info(f"Sample {sample_id}: {stats.raw_reads} raw read pairs, {stats.raw_bases} bases")
                else:
                    self.logger.warning(f"Raw files not found for {sample_id}: {raw_r1}, {raw_r2}")
                # Analyze trimmed files
                if os.path.exists(trimmed_r1) and os.path.exists(trimmed_r2):
                    trim_r1_reads, trim_r1_bases, avg_trim1_length = self._analyze_fastq_file(trimmed_r1)
                    trim_r2_reads, trim_r2_bases, avg_trim2_length = self._analyze_fastq_file(trimmed_r2)
                    stats.trimmed_reads = trim_r1_reads  # Number of read pairs
                    stats.trimmed_bases = trim_r1_bases + trim_r2_bases
                    stats.avg_trimmed_length = (avg_trim1_length + avg_trim2_length) / 2
                    stats.reads_retained = stats.trimmed_reads
                    stats.reads_discarded = stats.raw_reads - stats.trimmed_reads
                    stats.retention_rate = (stats.trimmed_reads / stats.raw_reads * 100) if stats.raw_reads > 0 else 0.0
                    stats.bases_retained = stats.trimmed_bases
                    stats.bases_discarded = stats.raw_bases - stats.trimmed_bases
                    stats.base_retention_rate = (stats.trimmed_bases / stats.raw_bases * 100) if stats.raw_bases > 0 else 0.0
                    self.logger.info(
                        f"Sample {sample_id}: {stats.trimmed_reads} trimmed read pairs ({stats.retention_rate:.1f}% retained)"
                    )
                else:
                    self.logger.warning(f"Trimmed files not found for {sample_id}: {trimmed_r1}, {trimmed_r2}")
            else:
                if len(sample_info) < 3:
                    sanitized_sample_info = str(sample_info).replace("\n", "\\n").replace("\r", "\\r")
                    self.logger.error(f"Insufficient sample info for {sample_id}: {sanitized_sample_info}")
                    return stats
                raw_file = sample_info[2]
                trimmed_file = self.trimmed_dict.get(sample_id, None)
                if not trimmed_file:
                    self.logger.warning(f"Trimmed file not found for {sample_id} in trimmed_dict: {trimmed_file}")
                    return stats
                # Analyze raw file
                if os.path.exists(raw_file):
                    raw_reads, raw_bases, avg_raw_length = self._analyze_fastq_file(raw_file)
                    stats.raw_reads = raw_reads
                    stats.raw_bases = raw_bases
                    stats.avg_raw_length = avg_raw_length
                    self.logger.info(f"Sample {sample_id}: {raw_reads} raw reads, {raw_bases} bases")
                else:
                    self.logger.warning(f"Raw file not found for {sample_id}: {raw_file}")
                # Analyze trimmed file
                if os.path.exists(trimmed_file):
                    trimmed_reads, trimmed_bases, avg_trimmed_length = self._analyze_fastq_file(trimmed_file)
                    stats.trimmed_reads = trimmed_reads
                    stats.trimmed_bases = trimmed_bases
                    stats.avg_trimmed_length = avg_trimmed_length
                    stats.reads_retained = trimmed_reads
                    stats.reads_discarded = stats.raw_reads - trimmed_reads
                    stats.retention_rate = (trimmed_reads / stats.raw_reads * 100) if stats.raw_reads > 0 else 0.0
                    stats.bases_retained = trimmed_bases
                    stats.bases_discarded = stats.raw_bases - trimmed_bases
                    stats.base_retention_rate = (trimmed_bases / stats.raw_bases * 100) if stats.raw_bases > 0 else 0.0
                    self.logger.info(
                        f"Sample {sample_id}: {trimmed_reads} trimmed reads ({stats.retention_rate:.1f}% retained)"
                    )
                else:
                    self.logger.warning(f"Trimmed file not found for {sample_id}: {trimmed_file}")
        except Exception as e:
            self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
        return stats

    def calculate_statistics(self) -> Dict[str, TrimmingSampleStats]:
        """
        Calculate trimming statistics for all samples using sequential processing.

        Returns:
            Dictionary mapping sample IDs to their statistics
        """
        self.logger.info("Starting trimming statistics calculation")

        # Create output directories
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created main output directory: {self.out_dir}")

        # Process each sample sequentially
        for sample_id, sample_info in self.samples_dict.items():
            if self.dryrun:
                self.logger.info(f"DRYRUN: Would process sample {sample_id}")
                # Create mock stats for dry run
                mock_stats = TrimmingSampleStats(
                    sample_id=sample_id,
                    raw_reads=10000,
                    raw_bases=750000,
                    trimmed_reads=9500,
                    trimmed_bases=700000,
                    avg_raw_length=75.0,
                    avg_trimmed_length=73.7,
                    reads_retained=9500,
                    reads_discarded=500,
                    retention_rate=95.0,
                    bases_retained=700000,
                    bases_discarded=50000,
                    base_retention_rate=93.3,
                )
                self.sample_stats[sample_id] = mock_stats
            else:
                stats = self._process_sample(sample_id, sample_info)
                self.sample_stats[sample_id] = stats

        self.logger.info(f"Completed trimming statistics for {len(self.sample_stats)} samples")
        return self.sample_stats

    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame from the calculated statistics.

        Returns:
            DataFrame containing summary statistics
        """
        if not self.sample_stats:
            self.logger.warning("No statistics available. Run calculate_statistics() first.")
            return pd.DataFrame()

        data = []
        for sample_id, stats in self.sample_stats.items():
            data.append(
                {
                    "Sample_ID": stats.sample_id,
                    "Raw_Reads": stats.raw_reads,
                    "Raw_Bases": stats.raw_bases,
                    "Avg_Raw_Length": stats.avg_raw_length,
                    "Trimmed_Reads": stats.trimmed_reads,
                    "Trimmed_Bases": stats.trimmed_bases,
                    "Avg_Trimmed_Length": stats.avg_trimmed_length,
                    "Reads_Retained": stats.reads_retained,
                    "Reads_Discarded": stats.reads_discarded,
                    "Read_Retention_Rate": stats.retention_rate,
                    "Bases_Retained": stats.bases_retained,
                    "Bases_Discarded": stats.bases_discarded,
                    "Base_Retention_Rate": stats.base_retention_rate,
                }
            )

        self.summary_stats = pd.DataFrame(data)
        return self.summary_stats

    def save_results(self) -> bool:
        """
        Save the trimming statistics results to files.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.sample_stats:
                self.logger.warning("No statistics to save")
                return False

            # Create summary DataFrame
            df = self.create_summary_dataframe()

            if self.dryrun:
                self.logger.info(f"DRYRUN: Would save results to {self.out_dir}")
                return True

            # Save as Excel
            excel_file = self.out_dir / "trimming_statistics.xlsx"
            df.to_excel(excel_file, index=False, sheet_name="Trimming_Stats")
            self.logger.info(f"Saved Excel report: {excel_file}")

            # Save as CSV
            csv_file = self.out_dir / "trimming_statistics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved CSV file: {csv_file}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False

    def create_visualizations(self) -> bool:
        """
        Create publication-ready visualizations of trimming statistics.

        Returns:
            True if successful, False otherwise
        """
        if not HAS_PLOTTING:
            self.logger.warning("Matplotlib/seaborn not available, skipping visualizations")
            return False

        if not self.sample_stats:
            self.logger.warning("No statistics available for plotting")
            return False

        try:
            if self.dryrun:
                self.logger.info("DRYRUN: Would create visualizations")
                return True

            # Create plots directory
            plots_dir = self.out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            df = self.create_summary_dataframe()

            # Set up the plotting style
            plt.style.use("default")
            sns.set_palette("husl")

            # 1. Read retention rate plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(
                df["Sample_ID"],
                df["Read_Retention_Rate"],
                color="steelblue",
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_xlabel("Sample ID", fontsize=12, fontweight="bold")
            ax.set_ylabel("Read Retention Rate (%)", fontsize=12, fontweight="bold")
            ax.set_title("Read Retention Rate After Trimming", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for bar, value in zip(bars, df["Read_Retention_Rate"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(plots_dir / "read_retention_rates.png", dpi=300, bbox_inches="tight")
            plt.close()

            # 2. Before/After comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Read counts
            x = range(len(df))
            width = 0.35
            ax1.bar(
                [i - width / 2 for i in x],
                df["Raw_Reads"],
                width,
                label="Raw Reads",
                color="lightcoral",
                alpha=0.7,
            )
            ax1.bar(
                [i + width / 2 for i in x],
                df["Trimmed_Reads"],
                width,
                label="Trimmed Reads",
                color="steelblue",
                alpha=0.7,
            )
            ax1.set_xlabel("Sample ID", fontweight="bold")
            ax1.set_ylabel("Number of Reads", fontweight="bold")
            ax1.set_title("Read Counts: Before vs After Trimming", fontweight="bold")
            ax1.set_xticks(x)
            ax1.set_xticklabels(df["Sample_ID"], rotation=45, ha="right")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Average lengths
            ax2.bar(
                [i - width / 2 for i in x],
                df["Avg_Raw_Length"],
                width,
                label="Raw Avg Length",
                color="lightcoral",
                alpha=0.7,
            )
            ax2.bar(
                [i + width / 2 for i in x],
                df["Avg_Trimmed_Length"],
                width,
                label="Trimmed Avg Length",
                color="steelblue",
                alpha=0.7,
            )
            ax2.set_xlabel("Sample ID", fontweight="bold")
            ax2.set_ylabel("Average Read Length (bp)", fontweight="bold")
            ax2.set_title("Average Read Length: Before vs After Trimming", fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels(df["Sample_ID"], rotation=45, ha="right")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(plots_dir / "before_after_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()

            # 3. Summary statistics heatmap
            if len(df) > 1:
                numeric_cols = [
                    "Read_Retention_Rate",
                    "Base_Retention_Rate",
                    "Avg_Raw_Length",
                    "Avg_Trimmed_Length",
                ]
                heatmap_data = df[["Sample_ID"] + numeric_cols].set_index("Sample_ID")

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    heatmap_data.T,
                    annot=True,
                    fmt=".1f",
                    cmap="RdYlBu_r",
                    center=heatmap_data.mean().mean(),
                    ax=ax,
                )
                ax.set_title("Trimming Statistics Overview", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(plots_dir / "statistics_heatmap.png", dpi=300, bbox_inches="tight")
                plt.close()

            self.logger.info(f"Created visualizations in {plots_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return False

    def summarize_results(self, results) -> str:
        """
        Generate a text summary of the trimming statistics results.

        Args:
            results: Results from the run() method (boolean or DataFrame)

        Returns:
            str: Text summary of results
        """
        if not self.sample_stats:
            return "No trimming statistics results available."

        # Calculate averages
        total_samples = len(self.sample_stats)
        avg_retention_rate = sum(stats.retention_rate for stats in self.sample_stats.values()) / total_samples
        avg_base_retention_rate = sum(stats.base_retention_rate for stats in self.sample_stats.values()) / total_samples

        # Find best and worst samples
        best_sample = max(self.sample_stats.values(), key=lambda x: x.retention_rate)
        worst_sample = min(self.sample_stats.values(), key=lambda x: x.retention_rate)

        # Calculate totals
        total_raw_reads = sum(stats.raw_reads for stats in self.sample_stats.values())
        total_trimmed_reads = sum(stats.trimmed_reads for stats in self.sample_stats.values())
        total_discarded_reads = sum(stats.reads_discarded for stats in self.sample_stats.values())

        # Build summary text
        summary = [
            "Trimming Statistics Summary",
            "---------------------------",
            f"Number of samples: {total_samples}",
            f"Total raw reads: {total_raw_reads:,}",
            f"Total trimmed reads: {total_trimmed_reads:,}",
            f"Total discarded reads: {total_discarded_reads:,}",
            f"Average read retention rate: {avg_retention_rate:.2f}%",
            f"Average base retention rate: {avg_base_retention_rate:.2f}%",
            "",
            f"Best retained sample: {best_sample.sample_id} ({best_sample.retention_rate:.2f}%)",
            f"Worst retained sample: {worst_sample.sample_id} ({worst_sample.retention_rate:.2f}%)",
        ]

        return "\n".join(summary)

    def run(self) -> bool:
        """
        Run the complete trimming statistics analysis.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting trimming statistics analysis")

            # Calculate statistics
            self.calculate_statistics()

            # Save results
            if not self.save_results():
                return False

            # Create visualizations
            if not self.create_visualizations():
                self.logger.warning("Failed to create visualizations, but continuing")

            self.logger.info("Trimming statistics analysis completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Trimming statistics analysis failed: {str(e)}")
            return False
