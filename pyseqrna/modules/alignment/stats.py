#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alignment Statistics Module

This module calculates alignment statistics from RNA-seq data. It processes BAM files
efficiently and generates comprehensive alignment metrics with a report-ready summary figure.
It supports parallel sample processing, automatic file validation, log file parsing, and BAM index checks.

Features:
    - Parallel per-sample processing with deterministic output ordering
    - Memory-efficient BAM file parsing and coordinate sort checking
    - Automatic BAM indexing and sorting integration using `pysam`
    - Read count statistics extraction from raw/trimmed FASTQ files
    - Support for multiple statistics sources (automated log parsing or direct BAM analysis)
    - Comprehensive spreadsheet reports (Excel/CSV) and publication-ready plots

Configuration:
    Configured programmatically via constructor arguments, specifying sample maps,
    trimmed file maps, BAM locations, output directories, thread allocations, and library type (paired-end).

Dependencies:
    - pandas: For data manipulation and Excel output
    - pysam: For BAM file processing and indexing
    - pyfastx: For fast FASTQ processing
    - matplotlib: For creating summary plots
    - seaborn: For enhanced publication-style visualizations

Classes / Functions / Exceptions:
    - SampleStats: Data class to store alignment statistics for a single sample.
    - AlignmentStats: Core class to calculate alignment statistics, export reports, and generate plots.
    - AlignmentStatsError: Custom exception for alignment statistics-related errors.

:Created: January 20, 2025
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utility modules
from pyseqrna.utils.file_manager import FileManager
from pyseqrna.utils.command_executor import CommandExecutor
from pyseqrna.utils.log_manager import LogManager
from pyseqrna.utils.resource_manager import ResourceManager
from pyseqrna.utils.dry_run_manager import DryRunManager

# Optional dependencies
try:
    import pysam

    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

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


class AlignmentStatsError(Exception):
    """Custom exception for alignment statistics-related errors."""

    pass


@dataclass
class SampleStats:
    """Data class to store alignment statistics for a single sample."""

    sample_id: str
    total_reads: int = 0
    aligned_reads: int = 0
    unique_mapped: int = 0
    multi_mapped: int = 0
    unmapped_reads: int = 0
    alignment_rate: float = 0.0
    unique_rate: float = 0.0
    multi_rate: float = 0.0
    trimmed_reads: int = 0


class AlignmentStats:
    """
    Calculate alignment statistics from raw reads, trimmed reads, and aligned BAM files.
    """

    def __init__(
        self,
        sample_dict: Dict[str, List[str]],
        trimmed_dict: Dict[str, Union[str, List[str]]],
        bam_dict: Dict[str, str],
        out_dir: str,
        cpu_threads: int = 1,
        paired: bool = False,
        dryrun: bool = False,
        logger: Optional[Any] = None,
        dry_run_manager: Optional[DryRunManager] = None,
        trimming_stats: Optional[Dict[str, Any]] = None,
        source: str = "auto",
        alignment_tool: Optional[str] = None,
    ):
        """
        Initialize the AlignmentStats calculator.
        Args:
            sample_dict: Dictionary mapping sample names to raw FASTQ info
            trimmed_dict: Dictionary mapping sample names to trimmed FASTQ files (trimmed)
            bam_dict: Dictionary mapping sample names to BAM file paths (aligned)
            out_dir: Output directory for results
            cpu_threads: Number of CPU threads reserved for future parallel statistics work
            paired: Whether the data is paired-end (affects read counting)
            dryrun: Whether to perform a dry run
            logger: Logger instance
            dry_run_manager: Dry run manager instance
            trimming_stats: Optional dict of precomputed trimming stats
            source: Statistics source: auto, logs, or bam
            alignment_tool: Aligner used to produce BAMs/logs
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.logger
        else:
            self.logger = logger

        # Store configuration
        self.sample_dict = sample_dict
        self.trimmed_dict = trimmed_dict
        self.bam_dict = bam_dict
        self.out_dir = Path(out_dir)
        self.cpu_threads = cpu_threads
        self.paired = paired
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager
        self.trimming_stats = trimming_stats
        self.source = str(source or "auto").lower()
        if self.source not in {"auto", "logs", "bam"}:
            self.logger.warning("Unknown alignment statistics source '%s'; using auto", self.source)
            self.source = "auto"
        self.alignment_tool = str(alignment_tool or "").lower()
        self.resolved_source = "bam"

        # Initialize utilities
        self.file_manager = FileManager(logger=self.logger)
        self.command_executor = CommandExecutor(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)

        # Results storage
        self.sample_stats: Dict[str, SampleStats] = {}
        self.summary_stats: Optional[pd.DataFrame] = None

        self.logger.info(f"Initialized AlignmentStats for {len(self.sample_dict)} samples")
        self.max_workers = max(1, min(len(self.sample_dict) or 1, max(1, int(self.cpu_threads or 1)), 8))
        self.logger.info(f"Using up to {self.max_workers} worker(s) for alignment statistics")
        self.logger.info("Paired-end mode: %s", self.paired)
        self.logger.info("Alignment statistics source: %s", self.source)

    def _get_bam_file(self, sample_id: str) -> Optional[str]:
        """Return the BAM path for a sample from a plain or metadata-rich BAM dict."""
        bam_data = self.bam_dict.get(sample_id) if self.bam_dict else None
        if isinstance(bam_data, dict):
            bam_file = bam_data.get("bam")
        elif isinstance(bam_data, str):
            bam_file = bam_data
        else:
            bam_file = None
        return str(bam_file) if bam_file else None

    def _candidate_log_files(self, sample_id: str) -> List[Path]:
        """Return likely aligner log files for a sample."""
        candidates: List[Path] = []
        bam_data = self.bam_dict.get(sample_id) if self.bam_dict else None
        if isinstance(bam_data, dict):
            for key in ("log", "summary", "stderr", "err"):
                path = bam_data.get(key)
                if path:
                    candidates.append(Path(path))

        bam_file = self._get_bam_file(sample_id)
        if bam_file:
            bam_path = Path(bam_file)
            results_dir = bam_path.parent
            alignment_dir = results_dir.parent
            logs_dir = alignment_dir / "logs"
            tool_names = [self.alignment_tool] if self.alignment_tool else []
            tool_names.extend([name for name in ("star", "hisat2", "bowtie2") if name not in tool_names])
            for tool in tool_names:
                if tool == "star":
                    candidates.extend(
                        [
                            results_dir / f"{sample_id}_Log.final.out",
                            logs_dir / f"{sample_id}_{tool}_alignment.slurm.err",
                            logs_dir / f"{sample_id}_{tool}_alignment.err",
                        ]
                    )
                else:
                    candidates.extend(
                        [
                            logs_dir / f"{sample_id}_{tool}_alignment.slurm.err",
                            logs_dir / f"{sample_id}_{tool}_alignment.err",
                            results_dir / f"{sample_id}_{tool}_alignment.log",
                        ]
                    )

        unique = []
        seen = set()
        for path in candidates:
            if path and str(path) not in seen:
                unique.append(path)
                seen.add(str(path))
        return unique

    def _parse_star_log(self, log_file: Path, sample_id: str) -> Optional[SampleStats]:
        """Parse STAR Log.final.out into SampleStats."""
        try:
            values = {}
            with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if "|" not in line:
                        continue
                    key, value = line.split("|", 1)
                    values[key.strip()] = value.strip().replace("%", "")

            def as_int(key: str) -> int:
                raw = values.get(key, "0").replace(",", "")
                try:
                    return int(float(raw))
                except ValueError:
                    return 0

            total = as_int("Number of input reads")
            unique = as_int("Uniquely mapped reads number")
            multi = as_int("Number of reads mapped to multiple loci") + as_int("Number of reads mapped to too many loci")
            aligned = unique + multi
            unmapped = max(total - aligned, 0)
            if total <= 0:
                return None

            stats = SampleStats(sample_id=sample_id)
            stats.total_reads = total
            stats.trimmed_reads = total
            stats.aligned_reads = aligned
            stats.unique_mapped = unique
            stats.multi_mapped = multi
            stats.unmapped_reads = unmapped
            stats.alignment_rate = (aligned / total) * 100
            stats.unique_rate = (unique / total) * 100
            stats.multi_rate = (multi / total) * 100
            return stats
        except Exception as exc:
            self.logger.debug("Could not parse STAR alignment log %s: %s", log_file, exc)
            return None

    def _parse_generic_alignment_log(self, log_file: Path, sample_id: str) -> Optional[SampleStats]:
        """Parse HISAT2/Bowtie2-style stderr logs when detailed BAM parsing is not needed."""
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            total_match = re.search(r"^\s*(\d+)\s+(?:reads|pairs);", text, flags=re.MULTILINE)
            rate_match = re.search(r"([\d.]+)%\s+overall alignment rate", text)
            if not total_match or not rate_match:
                return None

            total = int(total_match.group(1))
            rate = float(rate_match.group(1))
            aligned = int(round(total * rate / 100.0))
            unmapped = max(total - aligned, 0)

            stats = SampleStats(sample_id=sample_id)
            stats.total_reads = total
            stats.trimmed_reads = total
            stats.aligned_reads = aligned
            stats.unique_mapped = aligned
            stats.multi_mapped = 0
            stats.unmapped_reads = unmapped
            stats.alignment_rate = rate
            stats.unique_rate = rate
            stats.multi_rate = 0.0
            return stats
        except Exception as exc:
            self.logger.debug("Could not parse alignment log %s: %s", log_file, exc)
            return None

    def _parse_alignment_log(self, log_file: Path, sample_id: str) -> Optional[SampleStats]:
        """Parse the best supported alignment log format for one sample."""
        if not log_file.exists():
            return None
        if log_file.name.endswith("Log.final.out") or self.alignment_tool == "star":
            stats = self._parse_star_log(log_file, sample_id)
            if stats:
                return stats
            if self.source == "auto":
                return None
        if self.source != "logs":
            return None
        return self._parse_generic_alignment_log(log_file, sample_id)

    def _calculate_statistics_from_logs(self) -> bool:
        """Try to populate statistics from aligner logs for every sample."""
        completed: Dict[str, SampleStats] = {}
        missing: List[str] = []

        for sample_id in self.sample_dict:
            stats = None
            for log_file in self._candidate_log_files(sample_id):
                stats = self._parse_alignment_log(log_file, sample_id)
                if stats:
                    self.logger.info(
                        "Loaded alignment statistics for %s from %s",
                        sample_id,
                        log_file,
                    )
                    break
            if stats:
                completed[sample_id] = stats
            else:
                missing.append(sample_id)

        if missing:
            self.logger.info(
                "Alignment log statistics were unavailable for %d sample(s): %s",
                len(missing),
                ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
            )
            return False

        self.sample_stats = {sample_id: completed[sample_id] for sample_id in self.sample_dict}
        self.resolved_source = "logs"
        self.logger.info("Loaded alignment statistics from aligner logs for all samples")
        return True

    def _count_fastq_reads(self, file_paths: Union[str, List[str]]) -> int:
        """
        Count reads in FASTQ file(s), handling both single-end and paired-end data.

        Args:
            file_paths: Path to FASTQ file or list of paths for paired-end

        Returns:
            Total number of reads (for paired-end, counts each pair as one read unit)
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        total_reads = 0

        for file_path in file_paths:
            if not os.path.exists(file_path):
                self.logger.error(f"FASTQ file not found: {file_path}")
                continue
            try:
                # Always use pure Python method for portability
                count = 0
                # sanitize file path early for safe logging
                safe_file_path = str(file_path).replace("\n", "_").replace("\r", "_")
                if file_path.endswith(".gz"):
                    import gzip

                    open_func = gzip.open
                    mode = "rt"
                else:
                    open_func = open
                    mode = "r"
                with open_func(file_path, mode) as f:
                    for i, line in enumerate(f):
                        if i % 4 == 0:  # Header line
                            count += 1
                self.logger.debug(f"Counted {count} reads in {safe_file_path}")
                total_reads += count
            except Exception as e:
                # sanitize inputs to the log to mitigate log injection (CWE-117)
                safe_file_path = str(file_path).replace("\n", "_").replace("\r", "_")
                safe_error = str(e).replace("\n", " ").replace("\r", " ")
                self.logger.error(f"Error counting reads in {safe_file_path}: {safe_error}")

        # For paired-end data, we count each pair as one read unit
        # So if we have R1 and R2 files with 1000 reads each, total = 1000 (not 2000)
        if self.paired and len(file_paths) == 2:
            total_reads = total_reads // 2
            self.logger.debug(f"Paired-end data: adjusted count to {total_reads} read pairs")

        return total_reads

    def _analyze_bam_file(self, bam_file: str, sample_id: str) -> SampleStats:
        """
        Analyze a single BAM file to get alignment statistics.

        Args:
            bam_file: Path to BAM file
            sample_id: Sample identifier

        Returns:
            SampleStats object with alignment statistics
        """
        self.logger.info(f"Analyzing BAM file for sample {sample_id}: {bam_file}")

        stats = SampleStats(sample_id=sample_id)

        if not os.path.exists(bam_file):
            self.logger.error(f"BAM file not found: {bam_file}")
            return stats

        try:
            # Open BAM file
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                aligned = 0
                unique = 0
                multi = 0
                unmapped = 0

                # Process each read
                for read in bam.fetch():
                    if read.is_unmapped:
                        unmapped += 1
                    else:
                        # Skip secondary alignments for counting
                        if not read.is_secondary:
                            aligned += 1

                            # Check if uniquely mapped or multi-mapped
                            try:
                                nh_tag = read.get_tag("NH")  # Number of hits
                                if nh_tag == 1:
                                    unique += 1
                                elif nh_tag > 1:
                                    multi += 1
                            except KeyError:
                                # No NH tag, assume unique
                                unique += 1

                # For paired-end data, BAM contains both R1 and R2 reads
                # We need to adjust counts to represent read pairs, not individual reads
                if self.paired:
                    aligned = aligned // 2
                    unique = unique // 2
                    multi = multi // 2
                    unmapped = unmapped // 2
                    self.logger.debug("Paired-end BAM: adjusted counts to read pairs")

                # Calculate totals and rates
                total_reads = aligned + unmapped
                stats.total_reads = total_reads
                stats.aligned_reads = aligned
                stats.unique_mapped = unique
                stats.multi_mapped = multi
                stats.unmapped_reads = unmapped

                if total_reads > 0:
                    stats.alignment_rate = (aligned / total_reads) * 100
                    stats.unique_rate = (unique / total_reads) * 100
                    stats.multi_rate = (multi / total_reads) * 100

                # Sanitize sample_id to mitigate log injection (CWE-117)
                safe_sample_id = str(sample_id).replace("\n", "_").replace("\r", "_")
                # Use parameterized logging instead of f-strings and interpolate sanitized values
                self.logger.info(
                    "Sample %s: %d/%d reads aligned (%.1f%%)",
                    safe_sample_id,
                    aligned,
                    total_reads,
                    stats.alignment_rate,
                )
                self.logger.info(
                    "Sample %s: %d unique, %d multi-mapped",
                    safe_sample_id,
                    unique,
                    multi,
                )

        except Exception as e:
            self.logger.error(f"Error analyzing BAM file {bam_file}: {str(e)}")

        return stats

    def _is_bam_sorted(self, bam_file: str) -> bool:
        """
        Check if a BAM file is coordinate-sorted.
        """
        import pysam

        try:
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                hd = bam.header.get("HD", {})
                return hd.get("SO", "") == "coordinate"
        except Exception as e:
            self.logger.warning(f"Could not determine if BAM is sorted: {bam_file}: {e}")
            return False

    def _sort_bam(self, bam_file: str) -> str:
        """
        Sort a BAM file by coordinate and return the path to the sorted BAM.
        """
        import pysam

        sorted_bam = bam_file.replace(".bam", ".sorted.bam")
        try:
            pysam.sort("-o", sorted_bam, bam_file)
            self.logger.info(f"Sorted BAM file: {sorted_bam}")
            return sorted_bam
        except Exception as e:
            self.logger.error(f"Failed to sort BAM file {bam_file}: {e}")
            return bam_file

    def _calculate_sample_statistics(self, sample_id: str, sample_info: List[str]) -> Optional[SampleStats]:
        """Calculate alignment statistics for a single sample."""
        if sample_id not in self.bam_dict:
            self.logger.warning(f"No BAM file found for sample {sample_id}")
            return None

        bam_file = self._get_bam_file(sample_id)
        if not bam_file:
            self.logger.error(f"Invalid BAM data format for {sample_id}: {self.bam_dict.get(sample_id)}")
            return None

        if not os.path.exists(bam_file):
            self.logger.error(f"BAM file not found for {sample_id}: {bam_file}")
            return None

        if not self._is_bam_sorted(bam_file):
            self.logger.info(f"BAM file not sorted for {sample_id}, sorting now: {bam_file}")
            bam_file = self._sort_bam(bam_file)

        bai_file = bam_file + ".bai"
        bai_path = Path(bai_file)
        bam_path = Path(bam_file)

        needs_indexing = False
        if not bai_path.exists():
            self.logger.info(f"Index file does not exist for {sample_id}: {bam_file}")
            needs_indexing = True
        elif bai_path.stat().st_mtime < bam_path.stat().st_mtime:
            self.logger.info(f"Index file is older than BAM file for {sample_id}: {bam_file}")
            needs_indexing = True

        if needs_indexing:
            self.logger.info(f"Indexing BAM file for {sample_id}: {bam_file}")
            try:
                pysam.index(bam_file)
            except Exception as e:
                self.logger.error(f"Failed to index BAM file {bam_file}: {e}")
                return None
        else:
            self.logger.debug(f"BAM file {bam_file} is already indexed and current")

        stats = self._analyze_bam_file(bam_file, sample_id)

        if len(sample_info) >= 3:
            if self.paired and len(sample_info) >= 4:
                raw_count = self._count_fastq_reads([sample_info[2], sample_info[3]])
            else:
                raw_count = self._count_fastq_reads(sample_info[2])

            trimmed_count = 0
            if isinstance(self.trimming_stats, dict) and sample_id in self.trimming_stats:
                trim_stat = self.trimming_stats[sample_id]
                if hasattr(trim_stat, "trimmed_reads"):
                    trimmed_count = trim_stat.trimmed_reads
                elif isinstance(trim_stat, dict) and "trimmed_reads" in trim_stat:
                    trimmed_count = trim_stat["trimmed_reads"]
            else:
                trimmed_files = self.trimmed_dict.get(sample_id, None) if self.trimmed_dict else None
                if self.paired and trimmed_files and isinstance(trimmed_files, (list, tuple)) and len(trimmed_files) >= 2:
                    trimmed_count = self._count_fastq_reads(trimmed_files)
                elif trimmed_files:
                    trimmed_count = self._count_fastq_reads(trimmed_files)

            stats.total_reads = raw_count
            stats.trimmed_reads = trimmed_count
            if stats.total_reads > 0:
                stats.alignment_rate = (stats.aligned_reads / stats.total_reads) * 100
                stats.unique_rate = (stats.unique_mapped / stats.total_reads) * 100
                stats.multi_rate = (stats.multi_mapped / stats.total_reads) * 100

        return stats

    def calculate_statistics(self) -> Dict[str, SampleStats]:
        """
        Calculate alignment statistics for all samples.
        Returns:
            Dictionary mapping sample IDs to their statistics
        """
        self.logger.info("Starting alignment statistics calculation")
        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created main output directory: {self.out_dir}")
        else:
            self.logger.info(f"DRYRUN: Would create main output directory: {self.out_dir}")
            for sample_id, sample_info in self.sample_dict.items():
                stats = SampleStats(sample_id=sample_id)
                if len(sample_info) >= 3:
                    if self.paired and len(sample_info) >= 4:
                        fastq_files = [sample_info[2], sample_info[3]]
                        stats.total_reads = self._count_fastq_reads(fastq_files)
                    else:
                        stats.total_reads = self._count_fastq_reads(sample_info[2])
                self.sample_stats[sample_id] = stats
            self.logger.info(f"DRYRUN: Simulated alignment statistics for {len(self.sample_stats)} samples")
            return self.sample_stats

        if self.source in {"auto", "logs"}:
            if self._calculate_statistics_from_logs():
                return self.sample_stats
            if self.source == "logs":
                raise AlignmentStatsError(
                    "alignment_stats_source='logs' was requested, but complete aligner logs were not found"
                )
            self.logger.info("Falling back to BAM-based alignment statistics")

        self.resolved_source = "bam"

        if self.max_workers > 1:
            self.logger.info(f"Calculating alignment statistics in parallel with {self.max_workers} worker(s)")
            completed_stats = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_sample = {
                    executor.submit(self._calculate_sample_statistics, sample_id, sample_info): sample_id
                    for sample_id, sample_info in self.sample_dict.items()
                }
                for future in as_completed(future_to_sample):
                    sample_id = future_to_sample[future]
                    try:
                        stats = future.result()
                        if stats is not None:
                            completed_stats[sample_id] = stats
                    except Exception as e:
                        self.logger.error(f"Failed to calculate alignment statistics for {sample_id}: {e}")

            for sample_id in self.sample_dict:
                if sample_id in completed_stats:
                    self.sample_stats[sample_id] = completed_stats[sample_id]
        else:
            self.logger.info("Calculating alignment statistics sequentially")
            for sample_id, sample_info in self.sample_dict.items():
                stats = self._calculate_sample_statistics(sample_id, sample_info)
                if stats is not None:
                    self.sample_stats[sample_id] = stats
        self.logger.info(f"Completed alignment statistics for {len(self.sample_stats)} samples")
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
                    "Total_Reads": stats.total_reads,
                    "Aligned_Reads": stats.aligned_reads,
                    "Unique_Mapped": stats.unique_mapped,
                    "Multi_Mapped": stats.multi_mapped,
                    "Unmapped_Reads": stats.unmapped_reads,
                    "Alignment_Rate": stats.alignment_rate,
                    "Unique_Rate": stats.unique_rate,
                    "Multi_Rate": stats.multi_rate,
                    "Trimmed_Reads": stats.trimmed_reads,
                    "Stats_Source": self.resolved_source,
                }
            )

        self.summary_stats = pd.DataFrame(data)
        return self.summary_stats

    def save_results(self) -> bool:
        """
        Save the alignment statistics results to files.

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
            excel_file = self.out_dir / "alignment_statistics.xlsx"
            df.to_excel(excel_file, index=False, sheet_name="Alignment_Stats")
            self.logger.info(f"Saved Excel report: {excel_file}")

            # Save as CSV
            csv_file = self.out_dir / "alignment_statistics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved CSV file: {csv_file}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False

    def create_visualizations(self) -> bool:
        """
        Create publication-ready visualizations of alignment statistics.

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
            if df.empty:
                self.logger.warning("No statistics available for plotting")
                return False

            # Colorblind-safe Okabe-Ito inspired palette for publication-style QC figures.
            palette = {
                "raw": "#56B4E9",
                "trimmed": "#0072B2",
                "aligned": "#009E73",
                "multi": "#E69F00",
                "unmapped": "#999999",
                "accent": "#D55E00",
                "text": "#222222",
            }
            style_context = {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 120,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }

            sample_labels = df["Sample_ID"].astype(str).tolist()
            x = list(range(len(df)))
            unique_pct = df["Unique_Rate"].clip(lower=0, upper=100)
            multi_pct = df["Multi_Rate"].clip(lower=0, upper=100)
            unmapped_pct = (100 - df["Alignment_Rate"]).clip(lower=0, upper=100)
            count_cols = ["Total_Reads", "Trimmed_Reads", "Aligned_Reads"]
            counts_millions = df[count_cols].fillna(0) / 1_000_000

            with plt.rc_context(style_context):
                sns.set_theme(style="ticks", context="paper")
                fig_width = min(max(11.0, len(df) * 0.55), 18.0)
                fig = plt.figure(figsize=(fig_width, 9.0))
                grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.24)
                axes = [
                    fig.add_subplot(grid[0, 0]),
                    fig.add_subplot(grid[0, 1]),
                    fig.add_subplot(grid[1, 0]),
                    fig.add_subplot(grid[1, 1]),
                ]

                # A. Overall alignment rate with cohort median.
                ax = axes[0]
                bars = ax.bar(
                    x,
                    df["Alignment_Rate"],
                    color=palette["aligned"],
                    edgecolor="white",
                    linewidth=0.8,
                )
                median_rate = float(df["Alignment_Rate"].median())
                ax.axhline(
                    median_rate,
                    color=palette["accent"],
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Median {median_rate:.1f}%",
                )
                ax.set_title("Alignment rate by sample")
                ax.set_ylabel("Aligned reads (%)")
                ax.set_ylim(0, 100)
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=45, ha="right")
                ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
                ax.legend(frameon=False, loc="lower right")
                if len(df) <= 24:
                    for bar, value in zip(bars, df["Alignment_Rate"]):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            min(value + 2, 98),
                            f"{value:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color=palette["text"],
                        )

                # B. Mapping composition.
                ax = axes[1]
                ax.bar(
                    x,
                    unique_pct,
                    color=palette["aligned"],
                    edgecolor="white",
                    linewidth=0.5,
                    label="Unique",
                )
                ax.bar(
                    x,
                    multi_pct,
                    bottom=unique_pct,
                    color=palette["multi"],
                    edgecolor="white",
                    linewidth=0.5,
                    label="Multi",
                )
                ax.bar(
                    x,
                    unmapped_pct,
                    bottom=unique_pct + multi_pct,
                    color=palette["unmapped"],
                    edgecolor="white",
                    linewidth=0.5,
                    label="Unmapped",
                )
                ax.set_title("Mapping composition")
                ax.set_ylabel("Reads (%)")
                ax.set_ylim(0, 100)
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=45, ha="right")
                ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
                ax.legend(
                    frameon=False,
                    ncol=3,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.13),
                )

                # C. Read-count retention from raw to aligned.
                ax = axes[2]
                width = 0.25
                offsets = [-width, 0, width]
                count_series = [
                    ("Raw", "Total_Reads", palette["raw"]),
                    ("Trimmed", "Trimmed_Reads", palette["trimmed"]),
                    ("Aligned", "Aligned_Reads", palette["aligned"]),
                ]
                for offset, (label, col, color) in zip(offsets, count_series):
                    ax.bar(
                        [i + offset for i in x],
                        counts_millions[col],
                        width,
                        label=label,
                        color=color,
                        edgecolor="white",
                    )
                ax.set_title("Read counts retained")
                ax.set_ylabel("Read pairs (millions)" if self.paired else "Reads (millions)")
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=45, ha="right")
                ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
                ax.legend(frameon=False, ncol=3)

                # D. Compact rate heatmap.
                ax = axes[3]
                rate_cols = ["Alignment_Rate", "Unique_Rate", "Multi_Rate"]
                heatmap_data = df[["Sample_ID"] + rate_cols].set_index("Sample_ID").T
                heatmap_data.index = ["Aligned", "Unique", "Multi"]
                annot = len(df) <= 24
                sns.heatmap(
                    heatmap_data,
                    ax=ax,
                    cmap="cividis",
                    vmin=0,
                    vmax=100,
                    annot=annot,
                    fmt=".1f",
                    cbar_kws={"label": "Reads (%)"},
                    linewidths=0.35,
                    linecolor="white",
                )
                ax.set_title("Rate matrix")
                ax.set_xlabel("Sample ID")
                ax.set_ylabel("")

                for label, ax in zip(["A", "B", "C", "D"], axes):
                    ax.text(
                        -0.12,
                        1.08,
                        label,
                        transform=ax.transAxes,
                        fontsize=12,
                        fontweight="bold",
                        va="top",
                        ha="left",
                    )
                    sns.despine(ax=ax, left=False, bottom=False)

                fig.suptitle(
                    "Alignment statistics quality control",
                    fontsize=14,
                    fontweight="bold",
                    y=0.995,
                )
                fig.savefig(
                    plots_dir / "alignment_stats_overview.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                fig.savefig(plots_dir / "alignment_stats_overview.pdf", bbox_inches="tight")
                plt.close(fig)

                # Separate report-ready plots make the QC figures readable in HTML/PDF reports.
                rate_fig_width = min(max(8.0, len(df) * 0.38), 14.0)
                fig, ax = plt.subplots(figsize=(rate_fig_width, 5.2))
                bars = ax.bar(
                    x,
                    df["Alignment_Rate"],
                    color=palette["aligned"],
                    edgecolor="white",
                    linewidth=0.8,
                )
                median_rate = float(df["Alignment_Rate"].median())
                ax.axhline(
                    median_rate,
                    color=palette["accent"],
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Median {median_rate:.1f}%",
                )
                ax.set_title("Alignment rate by sample", fontsize=13, fontweight="bold", pad=14)
                ax.set_ylabel("Aligned reads (%)")
                ax.set_ylim(0, 105)
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=45, ha="right")
                ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
                ax.legend(frameon=False, loc="lower right")
                if len(df) <= 24:
                    for bar, value in zip(bars, df["Alignment_Rate"]):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            min(value + 1.8, 102),
                            f"{value:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            color=palette["text"],
                        )
                sns.despine(ax=ax, left=False, bottom=False)
                fig.tight_layout()
                fig.savefig(
                    plots_dir / "alignment_rate_by_sample.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                fig.savefig(plots_dir / "alignment_rate_by_sample.pdf", bbox_inches="tight")
                plt.close(fig)

                comp_height = min(max(5.0, len(df) * 0.30), 10.0)
                fig, ax = plt.subplots(figsize=(9.0, comp_height))
                y = list(range(len(df)))
                ax.barh(
                    y,
                    unique_pct,
                    color=palette["aligned"],
                    edgecolor="white",
                    linewidth=0.5,
                    label="Unique",
                )
                ax.barh(
                    y,
                    multi_pct,
                    left=unique_pct,
                    color=palette["multi"],
                    edgecolor="white",
                    linewidth=0.5,
                    label="Multi",
                )
                ax.barh(
                    y,
                    unmapped_pct,
                    left=unique_pct + multi_pct,
                    color=palette["unmapped"],
                    edgecolor="white",
                    linewidth=0.5,
                    label="Unmapped",
                )
                ax.set_title("Mapping composition", fontsize=13, fontweight="bold", pad=14)
                ax.set_xlabel("Reads (%)")
                ax.set_xlim(0, 100)
                ax.set_yticks(y)
                ax.set_yticklabels(sample_labels)
                ax.invert_yaxis()
                ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
                ax.legend(
                    frameon=False,
                    ncol=3,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                )
                sns.despine(ax=ax, left=False, bottom=False)
                fig.tight_layout()
                fig.savefig(plots_dir / "mapping_composition.png", dpi=300, bbox_inches="tight")
                fig.savefig(plots_dir / "mapping_composition.pdf", bbox_inches="tight")
                plt.close(fig)

                count_height = min(max(5.5, len(df) * 0.34), 11.0)
                fig, ax = plt.subplots(figsize=(9.5, count_height))
                bar_height = 0.24
                y_positions = [i for i in range(len(df))]
                count_series = [
                    ("Raw", "Total_Reads", palette["raw"]),
                    ("Trimmed", "Trimmed_Reads", palette["trimmed"]),
                    ("Aligned", "Aligned_Reads", palette["aligned"]),
                ]
                for offset, (label, col, color) in zip([-bar_height, 0, bar_height], count_series):
                    ax.barh(
                        [i + offset for i in y_positions],
                        counts_millions[col],
                        bar_height,
                        label=label,
                        color=color,
                        edgecolor="white",
                    )
                ax.set_title("Read-count retention", fontsize=13, fontweight="bold", pad=14)
                ax.set_xlabel("Read pairs (millions)" if self.paired else "Reads (millions)")
                ax.set_yticks(y_positions)
                ax.set_yticklabels(sample_labels)
                ax.invert_yaxis()
                ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
                ax.legend(
                    frameon=False,
                    ncol=3,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                )
                sns.despine(ax=ax, left=False, bottom=False)
                fig.tight_layout()
                fig.savefig(plots_dir / "read_count_retention.png", dpi=300, bbox_inches="tight")
                fig.savefig(plots_dir / "read_count_retention.pdf", bbox_inches="tight")
                plt.close(fig)

                matrix_fig_width = min(max(8.5, len(df) * 0.42), 16.0)
                matrix_fig_height = 3.4 if len(df) <= 24 else 3.8
                fig, ax = plt.subplots(figsize=(matrix_fig_width, matrix_fig_height))
                matrix_annot = len(df) <= 18
                sns.heatmap(
                    heatmap_data,
                    ax=ax,
                    cmap="cividis",
                    vmin=0,
                    vmax=100,
                    annot=matrix_annot,
                    fmt=".1f",
                    cbar_kws={"label": "Reads (%)"},
                    linewidths=0.35,
                    linecolor="white",
                )
                ax.set_title("Alignment rate matrix", fontsize=13, fontweight="bold", pad=14)
                ax.set_xlabel("Sample ID")
                ax.set_ylabel("")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(
                    plots_dir / "alignment_rate_matrix.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                fig.savefig(plots_dir / "alignment_rate_matrix.pdf", bbox_inches="tight")
                plt.close(fig)

            self.logger.info(f"Created visualizations in {plots_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return False

    def summarize_results(self, results) -> str:
        """
        Generate a text summary of the alignment statistics results.

        Args:
            results: Results from the run() method (boolean or DataFrame)

        Returns:
            str: Text summary of results
        """
        if not self.sample_stats:
            return "No alignment statistics results available."

        # Calculate averages
        total_samples = len(self.sample_stats)
        avg_alignment_rate = sum(stats.alignment_rate for stats in self.sample_stats.values()) / total_samples
        avg_unique_rate = sum(stats.unique_rate for stats in self.sample_stats.values()) / total_samples
        avg_multi_rate = sum(stats.multi_rate for stats in self.sample_stats.values()) / total_samples

        # Find best and worst samples
        best_sample = max(self.sample_stats.values(), key=lambda x: x.alignment_rate)
        worst_sample = min(self.sample_stats.values(), key=lambda x: x.alignment_rate)

        # Build summary text
        summary = [
            "Alignment Statistics Summary",
            "-------------------------",
            f"Number of samples: {total_samples}",
            f"Average alignment rate: {avg_alignment_rate:.2f}%",
            f"Average unique mapping rate: {avg_unique_rate:.2f}%",
            f"Average multi-mapping rate: {avg_multi_rate:.2f}%",
            "",
            f"Best aligned sample: {best_sample.sample_id} ({best_sample.alignment_rate:.2f}%)",
            f"Worst aligned sample: {worst_sample.sample_id} ({worst_sample.alignment_rate:.2f}%)",
        ]

        return "\n".join(summary)

    def run(self) -> bool:
        """
        Run the complete alignment statistics analysis.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting alignment statistics analysis")

            # Calculate statistics
            self.calculate_statistics()

            # Save results
            if not self.save_results():
                return False

            # Create visualizations
            if not self.create_visualizations():
                self.logger.warning("Failed to create visualizations, but continuing")

            self.logger.info("Alignment statistics analysis completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Alignment statistics analysis failed: {str(e)}")
            return False
