#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimapped Groups Module

This module provides functionality to count multimapped read groups in aligned BAM files.
It identifies groups of genes that share multimapped reads and provides count matrices
for downstream analysis.

Features:
    - Count and identify groups of genes sharing multimapped RNA-seq reads
    - Support for parallel processing of multiple BAM files
    - Multi-mapping annotation interval overlap logic via genomic binning
    - Option to collapse nested sub-groups into larger parent groups
    - Custom filtering thresholds for count counts and sample recurrence percentage
    - QC plotting functions producing heatmaps and barplots

Configuration:
    The analyzer parameters are configurable:
    - minimum count count threshold (min_count)
    - sample frequency detection fraction (percent_sample)
    - thread count (cpu_threads or threads)
    - overlap size rules (min_overlap, fraction_overlap)

Dependencies:
    - pysam
    - matplotlib
    - seaborn
    - pandas
    - numpy

Classes:
    MultimappedGroupsAnalyzer - Analyzer for multimapped gene groups from BAM files.

Exceptions:
    MultimappedGroupsError - Custom exception for multimapped groups analysis errors.

:Created: May 20, 2021
:Updated: April 15, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import logging
import pandas as pd
import numpy as np
import re
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ...utils.file_manager import FileManager
from ...utils.resource_manager import ResourceManager
from ...utils.command_executor import CommandExecutor

try:
    import pysam

    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
except ImportError:
    HAS_PLOTTING = False


class MultimappedGroupsError(Exception):
    """Custom exception for multimapped groups analysis errors."""

    pass


class MultimappedGroupsAnalyzer:
    """
    Analyzer for multimapped gene groups from BAM files.

    This class implements the multimapped groups counting functionality,
    identifying groups of genes that share multimapped reads and providing
    count matrices for downstream analysis.

    Attributes:
        logger: Logger instance for tracking progress
        file_manager: FileManager instance for file operations
        resource_manager: ResourceManager instance for resource management
        command_executor: CommandExecutor instance for executing commands
    """

    def __init__(
        self,
        bam_files: Dict[str, str],
        gff_file: str,
        out_dir: str = ".",
        feature: str = "gene",
        min_count: int = 100,
        percent_sample: float = 0.5,
        dryrun: bool = False,
        logger: Optional[Any] = None,
        dry_run_manager=None,
        **kwargs: Any,
    ):
        """
        Initialize multimapped groups analyzer.

        Args:
            bam_files: Dictionary mapping sample names to BAM file paths
            gff_file: Path to GFF/GTF annotation file
            out_dir: Output directory for results
            feature: Feature type to extract from GFF (default: 'gene')
            min_count: Minimum number of reads per sample for filtering
            percent_sample: Minimum percentage of samples that must meet min_count
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

        # Store parameters
        self.bam_files = bam_files
        self.gff_file = gff_file
        self.out_dir = Path(out_dir)
        self.feature = feature
        self.min_count = min_count
        self.percent_sample = percent_sample
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager
        self.cpu_threads = int(kwargs.get("cpu_threads", kwargs.get("threads", 1)) or 1)
        self.max_workers = min(len(bam_files), max(1, self.cpu_threads), 8)
        self.bin_size = int(kwargs.get("bin_size", 100000) or 100000)
        self.min_overlap = max(1, int(kwargs.get("min_overlap", 1) or 1))
        self.fraction_overlap = max(0.0, min(1.0, float(kwargs.get("fraction_overlap", 0.0) or 0.0)))
        self.include_ambiguous_unique = bool(kwargs.get("include_ambiguous_unique", True))
        self.collapse_contained_groups = bool(kwargs.get("collapse_contained_groups", True))

        # Store additional kwargs
        self.kwargs = kwargs

        # Initialize managers
        self.file_manager = FileManager(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)
        self.command_executor = CommandExecutor(logger=self.logger)

        # Set up output directory
        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.results = {}
        self.gene_bins: Dict[str, Dict[int, List[Tuple[int, int, str, str]]]] = {}
        self.sample_summaries: Dict[str, Dict[str, int]] = {}

        bam_count = str(len(bam_files)).replace("\n", "").replace("\r", "")
        self.logger.info(f"Multimapped groups analyzer initialized with {bam_count} BAM files")
        self.logger.info(f"Using up to {self.max_workers} worker(s) for multimapped groups analysis")
        self.logger.info(
            "MMG overlap rules: min_overlap=%d, fraction_overlap=%.2f, include_ambiguous_unique=%s, collapse_contained_groups=%s",
            self.min_overlap,
            self.fraction_overlap,
            self.include_ambiguous_unique,
            self.collapse_contained_groups,
        )
        self.logger.info(
            "STAR alignments are recommended for multimapped groups because STAR reliably reports NH tags for multi-mapping RNA-seq reads"
        )

    def _create_mmg_log(self, log_file: Path) -> None:
        """Create a detailed log file for multimapped groups analysis."""
        with open(log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MULTIMAPPED GROUPS ANALYSIS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GFF File: {self.gff_file}\n")
            f.write(f"Output Directory: {self.out_dir}\n")
            f.write(f"Feature Type: {self.feature}\n")
            f.write(f"Min Count: {self.min_count}\n")
            f.write(f"Percent Sample: {self.percent_sample}\n")
            f.write(f"Minimum Overlap: {self.min_overlap}\n")
            f.write(f"Fraction Overlap: {self.fraction_overlap}\n")
            f.write(f"Include Ambiguous Unique Reads: {self.include_ambiguous_unique}\n")
            f.write(f"Collapse Contained Groups: {self.collapse_contained_groups}\n")
            f.write("Aligner Recommendation: STAR is recommended for MMG analysis because it reports NH tags consistently.\n")
            f.write(f"BAM Files: {len(self.bam_files)}\n")
            for sample, bam_file in self.bam_files.items():
                f.write(f"  {sample}: {bam_file}\n")
            f.write("-" * 80 + "\n")
            f.write("PROCESSING LOG:\n")

    def _update_mmg_log(self, log_file: Path, result_data: dict) -> None:
        """Update the multimapped groups log file with completion status."""
        with open(log_file, "a") as f:
            f.write("-" * 80 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Status: SUCCESS\n")
            f.write(f"Total Multimapped Groups: {result_data.get('total_groups', 0)}\n")
            f.write(f"Total Samples: {result_data.get('total_samples', 0)}\n")
            f.write(f"Output Files Created: {result_data.get('output_files', 0)}\n")
            f.write("=" * 80 + "\n")

    def _extract_attribute(self, attributes: str, keys: List[str]) -> Optional[str]:
        """Extract the first matching GFF/GTF attribute value."""
        if not isinstance(attributes, str):
            return None

        for key in keys:
            gtf_match = re.search(rf'{re.escape(key)}\s+"([^"]+)"', attributes)
            if gtf_match:
                return gtf_match.group(1)

            gff_match = re.search(rf"(?:^|;){re.escape(key)}=([^;]+)", attributes)
            if gff_match:
                return gff_match.group(1)

        return None

    def _clean_gene_id(self, gene_id: str) -> str:
        """Normalize common GFF/GTF gene identifier prefixes."""
        gene_id = str(gene_id).strip()
        for prefix in ("gene:", "gene-", "GeneID:", "transcript:"):
            if gene_id.startswith(prefix):
                gene_id = gene_id[len(prefix) :]
        return gene_id

    def _load_gene_intervals(
        self,
    ) -> Dict[str, Dict[int, List[Tuple[int, int, str, str]]]]:
        """Load annotation intervals into genomic bins for fast overlap lookup."""
        if self.gene_bins:
            return self.gene_bins

        if not Path(self.gff_file).exists():
            raise MultimappedGroupsError(f"Annotation file not found: {self.gff_file}")

        self.logger.info(f"Loading {self.feature} intervals from annotation: {self.gff_file}")
        gene_bins: Dict[str, Dict[int, List[Tuple[int, int, str, str]]]] = defaultdict(lambda: defaultdict(list))
        interval_count = 0

        try:
            with open(self.gff_file, "r") as handle:
                for line in handle:
                    if not line.strip() or line.startswith("#"):
                        continue

                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 9 or fields[2] != self.feature:
                        continue

                    chrom = fields[0]
                    start = max(0, int(fields[3]) - 1)
                    end = int(fields[4])
                    strand = fields[6]
                    attributes = fields[8]
                    gene_id = self._extract_attribute(attributes, ["gene_id", "ID", "Name", "gene_name", "Parent"])

                    if not gene_id or end <= start:
                        continue

                    gene_id = self._clean_gene_id(gene_id)
                    interval = (start, end, gene_id, strand)
                    start_bin = start // self.bin_size
                    end_bin = max(start_bin, (end - 1) // self.bin_size)

                    for bin_id in range(start_bin, end_bin + 1):
                        gene_bins[chrom][bin_id].append(interval)

                    interval_count += 1

            self.gene_bins = {chrom: dict(bins) for chrom, bins in gene_bins.items()}
            self.logger.info(
                f"Loaded {interval_count} {self.feature} intervals across {len(self.gene_bins)} reference sequences"
            )
            return self.gene_bins

        except Exception as e:
            raise MultimappedGroupsError(f"Failed to load annotation intervals: {str(e)}")

    def _read_reference_blocks(self, read) -> List[Tuple[int, int]]:
        """Return aligned reference blocks for a read."""
        try:
            blocks = read.get_blocks()
        except Exception:
            blocks = []

        if blocks:
            return blocks

        if read.reference_start is None or read.reference_end is None:
            return []

        return [(read.reference_start, read.reference_end)]

    def _read_aligned_length(self, read) -> int:
        """Return the number of reference bases covered by aligned blocks."""
        return sum(max(0, end - start) for start, end in self._read_reference_blocks(read))

    def _genes_overlapping_read(self, read) -> Dict[str, int]:
        """Return genes overlapped by a read alignment with overlap lengths."""
        if read.reference_name not in self.gene_bins:
            return {}

        read_blocks = self._read_reference_blocks(read)
        if not read_blocks:
            return {}

        read_aligned_length = self._read_aligned_length(read)
        if read_aligned_length <= 0:
            return {}

        genes: Dict[str, int] = defaultdict(int)
        seen_intervals = set()
        chrom_bins = self.gene_bins[read.reference_name]
        read_start = min(start for start, _end in read_blocks)
        read_end = max(end for _start, end in read_blocks)
        start_bin = read_start // self.bin_size
        end_bin = max(start_bin, (read_end - 1) // self.bin_size)

        for bin_id in range(start_bin, end_bin + 1):
            for interval in chrom_bins.get(bin_id, []):
                if interval in seen_intervals:
                    continue
                seen_intervals.add(interval)

                gene_start, gene_end, gene_id, _strand = interval
                overlap = 0
                for block_start, block_end in read_blocks:
                    overlap += max(0, min(block_end, gene_end) - max(block_start, gene_start))

                if overlap >= self.min_overlap and overlap >= self.fraction_overlap * read_aligned_length:
                    genes[gene_id] += overlap

        return dict(genes)

    def _count_sample_groups(self, sample_name: str, bam_file: str) -> Tuple[str, Counter, Dict[str, int]]:
        """Count multimapped gene groups for one sample directly from a BAM file."""
        if not HAS_PYSAM:
            raise MultimappedGroupsError("pysam is required for multimapped groups analysis")

        if not Path(bam_file).exists():
            raise MultimappedGroupsError(f"BAM file not found for {sample_name}: {bam_file}")

        self.logger.info(f"Counting multimapped groups for sample {sample_name}")
        read_to_genes: Dict[str, set] = defaultdict(set)
        total_alignments = 0
        primary_alignments = 0
        multimapped_alignments = 0
        ambiguous_unique_alignments = 0

        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                total_alignments += 1
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue

                primary_alignments += 1
                try:
                    nh = int(read.get_tag("NH"))
                except KeyError:
                    nh = 1

                gene_overlaps = self._genes_overlapping_read(read)
                if not gene_overlaps:
                    continue

                is_multimapped = nh > 1
                is_ambiguous_unique = self.include_ambiguous_unique and len(gene_overlaps) > 1
                if not is_multimapped and not is_ambiguous_unique:
                    continue

                if is_multimapped:
                    multimapped_alignments += 1
                elif is_ambiguous_unique:
                    ambiguous_unique_alignments += 1

                read_to_genes[read.query_name].update(gene_overlaps)

        group_counts = Counter()
        for genes in read_to_genes.values():
            if len(genes) > 1:
                group_counts["-".join(sorted(genes))] += 1

        sample_summary = {
            "total_alignments": total_alignments,
            "primary_alignments": primary_alignments,
            "multimapped_alignments": multimapped_alignments,
            "ambiguous_unique_alignments": ambiguous_unique_alignments,
            "reads_touching_features": len(read_to_genes),
            "multimapped_groups": len(group_counts),
            "grouped_reads": int(sum(group_counts.values())),
        }

        self.logger.info(
            "Sample %s: %d MMG read(s) across %d group(s)",
            sample_name,
            sample_summary["grouped_reads"],
            sample_summary["multimapped_groups"],
        )

        return sample_name, group_counts, sample_summary

    def _filter_genes(self, mmg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter multimapped gene groups by user thresholds and assign stable IDs.

        Args:
            mmg_df: DataFrame with multimapped gene groups

        Returns:
            Filtered DataFrame
        """
        self.logger.info("Filtering multimapped gene groups")

        if mmg_df.empty:
            return mmg_df

        sample_cols = [col for col in mmg_df.columns if col not in ("MMG", "Gene")]
        filtered_df = mmg_df[mmg_df["Gene"].astype(str).str.contains("-", regex=False)].copy()

        if sample_cols:
            required_samples = len(sample_cols) * self.percent_sample
            above_threshold = (filtered_df[sample_cols] >= self.min_count).sum(axis=1)
            filtered_df = filtered_df[above_threshold >= required_samples].copy()
            filtered_df["Total_Counts"] = filtered_df[sample_cols].sum(axis=1).astype(int)
            filtered_df["Samples_Detected"] = (filtered_df[sample_cols] > 0).sum(axis=1).astype(int)
            filtered_df = filtered_df.sort_values(
                ["Total_Counts", "Samples_Detected", "Gene"],
                ascending=[False, False, True],
            )

        filtered_df.insert(0, "MMG", [f"MMG_{i + 1}" for i in range(len(filtered_df))])
        filtered_df = filtered_df.drop(columns=["Total_Counts", "Samples_Detected"], errors="ignore")

        self.logger.info(f"Filtered from {len(mmg_df)} to {len(filtered_df)} multimapped groups")
        return filtered_df

    def _collapse_groups(self, group_counts_by_sample: Dict[str, Counter]) -> Dict[str, Counter]:
        """Collapse groups wholly contained within larger observed groups."""
        if not self.collapse_contained_groups:
            return group_counts_by_sample

        all_groups = {group for counts in group_counts_by_sample.values() for group in counts}
        group_sets = {group: frozenset(group.split("-")) for group in all_groups}
        sorted_groups = sorted(all_groups, key=lambda group: (-len(group_sets[group]), group))
        parent_map = {group: group for group in all_groups}

        for group in sorted_groups:
            group_set = group_sets[group]
            containing = [
                candidate
                for candidate in sorted_groups
                if candidate != group
                and len(group_sets[candidate]) > len(group_set)
                and group_set.issubset(group_sets[candidate])
            ]
            if containing:
                parent_map[group] = min(
                    containing,
                    key=lambda candidate: (len(group_sets[candidate]), candidate),
                )

        if all(parent_map[group] == group for group in all_groups):
            return group_counts_by_sample

        collapsed_counts = {}
        for sample, counts in group_counts_by_sample.items():
            merged = Counter()
            for group, count in counts.items():
                merged[parent_map[group]] += count
            collapsed_counts[sample] = merged

        collapsed_groups = len({parent_map[group] for group in all_groups})
        self.logger.info(f"Collapsed contained multimapped groups from {len(all_groups)} to {collapsed_groups}")
        return collapsed_counts

    def analyze_multimapped_groups(self) -> pd.DataFrame:
        """
        Perform complete multimapped groups analysis.

        Returns:
            DataFrame with multimapped gene group counts

        Raises:
            MultimappedGroupsError: If analysis fails
        """
        try:
            self.logger.info("Starting multimapped groups analysis")

            # Create log file
            if not self.dryrun:
                log_file = self.out_dir / "multimapped_groups_analysis.log"
                self._create_mmg_log(log_file)

            if self.dryrun:
                self.logger.info("DRYRUN: Would count multimapped groups directly from BAM files")
                return pd.DataFrame()

            self._load_gene_intervals()
            if not self.gene_bins:
                self.logger.warning("No annotation intervals were loaded for multimapped groups")
                return pd.DataFrame()

            sample_group_counts: Dict[str, Counter] = {}
            self.sample_summaries = {}

            if self.max_workers > 1:
                self.logger.info(f"Counting multimapped groups in parallel with {self.max_workers} worker(s)")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_sample = {
                        executor.submit(self._count_sample_groups, sample, bam_file): sample
                        for sample, bam_file in self.bam_files.items()
                    }
                    for future in as_completed(future_to_sample):
                        sample = future_to_sample[future]
                        try:
                            sample_name, group_counts, sample_summary = future.result()
                            sample_group_counts[sample_name] = group_counts
                            self.sample_summaries[sample_name] = sample_summary
                        except Exception as e:
                            self.logger.error(f"Failed to count multimapped groups for {sample}: {e}")
            else:
                for sample, bam_file in self.bam_files.items():
                    sample_name, group_counts, sample_summary = self._count_sample_groups(sample, bam_file)
                    sample_group_counts[sample_name] = group_counts
                    self.sample_summaries[sample_name] = sample_summary

            sample_group_counts = self._collapse_groups(sample_group_counts)
            all_groups = sorted({group for counts in sample_group_counts.values() for group in counts})
            if not all_groups:
                self.logger.warning("No multimapped groups found in any sample")
                return pd.DataFrame()

            rows = []
            sample_names = list(self.bam_files.keys())
            for group in all_groups:
                rows.append(
                    [group] + [int(sample_group_counts.get(sample, Counter()).get(group, 0)) for sample in sample_names]
                )

            df_result = pd.DataFrame(rows, columns=["Gene"] + sample_names)
            df_filtered = self._filter_genes(df_result)

            if df_filtered.empty:
                self.logger.warning("No multimapped groups met the filtering criteria")
                return pd.DataFrame()

            self.logger.info(f"Final result: {len(df_filtered)} multimapped gene groups")

            if not self.dryrun:
                result_data = {
                    "total_groups": len(df_filtered),
                    "total_samples": len(self.bam_files),
                    "output_files": 1,
                }
                self._update_mmg_log(log_file, result_data)

            return df_filtered

        except Exception as e:
            self.logger.error(f"Multimapped groups analysis failed: {str(e)}")
            raise MultimappedGroupsError(f"Multimapped groups analysis failed: {str(e)}")

    def _create_visualizations(self, results_df: pd.DataFrame) -> List[str]:
        """Create publication-style summary plots for multimapped groups."""
        if not HAS_PLOTTING:
            self.logger.warning("Matplotlib/seaborn not available, skipping MMG visualizations")
            return []

        if results_df.empty:
            return []

        plots_dir = self.out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        sample_cols = [col for col in results_df.columns if col not in ("MMG", "Gene", "Total_Counts", "Samples_Detected")]
        if not sample_cols:
            return []

        plot_df = results_df.copy()
        if "Total_Counts" not in plot_df.columns:
            plot_df["Total_Counts"] = plot_df[sample_cols].sum(axis=1)
        if "Samples_Detected" not in plot_df.columns:
            plot_df["Samples_Detected"] = (plot_df[sample_cols] > 0).sum(axis=1)

        top_df = plot_df.nlargest(min(30, len(plot_df)), "Total_Counts").copy()
        heatmap_data = np.log1p(top_df[sample_cols].astype(float))
        heatmap_data.index = top_df["MMG"]
        sample_totals = plot_df[sample_cols].sum().sort_values(ascending=False)

        palette = {
            "counts": "#0072B2",
            "samples": "#009E73",
            "sample_total": "#E69F00",
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

        output_files = []
        with plt.rc_context(style_context):
            sns.set_theme(style="ticks", context="paper")
            fig_width = min(max(11.0, len(sample_cols) * 0.7), 18.0)
            fig_height = 9.0
            fig = plt.figure(figsize=(fig_width, fig_height))
            grid = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
            axes = [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[1, 1]),
            ]

            ax = axes[0]
            bar_df = top_df.sort_values("Total_Counts", ascending=True)
            ax.barh(bar_df["MMG"], bar_df["Total_Counts"], color=palette["counts"])
            ax.set_title("Top multimapped groups")
            ax.set_xlabel("Total grouped reads")
            ax.set_ylabel("")
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)

            ax = axes[1]
            sns.heatmap(
                heatmap_data,
                ax=ax,
                cmap="cividis",
                linewidths=0.25,
                linecolor="white",
                cbar_kws={"label": "log1p(count)"},
            )
            ax.set_title("Top group counts by sample")
            ax.set_xlabel("Sample")
            ax.set_ylabel("MMG")

            ax = axes[2]
            detected_counts = plot_df["Samples_Detected"].value_counts().sort_index()
            detected_x = list(range(len(detected_counts)))
            ax.bar(detected_x, detected_counts.values, color=palette["samples"])
            ax.set_title("Sample recurrence")
            ax.set_xlabel("Samples with count > 0")
            ax.set_ylabel("Number of MMGs")
            ax.set_xticks(detected_x)
            ax.set_xticklabels(detected_counts.index.astype(str))
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)

            ax = axes[3]
            sample_x = list(range(len(sample_totals)))
            ax.bar(sample_x, sample_totals.values, color=palette["sample_total"])
            ax.set_title("MMG burden by sample")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Grouped reads")
            ax.set_xticks(sample_x)
            ax.set_xticklabels(sample_totals.index.astype(str), rotation=45, ha="right")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)

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
                "Multimapped group quality control",
                fontsize=14,
                fontweight="bold",
                y=0.995,
            )
            png_file = plots_dir / "multimapped_groups_overview.png"
            pdf_file = plots_dir / "multimapped_groups_overview.pdf"
            fig.savefig(png_file, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_file, bbox_inches="tight")
            plt.close(fig)
            output_files.extend([str(png_file), str(pdf_file)])

        self.logger.info(f"Created multimapped groups visualizations in {plots_dir}")
        return output_files

    def run(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete multimapped groups analysis process.

        Args:
            save_results: Whether to save results to files

        Returns:
            Dictionary containing analysis results

        Raises:
            MultimappedGroupsError: If analysis fails
        """
        try:
            self.logger.info("Starting multimapped groups analysis")

            # Perform analysis
            results_df = self.analyze_multimapped_groups()

            # Save results if requested
            output_files = []
            if save_results and not self.dryrun and not results_df.empty:
                # Save results to Excel
                output_file = self.out_dir / "multimapped_groups_results.xlsx"
                results_df.to_excel(output_file, index=False)
                output_files.append(str(output_file))
                self.logger.info(f"Results saved to: {output_file}")

                csv_file = self.out_dir / "multimapped_groups_results.csv"
                results_df.to_csv(csv_file, index=False)
                output_files.append(str(csv_file))
                self.logger.info(f"Results saved to: {csv_file}")

                if self.sample_summaries:
                    sample_summary_file = self.out_dir / "multimapped_groups_sample_summary.tsv"
                    sample_summary_df = pd.DataFrame.from_dict(self.sample_summaries, orient="index")
                    sample_summary_df.index.name = "Sample_ID"
                    sample_summary_df.reset_index().to_csv(sample_summary_file, sep="\t", index=False)
                    output_files.append(str(sample_summary_file))
                    self.logger.info(f"Sample summary saved to: {sample_summary_file}")

                    sample_summary_json = self.out_dir / "multimapped_groups_sample_summary.json"
                    with open(sample_summary_json, "w") as handle:
                        json.dump(self.sample_summaries, handle, indent=2)
                    output_files.append(str(sample_summary_json))

                output_files.extend(self._create_visualizations(results_df))

            # Generate summary statistics
            summary_stats = {
                "total_groups": len(results_df) if not results_df.empty else 0,
                "total_samples": len(self.bam_files),
                "min_count": self.min_count,
                "percent_sample": self.percent_sample,
                "output_files": len(output_files),
                "total_grouped_reads": (
                    int(results_df[[col for col in results_df.columns if col in self.bam_files]].sum().sum())
                    if not results_df.empty
                    else 0
                ),
            }

            self.logger.info("Multimapped groups analysis completed successfully")

            return {
                "results": results_df,
                "summary": summary_stats,
                "output_files": output_files,
            }

        except Exception as e:
            self.logger.error(f"Multimapped groups analysis failed: {str(e)}")
            raise MultimappedGroupsError(f"Multimapped groups analysis failed: {str(e)}")

    def get_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for multimapped groups results.

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

                return summary
            else:
                # Fallback if no summary exists
                return {
                    "total_groups": 0,
                    "total_samples": len(self.bam_files),
                    "min_count": self.min_count,
                    "percent_sample": self.percent_sample,
                    "output_files": 0,
                }

        except Exception as e:
            self.logger.warning(f"Failed to generate summary stats: {str(e)}")
            return {
                "total_groups": 0,
                "total_samples": len(self.bam_files),
                "min_count": self.min_count,
                "percent_sample": self.percent_sample,
                "output_files": 0,
            }
