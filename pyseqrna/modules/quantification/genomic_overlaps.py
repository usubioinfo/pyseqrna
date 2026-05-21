"""
Novel Genomic Overlaps Quantification Module

This module implements advanced, custom overlap-counting algorithms for RNA-seq data.
It allows counting reads overlapping genomic features using different strategies such as
union, strict intersection, or non-empty intersection, with configurable filtering options
for mapping quality, strand specificity, and multi-mapping to increase accuracy.

Features:
    - Custom Python implementation of overlap counting using pysam
    - Multiple overlap modes: Union, IntersectionStrict, and IntersectionNotEmpty
    - Filtering of reads by mapping quality, strand orientation, and multi-mapping flags
    - Concurrent sample processing using a ThreadPoolExecutor or SLURM array execution
    - Support for GTF/GFF parsing and precomputing merged exonic regions per gene
    - Structured logging, worker coordination, and count matrix aggregation

Configuration:
    Configured via constructor parameters (such as overlap_mode, min_mapping_quality,
    ignore_strand, count_multi_mapping) or by loading a tool-specific genomic_overlaps.ini
    configuration file via ConfigManager.

Dependencies:
    - pysam
    - pandas
    - pyseqrna.modules.quantification.base (BaseQuantifier, QuantificationError)

Classes / Functions / Exceptions:
    - OverlapMode: Overlap counting modes for genomic feature quantification.
    - GenomicInterval: Represents a genomic interval with chromosome, start, end, and strand.
    - Gene: Represents a gene with multiple transcripts.
    - Transcript: Represents a transcript with exons.
    - ReadAlignment: Represents a read alignment with potential gaps (spliced reads).
    - GenomicOverlapsQuantifier: Novel genomic overlaps quantifier implementing advanced counting algorithms.
    - _run_worker: Run one genomic-overlaps sample worker for SLURM array execution.
    - main: Command-line worker entry point for SLURM array tasks.

:Created: May 20, 2021
:Updated: February 25, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import argparse
import json
import shlex
import shutil
import sys
import pysam
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseQuantifier, QuantificationError


def _run_worker(config_file: str, sample_id: str) -> None:
    """Run one genomic-overlaps sample worker for SLURM array execution."""
    with open(config_file, "r") as handle:
        config = json.load(handle)

    bam_file = config["samples"][sample_id]
    out_dir = Path(config["out_dir"])
    worker_output_dir = Path(config.get("worker_output_dir", out_dir / "genomic_overlaps_intermediates"))
    quantifier = GenomicOverlapsQuantifier(
        bam_dict={sample_id: bam_file},
        annotation_file=config["annotation_file"],
        out_dir=str(out_dir),
        param_dir=config.get("param_dir"),
        overlap_mode=OverlapMode(config.get("overlap_mode", OverlapMode.UNION.value)),
        min_mapping_quality=int(config.get("min_mapping_quality", 10)),
        ignore_strand=bool(config.get("ignore_strand", False)),
        count_multi_mapping=bool(config.get("count_multi_mapping", False)),
        filter_strategy=config.get("filter_strategy", "none"),
        slurm=False,
        dryrun=False,
        cpu_threads=int(config.get("cpu_threads", 1)),
        memory=config.get("memory"),
    )
    quantifier.worker_output_dir = worker_output_dir
    quantifier.bam_index_threads = int(config.get("bam_index_threads", 1))
    gene_counts = quantifier._process_sample(sample_id, bam_file)
    quantifier._write_sample_counts(sample_id, gene_counts)


class OverlapMode(Enum):
    """
    Overlap counting modes for genomic feature quantification.

    - UNION: A read is counted for a feature if it overlaps any part of it
    - INTERSECTION_STRICT: A read is counted only if it completely falls within exonic regions
    - INTERSECTION_NOT_EMPTY: A read is counted if it overlaps exonic regions (less strict than STRICT)
    """

    UNION = "Union"
    INTERSECTION_STRICT = "IntersectionStrict"
    INTERSECTION_NOT_EMPTY = "IntersectionNotEmpty"


@dataclass
class GenomicInterval:
    """Represents a genomic interval with chromosome, start, end, and strand."""

    chromosome: str
    start: int
    end: int
    strand: str = "."

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"Start ({self.start}) cannot be greater than end ({self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: "GenomicInterval") -> bool:
        """Check if this interval overlaps with another interval."""
        if self.chromosome != other.chromosome:
            return False
        return not (self.end < other.start or self.start > other.end)

    def intersection(self, other: "GenomicInterval") -> Optional["GenomicInterval"]:
        """Return the intersection of two intervals, or None if no overlap."""
        if not self.overlaps(other):
            return None

        return GenomicInterval(
            chromosome=self.chromosome,
            start=max(self.start, other.start),
            end=min(self.end, other.end),
            strand=self.strand,
        )

    def contains(self, other: "GenomicInterval") -> bool:
        """Check if this interval completely contains another interval."""
        if self.chromosome != other.chromosome:
            return False
        return self.start <= other.start and self.end >= other.end


@dataclass
class Gene:
    """Represents a gene with multiple transcripts."""

    gene_id: str
    gene_name: str = ""
    chromosome: str = ""
    strand: str = "."
    transcripts: Dict[str, "Transcript"] = field(default_factory=dict)

    def add_transcript(self, transcript: "Transcript"):
        """Add a transcript to this gene."""
        self.transcripts[transcript.transcript_id] = transcript
        transcript.gene_id = self.gene_id

    @property
    def exonic_regions(self) -> List[GenomicInterval]:
        """Get all exonic regions across all transcripts (union)."""
        all_exons = []
        for transcript in self.transcripts.values():
            all_exons.extend(transcript.exons)

        # Merge overlapping exons
        if not all_exons:
            return []

        # Sort by start position
        all_exons.sort(key=lambda x: x.start)
        merged = [all_exons[0]]

        for current in all_exons[1:]:
            last = merged[-1]
            if current.start <= last.end + 1:  # Adjacent or overlapping
                merged[-1] = GenomicInterval(
                    chromosome=last.chromosome,
                    start=last.start,
                    end=max(last.end, current.end),
                    strand=last.strand,
                )
            else:
                merged.append(current)

        return merged


@dataclass
class Transcript:
    """Represents a transcript with exons."""

    transcript_id: str
    gene_id: str = ""
    chromosome: str = ""
    strand: str = "."
    exons: List[GenomicInterval] = field(default_factory=list)

    def add_exon(self, exon: GenomicInterval):
        """Add an exon to this transcript."""
        self.exons.append(exon)
        # Sort exons by start position
        self.exons.sort(key=lambda x: x.start)

    @property
    def exonic_regions(self) -> List[GenomicInterval]:
        """Get exonic regions for this transcript."""
        return self.exons.copy()


@dataclass
class ReadAlignment:
    """Represents a read alignment with potential gaps (spliced reads)."""

    read_id: str
    chromosome: str
    strand: str
    segments: List[GenomicInterval] = field(default_factory=list)
    is_paired: bool = False
    mapping_quality: int = 0

    def add_segment(self, segment: GenomicInterval):
        """Add an alignment segment."""
        self.segments.append(segment)
        self.segments.sort(key=lambda x: x.start)

    @property
    def total_span(self) -> Optional[GenomicInterval]:
        """Get the total genomic span of this alignment."""
        if not self.segments:
            return None

        return GenomicInterval(
            chromosome=self.chromosome,
            start=min(seg.start for seg in self.segments),
            end=max(seg.end for seg in self.segments),
            strand=self.strand,
        )


class GenomicOverlapsQuantifier(BaseQuantifier):
    """
    Novel genomic overlaps quantifier implementing advanced counting algorithms.

    This quantifier implements three overlap counting modes:
    1. Union: Count reads overlapping any part of gene regions
    2. IntersectionStrict: Count reads completely within exonic regions
    3. IntersectionNotEmpty: Count reads with non-empty intersection with exons
    """

    def __init__(
        self,
        bam_dict: Dict[str, List[str]],
        annotation_file: str,
        out_dir: str,
        overlap_mode: OverlapMode = OverlapMode.UNION,
        min_mapping_quality: int = 10,
        ignore_strand: bool = False,
        count_multi_mapping: bool = False,
        filter_strategy: str = None,
        config_file: str = None,
        **kwargs,
    ):
        # First call base class init so self.config_manager is available
        super().__init__(bam_dict, annotation_file, out_dir, **kwargs)
        self.overlap_mode = overlap_mode
        self.min_mapping_quality = min_mapping_quality
        self.ignore_strand = ignore_strand
        self.count_multi_mapping = count_multi_mapping

        # Read config file
        config = (
            self.config_manager.read_tool_config("genomic_overlaps.ini", config_file)
            if config_file
            else self.load_config("genomic_overlaps.ini")
        )
        section = config.get("genomic_overlaps", {}) if config else {}

        # Set filter_strategy with explicit parameter taking precedence over config file
        if filter_strategy is not None:
            self.filter_strategy = filter_strategy.lower()
            self.logger.info(f"Using explicitly provided filter strategy: {self.filter_strategy}")
        else:
            self.filter_strategy = section.get("filter_strategy", "none").lower()
            self.logger.info(f"Using filter strategy from config: {self.filter_strategy}")

        self.parallel_samples = self._resolve_parallel_samples(section.get("parallel_samples", "auto"))
        self.bam_index_threads = self._resolve_positive_int(section.get("bam_index_threads", 1), default=1)
        self.worker_output_dir = self.out_dir / "genomic_overlaps_intermediates"

        # Initialize data structures - will be populated by _parse_annotation_file
        self.genes: Dict[str, Dict] = {}
        self.transcripts: Dict[str, Dict] = {}
        self.gene_to_transcripts: Dict[str, List[str]] = defaultdict(list)
        self.transcript_to_exons: Dict[str, List[Tuple]] = defaultdict(list)
        self.gene_exons: Dict[str, List[Tuple]] = defaultdict(list)

        # Load annotation file
        self._parse_annotation_file()
        self._build_gene_exon_intervals()

        self.logger.info(f"Initialized GenomicOverlaps quantifier with {overlap_mode.value} mode")
        # Sanitize log inputs to prevent log injection
        safe_filter_strategy = str(self.filter_strategy).replace("\n", " ").replace("\r", " ")
        safe_min_mapping_quality = str(self.min_mapping_quality).replace("\n", " ").replace("\r", " ")
        safe_ignore_strand = str(self.ignore_strand).replace("\n", " ").replace("\r", " ")
        safe_count_multi_mapping = str(self.count_multi_mapping).replace("\n", " ").replace("\r", " ")
        self.logger.info(
            f"Filter strategy: {safe_filter_strategy}, Min mapping quality: {safe_min_mapping_quality}, Ignore strand: {safe_ignore_strand}, Count multi-mapping: {safe_count_multi_mapping}"
        )
        self.logger.info(f"Loaded {len(self.genes)} genes and {len(self.transcripts)} transcripts")

    def _resolve_parallel_samples(self, value: Any) -> Optional[int]:
        """Resolve the number of samples to process concurrently."""
        if value is None:
            return None

        value_str = str(value).strip().lower()
        if value_str in {"", "auto", "none"}:
            return None

        try:
            workers = int(value_str)
        except ValueError:
            self.logger.warning(f"Invalid parallel_samples value '{value}', using auto")
            return None

        return max(1, workers)

    def _resolve_positive_int(self, value: Any, default: int = 1) -> int:
        """Resolve a positive integer configuration value."""
        try:
            resolved = int(str(value).strip())
        except (TypeError, ValueError):
            return default
        return max(1, resolved)

    def check_tool_availability(self) -> bool:
        """Check if required tools are available."""
        try:
            import pysam

            return True
        except ImportError:
            self.logger.error("pysam is required for GenomicOverlaps quantifier")
            return False

    def _record_internal_operation(self, operation_type: str, details: str, sample_id: str = None) -> None:
        """
        Record internal operations for execution reporting.

        Since GenomicOverlaps doesn't execute external commands, we record
        its internal operations for the execution report.

        Args:
            operation_type: Type of operation (e.g., 'genome_parsing', 'overlap_counting')
            details: Details about the operation
            sample_id: Sample ID if operation is sample-specific
        """
        if hasattr(self, "dry_run_manager") and self.dry_run_manager:
            operation_record = {
                "operation": "genomic_overlaps_internal",
                "operation_type": operation_type,
                "details": details,
                "stage": "genomic_overlaps_quantification",
                "timestamp": self.dry_run_manager._get_timestamp(),
                "filter_strategy": self.filter_strategy,
                "overlap_mode": self.overlap_mode.value,
            }

            if sample_id:
                operation_record["sample"] = sample_id

            # Add to executed operations list
            self.dry_run_manager.executed_operations.append(operation_record)

    def _parse_annotation_file(self):
        """Parse GTF/GFF annotation file to extract gene and transcript structures."""
        self.logger.info(f"Parsing annotation file: {self.annotation_file}")

        # Record this operation for the execution report
        self._record_internal_operation("annotation_parsing", f"Parsing annotation file: {self.annotation_file}")

        annotation_path = Path(self.annotation_file)
        if not annotation_path.exists():
            raise QuantificationError(f"Annotation file not found: {self.annotation_file}")

        # Initialize data structures like in the debug script
        self.genes: Dict[str, Dict] = {}
        self.transcripts: Dict[str, Dict] = {}
        self.gene_to_transcripts: Dict[str, List[str]] = defaultdict(list)
        self.transcript_to_exons: Dict[str, List[Tuple]] = defaultdict(list)
        exon_count = 0

        with open(annotation_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith("#") or not line.strip():
                    continue

                try:
                    fields = line.strip().split("\t")
                    if len(fields) < 9:
                        continue

                    chrom, _, feature, start, end, _, strand, _, attrs = fields
                    start, end = int(start), int(end)

                    # Parse attributes (robust parsing like debug script)
                    attr_dict = {}
                    for pair in attrs.split(";"):
                        pair = pair.strip()
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            attr_dict[k.strip()] = v.strip()
                        elif " " in pair:
                            k, v = pair.split(" ", 1)
                            attr_dict[k.strip()] = v.strip().strip('"')

                    # Process genes
                    if feature == "gene":
                        gene_id = attr_dict.get("ID", attr_dict.get("gene_id", ""))
                        if gene_id:
                            self.genes[gene_id] = {
                                "chrom": chrom,
                                "strand": strand,
                                "transcripts": [],
                                "name": attr_dict.get("Name", gene_id),
                            }

                    # Process transcripts (mRNA or transcript)
                    elif feature in ("mRNA", "transcript"):
                        transcript_id = attr_dict.get("ID", attr_dict.get("transcript_id", ""))
                        parent_gene = attr_dict.get("Parent", attr_dict.get("gene_id", ""))
                        if transcript_id and parent_gene:
                            self.transcripts[transcript_id] = {
                                "gene": parent_gene,
                                "chrom": chrom,
                                "strand": strand,
                                "exons": [],
                            }
                            self.gene_to_transcripts[parent_gene].append(transcript_id)

                    # Process exons
                    elif feature == "exon":
                        parent_transcript = attr_dict.get("Parent", attr_dict.get("transcript_id", ""))
                        if not parent_transcript:
                            continue

                        # If transcript doesn't exist, create it and try to infer gene
                        if parent_transcript not in self.transcripts:
                            # Try to infer gene from parent_transcript prefix
                            gene_guess = parent_transcript.split(".")[0]
                            self.transcripts[parent_transcript] = {
                                "gene": gene_guess,
                                "chrom": chrom,
                                "strand": strand,
                                "exons": [],
                            }
                            self.gene_to_transcripts[gene_guess].append(parent_transcript)

                        # Add exon to transcript
                        exon_tuple = (chrom, start, end, strand)
                        self.transcripts[parent_transcript]["exons"].append(exon_tuple)
                        self.transcript_to_exons[parent_transcript].append(exon_tuple)
                        exon_count += 1

                except Exception as e:
                    self.logger.warning(f"Error parsing line {line_num}: {str(e)}")
                    continue

        self.logger.info(f"Parsed {len(self.genes)} genes, {len(self.transcripts)} transcripts, {exon_count} exons")

        # Record parsing completion
        self._record_internal_operation(
            "annotation_parsed",
            f"Parsed {len(self.genes)} genes, {len(self.transcripts)} transcripts, {exon_count} exons",
        )

    def _build_gene_exon_intervals(self) -> None:
        """Precompute merged exonic intervals per gene to reduce BAM fetch calls."""
        self.gene_exons = defaultdict(list)

        for gene_id, transcript_ids in self.gene_to_transcripts.items():
            exons = []
            for transcript_id in transcript_ids:
                exons.extend(self.transcript_to_exons.get(transcript_id, []))
            self.gene_exons[gene_id] = self._merge_exons(exons)

        self.logger.info(f"Built merged exon intervals for {len(self.gene_exons)} genes")

    def _merge_exons(self, exons: List[Tuple]) -> List[Tuple]:
        """Merge overlapping exons for a gene while preserving chromosome/strand."""
        if not exons:
            return []

        sorted_exons = sorted(exons, key=lambda exon: (exon[0], exon[3], exon[1], exon[2]))
        merged = []
        for chrom, start, end, strand in sorted_exons:
            if not merged:
                merged.append([chrom, start, end, strand])
                continue

            last = merged[-1]
            if chrom == last[0] and strand == last[3] and start <= last[2] + 1:
                last[2] = max(last[2], end)
            else:
                merged.append([chrom, start, end, strand])

        return [tuple(exon) for exon in merged]

    def _sort_and_index_bam(self, bam_file: str) -> str:
        """Sort and index BAM file if needed."""
        bam_path = Path(bam_file)

        # Check if BAM is sorted
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                if bam.header.get("HD", {}).get("SO") == "coordinate":
                    self.logger.debug(f"BAM file {bam_file} is already coordinate sorted")
                    sorted_bam = str(bam_path)
                else:
                    self.logger.info(f"Sorting BAM file {bam_file}")
                    sorted_bam = str(bam_path.with_suffix(".sorted.bam"))
                    pysam.sort("-o", sorted_bam, bam_file)
        except Exception as e:
            self.logger.warning(f"Could not check BAM sort status, attempting to sort: {e}")
            sorted_bam = str(bam_path.with_suffix(".sorted.bam"))
            pysam.sort("-o", sorted_bam, bam_file)

        # Index BAM file
        index_file = sorted_bam + ".bai"
        index_path = Path(index_file)
        bam_path = Path(sorted_bam)

        # Check if index exists and is current (newer than or equal to BAM file)
        needs_indexing = False
        if not index_path.exists():
            self.logger.info(f"Index file does not exist for {sorted_bam}")
            needs_indexing = True
        elif index_path.stat().st_mtime < bam_path.stat().st_mtime:
            self.logger.info(f"Index file is older than BAM file for {sorted_bam}")
            needs_indexing = True

        if needs_indexing:
            safe_sorted_bam = str(sorted_bam).replace("\n", " ").replace("\r", " ")
            self.logger.info(f"Indexing BAM file {safe_sorted_bam}")
            if self.bam_index_threads > 1:
                pysam.index("-@", str(self.bam_index_threads), sorted_bam)
            else:
                pysam.index(sorted_bam)
        else:
            self.logger.debug(f"BAM file {sorted_bam} is already indexed and current")

        return sorted_bam

    def _count_overlaps_union(self, bam: pysam.AlignmentFile, gene_id: str, gene_info: Dict) -> int:
        """
        Count overlaps using Union mode with unique read names.
        If filter_strategy is 'strict', apply mapping quality, multi-mapping, and strand filters.
        If 'none', count all overlaps (debug script logic).
        """
        try:
            gene_exons = self.gene_exons.get(gene_id)
            if not gene_exons:
                tlist = self.gene_to_transcripts[gene_id]
                gene_exons = []
                for tid in tlist:
                    gene_exons.extend(self.transcript_to_exons[tid])
            gene_read_names = set()
            filtered_reads = {
                "unmapped": 0,
                "low_quality": 0,
                "multi_mapped": 0,
                "wrong_strand": 0,
            }

            for exon_chrom, exon_start, exon_end, exon_strand in gene_exons:
                try:
                    for read in bam.fetch(exon_chrom, exon_start - 1, exon_end):
                        if self.filter_strategy == "strict":
                            # Skip unmapped reads
                            if read.is_unmapped:
                                filtered_reads["unmapped"] += 1
                                continue
                            # Skip low quality reads
                            if read.mapping_quality < self.min_mapping_quality:
                                filtered_reads["low_quality"] += 1
                                continue
                            # Skip multi-mapping reads if not counting them
                            if not self.count_multi_mapping and read.has_tag("NH") and read.get_tag("NH") > 1:
                                filtered_reads["multi_mapped"] += 1
                                continue
                            # Check strand if not ignoring
                            if not self.ignore_strand:
                                read_strand = "-" if read.is_reverse else "+"
                                if read_strand != exon_strand and exon_strand != ".":
                                    filtered_reads["wrong_strand"] += 1
                                    continue
                        # Always add read name if not filtered
                        gene_read_names.add(read.query_name)
                except ValueError:
                    continue

            # Log filtering statistics for debugging
            if self.filter_strategy == "strict" and sum(filtered_reads.values()) > 0:
                self.logger.debug(
                    f"Gene {gene_id} filtering stats: "
                    + f"unmapped={filtered_reads['unmapped']}, "
                    + f"low_quality={filtered_reads['low_quality']}, "
                    + f"multi_mapped={filtered_reads['multi_mapped']}, "
                    + f"wrong_strand={filtered_reads['wrong_strand']}, "
                    + f"final_count={len(gene_read_names)}"
                )

            return len(gene_read_names)
        except Exception as e:
            self.logger.error(f"Error counting overlaps for gene {gene_id}: {e}")
            return 0

    def _process_sample(self, sample_id: str, bam_file: str) -> Dict[str, int]:
        """Process a single sample and return gene counts."""
        safe_sample_id = str(sample_id).replace("\n", " ").replace("\r", " ")
        safe_bam_file = str(bam_file).replace("\n", " ").replace("\r", " ")
        self.logger.info(f"Processing sample {safe_sample_id}: {safe_bam_file}")

        # Record this operation for the execution report
        self._record_internal_operation(
            "sample_quantification",
            f"Quantifying reads using {self.overlap_mode.value} mode with {self.filter_strategy} filtering",
            sample_id,
        )

        # Verify BAM file exists
        if not os.path.exists(bam_file):
            self.logger.error(f"BAM file not found: {bam_file}")
            raise QuantificationError(f"BAM file not found: {bam_file}")

        # Sort and index BAM file if needed
        sorted_bam = self._sort_and_index_bam(bam_file)

        # Count overlaps for each gene using Union mode. Keep one BAM handle
        # open for the whole sample instead of reopening it once per gene.
        gene_counts = {}

        with pysam.AlignmentFile(sorted_bam, "rb") as bam:
            for gene_id, gene_info in self.genes.items():
                count = self._count_overlaps_union(bam, gene_id, gene_info)
                gene_counts[gene_id] = count

        # Record completion for this sample
        total_reads = sum(gene_counts.values())
        genes_with_reads = sum(1 for count in gene_counts.values() if count > 0)
        self._record_internal_operation(
            "sample_completed",
            f"Processed {len(gene_counts)} genes, {total_reads} total reads, {genes_with_reads} genes with reads",
            sample_id,
        )

        self.logger.info(f"Completed quantification for sample {sample_id}")
        return gene_counts

    def _write_sample_counts(self, sample_id: str, gene_counts: Dict[str, int]) -> Path:
        """Write one sample's genomic-overlaps counts for SLURM worker merging."""
        self.worker_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.worker_output_dir / f"{sample_id}_genomic_overlaps_counts.tsv"
        with open(output_file, "w") as handle:
            handle.write("Gene\tCount\n")
            for gene_id in sorted(gene_counts):
                handle.write(f"{self._clean_gene_id(gene_id)}\t{gene_counts[gene_id]}\n")
        return output_file

    def _read_sample_counts(self, sample_id: str) -> Dict[str, int]:
        """Read one sample worker count file."""
        output_file = self.worker_output_dir / f"{sample_id}_genomic_overlaps_counts.tsv"
        if not output_file.exists():
            raise QuantificationError(f"Missing genomic-overlaps worker output for {sample_id}: {output_file}")

        df = pd.read_csv(output_file, sep="\t")
        return dict(zip(df["Gene"].astype(str), df["Count"].astype(int)))

    def _build_command(self, sample_id: str, bam_file: str) -> str:
        """
        Build command for genomic overlaps quantification.

        Since this is a pure Python implementation, no external command is needed.
        This method is implemented to satisfy the abstract base class requirement.

        Args:
            sample_id: Sample identifier
            bam_file: Path to BAM file

        Returns:
            str: Description of the quantification process
        """
        return f"genomic_overlaps_quantify --mode {self.overlap_mode.value} --input {bam_file} --annotation {self.annotation_file} --output {self.out_dir}/{sample_id}_genomic_overlaps_counts.txt"

    def _build_worker_config(self) -> Path:
        """Write worker configuration for SLURM array execution."""
        self.worker_output_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.worker_output_dir / "genomic_overlaps_worker_config.json"
        samples = {sample_id: self._extract_bam_path(sample_info) for sample_id, sample_info in self.bam_dict.items()}
        config = {
            "samples": samples,
            "annotation_file": self.annotation_file,
            "out_dir": str(self.out_dir),
            "worker_output_dir": str(self.worker_output_dir),
            "param_dir": self.param_dir,
            "overlap_mode": self.overlap_mode.value,
            "filter_strategy": self.filter_strategy,
            "min_mapping_quality": self.min_mapping_quality,
            "ignore_strand": self.ignore_strand,
            "count_multi_mapping": self.count_multi_mapping,
            "cpu_threads": 1,
            "memory": self.memory,
            "bam_index_threads": self.bam_index_threads,
        }
        with open(config_file, "w") as handle:
            json.dump(config, handle, indent=2)
        return config_file

    def _run_slurm_quantification(self) -> Dict[str, Any]:
        """Run per-sample genomic-overlaps workers through SLURM arrays."""
        self.logger.info("Running genomic overlaps quantification as SLURM array workers")
        config_file = self._build_worker_config()
        commands = {}
        for sample_id in self.bam_dict:
            commands[sample_id] = (
                f"{shlex.quote(sys.executable)} -m pyseqrna.modules.quantification.genomic_overlaps "
                f"--worker {shlex.quote(str(config_file))} {shlex.quote(str(sample_id))}"
            )

        self.execute_command(commands, str(self.out_dir), "genomic_overlaps")

        all_counts = {}
        results = {}
        for sample_id in self.bam_dict:
            gene_counts = self._read_sample_counts(sample_id)
            all_counts[sample_id] = gene_counts
            results[sample_id] = {
                "total_genes": len(gene_counts),
                "genes_with_reads": sum(1 for count in gene_counts.values() if count > 0),
                "total_reads_counted": sum(gene_counts.values()),
            }

        count_matrix_file = self.out_dir / "genomic_overlaps_count_matrix.txt"
        self._write_count_matrix(count_matrix_file, all_counts)
        summary_stats = self._calculate_summary_statistics(all_counts)
        shutil.rmtree(self.worker_output_dir, ignore_errors=True)
        self.logger.info("Cleaned genomic-overlaps intermediate worker outputs")

        return {
            "method": f"GenomicOverlaps_{self.overlap_mode.value}",
            "filter_strategy": self.filter_strategy,
            "samples": results,
            "count_matrix_file": str(count_matrix_file),
            "summary_statistics": summary_stats,
            "parameters": {
                "overlap_mode": self.overlap_mode.value,
                "filter_strategy": self.filter_strategy,
                "min_mapping_quality": self.min_mapping_quality,
                "ignore_strand": self.ignore_strand,
                "count_multi_mapping": self.count_multi_mapping,
                "total_genes": len(self.genes),
                "total_transcripts": len(self.transcripts),
                "execution": "slurm_array",
            },
        }

    def run(self) -> pd.DataFrame:
        """
        Run the genomic overlaps quantification process.

        Returns:
            DataFrame containing the count matrix

        Raises:
            QuantificationError: If quantification fails
        """
        self.logger.info(f"Starting {self.tool_name} quantification with filter strategy: {self.filter_strategy}")

        try:
            # Check tool availability (pysam)
            if not self.check_tool_availability():
                raise QuantificationError(f"{self.tool_name} is not available (pysam required)")

            # Create output directory
            if not self.dryrun:
                self.file_manager.create_subdirectory(str(self.out_dir), dry_run=False, preserve_existing=True)

                # Create log file for genomic_overlaps quantification
                log_file = self.out_dir / "genomic_overlaps_quantification.log"
                self._create_quantification_log(log_file)

            # Run quantification using the quantify method
            if not self.dryrun:
                if self.slurm:
                    quantification_result = self._run_slurm_quantification()
                else:
                    quantification_result = self.quantify()

                # Convert result to DataFrame format expected by pipeline
                count_matrix_file = quantification_result["count_matrix_file"]
                if os.path.exists(count_matrix_file):
                    # Read the count matrix and convert to Excel format for consistency
                    count_df = pd.read_csv(count_matrix_file, sep="\t", index_col=None)

                    # Add filter strategy as an attribute to the DataFrame
                    count_df.attrs["filter_strategy"] = self.filter_strategy

                    # Save as Excel file for consistency with other quantifiers
                    output_excel = self.out_dir / "Raw_Counts.xlsx"
                    count_df.to_excel(str(output_excel), index=False)
                    self.logger.info(f"Count matrix saved to: {output_excel} (filter strategy: {self.filter_strategy})")

                    # Clean up temporary TSV file
                    os.remove(count_matrix_file)

                    # Update log file with completion status
                    if not self.dryrun:
                        self._update_quantification_log(log_file, quantification_result)

                    self.logger.info(
                        f"{self.tool_name} quantification completed successfully with filter strategy: {self.filter_strategy}"
                    )
                    return count_df
                else:
                    raise QuantificationError("Count matrix file was not created")

            else:
                self.logger.info(f"DRYRUN: {self.tool_name} quantification simulation completed")
                # Return mock DataFrame for dry run
                sample_names = ["Gene"] + list(self.bam_dict.keys())
                mock_data = [
                    ["gene1"] + [100] * len(self.bam_dict),
                    ["gene2"] + [200] * len(self.bam_dict),
                ]
                mock_df = pd.DataFrame(mock_data, columns=sample_names)
                mock_df.attrs["filter_strategy"] = self.filter_strategy
                return mock_df

        except Exception as e:
            self.logger.error(f"{self.tool_name} quantification failed: {str(e)}")
            raise QuantificationError(f"{self.tool_name} quantification failed: {str(e)}")

    def _create_quantification_log(self, log_file: Path) -> None:
        """Create a detailed log file for genomic overlaps quantification."""
        from datetime import datetime

        with open(log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("GENOMIC OVERLAPS QUANTIFICATION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.overlap_mode.value}\n")
            f.write(f"Filter Strategy: {self.filter_strategy}\n")
            f.write(f"Min Mapping Quality: {self.min_mapping_quality}\n")
            f.write(f"Ignore Strand: {self.ignore_strand}\n")
            f.write(f"Count Multi-mapping: {self.count_multi_mapping}\n")
            f.write(f"Annotation File: {self.annotation_file}\n")
            f.write(f"Output Directory: {self.out_dir}\n")
            f.write(f"Total Samples: {len(self.bam_dict)}\n")
            f.write(f"Total Genes: {len(self.genes)}\n")
            f.write(f"Total Transcripts: {len(self.transcripts)}\n")
            f.write("-" * 80 + "\n")
            f.write("SAMPLES:\n")
            for sample_id, sample_info in self.bam_dict.items():
                bam_file = self._extract_bam_path(sample_info)
                f.write(f"  {sample_id}: {bam_file}\n")
            f.write("-" * 80 + "\n")
            f.write("PROCESSING LOG:\n")

    def _update_quantification_log(self, log_file: Path, quantification_result: Dict[str, Any]) -> None:
        """Update the quantification log file with completion details."""
        from datetime import datetime

        with open(log_file, "a") as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write("QUANTIFICATION COMPLETED\n")
            f.write("-" * 80 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: {quantification_result['method']}\n")
            f.write(f"Filter Strategy: {quantification_result['filter_strategy']}\n")

            # Sample statistics
            f.write("\nSAMPLE STATISTICS:\n")
            for sample_id, stats in quantification_result["samples"].items():
                f.write(f"  {sample_id}:\n")
                f.write(f"    Total genes: {stats['total_genes']}\n")
                f.write(f"    Genes with reads: {stats['genes_with_reads']}\n")
                f.write(f"    Total reads counted: {stats['total_reads_counted']}\n")

            # Summary statistics
            if "summary_statistics" in quantification_result:
                f.write("\nSUMMARY STATISTICS:\n")
                summary = quantification_result["summary_statistics"]
                for key, value in summary.items():
                    f.write(f"  {key}: {value}\n")

            # Parameters
            f.write("\nPARAMETERS:\n")
            for key, value in quantification_result["parameters"].items():
                f.write(f"  {key}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")

    def quantify(self) -> Dict[str, Any]:
        """
        Perform quantification using genomic overlaps.

        Returns:
            Dict containing quantification results and statistics
        """
        self.logger.info(
            f"Starting genomic overlaps quantification with {self.overlap_mode.value} mode and {self.filter_strategy} filter strategy"
        )

        results = {}
        all_counts = {}
        sample_items = list(self.bam_dict.items())
        auto_workers = min(len(sample_items), max(1, int(self.cpu_threads or 1) // 4), 8)
        max_workers = self.parallel_samples or auto_workers
        max_workers = max(1, min(max_workers, len(sample_items)))

        if max_workers > 1:
            self.logger.info(f"Processing {len(sample_items)} samples in parallel with {max_workers} worker(s)")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_sample = {}
                for sample_id, sample_info in sample_items:
                    bam_file = self._extract_bam_path(sample_info)
                    future = executor.submit(self._process_sample, sample_id, bam_file)
                    future_to_sample[future] = sample_id

                for future in as_completed(future_to_sample):
                    sample_id = future_to_sample[future]
                    try:
                        gene_counts = future.result()
                        all_counts[sample_id] = gene_counts
                        results[sample_id] = {
                            "total_genes": len(gene_counts),
                            "genes_with_reads": sum(1 for count in gene_counts.values() if count > 0),
                            "total_reads_counted": sum(gene_counts.values()),
                        }
                    except Exception as e:
                        self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
                        raise QuantificationError(f"Failed to process sample {sample_id}: {str(e)}")
        else:
            self.logger.info(f"Processing {len(sample_items)} samples sequentially")
            for sample_id, sample_info in sample_items:
                try:
                    # Extract BAM file path from sample info
                    bam_file = self._extract_bam_path(sample_info)

                    # Process sample
                    gene_counts = self._process_sample(sample_id, bam_file)
                    all_counts[sample_id] = gene_counts

                    # Store results (no individual file creation)
                    results[sample_id] = {
                        "total_genes": len(gene_counts),
                        "genes_with_reads": sum(1 for count in gene_counts.values() if count > 0),
                        "total_reads_counted": sum(gene_counts.values()),
                    }

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
                    raise QuantificationError(f"Failed to process sample {sample_id}: {str(e)}")

        # Create combined count matrix only
        count_matrix_file = self.out_dir / "genomic_overlaps_count_matrix.txt"
        self._write_count_matrix(count_matrix_file, all_counts)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(all_counts)

        quantification_result = {
            "method": f"GenomicOverlaps_{self.overlap_mode.value}",
            "filter_strategy": self.filter_strategy,
            "samples": results,
            "count_matrix_file": str(count_matrix_file),
            "summary_statistics": summary_stats,
            "parameters": {
                "overlap_mode": self.overlap_mode.value,
                "filter_strategy": self.filter_strategy,
                "min_mapping_quality": self.min_mapping_quality,
                "ignore_strand": self.ignore_strand,
                "count_multi_mapping": self.count_multi_mapping,
                "total_genes": len(self.genes),
                "total_transcripts": len(self.transcripts),
            },
        }

        self.logger.info("Genomic overlaps quantification completed successfully")

        # Record overall quantification completion
        total_samples = len(all_counts)
        total_genes = len(all_counts[next(iter(all_counts))]) if all_counts else 0
        total_reads_all_samples = sum(sum(counts.values()) for counts in all_counts.values())
        self._record_internal_operation(
            "quantification_completed",
            f"Completed quantification for {total_samples} samples, {total_genes} genes, {total_reads_all_samples} total reads across all samples",
        )

        return quantification_result

    def _write_count_matrix(self, output_file: Path, all_counts: Dict[str, Dict[str, int]]):
        """Write combined count matrix to file."""
        if not all_counts:
            return

        # Get all gene IDs
        all_gene_ids = set()
        for counts in all_counts.values():
            all_gene_ids.update(counts.keys())
        all_gene_ids = sorted(all_gene_ids)

        # Get sample IDs in the original order from bam_dict (not alphabetically sorted)
        sample_ids = list(self.bam_dict.keys())

        with open(output_file, "w") as f:
            # Write header
            f.write("Gene\t" + "\t".join(sample_ids) + "\n")

            # Write counts for each gene
            for gene_id in all_gene_ids:
                counts_row = [str(all_counts[sample_id].get(gene_id, 0)) for sample_id in sample_ids]
                f.write(f"{self._clean_gene_id(gene_id)}\t" + "\t".join(counts_row) + "\n")

    @staticmethod
    def _clean_gene_id(gene_id: str) -> str:
        """Normalize common annotation prefixes from stable gene IDs."""
        gene_id = str(gene_id).strip()
        for prefix in ("gene:", "gene-"):
            if gene_id.startswith(prefix):
                return gene_id[len(prefix) :]
        return gene_id

    def _calculate_summary_statistics(self, all_counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Calculate summary statistics across all samples."""
        if not all_counts:
            return {}

        # Convert to DataFrame for easier calculation
        count_data = []
        for sample_id, counts in all_counts.items():
            for gene_id, count in counts.items():
                count_data.append({"sample_id": sample_id, "gene_id": gene_id, "count": count})

        if not count_data:
            return {}

        df = pd.DataFrame(count_data)

        # Calculate statistics
        stats = {
            "total_samples": len(all_counts),
            "total_genes": len(set(df["gene_id"])),
            "mean_reads_per_sample": df.groupby("sample_id")["count"].sum().mean(),
            "mean_reads_per_gene": df.groupby("gene_id")["count"].sum().mean(),
            "genes_with_zero_counts": len(df.groupby("gene_id")["count"].sum()[df.groupby("gene_id")["count"].sum() == 0]),
            "percentage_genes_detected": (
                len(df.groupby("gene_id")["count"].sum()[df.groupby("gene_id")["count"].sum() > 0]) / len(set(df["gene_id"]))
            )
            * 100,
        }

        return stats


def main() -> int:
    """Command-line worker entry point for SLURM array tasks."""
    parser = argparse.ArgumentParser(description="Internal PySeqRNA genomic-overlaps worker")
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "SAMPLE_ID"))
    args = parser.parse_args()

    if not args.worker:
        parser.error("This module is intended for internal --worker execution")

    config_file, sample_id = args.worker
    _run_worker(config_file, sample_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
