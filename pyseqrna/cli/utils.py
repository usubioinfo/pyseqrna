# -*- coding: utf-8 -*-

"""
PySeqRNA CLI Utility Helper Functions

This module contains utility helper functions for resolving inputs, building SLURM
configurations, and preparing files for various PySeqRNA CLI commands. It handles
path scanning for FASTQ and BAM files and parses tables for downstream functional annotations.

Features:
    - Resolves raw/trimmed FASTQ files dynamically from sample configurations and directories
    - Reconstructs sample dictionaries from resolved read paths
    - Resolves sorted BAM files from alignment output directories
    - Extracts CLI options to build SLURM CommandExecutor-compatible configurations
    - Extracts gene lists from various text, CSV, TSV, or Excel formats for functional enrichment

Configuration:
    - Configured via function parameters, sample dictionaries, and CLI argument objects.

Dependencies:
    - Python packages: pathlib, typing, pandas

Classes / Functions / Exceptions:
    - _existing_fastq: Returns True for existing FASTQ-like files.
    - _format_trimmed_pattern: Expands user/file pattern for sample ID/label.
    - _find_first_trimmed_file: Finds the first existing trimmed FASTQ matching a relative pattern.
    - _infer_trimmed_pair: Infers paired trimmed FASTQs from naming conventions.
    - _resolve_trimmed_files: Resolves existing trimmed FASTQ files into a results dictionary.
    - _build_sample_dict_from_reads: Rebuilds sample dictionary from resolved reads.
    - _build_alignment_target: Formats samples for aligner input.
    - _build_slurm_config: Extracts SLURM CLI parameters into a configuration dictionary.
    - _format_bam_pattern: Expands BAM pattern for sample ID/label.
    - _existing_bam: Checks if a BAM file exists.
    - _find_first_bam: Finds the first BAM matching any relative pattern.
    - _default_bam_patterns: Default patterns for BAM files.
    - _resolve_bam_files: Resolves BAM files for each sample.
    - _collect_annotation_gene_files: Collects gene files from options.
    - _read_gene_ids_for_annotation: Reads gene IDs from files.
    - _prepare_annotation_inputs: Prepares input files for annotation.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd


def _existing_fastq(path: Path) -> bool:
    """Return True for existing FASTQ-like files."""
    if not path.exists() or not path.is_file():
        return False
    name = path.name.lower()
    return name.endswith(
        (
            ".fastq",
            ".fq",
            ".fastq.gz",
            ".fq.gz",
            ".fastq.bz2",
            ".fq.bz2",
        )
    )


def _format_trimmed_pattern(pattern: str, sample_id: str, sample_label: str) -> List[str]:
    """Expand a user/file pattern for sample id and sample label."""
    candidates = []
    for token in [sample_id, sample_label]:
        if token and token not in candidates:
            candidates.append(pattern.format(sample=token, sample_id=sample_id, sample_name=sample_label))
    return candidates


def _find_first_trimmed_file(trimmed_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Find the first existing trimmed FASTQ matching any relative pattern."""
    for pattern in patterns:
        direct = trimmed_dir / pattern
        if _existing_fastq(direct):
            return direct
        matches = sorted(path for path in trimmed_dir.rglob(pattern) if _existing_fastq(path))
        if matches:
            return matches[0]
    return None


def _infer_trimmed_pair(trimmed_dir: Path, identifiers: List[str]) -> Optional[List[Path]]:
    """Infer paired trimmed FASTQs from common external naming conventions."""
    fastqs = sorted(path for path in trimmed_dir.rglob("*") if _existing_fastq(path))
    for identifier in identifiers:
        if not identifier:
            continue
        matching = [path for path in fastqs if identifier in path.name]
        r1 = [
            path
            for path in matching
            if any(token in path.name for token in ("_R1", "_1", ".R1", ".1", "_read1", "_READ1", "_val_1"))
            and "unpaired" not in path.name.lower()
        ]
        r2 = [
            path
            for path in matching
            if any(token in path.name for token in ("_R2", "_2", ".R2", ".2", "_read2", "_READ2", "_val_2"))
            and "unpaired" not in path.name.lower()
        ]
        if r1 and r2:
            return [r1[0], r2[0]]
    return None


def _resolve_trimmed_files(
    sample_dict: dict,
    trimmed_dir: Path,
    paired: bool,
    pattern_r1: Optional[str] = None,
    pattern_r2: Optional[str] = None,
    pattern_single: Optional[str] = None,
) -> dict:
    """
    Resolve existing trimmed FASTQ files into a trimming-results dictionary.

    Supports PySeqRNA defaults and common external naming schemes. Custom
    patterns may use {sample}, {sample_id}, or {sample_name}.
    """
    if not trimmed_dir.exists() or not trimmed_dir.is_dir():
        raise ValueError(f"Trimmed directory not found: {trimmed_dir}")

    trimmed_results = {}
    missing = []

    default_paired_patterns = [
        ("{sample}_val_1.fq.gz", "{sample}_val_2.fq.gz"),
        ("{sample}_R1_trimmed.fastq.gz", "{sample}_R2_trimmed.fastq.gz"),
        ("{sample}_R1_trimmed.fq.gz", "{sample}_R2_trimmed.fq.gz"),
        ("{sample}_1_trimmed.fastq.gz", "{sample}_2_trimmed.fastq.gz"),
        ("{sample}_1_paired.fastq.gz", "{sample}_2_paired.fastq.gz"),
        ("{sample}_R1_paired.fastq.gz", "{sample}_R2_paired.fastq.gz"),
        ("{sample}_R1.fastq.gz", "{sample}_R2.fastq.gz"),
        ("{sample}_1.fastq.gz", "{sample}_2.fastq.gz"),
    ]
    default_single_patterns = [
        "{sample}_trimmed.fq.gz",
        "{sample}_trimmed.fastq.gz",
        "{sample}.trimmed.fq.gz",
        "{sample}.trimmed.fastq.gz",
        "{sample}.fastq.gz",
        "{sample}.fq.gz",
    ]

    for sample_id, sample_info in sample_dict.items():
        sample_label = str(sample_info[0]) if sample_info else sample_id
        identifiers = []
        for value in [sample_id, sample_label]:
            if value and value not in identifiers:
                identifiers.append(str(value))

        if paired:
            paired_patterns = [(pattern_r1, pattern_r2)] if pattern_r1 and pattern_r2 else default_paired_patterns
            found_pair = None
            for r1_pattern, r2_pattern in paired_patterns:
                r1 = _find_first_trimmed_file(
                    trimmed_dir,
                    [
                        expanded
                        for identifier in identifiers
                        for expanded in _format_trimmed_pattern(r1_pattern, identifier, sample_label)
                    ],
                )
                r2 = _find_first_trimmed_file(
                    trimmed_dir,
                    [
                        expanded
                        for identifier in identifiers
                        for expanded in _format_trimmed_pattern(r2_pattern, identifier, sample_label)
                    ],
                )
                if r1 and r2:
                    found_pair = [r1, r2]
                    break

            if found_pair is None and not (pattern_r1 or pattern_r2):
                found_pair = _infer_trimmed_pair(trimmed_dir, identifiers)

            if found_pair is None:
                missing.append(sample_id)
            else:
                trimmed_results[sample_id] = [str(found_pair[0]), str(found_pair[1])]
        else:
            single_patterns = [pattern_single] if pattern_single else default_single_patterns
            found = _find_first_trimmed_file(
                trimmed_dir,
                [
                    expanded
                    for single_pattern in single_patterns
                    for identifier in identifiers
                    for expanded in _format_trimmed_pattern(single_pattern, identifier, sample_label)
                ],
            )
            if found is None and not pattern_single:
                matches = sorted(
                    path
                    for path in trimmed_dir.rglob("*")
                    if _existing_fastq(path) and any(identifier in path.name for identifier in identifiers)
                )
                found = matches[0] if matches else None

            if found is None:
                missing.append(sample_id)
            else:
                trimmed_results[sample_id] = str(found)

    if missing:
        raise ValueError("Could not find trimmed FASTQ files for sample(s): " + ", ".join(missing) + f" in {trimmed_dir}")

    return trimmed_results


def _build_sample_dict_from_reads(sample_dict: dict, read_dict: dict, paired: bool) -> dict:
    """Build a sample_dict-compatible structure from resolved read paths."""
    rebuilt = {}
    for sample_id, sample_info in sample_dict.items():
        sample_label = sample_info[0] if len(sample_info) > 0 else sample_id
        condition = sample_info[1] if len(sample_info) > 1 else "sample"
        reads = read_dict[sample_id]
        if paired:
            rebuilt[sample_id] = [sample_label, condition, reads[0], reads[1]]
        else:
            rebuilt[sample_id] = [
                sample_label,
                condition,
                reads if isinstance(reads, str) else reads[0],
            ]
    return rebuilt


def _build_alignment_target(sample_dict: dict, paired: bool, read_dict: Optional[dict] = None) -> dict:
    """Build aligner target dictionary from raw or resolved trimmed reads."""
    alignment_target = {}
    for sample_id, sample_info in sample_dict.items():
        if read_dict and sample_id in read_dict:
            reads = read_dict[sample_id]
            if paired:
                alignment_target[sample_id] = [reads[0], reads[1]]
            else:
                alignment_target[sample_id] = [reads if isinstance(reads, str) else reads[0]]
        elif paired and len(sample_info) >= 4:
            alignment_target[sample_id] = [sample_info[2], sample_info[3]]
        else:
            alignment_target[sample_id] = [sample_info[2]]
    return alignment_target


def _build_slurm_config(args) -> dict:
    """Build CommandExecutor-compatible SLURM config from standalone args."""
    slurm_config = {
        "partition": args.slurm_partition,
        "time": args.slurm_time,
        "memory": str(args.slurm_memory or getattr(args, "memory", None) or 16),
        "cpus": str(args.threads or 1),
        "ntasks": "1",
    }
    if args.slurm_account:
        slurm_config["account"] = args.slurm_account
    if args.slurm_email:
        slurm_config["email"] = args.slurm_email
    if args.slurm_qos:
        slurm_config["qos"] = args.slurm_qos
    return slurm_config


def _format_bam_pattern(pattern: str, sample_id: str, sample_label: str) -> List[str]:
    """Expand a BAM file pattern for sample id and sample label."""
    candidates = []
    for token in [sample_id, sample_label]:
        if token and token not in candidates:
            candidates.append(pattern.format(sample=token, sample_id=sample_id, sample_name=sample_label))
    return candidates


def _existing_bam(path: Path, dryrun: bool = False) -> bool:
    """Return True for usable BAM paths, allowing simulated BAMs in dry-run."""
    if dryrun:
        return path.name.endswith(".bam")
    return path.exists() and path.is_file() and path.name.endswith(".bam")


def _find_first_bam(alignment_dir: Path, patterns: List[str], dryrun: bool = False) -> Optional[Path]:
    """Find the first BAM matching any relative pattern."""
    for pattern in patterns:
        direct = alignment_dir / pattern
        if _existing_bam(direct, dryrun=dryrun):
            return direct
        if not dryrun:
            matches = sorted(path for path in alignment_dir.rglob(pattern) if _existing_bam(path))
            if matches:
                return matches[0]
    return None


def _default_bam_patterns(alignment_tool: Optional[str]) -> List[str]:
    """Return PySeqRNA and common external BAM patterns."""
    tool_patterns = {
        "star": ["star_results/{sample}_Aligned.out.bam"],
        "hisat2": ["hisat2_results/{sample}_aligned.bam"],
        "bowtie2": ["bowtie2_results/{sample}_aligned.bam"],
        "bwa": ["bwa_results/{sample}_aligned.bam"],
        "minimap2": ["minimap2_results/{sample}_aligned.bam"],
    }
    patterns = []
    if alignment_tool:
        patterns.extend(tool_patterns.get(alignment_tool, []))
    else:
        for values in tool_patterns.values():
            patterns.extend(values)
    patterns.extend(
        [
            "{sample}_Aligned.out.bam",
            "{sample}_aligned.bam",
            "{sample}.bam",
            "{sample}_sorted.bam",
            "{sample}.sorted.bam",
        ]
    )
    return patterns


def _resolve_bam_files(
    sample_dict: dict,
    alignment_dir: Path,
    alignment_tool: Optional[str],
    bam_pattern: Optional[str],
    dryrun: bool = False,
) -> dict:
    """Resolve BAM files for each sample from PySeqRNA or external naming schemes."""
    if not alignment_dir.exists() or not alignment_dir.is_dir():
        if dryrun:
            alignment_dir = alignment_dir
        else:
            raise ValueError(f"Alignment/BAM directory not found: {alignment_dir}")

    bam_files = {}
    missing = []
    default_patterns = [bam_pattern] if bam_pattern else _default_bam_patterns(alignment_tool)

    for sample_id, sample_info in sample_dict.items():
        sample_label = str(sample_info[0]) if sample_info else sample_id
        patterns = []
        for base_pattern in default_patterns:
            patterns.extend(_format_bam_pattern(base_pattern, sample_id, sample_label))

        found = _find_first_bam(alignment_dir, patterns, dryrun=dryrun)

        if found is None and not bam_pattern and not dryrun:
            identifiers = [str(value) for value in [sample_id, sample_label] if value]
            matches = sorted(
                path for path in alignment_dir.rglob("*.bam") if any(identifier in path.name for identifier in identifiers)
            )
            found = matches[0] if matches else None

        if found is None:
            missing.append(sample_id)
        else:
            bam_files[sample_id] = str(found)

    if missing:
        raise ValueError(
            "Could not find BAM files for sample(s): "
            + ", ".join(missing)
            + f" in {alignment_dir}. Use --alignment-tool or --bam-pattern if needed."
        )

    return bam_files


def _collect_annotation_gene_files(args) -> List[Path]:
    """Collect gene files from --gene-files or --deg-dir."""
    if args.gene_files:
        return [Path(item.strip()) for item in args.gene_files.split(",") if item.strip()]

    deg_dir = Path(args.deg_dir)
    files = []
    for pattern in ("*.txt", "*.csv", "*.tsv", "*.xlsx", "*.xls"):
        files.extend(sorted(deg_dir.glob(pattern)))
    return files


def _read_gene_ids_for_annotation(file_path: Path) -> List[str]:
    """Read gene IDs from one-gene-per-line or tabular files with a Gene column."""
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        try:
            df = pd.read_csv(file_path)
            if "Gene" in df.columns:
                return df["Gene"].dropna().astype(str).tolist()
        except Exception:
            pass
        with open(file_path, "r") as handle:
            return [line.strip() for line in handle if line.strip() and not line.startswith("#")]

    if "Gene" not in df.columns:
        raise ValueError(f"Annotation input file must contain a 'Gene' column or one gene ID per line: {file_path}")
    return df["Gene"].dropna().astype(str).tolist()


def _prepare_annotation_inputs(gene_files: List[Path], outdir: Path, dryrun: bool) -> List[Dict[str, object]]:
    """Prepare GO txt and KEGG csv inputs for annotation modules."""
    prepared = []
    prep_dir = outdir / "annotation_inputs"
    if not dryrun:
        prep_dir.mkdir(parents=True, exist_ok=True)

    for file_path in gene_files:
        genes = pd.Series(_read_gene_ids_for_annotation(file_path)).dropna().astype(str).drop_duplicates()
        stem = file_path.stem
        go_file = prep_dir / f"{stem}_genes.txt"
        kegg_file = prep_dir / f"{stem}_genes.csv"

        if not dryrun:
            genes.to_csv(go_file, index=False, header=False)
            pd.DataFrame({"Gene": genes}).to_csv(kegg_file, index=False)

        prepared.append(
            {
                "source": file_path,
                "go_file": go_file,
                "kegg_file": kegg_file,
                "gene_count": len(genes),
            }
        )

    return prepared
