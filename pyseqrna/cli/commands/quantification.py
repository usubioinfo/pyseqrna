"""
PySeqRNA CLI Gene Expression Quantification Command

This module implements the standalone gene expression quantification subcommand for the PySeqRNA CLI.
It resolves aligned BAM files and runs FeatureCounts, HTSeq, or Genomic Overlaps, with optional multimapped groups analysis.

Features:
    - Standalone command-line runner for gene/transcript expression quantification
    - Interfaces with FeatureCounts, HTSeq, and Genomic Overlaps quantification backends
    - Optional multimapped groups (MMG) expression modeling and filtering
    - Resolves and verifies coordinate-sorted BAM files dynamically from directories
    - Sets parallel CPU threads, memory allocations, and SLURM configurations
    - Logs summary statistics (total mapping reads, unique features, sample columns)

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib
    - Internal modules: pyseqrna.cli.utils, pyseqrna.modules.quantification, pyseqrna.modules.multimapped_groups, pyseqrna.utils

Classes / Functions / Exceptions:
    - run_quantification_subcommand: Standard entry point function for running standalone gene quantification and optional MMG analysis.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from ..utils import (
    _build_slurm_config,
    _resolve_bam_files,
)


def run_quantification_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone gene quantification and optional multimapped groups analysis."""
    from ...modules.multimapped_groups import create_multimapped_groups_analyzer
    from ...modules.quantification import create_quantifier, get_available_quantifiers
    from ...utils import InputProcessor
    from ...utils.dry_run_manager import DryRunManager

    input_path = Path(args.input_file)
    samples_path = Path(args.samples_path)
    feature_path = Path(args.feature_file)
    alignment_dir = Path(args.alignment_dir)

    missing = []
    for label, path in [
        ("input file", input_path),
        ("samples directory", samples_path),
        ("feature file", feature_path),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    if not args.dryrun and not alignment_dir.exists():
        missing.append(f"alignment/BAM directory: {alignment_dir}")
    if missing:
        raise ValueError("Required quantification input(s) not found: " + "; ".join(missing))

    quant_method = "featurecounts" if args.quant_method == "featureCounts" else args.quant_method.lower()
    available_quantifiers = get_available_quantifiers()
    if quant_method not in available_quantifiers:
        raise ValueError(
            f"Unsupported quantification method: {args.quant_method}. Available: {', '.join(available_quantifiers)}"
        )

    outdir = Path(args.outdir)
    quantification_dir = outdir / "3.Quantification"
    if not args.dryrun:
        quantification_dir.mkdir(parents=True, exist_ok=True)
        pyseqrna_logger.info(f"Created quantification output directory: {quantification_dir}")
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create quantification output directory: {quantification_dir}")

    dry_run_manager = DryRunManager(enabled=args.dryrun, logger=pyseqrna_logger.logger)
    slurm_config = _build_slurm_config(args)
    if args.slurm:
        pyseqrna_logger.info(
            "SLURM mode enabled: partition=%s, time=%s, memory=%sGB, cpus=%s",
            slurm_config.get("partition"),
            slurm_config.get("time"),
            slurm_config.get("memory"),
            slurm_config.get("cpus"),
        )
        if args.dryrun:
            dry_run_manager.simulate_slurm_config_creation(
                str(outdir / "quantification_slurm_config.ini"),
                slurm_config,
            )

    pyseqrna_logger.info("Running standalone quantification")
    pyseqrna_logger.info(f"Quantification method: {quant_method}")
    pyseqrna_logger.info(f"Input sample sheet: {input_path}")
    pyseqrna_logger.info(f"Alignment/BAM directory: {alignment_dir}")
    pyseqrna_logger.info(f"Feature file: {feature_path}")

    input_processor = InputProcessor(pyseqrna_logger.logger)
    sample_data = input_processor.process_sample_file(
        str(input_path),
        str(samples_path),
        paired=args.paired,
    )
    sample_dict = sample_data["samples"]
    paired = bool(sample_data.get("paired", args.paired))
    pyseqrna_logger.info(f"Processed {len(sample_dict)} samples for quantification")
    pyseqrna_logger.info(f"Effective paired-end mode: {paired}")

    bam_files = _resolve_bam_files(
        sample_dict=sample_dict,
        alignment_dir=alignment_dir,
        alignment_tool=args.alignment_tool,
        bam_pattern=args.bam_pattern,
        dryrun=args.dryrun,
    )
    pyseqrna_logger.info(f"Resolved BAM files for {len(bam_files)} samples")

    if not args.skip_unique_counts:
        quantifier = create_quantifier(
            quantifier_name=quant_method,
            bam_dict=bam_files,
            annotation_file=str(feature_path),
            out_dir=str(quantification_dir),
            param_dir=args.param_dir,
            paired=paired,
            slurm=args.slurm,
            dryrun=args.dryrun,
            cpu_threads=args.threads,
            memory=args.memory,
            logger=pyseqrna_logger.logger,
            dry_run_manager=dry_run_manager,
            slurm_config=slurm_config,
        )
        count_matrix = quantifier.run()
        stats = quantifier.get_summary_stats(count_matrix)
        pyseqrna_logger.info("Gene quantification completed successfully")
        pyseqrna_logger.info(f"  - Tool: {quant_method}")
        pyseqrna_logger.info(f"  - Genes: {stats.get('total_genes', 0)}")
        pyseqrna_logger.info(f"  - Samples: {stats.get('total_samples', 0)}")
        pyseqrna_logger.info(f"  - Total reads: {stats.get('total_reads', 0):,}")
    else:
        pyseqrna_logger.info("Skipping normal gene count quantification as requested")

    if args.run_multimapped_groups:
        mmg_dir = quantification_dir / "multimapped_groups"
        if args.dryrun:
            pyseqrna_logger.info(f"DRYRUN: Would create multimapped groups directory: {mmg_dir}")
        else:
            mmg_dir.mkdir(parents=True, exist_ok=True)

        mmg_analyzer = create_multimapped_groups_analyzer(
            bam_files=bam_files,
            gff_file=str(feature_path),
            out_dir=str(mmg_dir),
            feature=args.mmg_feature,
            min_count=args.mmg_min_count,
            percent_sample=args.mmg_percent_sample,
            logger=pyseqrna_logger.logger,
            dryrun=args.dryrun,
            dry_run_manager=dry_run_manager,
            cpu_threads=args.threads,
            min_overlap=args.mmg_min_overlap,
            fraction_overlap=args.mmg_fraction_overlap,
            include_ambiguous_unique=not args.mmg_no_ambiguous_unique,
            collapse_contained_groups=not args.mmg_no_collapse_contained,
        )
        mmg_results = mmg_analyzer.run()
        mmg_stats = mmg_analyzer.get_summary_stats(mmg_results)
        pyseqrna_logger.info("Multimapped groups analysis completed successfully")
        pyseqrna_logger.info(f"  - Groups: {mmg_stats.get('total_groups', 0)}")
        pyseqrna_logger.info(f"  - Samples: {mmg_stats.get('total_samples', 0)}")

    pyseqrna_logger.info("Standalone quantification completed successfully")
    return 0
