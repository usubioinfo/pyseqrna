"""
PySeqRNA CLI Sequence Alignment Command

This module implements the standalone sequence alignment command-line interface. It coordinates
reference indexing, raw read quality checks, trimming, alignment, and final stats generation.

Features:
    - Standalone sequence alignment using STAR, HISAT2, Bowtie2, BWA, or Minimap2
    - Optional raw read quality control using FastQC
    - Optional adapter and quality trimming using Trim Galore, Flexbar, or Trimmomatic
    - Automatic mapping and resolution of raw/trimmed read paths and sample identifiers
    - Multi-sample parallel execution and SLURM job submission configurations
    - Alignment statistics collection and summaries

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib
    - Internal modules: pyseqrna.cli.utils, pyseqrna.modules.alignment, pyseqrna.modules.quality, pyseqrna.modules.trimming

Classes / Functions / Exceptions:
    - run_alignment_subcommand: Standard entry point function for running standalone read alignment and quality control.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from ..utils import (
    _build_slurm_config,
    _build_sample_dict_from_reads,
    _resolve_trimmed_files,
    _build_alignment_target,
)


def run_alignment_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone read alignment."""
    from ...modules.alignment import (
        AlignmentStats,
        create_aligner,
        get_available_aligners,
    )
    from ...modules.quality import create_quality_control, get_available_quality_tools
    from ...modules.trimming import (
        TrimmingStats,
        create_trimmer,
        get_available_trimmers,
    )
    from ...utils import InputProcessor
    from ...utils.dry_run_manager import DryRunManager

    input_path = Path(args.input_file)
    samples_path = Path(args.samples_path)
    reference_path = Path(args.reference_genome)
    feature_path = Path(args.feature_file) if args.feature_file else None

    missing = []
    for label, path in [
        ("input file", input_path),
        ("samples directory", samples_path),
        ("reference genome", reference_path),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    if feature_path and not feature_path.exists():
        missing.append(f"feature file: {feature_path}")
    if missing:
        raise ValueError("Required alignment input(s) not found: " + "; ".join(missing))

    available_aligners = get_available_aligners()
    if args.alignment_tool not in available_aligners:
        raise ValueError(f"Unsupported alignment tool: {args.alignment_tool}. Available: {', '.join(available_aligners)}")

    if args.run_quality_trimming:
        args.run_quality = True
        args.run_trimming = True
        args.quality_trim = True

    if args.trimmed_dir and args.run_trimming:
        raise ValueError("Use either --trimmed-dir or --run-trimming, not both")

    if args.run_quality and args.quality_tool not in get_available_quality_tools():
        raise ValueError(f"Unsupported quality tool: {args.quality_tool}")

    if args.run_trimming and args.trimming_tool not in get_available_trimmers():
        raise ValueError(f"Unsupported trimming tool: {args.trimming_tool}")

    outdir = Path(args.outdir)
    if not args.dryrun:
        outdir.mkdir(parents=True, exist_ok=True)
        pyseqrna_logger.info(f"Created alignment output directory: {outdir}")
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create alignment output directory: {outdir}")

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
                str(outdir / "alignment_slurm_config.ini"),
                slurm_config,
            )

    pyseqrna_logger.info("Running standalone read alignment")
    pyseqrna_logger.info(f"Alignment tool: {args.alignment_tool}")
    pyseqrna_logger.info(f"Input sample sheet: {input_path}")
    pyseqrna_logger.info(f"Samples directory: {samples_path}")
    pyseqrna_logger.info(f"Reference genome: {reference_path}")
    if feature_path:
        pyseqrna_logger.info(f"Feature file: {feature_path}")

    input_processor = InputProcessor(pyseqrna_logger.logger)
    sample_data = input_processor.process_sample_file(
        str(input_path),
        str(samples_path),
        paired=args.paired,
    )
    sample_dict = sample_data["samples"]
    paired = bool(sample_data.get("paired", args.paired))
    pyseqrna_logger.info(f"Processed {len(sample_dict)} samples for alignment")
    pyseqrna_logger.info(f"Effective paired-end mode: {paired}")

    selected_read_dict = None
    if args.run_quality:
        pyseqrna_logger.info(f"Running pre-alignment quality control with {args.quality_tool}")
        quality_module = create_quality_control(
            tool_name=args.quality_tool,
            sample_dict=sample_dict,
            out_dir=str(outdir),
            param_dir=args.param_dir,
            paired=paired,
            slurm=args.slurm,
            dryrun=args.dryrun,
            cpu_threads=args.threads,
            logger=pyseqrna_logger.logger,
            dry_run_manager=dry_run_manager,
            slurm_config=slurm_config,
        )
        quality_results = quality_module.run()
        pyseqrna_logger.info(f"{args.quality_tool} quality control completed for {len(quality_results)} samples")

    if args.run_trimming:
        pyseqrna_logger.info(f"Running read trimming with {args.trimming_tool}")
        trimming_module = create_trimmer(
            trimmer_name=args.trimming_tool,
            sample_dict=sample_dict,
            out_dir=str(outdir),
            param_dir=args.param_dir,
            paired=paired,
            slurm=args.slurm,
            dryrun=args.dryrun,
            cpu_threads=args.threads,
            logger=pyseqrna_logger.logger,
            dry_run_manager=dry_run_manager,
            slurm_config=slurm_config,
        )
        selected_read_dict = trimming_module.run()
        pyseqrna_logger.info(f"{args.trimming_tool} trimming completed for {len(selected_read_dict)} samples")

        if args.dryrun:
            pyseqrna_logger.info("DRYRUN: Would calculate trimming statistics for trimmed reads")
        else:
            pyseqrna_logger.info("Calculating trimming statistics")
            trim_stats_dir = outdir / "1.Quality_and_trimming" / "trimming_stats"
            trim_stats_module = TrimmingStats(
                samples_dict=sample_dict,
                trimmed_dict=selected_read_dict,
                out_dir=str(trim_stats_dir),
                paired=paired,
                cpu_threads=args.threads,
                logger=pyseqrna_logger.logger,
                dryrun=args.dryrun,
                dry_run_manager=dry_run_manager,
            )
            trimming_stats = trim_stats_module.run()
            trim_summary = trim_stats_module.summarize_results(trimming_stats)
            pyseqrna_logger.info("Trimming statistics summary:")
            for line in trim_summary.split("\n"):
                if line.strip():
                    pyseqrna_logger.info(line)

        if args.quality_trim:
            pyseqrna_logger.info(f"Running post-trimming quality control with {args.quality_tool}")
            trimmed_sample_dict = _build_sample_dict_from_reads(sample_dict, selected_read_dict, paired)
            quality_trim_module = create_quality_control(
                tool_name=args.quality_tool,
                sample_dict=trimmed_sample_dict,
                out_dir=str(outdir),
                param_dir=args.param_dir,
                paired=paired,
                slurm=args.slurm,
                dryrun=args.dryrun,
                cpu_threads=args.threads,
                logger=pyseqrna_logger.logger,
                dry_run_manager=dry_run_manager,
                slurm_config=slurm_config,
            )
            quality_trim_module.name = f"{args.quality_tool}_trim"
            quality_trim_results = quality_trim_module.run()
            pyseqrna_logger.info(
                f"Post-trimming {args.quality_tool} quality control completed for {len(quality_trim_results)} samples"
            )

    if args.trimmed_dir:
        trimmed_dir = Path(args.trimmed_dir)
        pyseqrna_logger.info(f"Resolving existing trimmed reads from: {trimmed_dir}")
        selected_read_dict = _resolve_trimmed_files(
            sample_dict=sample_dict,
            trimmed_dir=trimmed_dir,
            paired=paired,
            pattern_r1=args.trimmed_pattern_r1,
            pattern_r2=args.trimmed_pattern_r2,
            pattern_single=args.trimmed_pattern_single,
        )
        pyseqrna_logger.info(f"Resolved trimmed reads for {len(selected_read_dict)} samples")

    aligner = create_aligner(
        aligner_name=args.alignment_tool,
        genome=str(reference_path),
        out_dir=str(outdir),
        param_dir=args.param_dir,
        logger=pyseqrna_logger.logger,
        dryrun=args.dryrun,
        cpu_threads=args.threads,
        slurm=args.slurm,
        dry_run_manager=dry_run_manager,
        slurm_config=slurm_config,
    )

    if not aligner.check_index():
        pyseqrna_logger.info(f"Building {args.alignment_tool} index")
        aligner.build_index(gff=str(feature_path) if feature_path else None)
        pyseqrna_logger.info(f"{args.alignment_tool} index build step completed")

    if not args.dryrun and not aligner.check_index():
        raise RuntimeError(f"{args.alignment_tool} index was not found after index build")

    alignment_target = _build_alignment_target(sample_dict, paired, selected_read_dict)
    if selected_read_dict:
        pyseqrna_logger.info("Using trimmed reads for alignment")
    else:
        pyseqrna_logger.info("Using raw input reads for alignment")

    results = aligner.run_alignment(target=alignment_target, paired=paired)
    pyseqrna_logger.info(f"{args.alignment_tool} alignment completed successfully")
    pyseqrna_logger.info(f"{args.alignment_tool} results: {len(results)} samples processed")

    if not args.skip_stats:
        pyseqrna_logger.info("Calculating alignment statistics")
        stats_dir = outdir / "2.Alignment" / "alignment_stats"
        stats_module = AlignmentStats(
            sample_dict=sample_dict,
            trimmed_dict=selected_read_dict,
            bam_dict=results,
            trimming_stats=None,
            out_dir=str(stats_dir),
            cpu_threads=args.threads,
            paired=paired,
            logger=pyseqrna_logger.logger,
            dryrun=args.dryrun,
            dry_run_manager=dry_run_manager,
            source=args.alignment_stats_source,
            alignment_tool=args.alignment_tool,
        )
        alignment_stats = stats_module.run()
        summary = stats_module.summarize_results(alignment_stats)
        pyseqrna_logger.info("Alignment statistics summary:")
        for line in summary.split("\n"):
            if line.strip():
                pyseqrna_logger.info(line)
    else:
        pyseqrna_logger.info("Skipping alignment statistics as requested")

    pyseqrna_logger.info("Standalone alignment completed successfully")
    return 0
