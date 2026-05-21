"""
PySeqRNA CLI Visualization Plots Command

This module implements the standalone visualization CLI subcommand, generating
differential expression charts like volcano, MA, heatmaps, and Venn diagrams.

Features:
    - Standalone rendering of expression analysis graphs from differential matrices
    - Customizable labels, thresholds, and dimensions

Functions:
    - run_visualization_subcommand: Standard entry point for standalone visualizations

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
import numpy as np


def run_visualization_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone visualization."""
    from ...modules.visualization import Visualization
    from ...utils import InputProcessor
    from ...utils.dry_run_manager import DryRunManager

    norm_counts_file = Path(args.normalized_counts) if args.normalized_counts else None
    de_results_file = Path(args.de_results) if args.de_results else None
    filtered_deg_file = Path(args.filtered_degs) if args.filtered_degs else None

    missing = []
    if norm_counts_file and not norm_counts_file.exists() and not args.dryrun:
        missing.append(f"normalized counts: {norm_counts_file}")
    if de_results_file and not de_results_file.exists() and not args.dryrun:
        missing.append(f"DE results: {de_results_file}")
    if filtered_deg_file and not filtered_deg_file.exists() and not args.dryrun:
        missing.append(f"filtered DEGs: {filtered_deg_file}")
    if args.input_file and not Path(args.input_file).exists():
        missing.append(f"input file: {args.input_file}")
    if args.samples_path and not Path(args.samples_path).exists():
        missing.append(f"samples directory: {args.samples_path}")
    if missing:
        raise ValueError("Required visualization input(s) not found: " + "; ".join(missing))

    outdir = Path(args.outdir)
    if not args.dryrun:
        outdir.mkdir(parents=True, exist_ok=True)
        pyseqrna_logger.info(f"Created visualization output directory: {outdir}")
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create visualization output directory: {outdir}")

    sample_dict = None
    if args.input_file:
        input_processor = InputProcessor(pyseqrna_logger.logger)
        sample_data = input_processor.process_sample_file(
            args.input_file,
            args.samples_path,
            paired=args.paired,
        )
        sample_dict = sample_data["samples"]
        pyseqrna_logger.info(f"Loaded sample metadata for {len(sample_dict)} samples")

    pyseqrna_logger.info("Running standalone visualization")
    if norm_counts_file:
        pyseqrna_logger.info(f"Normalized counts: {norm_counts_file}")
    if de_results_file:
        pyseqrna_logger.info(f"DE results: {de_results_file}")
    if filtered_deg_file:
        pyseqrna_logger.info(f"Filtered DEGs: {filtered_deg_file}")

    visualizer = Visualization(
        outdir=str(outdir),
        logger=pyseqrna_logger.logger,
        dryrun=args.dryrun,
        dry_run_manager=DryRunManager(enabled=args.dryrun, logger=pyseqrna_logger.logger),
    )
    visualizer.run(
        norm_counts_file=str(norm_counts_file) if norm_counts_file else None,
        de_results_file=str(de_results_file) if de_results_file else None,
        filtered_deg_file=str(filtered_deg_file) if filtered_deg_file else None,
        sample_dict=sample_dict,
        log2fc_threshold=np.log2(args.fold_threshold),
        fdr_threshold=args.fdr_threshold,
        venn=not args.no_venn,
        venn_comparisons=[item.strip() for item in args.venn_comparisons.split(",")] if args.venn_comparisons else None,
        venn_label=args.venn_label,
        upset=not args.no_upset,
    )

    pyseqrna_logger.info("Standalone visualization completed successfully")
    return 0
