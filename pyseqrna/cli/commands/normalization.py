"""
PySeqRNA CLI Counts Normalization Command

This module implements the standalone counts normalization subcommand for the PySeqRNA CLI.
It normalizes raw RNA-seq expression counts using CPM, RPKM, TPM, FPKM, TMM, or Median Ratio methods.

Features:
    - Standalone command-line runner for expression counts normalization
    - Supports multiple algorithms (CPM, RPKM, TPM, FPKM, TMM, Median Ratio)
    - Validates inputs including count matrices, gene columns, and annotation structures
    - Performs diagnostic plot generation to visualize normalized distributions
    - Outputs summary metrics such as normalized gene counts and average sample depths

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib
    - Internal modules: pyseqrna.modules.normalization, pyseqrna.utils.dry_run_manager

Classes / Functions / Exceptions:
    - run_normalization_subcommand: Standard entry point function for running standalone count normalization.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path


def run_normalization_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone count normalization."""
    from ...modules.normalization import create_normalizer, get_available_normalizers
    from ...utils.dry_run_manager import DryRunManager

    count_path = Path(args.counts)
    feature_path = Path(args.feature_file) if args.feature_file else None

    missing = []
    if not count_path.exists():
        missing.append(f"count matrix: {count_path}")
    if feature_path and not feature_path.exists():
        missing.append(f"feature file: {feature_path}")
    if missing:
        raise ValueError("Required normalization input(s) not found: " + "; ".join(missing))

    available_normalizers = get_available_normalizers()
    if args.normalization_method not in available_normalizers:
        raise ValueError(
            f"Unsupported normalization method: {args.normalization_method}. Available: {', '.join(available_normalizers)}"
        )

    outdir = Path(args.outdir)
    normalization_dir = outdir / "4.Normalization"
    if not args.dryrun:
        normalization_dir.mkdir(parents=True, exist_ok=True)
        pyseqrna_logger.info(f"Created normalization output directory: {normalization_dir}")
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create normalization output directory: {normalization_dir}")

    pyseqrna_logger.info("Running standalone count normalization")
    pyseqrna_logger.info(f"Normalization method: {args.normalization_method}")
    pyseqrna_logger.info(f"Count matrix: {count_path}")
    if feature_path:
        pyseqrna_logger.info(f"Feature file: {feature_path}")
    pyseqrna_logger.info(f"Gene column: {args.gene_column}")

    dry_run_manager = DryRunManager(enabled=args.dryrun, logger=pyseqrna_logger.logger)
    normalizer = create_normalizer(
        normalizer_name=args.normalization_method,
        count_matrix_file=str(count_path),
        annotation_file=str(feature_path) if feature_path else None,
        out_dir=str(normalization_dir),
        gene_column=args.gene_column,
        logger=pyseqrna_logger.logger,
        dryrun=args.dryrun,
        dry_run_manager=dry_run_manager,
    )

    normalized_data = normalizer.run(
        plot=not args.skip_plots,
        save_results=not args.no_save,
    )
    stats = normalizer.get_summary_statistics(normalized_data)

    pyseqrna_logger.info("Standalone normalization completed successfully")
    pyseqrna_logger.info(f"  - Method: {args.normalization_method}")
    pyseqrna_logger.info(f"  - Genes: {stats.get('total_genes', 0)}")
    pyseqrna_logger.info(f"  - Samples: {stats.get('total_samples', 0)}")
    pyseqrna_logger.info(f"  - Mean counts per sample: {stats.get('mean_counts_per_sample', 0):.2f}")
    return 0
