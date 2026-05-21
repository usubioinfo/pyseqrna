"""
PySeqRNA CLI Sample Clustering Command

This module implements the standalone sample clustering subcommand for the PySeqRNA CLI.
It performs hierarchical or k-means clustering on gene expression count matrices.

Features:
    - Standalone command-line runner for sample clustering
    - Supports hierarchical and k-means clustering algorithms
    - Filters expression matrix using variance or minimum mean expression thresholds
    - Configures row/column scaling, log2 transforms, and custom color maps
    - Generates clustering heatmaps and exports cluster assignments

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib
    - Internal modules: pyseqrna.modules.clustering

Classes / Functions / Exceptions:
    - run_clustering_subcommand: Standard entry point function for running standalone sample clustering.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path


def run_clustering_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone sample clustering from argparse-compatible args."""
    from ...modules.clustering import ClusteringAnalyzer

    matrix_file = args.matrix
    outdir = Path(args.outdir)

    if not args.dryrun:
        outdir.mkdir(parents=True, exist_ok=True)
        pyseqrna_logger.info(f"Created clustering output directory: {outdir}")
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create clustering output directory: {outdir}")

    pyseqrna_logger.info("Running standalone clustering analysis")
    pyseqrna_logger.info(f"Matrix file: {matrix_file}")
    pyseqrna_logger.info(f"Clustering target: {args.cluster_target}")
    pyseqrna_logger.info(f"Clustering method: {args.cluster_method}")

    analyzer = ClusteringAnalyzer(
        matrix_file=matrix_file,
        out_dir=str(outdir),
        gene_column=args.gene_column,
        logger=pyseqrna_logger.logger,
        dryrun=args.dryrun,
    )
    analyzer.run(
        cluster_target=args.cluster_target,
        method=args.cluster_method,
        n_clusters=args.cluster_count,
        metric=args.metric,
        linkage_method=args.linkage,
        top_variable=None if args.top_variable == 0 else args.top_variable,
        min_mean=args.min_mean,
        log_transform=not args.no_log,
        scale=args.scale,
        heatmap=not args.no_heatmap,
        color_map=args.cmap,
        prefix=args.prefix,
    )

    pyseqrna_logger.info("Standalone clustering completed successfully")
    return 0
