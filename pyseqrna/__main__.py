#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Main Entry Point

This module serves as the main entry point for the PySeqRNA package,
parsing subcommand arguments and launching the pipeline or specific analyses.

Features:
    - Main package entry point and banner display
    - Routing and execution of standalone subcommands (alignment, diffexp, etc.)

Functions:
    - should_print_banner: Helper to decide if banner should be displayed
    - print_pyseqrna_banner: Prints the PySeqRNA ASCII banner and system resources
    - main: Primary main execution entry point

:Created: May 20, 2021
:Updated: May 20, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import sys
import traceback
from .cli.argument_manager import ArgumentManager
from .pipeline import Pipeline
from .utils import LogManager
from .utils.supported_species import SupportedSpecies
from .__version__ import __version__
import numpy as np


def should_print_banner(argv=None) -> bool:
    """Return True when the CLI should print the startup banner."""
    args = argv if argv is not None else sys.argv[1:]
    help_flags = {"-h", "--help"}
    return not any(arg in help_flags for arg in args)


def print_pyseqrna_banner():
    """Print the PySeqRNA banner with system information."""
    import platform
    import psutil

    # Get system information
    os_name = platform.system()
    os_version = platform.release()
    python_version = platform.python_version()
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)
    available_memory_gb = memory.available / (1024**3)
    memory_used_percent = memory.percent

    banner = f"""
           ----------------------------------------------------------
                                PySeqRNA {__version__}
                Written by Naveen Duhan (naveen.duhan@outlook.com),
               Kaundal Bioinformatics Lab, Utah State University,
            Released under the terms of GNU General Public License v3
           ----------------------------------------------------------

           System Information
           -----------------
           OS: {os_name} {os_version}
           Python Version: {python_version}
           CPU Cores: {cpu_count_physical} (Physical), {cpu_count_logical} (Logical)
           Total Memory: {total_memory_gb:.1f} GB
           Memory Available: {available_memory_gb:.1f} GB
           Memory Used: {memory_used_percent:.1f}%
            ----------------------------------------------------------
    """
    print(banner, flush=True)


def main():
    """Main entry point for PySeqRNA."""
    pyseqrna_logger = None
    try:
        subcommand = sys.argv[1] if len(sys.argv) > 1 else None
        is_diffexp_subcommand = subcommand == "diffexp"
        is_alignment_subcommand = subcommand == "alignment"
        is_quantification_subcommand = subcommand == "quantification"
        is_normalization_subcommand = subcommand == "normalization"
        is_visualization_subcommand = subcommand == "visualization"
        is_annotation_subcommand = subcommand == "annotation"
        is_clustering_subcommand = subcommand == "clustering"
        is_report_subcommand = subcommand == "report"

        # Keep help output focused on usage instead of runtime system information.
        print_runtime_banners = should_print_banner()
        if print_runtime_banners:
            print_pyseqrna_banner()

        argument_manager = ArgumentManager()

        # Initialize logger AFTER banner
        pyseqrna_logger = LogManager(print_end_banner=print_runtime_banners)

        if is_diffexp_subcommand:
            args = argument_manager.parse_diffexp_args()
            from .cli.commands.diffexp import run_diffexp_subcommand

            sys.exit(run_diffexp_subcommand(args, pyseqrna_logger))

        if is_alignment_subcommand:
            args = argument_manager.parse_alignment_args()
            from .cli.commands.alignment import run_alignment_subcommand

            sys.exit(run_alignment_subcommand(args, pyseqrna_logger))

        if is_quantification_subcommand:
            args = argument_manager.parse_quantification_args()
            from .cli.commands.quantification import run_quantification_subcommand

            sys.exit(run_quantification_subcommand(args, pyseqrna_logger))

        if is_normalization_subcommand:
            args = argument_manager.parse_normalization_args()
            from .cli.commands.normalization import run_normalization_subcommand

            sys.exit(run_normalization_subcommand(args, pyseqrna_logger))

        if is_visualization_subcommand:
            args = argument_manager.parse_visualization_args()
            from .cli.commands.visualization import run_visualization_subcommand

            sys.exit(run_visualization_subcommand(args, pyseqrna_logger))

        if is_annotation_subcommand:
            args = argument_manager.parse_annotation_args()
            from .cli.commands.annotation import run_annotation_subcommand

            sys.exit(run_annotation_subcommand(args, pyseqrna_logger))

        if is_clustering_subcommand:
            args = argument_manager.parse_clustering_args()
            from .cli.commands.clustering import run_clustering_subcommand

            sys.exit(run_clustering_subcommand(args, pyseqrna_logger))

        if is_report_subcommand:
            args = argument_manager.parse_report_args()
            from .cli.commands.report import run_report_subcommand

            sys.exit(run_report_subcommand(args, pyseqrna_logger))

        # Parse full-pipeline arguments
        args = argument_manager.parse_args()

        # Handle organism display request
        if args.organism:
            SupportedSpecies().display_organisms()

        # Validate species if provided
        if args.species:
            species_validator = SupportedSpecies(logger=pyseqrna_logger.logger)
            species_validator.validate_options(args)

        # Create and run pipeline
        pipeline = Pipeline(
            input_file=args.input_file,
            samples_path=args.samples_path,
            reference_genome=args.reference_genome,
            feature_file=args.feature_file,
            output_dir=args.outdir,
            threads=args.threads,
            memory=args.memory,
            local_jobs=args.local_jobs,
            dryrun=args.dryrun,
            force=args.force,
            resume=args.resume,
            resume_policy=args.resume_policy,
            config_file=args.config_file,
            param_dir=args.param_dir,
            logger=pyseqrna_logger,
            force_paired=args.paired,
            force_single_end=False,
            species=args.species,
            organism_type=args.organism_type,
            source=args.source,
            skip_quality=args.skip_quality,
            quality_tool=args.quality_tool,
            quality_trim=args.quality_trim,
            skip_trim=args.skip_trim,
            trimming_tool=args.trimming_tool,
            skip_alignment=args.skip_alignment,
            alignment_tool=args.alignment_tool,
            alignment_stats=args.alignment_stats,
            alignment_stats_source=args.alignment_stats_source,
            quant_method=args.quant_method,
            skip_quantification=args.skip_quantification,
            normalize_counts=not args.skip_normalization,
            normalization_method=args.normalization_method,
            skip_normalization_plots=args.skip_normalization_plots,
            run_multimapped_groups=args.run_multimapped_groups,
            mmg_min_count=args.mmg_min_count,
            mmg_percent_sample=args.mmg_percent_sample,
            mmg_feature=args.mmg_feature,
            mmg_min_overlap=args.mmg_min_overlap,
            mmg_fraction_overlap=args.mmg_fraction_overlap,
            mmg_include_ambiguous_unique=not args.mmg_no_ambiguous_unique,
            mmg_collapse_contained_groups=not args.mmg_no_collapse_contained,
            skip_diffexp=args.skip_diffexp,
            diffexp_tool=args.diffexp_tool,
            diffexp_normalization=args.diffexp_normalization,
            diffexp_abundance=args.diffexp_abundance,
            diffexp_dispersion=args.diffexp_dispersion,
            diffexp_test=args.diffexp_test,
            fdr_threshold=args.fdr_threshold,
            log2fc_threshold=np.log2(args.fold_threshold) if args.fold_threshold > 0 else 0.0,
            pvalue_threshold=args.pvalue_threshold,
            subset=args.subset,
            add_gene_names=args.add_gene_names,
            skip_functional_annotation=args.skip_functional_annotation,
            gene_ontology=args.gene_ontology,
            kegg_pathway=args.kegg_pathway,
            go_pvalue_threshold=args.go_pvalue_threshold,
            kegg_pvalue_threshold=args.kegg_pvalue_threshold,
            run_clustering=args.run_clustering,
            cluster_target=args.cluster_target,
            cluster_method=args.cluster_method,
            cluster_count=args.cluster_count,
            cluster_metric=args.cluster_metric,
            cluster_linkage=args.cluster_linkage,
            cluster_top_variable=args.cluster_top_variable,
            cluster_scale=args.cluster_scale,
            cluster_no_log=args.cluster_no_log,
            cluster_no_heatmap=args.cluster_no_heatmap,
            cluster_cmap=args.cluster_cmap,
            run_coexpression=args.run_coexpression,
            coexpression_tool=args.coexpression_tool,
            coexpression_tightness=args.coexpression_tightness,
            coexpression_k_values=args.coexpression_k_values,
            coexpression_outlier=args.coexpression_outlier,
            coexpression_cluster_size=args.coexpression_cluster_size,
            coexpression_replicates=args.coexpression_replicates,
            coexpression_preprocessing=args.coexpression_preprocessing,
            pca_plot=args.pca_plot,
            tsne_plot=args.tsne_plot,
            volcano_plot=args.volcano_plot,
            ma_plot=args.ma_plot,
            deg_heatmap=args.deg_heatmap,
            heatmap_top_genes=args.heatmap_top_genes,
            venn=args.venn,
            upset=args.upset,
            venn_comparisons=args.venn_comparisons,
            venn_label=args.venn_label,
            skip_report=args.skip_report,
            report_formats=args.report_formats,
            report_title=args.report_title,
            slurm=args.slurm,
            slurm_partition=args.slurm_partition,
            slurm_account=args.slurm_account,
            slurm_time=args.slurm_time,
            slurm_email=args.slurm_email,
            slurm_qos=args.slurm_qos,
            slurm_array_max_parallel=args.slurm_array_max_parallel,
            slurm_cpus_per_task=args.slurm_cpus_per_task,
            slurm_memory_per_task=args.slurm_memory_per_task,
            slurm_wait_timeout_hours=args.slurm_wait_timeout_hours,
        )

        # Run pipeline
        success = pipeline.run()
        sys.exit(0 if success else 1)

    except Exception as e:
        traceback.print_exc()
        print(f"\nPySeqRNA failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if pyseqrna_logger is not None:
            pyseqrna_logger.close_logger()


if __name__ == "__main__":
    main()
