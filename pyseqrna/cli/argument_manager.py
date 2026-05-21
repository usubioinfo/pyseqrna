#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA CLI Argument Manager

This module provides command-line argument management for the PySeqRNA package. It defines
and configures parsers for the main pipeline as well as individual standalone subcommands
(such as alignment, quantification, differential expression, normalization, and functional annotation).

Features:
    - Defines centralized command-line argument schemas for main entry point and subcommands
    - Groups arguments logically (e.g., General, Species, Quality Control, Alignment, SLURM, etc.)
    - Parses and updates default parameter structures using INI-based runconfig files
    - Validates required inputs and parameter ranges to raise early command errors
    - Supports interactive help output formatting for terminal usage

Configuration:
    - Configured via sys.argv inputs and custom INI runconfig files passed to the program.

Dependencies:
    - Python packages: argparse, configparser, sys, itertools, textwrap
    - Internal modules: pyseqrna.__version__

Classes / Functions / Exceptions:
    - ArgumentManager: Manages and validates command-line arguments for the main pipeline and all subcommands.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import argparse
import configparser
import sys
from itertools import combinations
from textwrap import dedent

from pyseqrna.__version__ import __version__


class ArgumentManager:
    """
    A class to manage command-line arguments for the pySeqRNA package.
    """

    def __init__(self):
        """Initialize the ArgumentManager."""
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description=f"pyseqrna {__version__}: a Python-based RNAseq data analysis package",
            usage=(
                "%(prog)s input_file samples_path reference_genome feature_file [options]\n"
                "       %(prog)s <subcommand> [options]"
            ),
            epilog=dedent("""\
                Available Subcommands (run 'pyseqrna <subcommand> --help' for details):
                  alignment       Run read alignment standalone
                  annotation      Run functional annotation enrichment (GO/KEGG) standalone
                  clustering      Run sample similarity clustering standalone
                  diffexp         Run differential expression analysis standalone
                  normalization   Run count normalization standalone
                  quantification  Run gene expression quantification standalone
                  report          Generate comprehensive report standalone
                  visualization   Generate analysis plots standalone

                Written by Naveen Duhan (naveen.duhan@usu.edu),
                Kaundal Bioinformatics Lab, Utah State University,
                Released under the terms of GNU General Public License v3
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Add mandatory arguments
        self._add_mandatory_arguments(parser)

        # Add implemented arguments in organized order
        self._add_general_arguments(parser)
        self._add_species_arguments(parser)
        self._add_quality_control_arguments(parser)
        self._add_alignment_arguments(parser)
        self._add_quantification_arguments(parser)
        self._add_differential_expression_arguments(parser)
        self._add_visualization_arguments(parser)
        self._add_functional_annotation_arguments(parser)
        self._add_report_arguments(parser)
        self._add_computational_arguments(parser)
        self._add_configuration_arguments(parser)
        self._add_slurm_arguments(parser)

        return parser

    def _add_mandatory_arguments(self, parser):
        """Add mandatory arguments to the parser."""
        import sys

        # Only add mandatory arguments if special flags are not present
        if "--organism" not in sys.argv:
            mandatory = parser.add_argument_group("Required Arguments")
            mandatory.add_argument(
                "input_file",
                type=str,
                nargs="?",
                help="Tab-delimited file containing sample information",
            )
            mandatory.add_argument(
                "samples_path",
                type=str,
                nargs="?",
                help="Directory containing raw reads",
            )
            mandatory.add_argument(
                "reference_genome",
                type=str,
                nargs="?",
                help="Path to the reference genome file",
            )
            mandatory.add_argument("feature_file", type=str, nargs="?", help="Path to the GTF/GFF file")

    def _add_general_arguments(self, parser):
        """Add general arguments to the parser."""
        general = parser.add_argument_group("General Arguments")
        general.add_argument(
            "--outdir",
            default="pySeqRNA_results",
            help="Output directory name to store results [default: %(default)s]",
        )
        general.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Run in dry-run mode (show commands without executing) [default: %(default)s]",
        )
        general.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Force overwrite existing output directory without asking [default: %(default)s]",
        )

        # Paired-end detection and validation
        general.add_argument(
            "--paired",
            action="store_true",
            default=False,
            help="Enable paired-end mode for analysis [default: %(default)s]",
        )

        # Version information
        general.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna (version {__version__})",
            help="Show version information and exit",
        )

        # Show supported organisms
        general.add_argument(
            "--organism",
            action="store_true",
            help="Display supported organisms for functional annotation and exit",
        )

    def _add_species_arguments(self, parser):
        """Add species and annotation arguments to the parser."""
        species = parser.add_argument_group("Species and Gene Annotation Arguments")
        species.add_argument(
            "--species",
            type=str,
            default=None,
            help="Species name for functional annotation (e.g., athaliana for Arabidopsis thaliana). Use --organism to see supported species.",
        )
        species.add_argument(
            "--organism-type",
            choices=["plants", "animals"],
            default="plants",
            help="Type of organism - plants or animals [default: %(default)s]",
        )
        species.add_argument(
            "--source",
            choices=["ENSEMBL", "NCBI"],
            default="ENSEMBL",
            help="Source database for reference and feature files [default: %(default)s]",
        )

    def _add_quality_control_arguments(self, parser):
        """Add quality control arguments to the parser."""
        qc = parser.add_argument_group("Quality Control Arguments")
        qc.add_argument(
            "--skip-quality",
            action="store_true",
            default=False,
            help="Skip quality control on raw reads (quality control runs by default) [default: %(default)s]",
        )

        qc.add_argument(
            "--quality-tool",
            type=str,
            default="fastqc",
            choices=["fastqc"],
            help="Quality control tool to use [default: %(default)s]",
        )
        qc.add_argument(
            "--quality-trim",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run quality control on trimmed reads [default: %(default)s]",
        )
        qc.add_argument(
            "--skip-trim",
            action="store_true",
            default=False,
            help="Skip read trimming (trimming runs by default) [default: %(default)s]",
        )
        qc.add_argument(
            "--trimming-tool",
            choices=["trim_galore", "trimmomatic", "flexbar"],
            default="trim_galore",
            help="Select trimming tool to use [default: %(default)s]",
        )

    def _add_alignment_arguments(self, parser):
        """Add alignment arguments to the parser."""
        align = parser.add_argument_group("Alignment Arguments")
        align.add_argument(
            "--skip-alignment",
            action="store_true",
            default=False,
            help="Skip read alignment (alignment runs by default) [default: %(default)s]",
        )
        align.add_argument(
            "--alignment-tool",
            choices=["star", "hisat2", "bowtie2", "bwa", "minimap2"],
            default="star",
            help="Select alignment tool to use [default: %(default)s]",
        )
        align.add_argument(
            "--alignment-stats",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate alignment statistics after BAM preparation [default: %(default)s]",
        )
        align.add_argument(
            "--alignment-stats-source",
            choices=["auto", "logs", "bam"],
            default="auto",
            help="Alignment statistics source: auto uses aligner logs when complete, then falls back to BAM [default: %(default)s]",
        )

    def _add_quantification_arguments(self, parser):
        """Add quantification arguments to the parser."""
        quant = parser.add_argument_group("Quantification Arguments")
        quant.add_argument(
            "--quant-method",
            "-Q",
            choices=["featureCounts", "htseq", "genomic_overlaps"],
            default="genomic_overlaps",
            help="Quantification method to use: genomic_overlaps (default), featureCounts, or htseq.",
        )
        quant.add_argument(
            "--skip-quantification",
            action="store_true",
            default=False,
            help="Skip gene expression quantification [default: %(default)s]",
        )

        # Add multimapped groups arguments
        quant.add_argument(
            "--run-multimapped-groups",
            action="store_true",
            default=False,
            help="Run multimapped groups analysis after alignment [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-min-count",
            type=int,
            default=100,
            help="Minimum read count per sample for multimapped groups filtering [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-percent-sample",
            type=float,
            default=0.5,
            help="Minimum percentage of samples that must meet min_count for multimapped groups [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-feature",
            type=str,
            default="gene",
            help="Feature type to extract from GFF/GTF for multimapped groups analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-min-overlap",
            type=int,
            default=1,
            help="Minimum overlapping bases required for a read to match a feature in MMG analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-fraction-overlap",
            type=float,
            default=0.0,
            help="Minimum fraction of aligned read bases that must overlap a feature in MMG analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-no-ambiguous-unique",
            action="store_true",
            default=False,
            help="Do not include uniquely mapped reads that overlap multiple genes in MMG analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-no-collapse-contained",
            action="store_true",
            default=False,
            help="Do not collapse MMGs wholly contained within larger MMGs [default: %(default)s]",
        )

        # Add normalization arguments
        quant.add_argument(
            "--skip-normalization",
            action="store_true",
            default=False,
            help="Skip count normalization after quantification [default: %(default)s]",
        )
        quant.add_argument(
            "--normalization-method",
            choices=["cpm", "rpkm", "tpm", "fpkm", "median_ratio", "tmm"],
            default="rpkm",
            help="Normalization method to use: rpkm (default), cpm, tpm, fpkm, median_ratio, or tmm [default: %(default)s]",
        )
        quant.add_argument(
            "--skip-normalization-plots",
            action="store_true",
            default=False,
            help="Skip creating comparison plots during normalization [default: %(default)s]",
        )

        cluster = parser.add_argument_group("Clustering Arguments")
        cluster.add_argument(
            "--run-clustering",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run sample similarity clustering after normalization [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-target",
            choices=["samples"],
            default="samples",
            help="Clustering target [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-method",
            choices=["hierarchical", "kmeans"],
            default="hierarchical",
            help="Sample clustering method [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-count",
            type=int,
            default=6,
            help="Number of sample clusters [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-metric",
            default="euclidean",
            help="Distance metric for hierarchical clustering [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-linkage",
            default="average",
            help="Linkage method for hierarchical clustering [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-top-variable",
            type=int,
            default=1000,
            help="Top variable genes used for sample clustering; 0 keeps all genes [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-scale",
            choices=["row", "column", "none"],
            default="row",
            help="Scaling mode before sample clustering [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-no-log",
            action="store_true",
            default=False,
            help="Disable log2(x+1) transform before sample clustering [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-no-heatmap",
            action="store_true",
            default=False,
            help="Skip clustered sample heatmap [default: %(default)s]",
        )
        cluster.add_argument(
            "--cluster-cmap",
            default="vlag",
            help="Sample clustering heatmap color map [default: %(default)s]",
        )

        coexpression = parser.add_argument_group("Co-expression Arguments")
        coexpression.add_argument(
            "--run-coexpression",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run gene co-expression analysis after normalization [default: %(default)s]",
        )
        coexpression.add_argument(
            "--coexpression-tool",
            choices=["pycoexpression"],
            default="pycoexpression",
            help="Co-expression tool [default: %(default)s]",
        )
        coexpression.add_argument(
            "--coexpression-tightness",
            type=float,
            default=None,
            help="Outlier correlation threshold scaling factor [default: coexpression.ini]",
        )
        coexpression.add_argument(
            "--coexpression-k-values",
            default=None,
            help="Number of clusters K or range (e.g. '4' or '3 4 5') [default: coexpression.ini]",
        )
        coexpression.add_argument(
            "--coexpression-outlier",
            type=float,
            default=None,
            help="Z-score distance outlier threshold [default: coexpression.ini]",
        )
        coexpression.add_argument(
            "--coexpression-cluster-size",
            type=int,
            default=None,
            help="Minimum cluster size [default: coexpression.ini]",
        )
        coexpression.add_argument(
            "--coexpression-replicates",
            action="store_true",
            default=None,
            help="Collapse replicates by averaging [default: coexpression.ini]",
        )
        coexpression.add_argument(
            "--coexpression-preprocessing",
            action="store_true",
            default=None,
            help="Apply log2(expression + 1) transform [default: coexpression.ini]",
        )

    def _add_differential_expression_arguments(self, parser):
        """Add differential expression arguments to the parser."""
        diffexp = parser.add_argument_group("Differential Expression Arguments")
        diffexp.add_argument(
            "--skip-diffexp",
            action="store_true",
            default=False,
            help="Skip differential expression analysis [default: %(default)s]",
        )
        diffexp.add_argument(
            "--diffexp-tool",
            choices=["deseq2", "edger", "pydiffexpress"],
            default="pydiffexpress",
            help="Differential expression tool: deseq2, edger, pydiffexpress [default: %(default)s]",
        )
        diffexp.add_argument(
            "--diffexp-normalization",
            choices=["median_ratio", "poscounts", "iterate", "tmm"],
            default="median_ratio",
            help="PyDiffExpress normalization component [default: %(default)s]",
        )
        diffexp.add_argument(
            "--diffexp-abundance",
            choices=["base_mean", "ave_log_cpm"],
            default="base_mean",
            help="PyDiffExpress abundance summary component [default: %(default)s]",
        )
        diffexp.add_argument(
            "--diffexp-dispersion",
            choices=["map", "common", "trended", "tagwise"],
            default="map",
            help="PyDiffExpress dispersion component [default: %(default)s]",
        )
        diffexp.add_argument(
            "--diffexp-test",
            choices=["wald", "lrt"],
            default="wald",
            help="PyDiffExpress hypothesis-test component [default: %(default)s]",
        )
        diffexp.add_argument(
            "--fdr-threshold",
            type=float,
            default=0.05,
            help="False Discovery Rate (FDR) threshold for differential expression [default: %(default)s]",
        )
        diffexp.add_argument(
            "--fold-threshold",
            type=float,
            default=2.0,
            help="Fold change threshold for differential expression (e.g., 2.0 means 2-fold change). Internally converted to log2 [default: %(default)s]",
        )
        diffexp.add_argument(
            "--pvalue-threshold",
            type=float,
            default=0.05,
            help="P-value threshold for differential expression [default: %(default)s]",
        )
        diffexp.add_argument(
            "--add-gene-names",
            action="store_true",
            default=True,
            help="Add gene names and descriptions to differential expression results [default: %(default)s]",
        )
        diffexp.add_argument(
            "--subset",
            action="store_true",
            default=False,
            help="Subset count data for each comparison [default: %(default)s]",
        )

    def _add_functional_annotation_arguments(self, parser):
        """Add functional annotation arguments to the parser."""
        functional = parser.add_argument_group("Functional Annotation Arguments")
        functional.add_argument(
            "--skip-functional-annotation",
            action="store_true",
            default=False,
            help="Skip functional annotation analysis (Gene Ontology, KEGG pathways) [default: %(default)s]",
        )
        functional.add_argument(
            "--gene-ontology",
            action="store_true",
            default=False,
            help="Enable Gene Ontology functional enrichment analysis [default: %(default)s]",
        )
        functional.add_argument(
            "--kegg-pathway",
            action="store_true",
            default=False,
            help="Enable KEGG pathway functional enrichment analysis [default: %(default)s]",
        )
        functional.add_argument(
            "--go-pvalue-threshold",
            type=float,
            default=0.05,
            help="P-value threshold for Gene Ontology enrichment [default: %(default)s]",
        )
        functional.add_argument(
            "--kegg-pvalue-threshold",
            type=float,
            default=0.05,
            help="P-value threshold for KEGG pathway enrichment [default: %(default)s]",
        )

    def _add_visualization_arguments(self, parser):
        """Add full-pipeline visualization arguments."""
        visualization = parser.add_argument_group("Visualization Arguments")
        visualization.add_argument(
            "--pca-plot",
            dest="pca_plot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate PCA plot from normalized counts [default: %(default)s]",
        )
        visualization.add_argument(
            "--tsne-plot",
            dest="tsne_plot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate t-SNE plot from normalized counts [default: %(default)s]",
        )
        visualization.add_argument(
            "--volcano-plot",
            dest="volcano_plot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate volcano plots for differential-expression comparisons [default: %(default)s]",
        )
        visualization.add_argument(
            "--ma-plot",
            dest="ma_plot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate MA plots for differential-expression comparisons [default: %(default)s]",
        )
        visualization.add_argument(
            "--deg-heatmap",
            dest="deg_heatmap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate DEG heatmap from differential-expression results [default: %(default)s]",
        )
        visualization.add_argument(
            "--heatmap-top-genes",
            type=int,
            default=50,
            help="Number of top differential genes to use for DEG heatmap [default: %(default)s]",
        )
        visualization.add_argument(
            "--venn",
            dest="venn",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate Venn plots from filtered DEGs [default: %(default)s]",
        )
        visualization.add_argument(
            "--upset",
            dest="upset",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate UpSet-style DEG intersection plot [default: %(default)s]",
        )
        visualization.add_argument(
            "--venn-comparisons",
            default=None,
            help="Comma-separated list of 2-4 comparisons for one Venn plot; default chunks all comparisons by four",
        )
        visualization.add_argument(
            "--venn-label",
            choices=["updown", "total"],
            default="updown",
            help="Venn labels: up/down counts or total DEG counts [default: %(default)s]",
        )

    def _add_report_arguments(self, parser):
        """Add comprehensive report arguments."""
        report = parser.add_argument_group("Report Arguments")
        report.add_argument(
            "--skip-report",
            action="store_true",
            default=False,
            help="Skip final comprehensive report generation [default: %(default)s]",
        )
        report.add_argument(
            "--report-formats",
            default="html,md,json",
            help="Comma-separated report formats: html,md,json,docx,pdf [default: %(default)s]",
        )
        report.add_argument(
            "--report-title",
            default="PySeqRNA Analysis Report",
            help="Title used in generated reports [default: %(default)s]",
        )

    def _add_computational_arguments(self, parser):
        """Add computational resource arguments to the parser."""
        comp = parser.add_argument_group("Computational Arguments")
        comp.add_argument(
            "--threads",
            type=int,
            default=None,  # None means auto-detect
            help="Number of threads to use. If not specified, uses 80%% of available CPUs or SLURM allocation [default: auto-detect]",
        )
        comp.add_argument(
            "--memory",
            type=int,
            default=None,  # None means auto-detect
            help="Memory to use in GB. If not specified, uses 60%% of available memory or SLURM allocation [default: auto-detect]",
        )
        comp.add_argument(
            "--local-jobs",
            type=int,
            default=1,
            help="Maximum sample-level commands to run in parallel in local mode [default: %(default)s]",
        )
        comp.add_argument(
            "--resume",
            choices=[
                "all",
                "quality",
                "quality_trim",
                "trimming",
                "alignment",
                "bam_preparation",
                "alignment_stats",
                "quantification",
                "normalization",
                "sample_clustering",
                "coexpression",
                "differential",
                "annotation",
            ],
            default="all",
            help="Resume from a specific stage [default: %(default)s]",
        )
        comp.add_argument(
            "--resume-policy",
            choices=["skip", "rerun", "fail", "prompt"],
            default="skip",
            help=(
                "How to handle completed stages during resume: skip existing results, "
                "rerun and overwrite, fail fast, or prompt interactively [default: %(default)s]"
            ),
        )

    def _add_configuration_arguments(self, parser):
        """Add configuration arguments to the parser."""
        config = parser.add_argument_group("Configuration Arguments")
        config.add_argument(
            "-c",
            "--config",
            dest="run_config",
            type=str,
            default=None,
            help="Run configuration file with CLI arguments (INI format) [default: None]",
        )
        config.add_argument(
            "--config-file",
            type=str,
            default=None,
            help="Custom configuration file path [default: use package defaults]",
        )
        config.add_argument(
            "--param-dir",
            type=str,
            default=None,
            help="Custom parameter directory path [default: use package defaults]",
        )

    def _add_slurm_arguments(self, parser):
        """Add SLURM-related arguments to the parser."""
        slurm = parser.add_argument_group("SLURM Arguments")
        slurm.add_argument(
            "--slurm",
            action="store_true",
            default=False,
            help="Enable SLURM job scheduling on HPC [default: %(default)s]",
        )
        slurm.add_argument(
            "--slurm_partition",
            dest="slurm_partition",
            default="compute",
            help="Specify SLURM partition [default: %(default)s]",
        )
        slurm.add_argument(
            "--slurm_account",
            dest="slurm_account",
            default="",
            help="Specify SLURM account to charge for resource usage",
        )
        slurm.add_argument(
            "--slurm_time",
            dest="slurm_time",
            default="24:00:00",
            help="Set time limit for SLURM jobs (format: HH:MM:SS) [default: %(default)s]",
        )
        slurm.add_argument(
            "--slurm_email",
            dest="slurm_email",
            default="",
            help="Specify email address to receive notifications about job status",
        )
        slurm.add_argument(
            "--slurm_qos",
            dest="slurm_qos",
            default="",
            help="Set the SLURM Quality of Service (QoS) level",
        )
        slurm.add_argument(
            "--slurm-array-max-parallel",
            dest="slurm_array_max_parallel",
            type=int,
            default=10,
            help="Maximum simultaneous tasks for sample-level SLURM arrays [default: %(default)s]",
        )
        slurm.add_argument(
            "--slurm-cpus-per-task",
            dest="slurm_cpus_per_task",
            type=int,
            default=0,
            help="CPU cores per sample-level SLURM array task; 0 uses stage-aware PySeqRNA defaults [default: %(default)s]",
        )
        slurm.add_argument(
            "--slurm-memory-per-task",
            dest="slurm_memory_per_task",
            type=int,
            default=0,
            help="Memory in GB per sample-level SLURM array task; 0 uses stage-aware PySeqRNA defaults [default: %(default)s]",
        )
        slurm.add_argument(
            "--slurm-wait-timeout-hours",
            dest="slurm_wait_timeout_hours",
            type=float,
            default=72.0,
            help="Maximum hours to wait for blocking internal SLURM jobs before failing [default: %(default)s]",
        )

    def parse_args(self):
        """Parse command-line arguments."""
        run_config = self._preparse_run_config()
        if run_config:
            config_defaults = self._read_run_config(run_config)
            if config_defaults:
                self.parser.set_defaults(**config_defaults)

        args = self.parser.parse_args()
        self._validate_required_args(args)
        return args

    def get_parser(self):
        """Get the argument parser."""
        return self.parser

    def create_diffexp_parser(self):
        """Create a parser for the standalone differential expression subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna diffexp",
            usage=(
                "pyseqrna diffexp --counts COUNTS "
                "(--sample-info SAMPLE_INFO | --input-file INPUT_FILE --samples-path SAMPLES_PATH) "
                "[options]"
            ),
            description=dedent("""\
                Run differential expression analysis without running the full pipeline.

                You can provide sample metadata in one of two ways:
                  1. Direct metadata mode: use --sample-info with a table containing a 'condition' column
                  2. PySeqRNA sample-sheet mode: use --input-file and --samples-path to infer conditions/comparisons
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna diffexp --counts Raw_Counts.xlsx --sample-info samples.tsv --outdir diffexp_out
                  pyseqrna diffexp --counts Raw_Counts.xlsx --sample-info samples.tsv --comparisons M1-A1,V1-A1
                  pyseqrna diffexp --counts Raw_Counts.xlsx --input-file input_samples.txt --samples-path data_dir --outdir diffexp_out
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        required = parser.add_argument_group("Required")
        required.add_argument(
            "--counts",
            required=True,
            help="Count matrix file (.xlsx, .csv, .tsv, .txt)",
        )

        input_mode = parser.add_argument_group("Input Mode")
        input_mode.add_argument(
            "--sample-info",
            dest="sample_info_file",
            default=None,
            help="Sample metadata file with at least a 'condition' column",
        )
        input_mode.add_argument(
            "--input-file",
            dest="input_file",
            default=None,
            help="PySeqRNA sample sheet used to derive conditions and comparisons",
        )
        input_mode.add_argument(
            "--samples-path",
            dest="samples_path",
            default=None,
            help="Samples directory used together with --input-file",
        )

        analysis = parser.add_argument_group("Analysis")
        analysis.add_argument(
            "--comparisons",
            default=None,
            help="Comma-separated comparisons like M1-A1,V1-A1. If omitted, inferred from conditions.",
        )
        analysis.add_argument(
            "--outdir",
            default="pyseqrna_diffexp_results",
            help="Output directory for differential expression results [default: %(default)s]",
        )
        analysis.add_argument(
            "--diffexp-tool",
            choices=["deseq2", "edger", "pydiffexpress"],
            default="pydiffexpress",
            help="Differential expression tool [default: %(default)s]",
        )
        analysis.add_argument(
            "--diffexp-normalization",
            choices=["median_ratio", "poscounts", "iterate", "tmm"],
            default="median_ratio",
            help="PyDiffExpress normalization component [default: %(default)s]",
        )
        analysis.add_argument(
            "--diffexp-abundance",
            choices=["base_mean", "ave_log_cpm"],
            default="base_mean",
            help="PyDiffExpress abundance summary component [default: %(default)s]",
        )
        analysis.add_argument(
            "--diffexp-dispersion",
            choices=["map", "common", "trended", "tagwise"],
            default="map",
            help="PyDiffExpress dispersion component [default: %(default)s]",
        )
        analysis.add_argument(
            "--diffexp-test",
            choices=["wald", "lrt"],
            default="wald",
            help="PyDiffExpress hypothesis-test component [default: %(default)s]",
        )
        analysis.add_argument(
            "--gene-column",
            default="Gene",
            help="Gene ID column name in the count matrix [default: %(default)s]",
        )
        analysis.add_argument(
            "--fdr-threshold",
            type=float,
            default=0.05,
            help="False Discovery Rate (FDR) threshold [default: %(default)s]",
        )
        analysis.add_argument(
            "--fold-threshold",
            type=float,
            default=2.0,
            help="Fold-change threshold, converted internally to log2 [default: %(default)s]",
        )
        analysis.add_argument(
            "--subset",
            action="store_true",
            default=False,
            help="Subset count data for each comparison [default: %(default)s]",
        )

        annotation = parser.add_argument_group("Annotation")
        annotation.add_argument(
            "--species",
            type=str,
            default=None,
            help="Species name for gene annotation, for example athaliana",
        )
        annotation.add_argument(
            "--organism-type",
            choices=["plants", "animals"],
            default="plants",
            help="Type of organism [default: %(default)s]",
        )
        annotation.add_argument(
            "--add-gene-names",
            action="store_true",
            default=True,
            help="Add gene names and descriptions when species support is available [default: %(default)s]",
        )

        runtime = parser.add_argument_group("Runtime")
        runtime.add_argument(
            "--paired",
            action="store_true",
            default=False,
            help="Treat --input-file as paired-end input when deriving metadata [default: %(default)s]",
        )
        runtime.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Run in dry-run mode [default: %(default)s]",
        )
        runtime.add_argument(
            "--param-dir",
            default=None,
            help="Custom parameter directory path [default: use package defaults]",
        )

        misc = parser.add_argument_group("Misc")
        misc.add_argument(
            "--organism",
            action="store_true",
            help="Display supported organisms for functional annotation and exit",
        )
        misc.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna diffexp (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_diffexp_args(self):
        """Parse arguments for the standalone differential expression subcommand."""
        parser = self.create_diffexp_parser()
        args = parser.parse_args(sys.argv[2:])
        self._validate_diffexp_args(parser, args)
        return args

    def create_alignment_parser(self):
        """Create parser for the standalone alignment subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna alignment",
            usage="pyseqrna alignment --input-file INPUT_FILE --samples-path SAMPLES_PATH --reference-genome FASTA [options]",
            description=dedent("""\
                Run read alignment without running the full PySeqRNA pipeline.

                The sample sheet uses the same PySeqRNA format as the full pipeline.
                For trimmed-read alignment, point --samples-path and the sample sheet
                File1/File2 columns at the trimmed FASTQ files.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna alignment --input-file input_samples.txt --samples-path data --reference-genome tair10.fasta --feature-file tair10.gff
                  pyseqrna alignment --input-file input_samples.txt --samples-path data --reference-genome tair10.fasta --alignment-tool hisat2 --paired --dryrun
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        required = parser.add_argument_group("Required")
        required.add_argument(
            "--input-file",
            required=True,
            help="PySeqRNA sample sheet containing sample and FASTQ information",
        )
        required.add_argument(
            "--samples-path",
            required=True,
            help="Directory containing FASTQ files referenced by --input-file",
        )
        required.add_argument(
            "--reference-genome",
            required=True,
            help="Reference genome FASTA file",
        )

        alignment = parser.add_argument_group("Alignment")
        alignment.add_argument(
            "--feature-file",
            default=None,
            help="Optional GFF/GTF annotation file used during index building when supported",
        )
        alignment.add_argument(
            "--alignment-tool",
            choices=["star", "hisat2", "bowtie2", "bwa", "minimap2"],
            default="star",
            help="Alignment tool [default: %(default)s]",
        )
        alignment.add_argument(
            "--outdir",
            default="pyseqrna_alignment_results",
            help="Output directory for alignment results [default: %(default)s]",
        )
        alignment.add_argument(
            "--skip-stats",
            action="store_true",
            default=False,
            help="Skip alignment statistics after alignment [default: %(default)s]",
        )
        alignment.add_argument(
            "--alignment-stats-source",
            choices=["auto", "logs", "bam"],
            default="auto",
            help="Alignment statistics source: auto uses STAR logs when complete, then falls back to BAM [default: %(default)s]",
        )
        alignment.add_argument(
            "--run-quality",
            action="store_true",
            default=False,
            help="Run quality control before alignment [default: %(default)s]",
        )
        alignment.add_argument(
            "--run-trimming",
            action="store_true",
            default=False,
            help="Run read trimming before alignment and align trimmed reads [default: %(default)s]",
        )
        alignment.add_argument(
            "--run-quality-trimming",
            action="store_true",
            default=False,
            help="Convenience flag: run pre-QC, trimming, and post-trimming QC before alignment [default: %(default)s]",
        )
        alignment.add_argument(
            "--quality-trim",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run quality control on trimmed reads when trimming is enabled [default: %(default)s]",
        )
        alignment.add_argument(
            "--quality-tool",
            choices=["fastqc"],
            default="fastqc",
            help="Quality-control tool [default: %(default)s]",
        )
        alignment.add_argument(
            "--trimming-tool",
            choices=["trim_galore", "trimmomatic", "flexbar"],
            default="trim_galore",
            help="Trimming tool [default: %(default)s]",
        )
        alignment.add_argument(
            "--trimmed-dir",
            default=None,
            help="Directory containing existing trimmed FASTQ files to align instead of raw files",
        )
        alignment.add_argument(
            "--trimmed-pattern-r1",
            default=None,
            help="Custom paired R1 pattern relative to --trimmed-dir, using {sample}, for example {sample}_R1_trimmed.fastq.gz",
        )
        alignment.add_argument(
            "--trimmed-pattern-r2",
            default=None,
            help="Custom paired R2 pattern relative to --trimmed-dir, using {sample}, for example {sample}_R2_trimmed.fastq.gz",
        )
        alignment.add_argument(
            "--trimmed-pattern-single",
            default=None,
            help="Custom single-end pattern relative to --trimmed-dir, using {sample}, for example {sample}_trimmed.fq.gz",
        )

        runtime = parser.add_argument_group("Runtime")
        runtime.add_argument(
            "--paired",
            action="store_true",
            default=False,
            help="Force paired-end parsing for the sample sheet [default: %(default)s]",
        )
        runtime.add_argument(
            "--threads",
            type=int,
            default=1,
            help="CPU threads to use [default: %(default)s]",
        )
        runtime.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Print commands without executing them [default: %(default)s]",
        )
        runtime.add_argument(
            "--param-dir",
            default=None,
            help="Custom parameter directory path [default: use package defaults]",
        )
        runtime.add_argument(
            "--slurm",
            action="store_true",
            default=False,
            help="Submit alignment jobs through SLURM [default: %(default)s]",
        )
        runtime.add_argument(
            "--slurm-partition",
            dest="slurm_partition",
            default="compute",
            help="SLURM partition [default: %(default)s]",
        )
        runtime.add_argument(
            "--slurm-account",
            dest="slurm_account",
            default="",
            help="SLURM account to charge [default: none]",
        )
        runtime.add_argument(
            "--slurm-time",
            dest="slurm_time",
            default="24:00:00",
            help="SLURM time limit [default: %(default)s]",
        )
        runtime.add_argument(
            "--slurm-memory",
            dest="slurm_memory",
            default=None,
            help="SLURM memory in GB [default: 16]",
        )
        runtime.add_argument(
            "--slurm-email",
            dest="slurm_email",
            default="",
            help="SLURM notification email [default: none]",
        )
        runtime.add_argument(
            "--slurm-qos",
            dest="slurm_qos",
            default="",
            help="SLURM Quality of Service [default: none]",
        )
        runtime.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna alignment (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_alignment_args(self):
        """Parse arguments for the standalone alignment subcommand."""
        parser = self.create_alignment_parser()
        return parser.parse_args(sys.argv[2:])

    def create_quantification_parser(self):
        """Create parser for the standalone quantification subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna quantification",
            usage="pyseqrna quantification --input-file INPUT_FILE --samples-path SAMPLES_PATH --feature-file GFF --alignment-dir DIR [options]",
            description=dedent("""\
                Run gene quantification without running the full PySeqRNA pipeline.

                By default this command creates the normal gene count matrix from
                aligned BAM files. Add --run-multimapped-groups to also quantify
                multimapped gene groups, or use --skip-unique-counts when you only
                want the multimapped-groups output.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna quantification --input-file input_samples.txt --samples-path data --feature-file tair10.gff --alignment-dir run/2.Alignment
                  pyseqrna quantification --input-file input_samples.txt --samples-path data --feature-file tair10.gff --alignment-dir run/2.Alignment --alignment-tool star --run-multimapped-groups --dryrun
                  pyseqrna quantification --input-file input_samples.txt --samples-path data --feature-file tair10.gff --alignment-dir external_bams --bam-pattern "{sample}.bam"
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        required = parser.add_argument_group("Required")
        required.add_argument(
            "--input-file",
            required=True,
            help="PySeqRNA sample sheet used to identify sample IDs",
        )
        required.add_argument(
            "--samples-path",
            required=True,
            help="Directory containing files referenced by --input-file",
        )
        required.add_argument(
            "--feature-file",
            required=True,
            help="GFF/GTF annotation file",
        )
        required.add_argument(
            "--alignment-dir",
            required=True,
            help="Directory containing BAM files or PySeqRNA alignment result folders",
        )

        quant = parser.add_argument_group("Quantification")
        quant.add_argument(
            "--quant-method",
            "-Q",
            choices=["featureCounts", "featurecounts", "htseq", "genomic_overlaps"],
            default="genomic_overlaps",
            help="Quantification method [default: %(default)s]",
        )
        quant.add_argument(
            "--outdir",
            default="pyseqrna_quantification_results",
            help="Output directory for quantification results [default: %(default)s]",
        )
        quant.add_argument(
            "--alignment-tool",
            choices=["star", "hisat2", "bowtie2", "bwa", "minimap2"],
            default=None,
            help="Alignment tool used to create BAMs; helps infer PySeqRNA output names [default: auto-scan]",
        )
        quant.add_argument(
            "--bam-pattern",
            default=None,
            help="Custom BAM pattern relative to --alignment-dir using {sample}, for example {sample}.bam",
        )
        quant.add_argument(
            "--skip-unique-counts",
            action="store_true",
            default=False,
            help="Skip normal gene count quantification [default: %(default)s]",
        )
        quant.add_argument(
            "--run-multimapped-groups",
            action="store_true",
            default=False,
            help="Run multimapped groups analysis from the same BAM files [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-min-count",
            type=int,
            default=100,
            help="Minimum read count per sample for multimapped groups filtering [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-percent-sample",
            type=float,
            default=0.5,
            help="Minimum fraction of samples that must meet min_count for multimapped groups [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-feature",
            default="gene",
            help="Feature type to extract from GFF/GTF for multimapped groups [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-min-overlap",
            type=int,
            default=1,
            help="Minimum overlapping bases required for a read to match a feature in MMG analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-fraction-overlap",
            type=float,
            default=0.0,
            help="Minimum fraction of aligned read bases that must overlap a feature in MMG analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-no-ambiguous-unique",
            action="store_true",
            default=False,
            help="Do not include uniquely mapped reads that overlap multiple genes in MMG analysis [default: %(default)s]",
        )
        quant.add_argument(
            "--mmg-no-collapse-contained",
            action="store_true",
            default=False,
            help="Do not collapse MMGs wholly contained within larger MMGs [default: %(default)s]",
        )

        runtime = parser.add_argument_group("Runtime")
        runtime.add_argument(
            "--paired",
            action="store_true",
            default=False,
            help="Force paired-end parsing for the sample sheet [default: %(default)s]",
        )
        runtime.add_argument(
            "--threads",
            type=int,
            default=1,
            help="CPU threads to use [default: %(default)s]",
        )
        runtime.add_argument(
            "--memory",
            type=int,
            default=16,
            help="Memory in GB [default: %(default)s]",
        )
        runtime.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Print planned work without executing it [default: %(default)s]",
        )
        runtime.add_argument(
            "--param-dir",
            default=None,
            help="Custom parameter directory path [default: use package defaults]",
        )
        runtime.add_argument(
            "--slurm",
            action="store_true",
            default=False,
            help="Submit quantification jobs through SLURM when supported [default: %(default)s]",
        )
        runtime.add_argument(
            "--slurm-partition",
            dest="slurm_partition",
            default="compute",
            help="SLURM partition [default: %(default)s]",
        )
        runtime.add_argument(
            "--slurm-account",
            dest="slurm_account",
            default="",
            help="SLURM account to charge [default: none]",
        )
        runtime.add_argument(
            "--slurm-time",
            dest="slurm_time",
            default="24:00:00",
            help="SLURM time limit [default: %(default)s]",
        )
        runtime.add_argument(
            "--slurm-memory",
            dest="slurm_memory",
            default=None,
            help="SLURM memory in GB [default: --memory]",
        )
        runtime.add_argument(
            "--slurm-email",
            dest="slurm_email",
            default="",
            help="SLURM notification email [default: none]",
        )
        runtime.add_argument(
            "--slurm-qos",
            dest="slurm_qos",
            default="",
            help="SLURM Quality of Service [default: none]",
        )
        runtime.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna quantification (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_quantification_args(self):
        """Parse arguments for the standalone quantification subcommand."""
        parser = self.create_quantification_parser()
        args = parser.parse_args(sys.argv[2:])
        self._validate_quantification_args(parser, args)
        return args

    def create_normalization_parser(self):
        """Create parser for the standalone normalization subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna normalization",
            usage="pyseqrna normalization --counts COUNTS [options]",
            description=dedent("""\
                Run count normalization without running the full PySeqRNA pipeline.

                CPM, TMM, and median-ratio normalization only need a count matrix.
                RPKM, TPM, and FPKM also need --feature-file so gene lengths can
                be extracted from the annotation.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna normalization --counts Raw_Counts.xlsx --normalization-method cpm
                  pyseqrna normalization --counts Raw_Counts.xlsx --feature-file tair10.gff --normalization-method rpkm
                  pyseqrna normalization --counts Raw_Counts.xlsx --normalization-method median_ratio --skip-plots --dryrun
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        required = parser.add_argument_group("Required")
        required.add_argument(
            "--counts",
            required=True,
            help="Count matrix file (.xlsx, .csv, .tsv, .txt)",
        )

        analysis = parser.add_argument_group("Normalization")
        analysis.add_argument(
            "--normalization-method",
            choices=["cpm", "rpkm", "tpm", "fpkm", "median_ratio", "tmm"],
            default="rpkm",
            help="Normalization method [default: %(default)s]",
        )
        analysis.add_argument(
            "--feature-file",
            default=None,
            help="GFF/GTF annotation file required for rpkm, tpm, and fpkm",
        )
        analysis.add_argument(
            "--gene-column",
            default="Gene",
            help="Gene ID column name in the count matrix [default: %(default)s]",
        )
        analysis.add_argument(
            "--outdir",
            default="pyseqrna_normalization_results",
            help="Output directory for normalization results [default: %(default)s]",
        )
        analysis.add_argument(
            "--skip-plots",
            action="store_true",
            default=False,
            help="Skip raw-vs-normalized comparison plots [default: %(default)s]",
        )
        analysis.add_argument(
            "--no-save",
            action="store_true",
            default=False,
            help="Run normalization but do not save normalized counts [default: %(default)s]",
        )

        runtime = parser.add_argument_group("Runtime")
        runtime.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Run in dry-run mode [default: %(default)s]",
        )
        runtime.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna normalization (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_normalization_args(self):
        """Parse arguments for the standalone normalization subcommand."""
        parser = self.create_normalization_parser()
        args = parser.parse_args(sys.argv[2:])
        self._validate_normalization_args(parser, args)
        return args

    def create_visualization_parser(self):
        """Create parser for the standalone visualization subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna visualization",
            usage="pyseqrna visualization [--normalized-counts FILE] [--de-results FILE] [options]",
            description=dedent("""\
                Generate PySeqRNA plots without running the full pipeline.

                Provide --normalized-counts for PCA/t-SNE plots and --de-results
                for volcano, MA, DEG heatmap, Venn, and UpSet-style plots.
                Supplying both gives the most complete visualization set.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna visualization --normalized-counts RPKM_normalized_counts.xlsx --input-file input_samples.txt --samples-path data
                  pyseqrna visualization --de-results All_gene_expression.xlsx --normalized-counts RPKM_normalized_counts.xlsx
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        inputs = parser.add_argument_group("Inputs")
        inputs.add_argument(
            "--normalized-counts",
            default=None,
            help="Normalized count matrix for PCA/t-SNE and MA plots",
        )
        inputs.add_argument(
            "--de-results",
            default=None,
            help="Differential expression results file, usually All_gene_expression.xlsx",
        )
        inputs.add_argument(
            "--filtered-degs",
            default=None,
            help="Filtered DEG workbook for Venn/UpSet-style plots, usually Filtered_DEGs.xlsx",
        )
        inputs.add_argument(
            "--input-file",
            default=None,
            help="Optional PySeqRNA sample sheet for sample condition labels",
        )
        inputs.add_argument(
            "--samples-path",
            default=None,
            help="Samples directory used with --input-file",
        )

        plot = parser.add_argument_group("Plot Options")
        plot.add_argument(
            "--outdir",
            default="pyseqrna_visualization_results",
            help="Output directory [default: %(default)s]",
        )
        plot.add_argument(
            "--fold-threshold",
            type=float,
            default=2.0,
            help="Fold-change threshold [default: %(default)s]",
        )
        plot.add_argument(
            "--fdr-threshold",
            type=float,
            default=0.05,
            help="FDR/p-value threshold [default: %(default)s]",
        )
        plot.add_argument(
            "--no-venn",
            action="store_true",
            default=False,
            help="Skip Venn plots [default: %(default)s]",
        )
        plot.add_argument(
            "--no-upset",
            action="store_true",
            default=False,
            help="Skip UpSet-style intersection plot [default: %(default)s]",
        )
        plot.add_argument(
            "--venn-comparisons",
            default=None,
            help="Comma-separated list of 2-4 comparisons for one Venn plot; default chunks all comparisons by four",
        )
        plot.add_argument(
            "--venn-label",
            choices=["updown", "total"],
            default="updown",
            help="Venn labels: up/down counts or total DEG counts [default: %(default)s]",
        )

        runtime = parser.add_argument_group("Runtime")
        runtime.add_argument(
            "--paired",
            action="store_true",
            default=False,
            help="Paired-end parsing when using --input-file [default: %(default)s]",
        )
        runtime.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Run in dry-run mode [default: %(default)s]",
        )
        runtime.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna visualization (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_visualization_args(self):
        """Parse arguments for the standalone visualization subcommand."""
        parser = self.create_visualization_parser()
        args = parser.parse_args(sys.argv[2:])
        self._validate_visualization_args(parser, args)
        return args

    def create_annotation_parser(self):
        """Create parser for the standalone functional annotation subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna annotation",
            usage="pyseqrna annotation --species SPECIES (--gene-files FILES | --deg-dir DIR) [options]",
            description=dedent("""\
                Run functional annotation without running the full pipeline.

                Current GO and KEGG standalone annotation is validated for
                Ensembl-style gene IDs. Input files may be one-gene-per-line text
                files or tabular files with a Gene column.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna annotation --species athaliana --gene-files diff_genes/GA-GB.txt --go --kegg
                  pyseqrna annotation --species athaliana --deg-dir diff_genes --go --dryrun
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        inputs = parser.add_argument_group("Inputs")
        inputs.add_argument("--gene-files", default=None, help="Comma-separated gene/DEG files")
        inputs.add_argument(
            "--deg-dir",
            default=None,
            help="Directory containing .txt, .csv, .tsv, or .xlsx gene/DEG files",
        )
        inputs.add_argument("--species", required=True, help="Species identifier, for example athaliana")
        inputs.add_argument(
            "--organism-type",
            choices=["plants", "animals"],
            default="plants",
            help="Organism type [default: %(default)s]",
        )
        inputs.add_argument(
            "--key-type",
            choices=["ensembl"],
            default="ensembl",
            help="Gene ID type [default: %(default)s]",
        )

        analysis = parser.add_argument_group("Annotation")
        analysis.add_argument(
            "--go",
            action="store_true",
            default=False,
            help="Run Gene Ontology enrichment [default: %(default)s]",
        )
        analysis.add_argument(
            "--kegg",
            action="store_true",
            default=False,
            help="Run KEGG pathway enrichment [default: %(default)s]",
        )
        analysis.add_argument(
            "--outdir",
            default="pyseqrna_annotation_results",
            help="Output directory [default: %(default)s]",
        )
        analysis.add_argument(
            "--go-pvalue-threshold",
            type=float,
            default=0.05,
            help="GO p-value cutoff [default: %(default)s]",
        )
        analysis.add_argument(
            "--kegg-pvalue-threshold",
            type=float,
            default=0.05,
            help="KEGG p-value cutoff [default: %(default)s]",
        )
        analysis.add_argument(
            "--plot-type",
            choices=["dotplot", "barplot", "all"],
            default="all",
            help="Annotation plot type [default: %(default)s]",
        )
        analysis.add_argument(
            "--nrows",
            type=int,
            default=20,
            help="Top terms/pathways to show in plots [default: %(default)s]",
        )
        analysis.add_argument(
            "--color-by",
            choices=["logPvalues", "FDR"],
            default="logPvalues",
            help="Plot color metric [default: %(default)s]",
        )
        analysis.add_argument(
            "--no-plots",
            action="store_true",
            default=False,
            help="Skip annotation plots [default: %(default)s]",
        )

        runtime = parser.add_argument_group("Runtime")
        runtime.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Run in dry-run mode [default: %(default)s]",
        )
        runtime.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna annotation (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_annotation_args(self):
        """Parse arguments for the standalone annotation subcommand."""
        parser = self.create_annotation_parser()
        args = parser.parse_args(sys.argv[2:])
        self._validate_annotation_args(parser, args)
        return args

    def create_clustering_parser(self):
        """Create parser for the standalone clustering subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna clustering",
            usage="pyseqrna clustering --matrix MATRIX [options]",
            description=dedent("""\
                Run sample similarity clustering without running the full pipeline.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna clustering --matrix RPKM_normalized_counts.xlsx --cluster-target samples
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        required = parser.add_argument_group("Required")
        required.add_argument(
            "--matrix",
            required=True,
            help="Expression matrix file (.xlsx, .csv, .tsv, .txt)",
        )

        common = parser.add_argument_group("Common")
        common.add_argument(
            "--outdir",
            default="pyseqrna_clustering_results",
            help="Output directory [default: %(default)s]",
        )
        common.add_argument(
            "--gene-column",
            default="Gene",
            help="Gene ID column [default: %(default)s]",
        )
        common.add_argument(
            "--prefix",
            default="clustering",
            help="Output file prefix [default: %(default)s]",
        )
        common.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Run in dry-run mode [default: %(default)s]",
        )

        native = parser.add_argument_group("Sample Clustering")
        native.add_argument(
            "--cluster-target",
            choices=["samples"],
            default="samples",
            help="What to cluster [default: %(default)s]",
        )
        native.add_argument(
            "--cluster-method",
            choices=["hierarchical", "kmeans"],
            default="hierarchical",
            help="Clustering method [default: %(default)s]",
        )
        native.add_argument(
            "--cluster-count",
            type=int,
            default=6,
            help="Number of clusters [default: %(default)s]",
        )
        native.add_argument(
            "--metric",
            default="euclidean",
            help="Distance metric for hierarchical clustering [default: %(default)s]",
        )
        native.add_argument(
            "--linkage",
            default="average",
            help="Linkage method for hierarchical clustering [default: %(default)s]",
        )
        native.add_argument(
            "--top-variable",
            type=int,
            default=1000,
            help="Top variable genes; 0 keeps all genes [default: %(default)s]",
        )
        native.add_argument(
            "--min-mean",
            type=float,
            default=0.0,
            help="Minimum mean expression filter [default: %(default)s]",
        )
        native.add_argument(
            "--scale",
            choices=["row", "column", "none"],
            default="row",
            help="Scaling mode [default: %(default)s]",
        )
        native.add_argument(
            "--no-log",
            action="store_true",
            default=False,
            help="Disable log2(x+1) transform [default: %(default)s]",
        )
        native.add_argument(
            "--no-heatmap",
            action="store_true",
            default=False,
            help="Skip clustered heatmap [default: %(default)s]",
        )
        native.add_argument("--cmap", default="vlag", help="Heatmap color map [default: %(default)s]")

        native.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna clustering (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_clustering_args(self):
        """Parse arguments for the standalone clustering subcommand."""
        parser = self.create_clustering_parser()
        return parser.parse_args(sys.argv[2:])

    def create_report_parser(self):
        """Create parser for the standalone report subcommand."""
        parser = argparse.ArgumentParser(
            prog="pyseqrna report",
            usage="pyseqrna report --pipeline-dir OUTDIR [options]",
            description=dedent("""\
                Generate a comprehensive report from an existing PySeqRNA run directory.

                The report command inspects checkpoint metadata, output files, tables,
                and plots. It can be used after the full pipeline or after standalone
                modules if their outputs are collected in one results directory.
            """),
            epilog=dedent("""\
                Examples:
                  pyseqrna report --pipeline-dir pySeqRNA_results
                  pyseqrna report --pipeline-dir run_test4 --formats html,docx,pdf --input-file data/input_samples.txt
            """),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        required = parser.add_argument_group("Required")
        required.add_argument(
            "--pipeline-dir",
            required=True,
            help="Existing PySeqRNA output directory to summarize",
        )

        options = parser.add_argument_group("Report Options")
        options.add_argument(
            "--outdir",
            default=None,
            help="Report output directory [default: PIPELINE_DIR/7.Report]",
        )
        options.add_argument(
            "--formats",
            default="html,md,json",
            help="Comma-separated formats: html,md,json,docx,pdf [default: %(default)s]",
        )
        options.add_argument(
            "--title",
            default="PySeqRNA Analysis Report",
            help="Report title [default: %(default)s]",
        )
        options.add_argument(
            "--input-file",
            default=None,
            help="Optional PySeqRNA sample sheet to include in the report",
        )
        options.add_argument(
            "--samples-path",
            default=None,
            help="Optional samples directory recorded in the report",
        )
        options.add_argument(
            "--reference-genome",
            default=None,
            help="Optional reference genome path recorded in the report",
        )
        options.add_argument(
            "--feature-file",
            default=None,
            help="Optional GFF/GTF annotation path recorded in the report",
        )
        options.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Show planned report files without writing them [default: %(default)s]",
        )
        options.add_argument(
            "--version",
            action="version",
            version=f"pyseqrna report (version {__version__})",
            help="Show version information and exit",
        )
        return parser

    def parse_report_args(self):
        """Parse arguments for the standalone report subcommand."""
        parser = self.create_report_parser()
        return parser.parse_args(sys.argv[2:])

    def _preparse_run_config(self):
        """Pre-parse to find the run config file before full argument parsing."""
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("-c", "--config", dest="run_config", type=str, default=None)
        known, _ = pre_parser.parse_known_args()
        return known.run_config

    def _read_run_config(self, config_path):
        """Read INI-style run configuration and return defaults dict."""
        parser_actions = {a.dest: a for a in self.parser._actions}
        config = configparser.ConfigParser()
        read_ok = config.read(config_path)
        if not read_ok:
            raise SystemExit(f"Failed to read config file: {config_path}")

        # Merge DEFAULT and all named sections by matching INI keys to CLI destinations.
        defaults = {}
        sections = ["DEFAULT"] + config.sections()
        for section in sections:
            for key, value in config.items(section):
                normalized_key = key.strip()
                if normalized_key not in parser_actions:
                    raise SystemExit(
                        f"Unknown config key '{normalized_key}' in section [{section}] of {config_path}. "
                        "Fix the key name or remove it; PySeqRNA does not ignore unknown production config keys."
                    )

                defaults[normalized_key] = self._coerce_value(
                    value,
                    parser_actions[normalized_key],
                    key=normalized_key,
                    section=section,
                    config_path=config_path,
                )

        defaults["run_config"] = config_path
        return defaults

    def _coerce_value(self, value, action, key=None, section=None, config_path=None):
        """Convert string config values to appropriate types based on argparse action."""
        if action is None:
            return value

        # Handle flags
        if isinstance(action, argparse._StoreTrueAction):
            coerced = self._parse_bool(value, key=key, section=section, config_path=config_path)
            return self._validate_config_choice(coerced, action, key, section, config_path)
        if isinstance(action, argparse._StoreFalseAction):
            coerced = not self._parse_bool(value, key=key, section=section, config_path=config_path)
            return self._validate_config_choice(coerced, action, key, section, config_path)
        if isinstance(action, argparse.BooleanOptionalAction):
            coerced = self._parse_bool(value, key=key, section=section, config_path=config_path)
            return self._validate_config_choice(coerced, action, key, section, config_path)

        # Explicit type
        if action.type is not None:
            try:
                coerced = action.type(value)
            except Exception as exc:
                raise SystemExit(
                    f"Invalid value for config key '{key}' in section [{section}] of {config_path}: "
                    f"{value!r} cannot be converted to {getattr(action.type, '__name__', action.type)}"
                ) from exc
            return self._validate_config_choice(coerced, action, key, section, config_path)

        # Fallback for nargs or choices
        return self._validate_config_choice(value, action, key, section, config_path)

    def _validate_config_choice(self, value, action, key, section, config_path):
        """Validate argparse choices for values loaded from INI config."""
        if action is None or not action.choices:
            return value
        if value in action.choices:
            return value
        if isinstance(value, str):
            value_lower = value.lower()
            for choice in action.choices:
                if isinstance(choice, str) and choice.lower() == value_lower:
                    return choice
        choices = ", ".join(map(str, action.choices))
        raise SystemExit(
            f"Invalid value for config key '{key}' in section [{section}] of {config_path}: "
            f"{value!r}. Allowed values: {choices}"
        )

    def _parse_bool(self, value, key=None, section=None, config_path=None):
        """Parse common boolean strings."""
        if isinstance(value, bool):
            return value
        val = str(value).strip().lower()
        if val in ("true", "1", "yes", "y", "on"):
            return True
        if val in ("false", "0", "no", "n", "off"):
            return False
        raise SystemExit(
            f"Invalid boolean for config key '{key}' in section [{section}] of {config_path}: "
            f"{value!r}. Use true/false, yes/no, 1/0, or on/off."
        )

    def _validate_required_args(self, args):
        """Validate required positional args unless a run config provides them."""
        required = ["input_file", "samples_path", "reference_genome", "feature_file"]
        missing = [r for r in required if getattr(args, r, None) in (None, "")]
        if missing and "--organism" not in sys.argv:
            if args.run_config:
                self.parser.error(f"Missing required arguments in config: {', '.join(missing)}")
            else:
                self.parser.error(f"Missing required arguments: {', '.join(missing)}")

    def _validate_diffexp_args(self, parser, args):
        """Validate differential expression subcommand arguments."""
        has_sample_info = bool(args.sample_info_file)
        has_input_file = bool(args.input_file)

        if has_sample_info == has_input_file:
            parser.error("Provide exactly one of --sample-info or --input-file.")

        if has_input_file and not args.samples_path:
            parser.error("--samples-path is required when using --input-file.")

        if args.comparisons:
            parsed = [item.strip() for item in args.comparisons.split(",") if item.strip()]
            if not parsed:
                parser.error("--comparisons was provided but no valid comparisons were found.")
            args.comparisons = parsed

    def _validate_quantification_args(self, parser, args):
        """Validate quantification subcommand arguments."""
        if args.skip_unique_counts and not args.run_multimapped_groups:
            parser.error("--skip-unique-counts requires --run-multimapped-groups; otherwise there is nothing to run.")
        if args.bam_pattern and "{sample" not in args.bam_pattern:
            parser.error("--bam-pattern must include {sample}, {sample_id}, or {sample_name}.")

    def _validate_normalization_args(self, parser, args):
        """Validate normalization subcommand arguments."""
        length_based = {"rpkm", "tpm", "fpkm"}
        if args.normalization_method in length_based and not args.feature_file:
            parser.error(f"--feature-file is required for {args.normalization_method} normalization.")

    def _validate_visualization_args(self, parser, args):
        """Validate visualization subcommand arguments."""
        if not args.normalized_counts and not args.de_results:
            parser.error("Provide at least one of --normalized-counts or --de-results.")
        if bool(args.input_file) != bool(args.samples_path):
            parser.error("--input-file and --samples-path must be provided together.")

    def _validate_annotation_args(self, parser, args):
        """Validate annotation subcommand arguments."""
        if bool(args.gene_files) == bool(args.deg_dir):
            parser.error("Provide exactly one of --gene-files or --deg-dir.")
        if not args.go and not args.kegg:
            args.go = True
            args.kegg = True

    @staticmethod
    def infer_comparisons_from_conditions(conditions):
        """Infer pairwise comparisons from a sequence of condition labels."""
        ordered_conditions = []
        seen = set()
        for condition in conditions:
            condition_str = str(condition).strip()
            if condition_str and condition_str not in seen:
                ordered_conditions.append(condition_str)
                seen.add(condition_str)

        return [f"{left}-{right}" for left, right in combinations(ordered_conditions, 2)]
