#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Orchestrator

This module provides the main pipeline class for orchestrating RNA-seq analysis,
integrating all steps from preprocessing to differential expression and functional annotation.
It manages workflow execution, handles job scheduling on local machines or SLURM clusters,
and facilitates step-by-step resumption via checkpointing.

Features:
    - Command executor, log, configuration, and checkpoint management
    - Support for SLURM and local execution resource allocation
    - Intelligent stage resume policy with upstream validation
    - Robust parameter verification and input file processing
    - Seamless integration of quality control, trimming, alignment, quantification, differential expression, and clustering

Configuration:
    - Configured via instantiation parameters, optional INI configuration files, and resource settings.
      Parameters define directories, tools to use, thresholds, and cluster submission parameters.

Dependencies:
    - Python packages: pandas, numpy, psutil, tabulate
    - External tools: fastqc, trim_galore, star, hisat2, bowtie2, subread (featureCounts), clust
    - R packages: DESeq2, edgeR (via diffexp backends)

Classes / Functions / Exceptions:
    - Pipeline: Main orchestrator class for RNA-seq analysis.
    - _RawLoggerAdapter: Internal logger adapter class to wrap standard logging into LogManager.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
import sys
import logging
import json
import platform
import subprocess
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from ..__version__ import __version__

from ..utils import (
    LogManager,
    FileManager,
    CommandExecutor,
    InputProcessor,
    ResourceManager,
    ConfigManager,
    CheckpointManager,
)
from ..utils.dry_run_manager import DryRunManager
from ..utils.stage_registry import PIPELINE_STAGE_NAMES, get_pipeline_stages
from ..modules.quality import get_available_quality_tools
from ..modules.trimming import get_available_trimmers
from ..modules.alignment import get_available_aligners
from ..modules.quantification import get_available_quantifiers


from .runners.preprocessing import PreProcessingRunner
from .runners.alignment import AlignmentRunner
from .runners.quantification import QuantificationRunner
from .runners.diffexp import DifferentialExpressionRunner
from .runners.visualization import VisualizationRunner
from .runners.annotation import AnnotationRunner
from .runners.downstream import DownstreamRunner


class _RawLoggerAdapter:
    """Adapt a standard logging.Logger to the LogManager interface used internally."""

    def __init__(self, raw_logger: logging.Logger) -> None:
        self._raw = raw_logger

    @property
    def logger(self) -> logging.Logger:
        return self._raw

    def debug(self, msg, *args, **kwargs):
        self._raw.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._raw.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._raw.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._raw.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._raw.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._raw.exception(msg, *args, **kwargs)

    def update_log_directory(self, *args, **kwargs):
        """No-op for externally owned loggers."""

    def close_logger(self):
        """No-op for externally owned loggers."""


class Pipeline:
    """
    Main pipeline class for PySeqRNA analysis.
    """

    def __init__(
        self,
        input_file: str,
        samples_path: str,
        reference_genome: str,
        feature_file: str,
        output_dir: str = "pySeqRNA_results",
        species: Optional[str] = None,
        organism_type: str = "plants",
        source: str = "ENSEMBL",
        add_gene_names: bool = True,
        gene_ontology: bool = False,
        kegg_pathway: bool = False,
        skip_functional_annotation: bool = False,
        go_pvalue_threshold: float = 0.05,
        kegg_pvalue_threshold: float = 0.05,
        skip_diffexp: bool = False,
        diffexp_tool: str = "pydiffexpress",
        diffexp_normalization: str = "median_ratio",
        diffexp_abundance: str = "base_mean",
        diffexp_dispersion: str = "map",
        diffexp_test: str = "wald",
        fdr_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,  # log2(2.0) = 1.0, converted from fold_threshold internally
        pvalue_threshold: float = 0.05,
        subset: bool = True,
        threads: int = 1,
        memory: int = 8,  # Default 8GB memory, can be overridden by kwargs
        local_jobs: int = 1,
        skip_quality: bool = False,
        quality_tool: str = "fastqc",
        quality_trim: bool = True,
        skip_trim: bool = False,
        trimming_tool: str = "trim_galore",
        skip_alignment: bool = False,
        alignment_tool: str = "star",
        alignment_stats: bool = True,
        alignment_stats_source: str = "auto",
        quantification_tool: str = "genomic_overlaps",
        dryrun: bool = False,
        force: bool = False,
        resume_policy: str = "skip",
        slurm: bool = False,
        slurm_partition: Optional[str] = None,
        slurm_account: Optional[str] = None,
        slurm_time: Optional[str] = None,
        slurm_email: Optional[str] = None,
        slurm_qos: Optional[str] = None,
        slurm_array_max_parallel: int = 10,
        slurm_cpus_per_task: int = 0,
        slurm_memory_per_task: int = 0,
        slurm_wait_timeout_hours: float = 72.0,
        config_file: Optional[str] = None,
        param_dir: Optional[str] = None,
        logger: Optional[Any] = None,
        force_paired: bool = False,
        force_single_end: bool = False,
        quant_method: str = "genomic_overlaps",
        skip_quantification: bool = False,
        run_multimapped_groups: bool = False,
        mmg_min_count: int = 100,
        mmg_percent_sample: float = 0.5,
        mmg_feature: str = "gene",
        mmg_min_overlap: int = 1,
        mmg_fraction_overlap: float = 0.0,
        mmg_include_ambiguous_unique: bool = True,
        mmg_collapse_contained_groups: bool = True,
        normalize_counts: bool = False,
        normalization_method: str = "rpkm",
        skip_normalization_plots: bool = False,
        run_clustering: bool = True,
        cluster_target: str = "samples",
        cluster_method: str = "hierarchical",
        cluster_count: int = 6,
        cluster_metric: str = "euclidean",
        cluster_linkage: str = "average",
        cluster_top_variable: int = 1000,
        cluster_scale: str = "row",
        cluster_no_log: bool = False,
        cluster_no_heatmap: bool = False,
        cluster_cmap: str = "vlag",
        run_coexpression: bool = True,
        coexpression_tool: str = "pycoexpression",
        coexpression_tightness: Optional[float] = None,
        coexpression_k_values: Optional[str] = None,
        coexpression_outlier: Optional[float] = None,
        coexpression_cluster_size: Optional[int] = None,
        coexpression_replicates: Optional[bool] = None,
        coexpression_preprocessing: Optional[bool] = None,
        pca_plot: bool = True,
        tsne_plot: bool = True,
        volcano_plot: bool = True,
        ma_plot: bool = True,
        deg_heatmap: bool = True,
        heatmap_top_genes: int = 50,
        venn: bool = True,
        upset: bool = True,
        venn_comparisons: Optional[str] = None,
        venn_label: str = "updown",
        skip_report: bool = False,
        report_formats: str = "html,md,json",
        report_title: str = "PySeqRNA Analysis Report",
        resume: str = "all",
        **kwargs,
    ):
        """
        Initialize the RNA-seq pipeline.

        Args:
            input_file: Path to input file containing sample information
            samples_path: Directory containing sample files
            reference_genome: Path to reference genome file
            feature_file: Path to annotation file (GFF/GTF)
            output_dir: Directory for pipeline outputs
            species: Species name for functional annotation (e.g., 'athaliana')
            organism_type: Type of organism - 'plants' or 'animals'
            source: Source database for reference files - 'ENSEMBL' or 'NCBI'
            add_gene_names: Whether to add gene names to differential expression results
            gene_ontology: Whether to perform Gene Ontology enrichment analysis
            kegg_pathway: Whether to perform KEGG pathway enrichment analysis
            skip_functional_annotation: Whether to skip all functional annotation
            go_pvalue_threshold: P-value threshold for Gene Ontology enrichment
            kegg_pvalue_threshold: P-value threshold for KEGG pathway enrichment
            skip_diffexp: Whether to skip differential expression analysis
            diffexp_tool: Differential expression tool to use ('deseq2', 'edger', 'pydiffexpress')
            diffexp_normalization: PyDiffExpress normalization component
            diffexp_abundance: PyDiffExpress abundance summary component
            diffexp_dispersion: PyDiffExpress dispersion component
            diffexp_test: PyDiffExpress hypothesis-test component

            fdr_threshold: False Discovery Rate threshold for differential expression
            log2fc_threshold: Log2 fold change threshold (converted from fold_threshold internally)
            pvalue_threshold: P-value threshold for differential expression
            threads: Number of CPU threads to use
            memory: Memory in GB to allocate for jobs (default: 8)
            local_jobs: Maximum sample-level commands to run in parallel in local mode
            skip_quality: Skip quality control
            quality_tool: Quality control tool to use ('fastqc')
            quality_trim: Run quality control after trimming
            skip_trim: Skip trimming step
            trimming_tool: Read trimming tool to use
            skip_alignment: Skip alignment step
            alignment_tool: Read alignment tool to use
            alignment_stats: Whether to generate alignment statistics
            alignment_stats_source: Alignment statistics source: auto, logs, or bam
            quantification_tool: Gene expression quantification tool to use
            dryrun: Whether to perform a dry run
            force: Whether to force restart from beginning
            resume_policy: How to handle completed stages during resume: skip, rerun, fail, or prompt
            slurm: Whether to use SLURM for job scheduling
            slurm_partition: SLURM partition to use
            slurm_account: SLURM account to use
            slurm_time: SLURM time limit
            slurm_email: SLURM email for notifications
            slurm_qos: SLURM QOS to use
            slurm_wait_timeout_hours: Maximum hours to wait for blocking internal SLURM jobs
            config_file: Path to configuration file
            param_dir: Directory containing parameter files
            logger: Logger instance
            force_paired: Force paired-end mode (validate data is paired-end)
            force_single_end: Force single-end mode even if data appears paired-end
            quant_method: Gene expression quantification method to use
            skip_quantification: Skip gene expression quantification
            run_multimapped_groups: Whether to run multimapped groups analysis
            mmg_min_count: Minimum read count per sample for multimapped groups filtering
            mmg_percent_sample: Minimum percentage of samples for multimapped groups filtering
            mmg_feature: Feature type to extract from GFF/GTF for multimapped groups
            mmg_min_overlap: Minimum overlapping bases required for a read to match a feature
            mmg_fraction_overlap: Minimum fraction of aligned read bases that must overlap a feature
            mmg_include_ambiguous_unique: Include uniquely mapped reads that overlap multiple genes
            mmg_collapse_contained_groups: Collapse MMGs wholly contained within larger MMGs
            normalize_counts: Whether to normalize gene expression counts
            normalization_method: Method for normalization (e.g., 'rpkm', 'tpm')
            skip_normalization_plots: Whether to skip generating normalization plots
            run_clustering: Whether to run sample similarity clustering after normalization
            run_coexpression: Whether to run gene co-expression analysis after normalization
            coexpression_tool: Co-expression tool to use ('clust')
            pca_plot: Whether to generate PCA plot from normalized counts
            tsne_plot: Whether to generate t-SNE plot from normalized counts
            volcano_plot: Whether to generate volcano plots
            ma_plot: Whether to generate MA plots
            deg_heatmap: Whether to generate DEG heatmap
            heatmap_top_genes: Number of top differential genes to use for DEG heatmap
            venn: Whether to generate Venn plots from filtered DEGs
            upset: Whether to generate UpSet-style DEG intersection plots
            venn_comparisons: Optional comma-separated list of 2-4 comparisons for one Venn plot
            venn_label: Label mode for Venn plots ('updown' or 'total')
            skip_report: Whether to skip final comprehensive report generation
            report_formats: Comma-separated report formats (html,md,json,docx,pdf)
            report_title: Title used in generated reports
            **kwargs: Additional arguments
        """
        # Store basic configuration
        self.input_file = input_file
        self.samples_path = samples_path
        self.reference_genome = reference_genome
        self.feature_file = feature_file
        self.output_dir = output_dir
        self.threads = threads
        self.memory = memory  # Default 8GB memory, can be overridden by kwargs
        self.local_jobs = int(local_jobs if local_jobs is not None else 1)

        # Species and functional annotation configuration
        self.species = species
        self.organism_type = organism_type
        self.source = source
        self.add_gene_names = add_gene_names
        self.gene_ontology = gene_ontology
        self.kegg_pathway = kegg_pathway
        self.skip_functional_annotation = skip_functional_annotation

        # Differential expression parameters
        self.go_pvalue_threshold = go_pvalue_threshold
        self.kegg_pvalue_threshold = kegg_pvalue_threshold
        self.skip_diffexp = skip_diffexp
        self.diffexp_tool = diffexp_tool
        self.diffexp_normalization = diffexp_normalization
        self.diffexp_abundance = diffexp_abundance
        self.diffexp_dispersion = diffexp_dispersion
        self.diffexp_test = diffexp_test

        self.fdr_threshold = fdr_threshold
        self.log2fc_threshold = log2fc_threshold
        self.pvalue_threshold = pvalue_threshold
        self.subset = subset  # Add subset parameter

        # Pipeline configuration
        self.skip_quality = skip_quality
        self.quality_tool = quality_tool
        self.quality_trim = quality_trim
        self.resume = resume
        self.skip_trim = skip_trim
        self.trimming_tool = trimming_tool
        self.skip_alignment = skip_alignment
        self.alignment_tool = alignment_tool
        self.alignment_stats = bool(alignment_stats)
        self.alignment_stats_source = str(alignment_stats_source or "auto").lower()
        self.quantification_tool = quantification_tool
        self.dryrun = dryrun
        self.force = force
        self.resume_policy = str(resume_policy or "skip").lower()
        if self.resume_policy not in {"skip", "rerun", "fail", "prompt"}:
            raise ValueError("resume_policy must be one of: skip, rerun, fail, prompt")
        self.slurm = slurm
        self.slurm_partition = slurm_partition
        self.slurm_account = slurm_account
        self.slurm_time = slurm_time
        self.slurm_email = slurm_email
        self.slurm_qos = slurm_qos
        self.slurm_array_max_parallel = slurm_array_max_parallel
        self.slurm_cpus_per_task = slurm_cpus_per_task
        self.slurm_memory_per_task = slurm_memory_per_task
        self.slurm_wait_timeout_hours = slurm_wait_timeout_hours
        self.config_file = config_file
        self.param_dir = param_dir

        # Paired-end detection and validation settings
        self.force_paired = force_paired
        self.force_single_end = force_single_end

        # Quantification and multimapped groups settings
        self.quant_method = quant_method
        self.skip_quantification = skip_quantification
        self.enable_multimapped_groups = run_multimapped_groups
        self.mmg_min_count = mmg_min_count
        self.mmg_percent_sample = mmg_percent_sample
        self.mmg_feature = mmg_feature
        self.mmg_min_overlap = mmg_min_overlap
        self.mmg_fraction_overlap = mmg_fraction_overlap
        self.mmg_include_ambiguous_unique = mmg_include_ambiguous_unique
        self.mmg_collapse_contained_groups = mmg_collapse_contained_groups
        self.normalize_counts = normalize_counts
        self.normalization_method = normalization_method
        self.skip_normalization_plots = skip_normalization_plots
        self.run_clustering = run_clustering
        self.cluster_target = cluster_target
        self.cluster_method = cluster_method
        self.cluster_count = cluster_count
        self.cluster_metric = cluster_metric
        self.cluster_linkage = cluster_linkage
        self.cluster_top_variable = cluster_top_variable
        self.cluster_scale = cluster_scale
        self.cluster_no_log = cluster_no_log
        self.cluster_no_heatmap = cluster_no_heatmap
        self.cluster_cmap = cluster_cmap
        self.run_coexpression = run_coexpression
        self.enable_coexpression = run_coexpression
        self.coexpression_tool = coexpression_tool
        self.coexpression_tightness = coexpression_tightness
        self.coexpression_k_values = coexpression_k_values
        self.coexpression_outlier = coexpression_outlier
        self.coexpression_cluster_size = coexpression_cluster_size
        self.coexpression_replicates = coexpression_replicates
        self.coexpression_preprocessing = coexpression_preprocessing
        self.pca_plot = pca_plot
        self.tsne_plot = tsne_plot
        self.volcano_plot = volcano_plot
        self.ma_plot = ma_plot
        self.deg_heatmap = deg_heatmap
        self.heatmap_top_genes = heatmap_top_genes
        self.venn = venn
        self.upset = upset
        self.venn_comparisons = venn_comparisons
        self.venn_label = venn_label
        self.skip_report = skip_report
        self.report_formats = report_formats
        self.report_title = report_title

        pending_kwargs = dict(kwargs)

        # Initialize logger - always ensure we have a LogManager-like instance.
        if logger is None:
            self.logger = LogManager()
        elif hasattr(logger, "logger"):
            self.logger = logger
        elif isinstance(logger, logging.Logger):
            self.logger = _RawLoggerAdapter(logger)
        else:
            raise TypeError(f"logger must be a LogManager-like object, logging.Logger, or None; got {type(logger).__name__}")

        for key, value in pending_kwargs.items():
            if hasattr(self, key):
                self.logger.warning(
                    "Pipeline.__init__ kwarg '%s' overrides existing attribute (old=%r, new=%r)",
                    key,
                    getattr(self, key),
                    value,
                )
            setattr(self, key, value)

        if float(self.slurm_wait_timeout_hours) <= 0:
            raise ValueError(f"slurm_wait_timeout_hours must be > 0, got {self.slurm_wait_timeout_hours}")

        # Initialize utilities
        self.file_manager = FileManager(self.logger.logger)
        self.command_executor = CommandExecutor(
            self.logger.logger,
            slurm_wait_timeout_seconds=int(float(self.slurm_wait_timeout_hours) * 3600),
        )
        self.input_processor = InputProcessor(self.logger.logger)
        self.resource_manager = ResourceManager(self.logger.logger)
        self.config_manager = ConfigManager(self.logger.logger)

        # Initialize dry-run manager
        self.dry_run_manager = DryRunManager(enabled=self.dryrun, logger=self.logger.logger)

        # Intelligent resource allocation
        resource_allocation = self.resource_manager.allocate_resources(
            user_threads=threads, user_memory=memory, use_slurm=self.slurm
        )

        # Override original values with intelligently allocated ones
        self.threads = resource_allocation["threads"]
        self.memory = resource_allocation["memory"]

        self.logger.info(f"Pipeline configured with {self.threads} threads and {self.memory}GB memory")
        self._validate_init_params()

        # Pipeline state - initialize results storage
        self.checkpoint_manager = None  # Will be initialized in setup()
        self.sample_dict = {}
        self.quality_results = None
        self.quality_trim_results = None
        self.trimming_results = None
        self.trimming_stats_results = None
        self.alignment_results = None
        self.prepared_bam_results = None
        self.alignment_stats_results = None
        self.quantification_results = None
        self.normalization_results = None
        self.differential_expression_results = None
        self.comparisons = []

        self._paired_end_cache = None

        # Track SLURM job IDs for dependency chaining between stages
        self._last_slurm_job_id = ""

        # Initialize composition runners
        self.preprocessing = PreProcessingRunner(self)
        self.alignment = AlignmentRunner(self)
        self.quantification = QuantificationRunner(self)
        self.diffexp = DifferentialExpressionRunner(self)
        self.visualization = VisualizationRunner(self)
        self.annotation = AnnotationRunner(self)
        self.downstream = DownstreamRunner(self)

    def setup(self) -> bool:
        """
        Setup the pipeline (validate inputs, create directories).

        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Handle resume vs fresh run
            if self.resume != "all":
                # Resume uses the user-provided output directory to load an existing checkpoint.
                self.output_dir = str(Path(self.output_dir).resolve())
                self.output_dir = self.output_dir
                self.checkpoint_manager = CheckpointManager(self.output_dir, logger=self.logger.logger)

                # Load checkpoint data and log it for resume operations
                if self.checkpoint_manager.checkpoint_file.exists():
                    self.logger.info(f"📋 Loaded checkpoint from {self.checkpoint_manager.checkpoint_file}")

            else:
                # Fresh run - create directories with confirmation
                self.logger.info("Setting up PySeqRNA pipeline")

                # Validate input files
                if not self.file_manager.verify_files_exist(self.input_file, self.reference_genome, self.feature_file):
                    self.logger.error("One or more input files not found")
                    return False

                # Create output directory with user confirmation if exists
                self.output_dir = self.file_manager.create_main_output_directory(
                    self.output_dir,
                    dry_run=self.dryrun,
                    force_overwrite=self.force,
                    dry_run_manager=self.dry_run_manager,
                    allow_prompt=self._can_prompt_user(),
                )
                self.output_dir = str(Path(self.output_dir).resolve())
                self.output_dir = self.output_dir
                self.logger.info(f"Created output directory: {self.output_dir}")

                # Initialize checkpoints only after the final output path is known.
                # Initializing earlier creates the requested directory and makes
                # brand-new runs look like pre-existing outputs.
                self.checkpoint_manager = CheckpointManager(self.output_dir, logger=self.logger.logger)

                # Create quality and trimming subdirectory
                quality_trim_dir = Path(self.output_dir) / "1.Quality_and_trimming"
                if not self.dryrun:
                    quality_trim_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created quality and trimming directory: {quality_trim_dir}")
                else:
                    self.logger.info(f"Would create quality and trimming directory: {quality_trim_dir}")

            # Update LogManager to use the output directory for logs
            logs_dir = str(self.output_dir)
            self.logger.update_log_directory(logs_dir)

            # Set dry_run_completed flag if running in dry-run mode
            if self.dryrun:
                self.checkpoint_manager.dry_run_completed = True
                self.checkpoint_manager.save_checkpoint()
                self.logger.info("Set dry_run_completed flag to True for dry-run mode")

            self.logger.info("Checkpoint manager initialized")

            # Process sample file
            sample_data = self.input_processor.process_sample_file(
                self.input_file,
                self.samples_path,
                paired=self.force_paired,
            )
            self.sample_dict = sample_data["samples"]
            self.comparisons = sample_data.get("combinations", [])
            self._paired_end_cache = sample_data.get("paired")
            self.logger.info(f"Successfully processed {len(self.sample_dict)} samples")

            if self.resume != "all":
                # Validate resume prerequisites after sample parsing so recovery
                # paths can reconstruct per-sample outputs when needed.
                if not self.validate_resume_prerequisites(self.resume):
                    return False

                # Log comprehensive resume status based on checkpoint
                self._log_resume_status(self.resume)

                # Check directories based on what we know from checkpoint
                self._check_required_directories(self.resume)

            # Log sample information
            sample_labels = sample_data.get("sample_labels", {})
            detail_sample_stages = {
                "all",
                "quality",
                "trimming",
                "quality_trim",
                "alignment",
                "bam_preparation",
                "alignment_stats",
                "quantification",
                "multimapped_groups",
            }
            if self.resume in detail_sample_stages:
                for sample_name, sample_info in self.sample_dict.items():
                    sample_label = sample_labels.get(
                        sample_name,
                        sample_info[0] if len(sample_info) > 0 else sample_name,
                    )
                    replication = sample_name
                    identifier = sample_info[1]
                    file_path = sample_info[2]
                    self.logger.info(
                        f"Sample: {sample_label}, Replication: {replication}, Identifier: {identifier}, File: {file_path}"
                    )
            else:
                factor_count = len({sample_info[1] for sample_info in self.sample_dict.values() if len(sample_info) > 1})
                self.logger.info(
                    "Loaded %d samples across %d factor(s); per-sample FASTQ details suppressed for resume stage '%s'.",
                    len(self.sample_dict),
                    factor_count,
                    self.resume,
                )

            # Setup SLURM configuration if enabled
            if self.slurm:
                self._setup_slurm_config()

            self.logger.info("Pipeline setup completed successfully")
            return True

        except Exception:
            self.logger.exception("Pipeline setup failed")
            return False

    def _validate_init_params(self) -> None:
        """Validate critical initialization parameters before running the pipeline."""
        if not (0 < float(self.fdr_threshold) <= 1):
            raise ValueError(f"fdr_threshold must be in (0, 1], got {self.fdr_threshold}")
        if not (0 < float(self.pvalue_threshold) <= 1):
            raise ValueError(f"pvalue_threshold must be in (0, 1], got {self.pvalue_threshold}")
        if not (0 < float(self.go_pvalue_threshold) <= 1):
            raise ValueError(f"go_pvalue_threshold must be in (0, 1], got {self.go_pvalue_threshold}")
        if not (0 < float(self.kegg_pvalue_threshold) <= 1):
            raise ValueError(f"kegg_pvalue_threshold must be in (0, 1], got {self.kegg_pvalue_threshold}")
        if int(self.threads) < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")
        if int(self.memory) < 1:
            raise ValueError(f"memory must be >= 1 GB, got {self.memory}")
        if int(self.local_jobs) < 1:
            raise ValueError(f"local_jobs must be >= 1, got {self.local_jobs}")
        if int(self.slurm_array_max_parallel) < 1:
            raise ValueError(f"slurm_array_max_parallel must be >= 1, got {self.slurm_array_max_parallel}")
        if int(self.slurm_cpus_per_task) < 0:
            raise ValueError(f"slurm_cpus_per_task must be >= 0, got {self.slurm_cpus_per_task}")
        if int(self.slurm_memory_per_task) < 0:
            raise ValueError(f"slurm_memory_per_task must be >= 0, got {self.slurm_memory_per_task}")
        if float(self.slurm_wait_timeout_hours) <= 0:
            raise ValueError(f"slurm_wait_timeout_hours must be > 0, got {self.slurm_wait_timeout_hours}")
        if self.resume_policy not in {"skip", "rerun", "fail", "prompt"}:
            raise ValueError("resume_policy must be one of: skip, rerun, fail, prompt")
        if self.alignment_stats_source not in {"auto", "logs", "bam"}:
            raise ValueError("alignment_stats_source must be one of: auto, logs, bam")

        quality_tools = get_available_quality_tools()
        if str(self.quality_tool).lower() not in quality_tools:
            raise ValueError(f"Unknown quality_tool: {self.quality_tool!r}. Available: {', '.join(quality_tools)}")
        trimmers = get_available_trimmers()
        if str(self.trimming_tool).lower() not in trimmers:
            raise ValueError(f"Unknown trimming_tool: {self.trimming_tool!r}. Available: {', '.join(trimmers)}")
        aligners = get_available_aligners()
        if str(self.alignment_tool).lower() not in aligners:
            raise ValueError(f"Unknown alignment_tool: {self.alignment_tool!r}. Available: {', '.join(aligners)}")
        quantifiers = get_available_quantifiers()
        quant_method = str(self.quant_method).lower()
        if quant_method not in {str(item).lower() for item in quantifiers}:
            raise ValueError(f"Unknown quant_method: {self.quant_method!r}. Available: {', '.join(quantifiers)}")

    def validate_resume_prerequisites(self, resume_stage: str) -> bool:
        """Validate that all prerequisites for resuming from a stage are met."""

        # Define direct resume dependencies. A resume point should validate the
        # stage outputs it immediately consumes, not every historical upstream
        # stage. For example, quantification consumes BAMs from alignment; it
        # does not need quality/trimming checkpoints if alignment is complete.
        stage_dependencies = {
            "quality": [],
            "trimming": [] if self.skip_quality else ["quality"],
            "quality_trim": [] if self.skip_trim else ["trimming"],
            "alignment": [] if self.skip_trim else ["trimming"],
            "bam_preparation": ["alignment"],
            "alignment_stats": ["alignment"],
            "quantification": ["alignment"],
            "multimapped_groups": ["alignment", "quantification"],
            "normalization": ["quantification"],
            "sample_clustering": ["normalization"],
            "coexpression": ["normalization"],
            "differential": ["quantification"],
            "visualization": ["differential"],
            "annotation": ["differential"],
        }

        # Check if all required previous stages are completed
        required_stages = stage_dependencies.get(resume_stage, [])
        missing_stages = []
        invalid_stages = []

        for stage in required_stages:
            if not self.checkpoint_manager.is_stage_complete(stage):
                if stage == "differential":
                    recovered = self._discover_differential_results()
                    if recovered:
                        self.differential_expression_results = recovered["output_file"]
                        self.checkpoint_manager.mark_stage_complete("differential", metadata=recovered, dry_run=self.dryrun)
                        self.logger.info("Recovered differential expression results from existing files")
                        continue
                if stage == "alignment":
                    discovered = self._discover_alignment_results()
                    if discovered:
                        self.alignment_results = discovered
                        self.checkpoint_manager.mark_stage_complete(
                            "alignment",
                            metadata={
                                "tool": self.alignment_tool,
                                "output_files": discovered,
                                "recovered_from_filesystem": True,
                            },
                            dry_run=self.dryrun,
                        )
                        self.logger.info("Recovered alignment results from existing BAM files")
                        continue
                missing_stages.append(stage)
                continue

            metadata = self.checkpoint_manager.get_stage_metadata(stage)
            validation_targets = self._get_stage_validation_targets(stage, metadata, resume_stage)

            if validation_targets:
                if not self.validate_stage_results(stage, validation_targets):
                    if stage == "alignment":
                        discovered = self._discover_alignment_results()
                        if discovered and self.validate_stage_results(stage, discovered):
                            self.alignment_results = discovered
                            self.checkpoint_manager.mark_stage_complete(
                                "alignment",
                                metadata={
                                    "tool": self.alignment_tool,
                                    "output_files": discovered,
                                    "recovered_from_filesystem": True,
                                },
                                dry_run=self.dryrun,
                            )
                            self.logger.info("Recovered alignment results from existing BAM files")
                            continue
                    invalid_stages.append(stage)
                elif stage == "alignment":
                    output_files = metadata.get("output_files") if metadata else None
                    if isinstance(output_files, dict):
                        self.alignment_results = self._normalize_alignment_output_files(output_files)
            elif metadata and metadata.get("detailed_output_files"):
                if not self._validate_stage_files(stage, metadata):
                    invalid_stages.append(stage)

        if missing_stages and not self.dryrun:
            self.logger.error(f"Cannot resume from '{resume_stage}'. Missing completed stages: {missing_stages}")
            return False
        elif missing_stages:
            self.logger.info(f"DRYRUN: Would require completed stages before '{resume_stage}': {missing_stages}")

        if invalid_stages and not self.dryrun:
            self.logger.error(
                f"Cannot resume from '{resume_stage}'. Upstream stage outputs are missing or invalid: {invalid_stages}"
            )
            return False
        elif invalid_stages:
            self.logger.info(f"DRYRUN: Would require valid upstream outputs before '{resume_stage}': {invalid_stages}")

        return True

    def _discover_differential_results(self) -> Optional[Dict[str, Any]]:
        """Recover differential-expression outputs from the filesystem."""
        diffexp_dir = Path(self.output_dir) / "4.Differential_Expression"
        if not diffexp_dir.exists():
            return None

        output_files: List[str] = []
        main_output = diffexp_dir / "All_gene_expression.xlsx"
        if main_output.exists():
            output_files.append(str(main_output))

        sheet_output = diffexp_dir / "All_gene_expression_sheet.xlsx"
        if sheet_output.exists():
            output_files.append(str(sheet_output))

        for name in (
            "Filtered_DEGs.xlsx",
            "Filtered_upDEGs.xlsx",
            "Filtered_downDEGs.xlsx",
            "Filtered_DEGs_summary.xlsx",
            "Filtered_DEG.png",
        ):
            path = diffexp_dir / name
            if path.exists():
                output_files.append(str(path))

        gene_files = self._find_annotation_deg_files(diffexp_dir)
        output_files.extend(str(path) for path in gene_files)

        if not output_files:
            return None

        return {
            "tool": self.diffexp_tool,
            "output_file": str(main_output if main_output.exists() else output_files[0]),
            "output_files": output_files,
            "recovered_from_filesystem": True,
        }

    def _get_stage_validation_targets(
        self,
        stage_name: str,
        metadata: Optional[dict],
        resume_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract concrete checkpoint outputs that can be validated on disk.

        Args:
            stage_name: Stage that produced the checkpoint metadata
            metadata: Stage metadata stored in the checkpoint
            resume_stage: Stage that will consume the checkpoint outputs

        Returns:
            Dict of output labels to files/directories for validate_stage_results()
        """
        if not metadata:
            return {}

        if stage_name == "alignment":
            output_files = metadata.get("output_files")
            if isinstance(output_files, dict):
                output_files = self._normalize_alignment_output_files(output_files)
                bam_targets = {}
                for sample_name, file_info in output_files.items():
                    if isinstance(file_info, dict):
                        bam_path = file_info.get("bam")
                    else:
                        bam_path = file_info

                    if isinstance(bam_path, str):
                        bam_targets[sample_name] = bam_path
                return bam_targets

        if isinstance(metadata.get("output_directories"), dict):
            return metadata["output_directories"]

        output_files = metadata.get("output_files")
        if isinstance(output_files, dict):
            return output_files
        if isinstance(output_files, list):
            return {f"output_{idx + 1}": path for idx, path in enumerate(output_files)}

        output_file = metadata.get("output_file")
        if isinstance(output_file, str):
            return {"primary_output": output_file}

        return {}

    def _normalize_alignment_output_files(self, output_files: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize alignment output metadata to a sample-keyed mapping."""
        normalized = {}
        for sample_name, file_info in output_files.items():
            if isinstance(file_info, dict):
                normalized[sample_name] = file_info
            elif isinstance(file_info, str):
                normalized[sample_name] = file_info
            else:
                normalized[sample_name] = file_info
        return normalized

    def _discover_alignment_results(self) -> Dict[str, Dict[str, str]]:
        """
        Reconstruct alignment BAM outputs from the alignment results directory.

        This is used during resume when alignment finished but checkpoint
        metadata is missing or stale.
        """
        if not self.sample_dict:
            return {}

        results_dir = Path(self.output_dir) / "2.Alignment" / f"{self.alignment_tool.lower()}_results"
        if not results_dir.exists():
            return {}

        discovered = {}
        for sample_name in self.sample_dict:
            candidates = self._alignment_bam_candidates(results_dir, sample_name)
            bam_path = next((path for path in candidates if path.exists()), None)
            if bam_path is None:
                self.logger.warning(f"Could not find alignment BAM for sample {sample_name} in {results_dir}")
                return {}

            discovered[sample_name] = {"bam": str(bam_path)}

        return discovered

    def _alignment_bam_candidates(self, results_dir: Path, sample_name: str) -> List[Path]:
        """Return expected BAM candidates for the configured aligner."""
        tool = self.alignment_tool.lower()
        if tool == "star":
            return [
                results_dir / f"{sample_name}_Aligned.out.bam",
                results_dir / f"{sample_name}_Aligned.sortedByCoord.out.bam",
            ]
        if tool in {"hisat2", "bowtie2", "bwa", "minimap2"}:
            return [results_dir / f"{sample_name}_aligned.bam"]
        return [results_dir / f"{sample_name}_aligned.bam"]

    def _get_required_directories_for_stage(self, stage: str) -> List[str]:
        """Get list of directories required for a specific stage."""
        base_dir = self.output_dir

        # These are prerequisite directories that should already exist before a
        # stage can resume. Do not include the current stage's own output
        # directory here, because the stage itself may need to create it.
        stage_dirs = {
            "quality": [],
            "quality_trim": [f"{base_dir}/1.Quality_and_trimming"],
            "trimming": [f"{base_dir}/1.Quality_and_trimming"],
            "alignment": [f"{base_dir}/1.Quality_and_trimming"],
            "bam_preparation": [f"{base_dir}/2.Alignment"],
            "alignment_stats": [f"{base_dir}/2.Alignment"],
            "quantification": [
                f"{base_dir}/1.Quality_and_trimming",
                f"{base_dir}/2.Alignment",
            ],
            "multimapped_groups": [
                f"{base_dir}/1.Quality_and_trimming",
                f"{base_dir}/2.Alignment",
                f"{base_dir}/3.Quantification",
            ],
            "normalization": [
                f"{base_dir}/1.Quality_and_trimming",
                f"{base_dir}/2.Alignment",
                f"{base_dir}/3.Quantification",
            ],
            "sample_clustering": [f"{base_dir}/4.Normalization"],
            "coexpression": [f"{base_dir}/4.Normalization"],
            "differential": [
                f"{base_dir}/1.Quality_and_trimming",
                f"{base_dir}/2.Alignment",
                f"{base_dir}/3.Quantification",
            ],
            "annotation": [f"{base_dir}/4.Differential_Expression"],
            "visualization": [
                f"{base_dir}/1.Quality_and_trimming",
                f"{base_dir}/2.Alignment",
                f"{base_dir}/3.Quantification",
                f"{base_dir}/4.Differential_Expression",
            ],
        }

        return stage_dirs.get(stage, [])

    def _log_resume_status(self, resume_stage: str):
        """Log comprehensive resume status."""

        # Get completed and pending stages
        completed_stages = self.checkpoint_manager.get_completed_stages()
        pipeline_stages = list(PIPELINE_STAGE_NAMES)

        # Find stages that will be run
        resume_index = pipeline_stages.index(resume_stage)
        stages_to_run = pipeline_stages[resume_index:]

        self.logger.info(f"Resuming PySeqRNA pipeline from: {resume_stage}")
        self.logger.info(f"Completed stages: {', '.join(completed_stages)}")
        self.logger.info(f"Stages to run: {', '.join(stages_to_run)}")

    def _get_resume_stage_plan(self) -> Tuple[List[str], List[str]]:
        """Return stages treated as already satisfied and stages to run now."""
        pipeline_stages = list(PIPELINE_STAGE_NAMES)
        if self.resume == "all" or self.resume not in pipeline_stages:
            return [], pipeline_stages

        resume_index = pipeline_stages.index(self.resume)
        return pipeline_stages[:resume_index], pipeline_stages[resume_index:]

    def _log_pipeline_status_for_current_run(self) -> None:
        """Log status in terms of the current run/resume plan."""
        status = self.get_pipeline_status()
        self.logger.info(f"Pipeline: {status['pipeline_name']}")

        if self.resume == "all":
            self.logger.info(f"Total stages: {status['total_stages']}")
            self.logger.info(f"Completed stages: {status['completed_stages']}")
            self.logger.info(f"Incomplete stages: {status['incomplete_stages']}")
            self.logger.info(f"Completion percentage: {status['completion_percentage']:.1f}%")
            return

        satisfied_stages, stages_to_run = self._get_resume_stage_plan()
        self.logger.info(f"Resume stage: {self.resume}")
        self.logger.info(f"Stages satisfied before resume: {len(satisfied_stages)}")
        self.logger.info(f"Stages scheduled for this run: {', '.join(stages_to_run)}")

    def _check_required_directories(self, resume_stage: str):
        """Check and log directory status for resume."""
        self.logger.info("Checking required directories...")

        # Get all stages that will be run
        pipeline_stages = list(PIPELINE_STAGE_NAMES)
        resume_index = pipeline_stages.index(resume_stage)
        stages_to_run = pipeline_stages[resume_index:]

        # Check directories for each stage to be run
        for stage in stages_to_run:
            required_dirs = self._get_required_directories_for_stage(stage)
            for dir_path in required_dirs:
                if Path(dir_path).exists():
                    self.logger.info(f"Using existing directory: {dir_path}")
                else:
                    self.logger.info(f"Will create directory: {dir_path}")

    def _ask_user_rerun_stage(self, stage_name: str, completed_tool: str = None) -> bool:
        """
        Resolve whether to re-run a completed stage.

        Args:
            stage_name: Name of the stage
            completed_tool: Tool that was used for completion (if applicable)

        Returns:
            bool: True if user wants to re-run, False to skip
        """
        tool_info = f" with {completed_tool}" if completed_tool else ""
        safe_stage_name = str(stage_name).replace("\n", "").replace("\r", "").replace("\t", "")

        if self.resume_policy == "skip":
            self.logger.info(f"Skipping completed {safe_stage_name} stage{tool_info}; resume_policy=skip")
            return False
        if self.resume_policy == "rerun":
            self.logger.info(f"Re-running completed {safe_stage_name} stage{tool_info}; resume_policy=rerun")
            return True
        if self.resume_policy == "fail":
            raise ValueError(
                f"Stage '{safe_stage_name}' is already complete{tool_info}; "
                "resume_policy=fail prevents implicit reuse or overwrite."
            )
        if not self._can_prompt_user():
            raise ValueError(
                f"Stage '{safe_stage_name}' is already complete{tool_info}, but this run is "
                "non-interactive. Set resume_policy to skip, rerun, or fail."
            )

        print(f"\n{'=' * 60}")
        print(f"WARNING: STAGE ALREADY COMPLETED: {stage_name.upper()}{tool_info}")
        print(f"{'=' * 60}")
        print(f"Stage '{stage_name}' has already been completed successfully.")
        print("What would you like to do?")
        print("1. Skip this stage (use existing results)")
        print("2. Re-run this stage (overwrite existing results)")
        print("3. Cancel the entire pipeline")
        print(f"{'=' * 60}")

        while True:
            try:
                choice = input("Enter your choice (1/2/3): ").strip()
                if choice == "1":
                    self.logger.info(f"User chose to skip {stage_name} stage (using existing results)")
                    return False
                elif choice == "2":
                    self.logger.info(f"User chose to re-run {safe_stage_name} stage (will overwrite existing results)")
                    return True
                elif choice == "3":
                    self.logger.info("User chose to cancel the pipeline")
                    return None  # Special value to indicate cancellation
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\nPipeline cancelled by user.")
                return None
            except EOFError:
                print("\nPipeline cancelled by user.")
                return None

    def _can_prompt_user(self) -> bool:
        """Return True when it is safe to ask an interactive question."""
        return (not self.slurm) and sys.stdin.isatty()

    def _validate_stage_files(self, stage_name: str, metadata: dict) -> bool:
        """
        Enhanced validation of stage output files with detailed checks.

        Args:
            stage_name: Name of the stage
            metadata: Stage metadata from checkpoint

        Returns:
            bool: True if all files are valid
        """
        if not metadata:
            self.logger.warning(f"No metadata found for {stage_name}")
            return False

        detailed_files = metadata.get("detailed_output_files", {})
        if not detailed_files:
            self.logger.warning(f"No detailed file information for {stage_name}")
            return True  # Fallback to basic validation

        all_valid = True
        total_size = 0

        self.logger.info(f"Validating {stage_name} output files:")

        for file_path, file_info in detailed_files.items():
            if file_info.get("exists", False):
                current_path = Path(file_path)
                if current_path.exists():
                    current_size = current_path.stat().st_size
                    expected_size = file_info.get("size_bytes", 0)

                    # Check if file size matches (within 1% tolerance)
                    size_diff = abs(current_size - expected_size) / expected_size if expected_size > 0 else 0

                    if size_diff <= 0.01:  # 1% tolerance
                        self.logger.info(f"  OK: {current_path.name}: {file_info.get('size_mb', 0)} MB")
                        total_size += current_size
                    else:
                        self.logger.warning(
                            f"  WARNING: {current_path.name}: Size mismatch (expected {file_info.get('size_mb', 0)} MB, got {round(current_size / (1024 * 1024), 2)} MB)"
                        )
                        all_valid = False
                else:
                    # Sanitize filename to prevent log injection
                    safe_filename = str(current_path.name).replace("\n", "").replace("\r", "").replace("\t", "")
                    self.logger.error(f"  ERROR: {safe_filename}: File missing")
                    all_valid = False
            else:
                self.logger.error(f"  ERROR: {file_path}: {file_info.get('error', 'Unknown error')}")
                all_valid = False

        if all_valid:
            self.logger.info(
                f"All {stage_name} files validated successfully (Total: {round(total_size / (1024 * 1024), 2)} MB)"
            )
        else:
            self.logger.warning(f"WARNING: Some {stage_name} files failed validation")

        return all_valid

    def _record_internal_operation(self, operation_type: str, details: str) -> None:
        """
        Record internal pipeline operations for execution reporting.

        Args:
            operation_type: Type of operation (e.g., 'gene_ontology_start')
            details: Details about the operation
        """
        if hasattr(self, "dry_run_manager") and self.dry_run_manager:
            operation_record = {
                "operation": "pipeline_internal",
                "operation_type": operation_type,
                "details": details,
                "stage": "pipeline",
                "timestamp": self.dry_run_manager._get_timestamp(),
            }

            if self.dryrun:
                self.dry_run_manager.simulated_operations.append(operation_record)
            else:
                self.dry_run_manager.executed_operations.append(operation_record)

    def _convert_numpy_for_json(self, obj: Any) -> Any:
        """Convert NumPy values to JSON-serializable Python values."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        if isinstance(obj, tuple):
            return [self._convert_numpy_for_json(item) for item in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def _git_metadata(self) -> Dict[str, Any]:
        """Return lightweight git metadata for the current project directory."""
        repo_dir = Path(__file__).resolve().parents[2]

        def run_git(args: List[str]) -> Optional[str]:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=str(repo_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=10,
                )
            except Exception:
                return None
            if result.returncode != 0:
                return None
            return result.stdout.strip() or None

        return {
            "commit": run_git(["rev-parse", "HEAD"]),
            "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(run_git(["status", "--porcelain"])),
        }

    def _write_run_record(self, success: bool, error: Optional[str] = None) -> Path:
        """Write a structured run record for reproducibility and support."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_record_path = output_dir / "pyseqrna_run_record.json"

        payload = {
            "schema_version": 1,
            "pipeline": "pyseqrna",
            "pyseqrna_version": __version__,
            "success": bool(success),
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "git": self._git_metadata(),
            "inputs": {
                "input_file": self.input_file,
                "samples_path": self.samples_path,
                "reference_genome": self.reference_genome,
                "feature_file": self.feature_file,
            },
            "execution": {
                "output_dir": str(output_dir),
                "threads": self.threads,
                "memory_gb": self.memory,
                "local_jobs": self.local_jobs,
                "slurm": self.slurm,
                "slurm_partition": self.slurm_partition,
                "slurm_wait_timeout_hours": self.slurm_wait_timeout_hours,
                "resume": self.resume,
                "resume_policy": self.resume_policy,
                "dryrun": self.dryrun,
            },
            "tools": {
                "quality": self.quality_tool,
                "trimming": self.trimming_tool,
                "alignment": self.alignment_tool,
                "quantification": self.quant_method,
                "normalization": self.normalization_method,
                "diffexp": self.diffexp_tool,
            },
            "completed_stages": self.checkpoint_manager.get_completed_stages() if self.checkpoint_manager else [],
            "executed_operations": self.dry_run_manager.executed_operations if self.dry_run_manager else [],
        }

        run_record_path.write_text(
            json.dumps(self._convert_numpy_for_json(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.logger.info(f"Wrote structured run record: {run_record_path}")
        return run_record_path

    def _setup_slurm_config(self):
        """Setup SLURM configuration if SLURM is enabled."""
        try:
            self.logger.info("Setting up SLURM configuration")

            # Calculate memory in MB (assuming 8GB per thread as default)
            memory_mb = self.threads * 8000

            # Create SLURM configuration
            slurm_config_path = self.resource_manager.write_slurm_ini(
                partition=self.slurm_partition,
                cpus=self.threads,
                memory=memory_mb,
                ntasks=1,
                account=self.slurm_account if self.slurm_account else "",
                time=self.slurm_time,
                email=self.slurm_email if self.slurm_email else "",
                qos=self.slurm_qos if self.slurm_qos else "",
                output_dir=self.output_dir,
            )

            self.logger.info(f"SLURM configuration created: {slurm_config_path}")

            # Log SLURM parameters
            slurm_params = []
            if self.slurm_partition:
                slurm_params.append(f"partition={self.slurm_partition}")
            if self.slurm_account:
                slurm_params.append(f"account={self.slurm_account}")
            if self.slurm_time:
                slurm_params.append(f"time={self.slurm_time}")
            if self.slurm_email:
                slurm_params.append(f"email={self.slurm_email}")
            if self.slurm_qos:
                slurm_params.append(f"qos={self.slurm_qos}")

            if slurm_params:
                self.logger.info(f"SLURM parameters: {', '.join(slurm_params)}")

            if not self.dryrun:
                self.command_executor.validate_slurm_environment(partition=self.slurm_partition)

        except Exception as e:
            self.logger.error(f"SLURM configuration setup failed: {e}")
            raise

    def _get_slurm_config(self) -> Dict[str, str]:
        """Return SLURM settings used by modules that submit internal jobs."""
        config = {
            "partition": self.slurm_partition or "compute",
            "time": self.slurm_time or "24:00:00",
            "memory": str(self.memory or 16),
            "cpus": str(self.threads or 1),
            "ntasks": "1",
            "wait_timeout_seconds": str(int(float(self.slurm_wait_timeout_hours) * 3600)),
        }
        if self.slurm_account:
            config["account"] = self.slurm_account
        if self.slurm_email:
            config["email"] = self.slurm_email
        if self.slurm_qos:
            config["qos"] = self.slurm_qos
        return config

    def _get_sample_parallel_resources(self, stage_name: str, task_count: int) -> Tuple[int, Dict[str, str]]:
        """
        Return per-task CPU threads and SLURM config for sample-parallel stages.

        Pipeline-level ``threads`` and ``memory`` are reserved for single jobs
        such as index building. Sample-level arrays use explicit per-task
        resources so users can control scheduler pressure without hidden math.
        """
        base_config = self._get_slurm_config()
        if not self.slurm:
            local_jobs = max(1, min(self.local_jobs, max(1, int(task_count or 1))))
            cpus_per_task = max(1, int(self.threads or 1) // local_jobs)
            base_config.update(
                {
                    "local_jobs": str(local_jobs),
                    "cpus": str(cpus_per_task),
                }
            )
            self.logger.info(
                f"{stage_name} local resources: {task_count} task(s), "
                f"local_jobs={local_jobs}, threads_per_task={cpus_per_task}"
            )
            return cpus_per_task, base_config

        default_cpus, default_memory = self._get_stage_slurm_defaults(stage_name)
        cpus_per_task = max(1, int(self.slurm_cpus_per_task or default_cpus))
        memory_per_task = max(1, int(self.slurm_memory_per_task or default_memory))
        max_parallel = max(1, int(self.slurm_array_max_parallel or 10))
        max_parallel = min(max_parallel, max(1, task_count))

        base_config.update(
            {
                "cpus": str(cpus_per_task),
                "memory": str(memory_per_task),
                "array_max_parallel": str(max_parallel),
            }
        )

        self.logger.info(
            f"{stage_name} SLURM array resources: {task_count} task(s), "
            f"max_parallel={max_parallel}, cpus_per_task={cpus_per_task}, "
            f"memory_per_task={memory_per_task}GB"
        )
        return cpus_per_task, base_config

    def _get_stage_slurm_defaults(self, stage_name: str) -> Tuple[int, int]:
        """Return default CPU and memory per sample-array task for a stage."""
        stage = stage_name.lower()
        if "fastqc" in stage:
            return 2, 8
        if "trim" in stage or "trimmomatic" in stage or "flexbar" in stage:
            return 4, 16
        if "bam_preparation" in stage:
            return 8, 32
        if "star" in stage and "alignment" in stage:
            return 8, 64
        if any(tool in stage for tool in ("hisat2", "bowtie2", "bwa", "minimap2")):
            return 8, 32
        return 4, 16

    def run_quality_control(self) -> bool:
        """
        Run quality control using modular implementation.

        Returns:
            True if successful, False otherwise
        """
        return self.preprocessing.run_quality_control()

    def run_quality_control_trim(self) -> bool:
        """
        Run quality control on trimmed reads using modular implementation.

        Returns:
            True if successful, False otherwise
        """
        return self.preprocessing.run_quality_control_trim()

    def run_trimming(self) -> bool:
        """
        Run read trimming using the selected trimming tool.

        Returns:
            True if successful, False otherwise
        """
        return self.preprocessing.run_trimming()

    def run_alignment(self) -> bool:
        """
        Run read alignment using the selected alignment tool.

        Returns:
            True if successful, False otherwise
        """
        return self.alignment.run_alignment()

    def _extract_bam_path(self, sample_info: Any) -> Optional[str]:
        """Extract a BAM path from a string, list, or result dictionary."""
        return self.alignment._extract_bam_path(sample_info)

    def _load_alignment_results_from_checkpoint(self) -> bool:
        """Load alignment outputs from checkpoint into pipeline state."""
        return self.alignment._load_alignment_results_from_checkpoint()

    def _load_prepared_bams_from_checkpoint(self) -> bool:
        """Load prepared BAM outputs from checkpoint into pipeline state."""
        return self.alignment._load_prepared_bams_from_checkpoint()

    def _prepared_bam_is_current(self, bam_file: Path, bai_file: Path) -> bool:
        """Return True when a BAM and its index exist and the index is current."""
        return self.alignment._prepared_bam_is_current(bam_file, bai_file)

    def _build_bam_preparation_command(
        self,
        sample_id: str,
        input_bam: Path,
        prepared_bam: Path,
        marker_file: Path,
        threads: int,
        memory_gb: int,
    ) -> str:
        """Build a portable shell command that coordinate-sorts and indexes one BAM."""
        return self.alignment._build_bam_preparation_command(
            sample_id, input_bam, prepared_bam, marker_file, threads, memory_gb
        )

    def run_bam_preparation(self) -> bool:
        """
        Prepare alignment BAMs once for downstream reuse.

        Coordinate-sorted and indexed BAMs are reused by alignment statistics,
        quantification, and multimapped-group analysis.
        """
        return self.alignment.run_bam_preparation()

    def run_alignment_statistics(self) -> bool:
        """Run alignment statistics using logs first when configured."""
        return self.alignment.run_alignment_statistics()

    def run_multimapped_groups(self) -> bool:
        """
        Run multimapped groups analysis using aligned BAM files.

        Returns:
            True if successful, False otherwise
        """
        return self.quantification.run_multimapped_groups()

    def run_quantification(self) -> bool:
        """
        Run gene expression quantification using the selected quantification tool.

        Returns:
            True if successful, False otherwise
        """
        return self.quantification.run_quantification()

    def run_normalization(self) -> bool:
        """
        Run count normalization step.

        Returns:
            bool: True if normalization successful, False otherwise
        """
        return self.quantification.run_normalization()

    def run_sample_clustering(self) -> bool:
        """Run sample similarity clustering from normalized counts."""
        return self.downstream.run_sample_clustering()

    def run_coexpression_analysis(self) -> bool:
        """Run gene co-expression analysis from normalized counts."""
        return self.downstream.run_coexpression_analysis()

    def run_differential_expression(self) -> bool:
        """
        Run differential expression analysis using the selected tool.

        Returns:
            True if successful, False otherwise
        """
        return self.diffexp.run_differential_expression()

    def _run_mmg_differential_expression(
        self,
        diffexp_dir: Path,
        processed_targets: pd.DataFrame,
        actual_combinations: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Run differential expression and filtering for multimapped gene groups."""
        raw_mmg_file = Path(self.output_dir) / "3.Quantification" / "Raw_MMGcounts.xlsx"
        if not raw_mmg_file.exists():
            self.logger.warning(
                "Multimapped groups were requested, but %s was not found. Skipping MMG differential expression.",
                raw_mmg_file,
            )
            return None

        try:
            from ..modules.diffexp import create_diffexp_analyzer, create_deg_filter

            self.logger.info(
                "Starting MMG differential expression analysis with %s",
                self.diffexp_tool,
            )
            raw_mmg_counts = pd.read_excel(raw_mmg_file)
            if raw_mmg_counts.empty or "MMG" not in raw_mmg_counts.columns or "Gene" not in raw_mmg_counts.columns:
                self.logger.warning(
                    "Raw MMG counts file is empty or missing MMG/Gene columns; skipping MMG differential expression"
                )
                return None

            mmg_gene_map = raw_mmg_counts[["MMG", "Gene"]].drop_duplicates()
            mmg_count_matrix = raw_mmg_counts.drop(columns=["Gene"]).rename(columns={"MMG": "Gene"})

            mmg_module = create_diffexp_analyzer(
                tool_name=self.diffexp_tool,
                count_matrix_file=mmg_count_matrix,
                sample_info_file=processed_targets,
                comparisons=actual_combinations,
                out_dir=str(diffexp_dir),
                param_dir=self.param_dir,
                species=None,
                organism_type=self.organism_type,
                add_gene_names=False,
                gene_column="Gene",
                logger=self.logger.logger,
                fdr_threshold=self.fdr_threshold,
                log2fc_threshold=self.log2fc_threshold,
                dryrun=self.dryrun,
                dry_run_manager=self.dry_run_manager,
                subset=self.subset,
                diffexp_normalization=self.diffexp_normalization,
                diffexp_abundance=self.diffexp_abundance,
                diffexp_dispersion=self.diffexp_dispersion,
                diffexp_test=self.diffexp_test,
            )

            mmg_analysis = mmg_module.run(save_results=False, filter_results=False)
            combined_results = mmg_analysis.get("results", {}).get("combined_results")
            if combined_results is None or combined_results.empty:
                self.logger.warning("MMG differential expression produced no combined results")
                return None

            combined_results = combined_results.rename(columns={"Gene": "MMG"})
            combined_results = combined_results.merge(mmg_gene_map, on="MMG", how="left")
            ordered_columns = ["MMG", "Gene"] + [col for col in combined_results.columns if col not in {"MMG", "Gene"}]
            combined_results = combined_results[ordered_columns]

            output_files: List[str] = []
            all_mmg_file = diffexp_dir / "All_MMG_expression.xlsx"
            combined_results.to_excel(all_mmg_file, index=False)
            output_files.append(str(all_mmg_file))
            self.logger.info(f"Saved MMG differential expression results to: {all_mmg_file}")

            mmg_filter = create_deg_filter(
                fdr_threshold=self.fdr_threshold,
                fold_threshold=2**self.log2fc_threshold,
                has_replicates=True,
                mmg=True,
                extra_columns=False,
                logger=self.logger.logger,
            )
            filtered_mmg = mmg_filter.filter_degs(
                deg_df=combined_results,
                compare_list=actual_combinations,
                create_plot=True,
                save_plot_path=str(diffexp_dir / "Filtered_MMG.png"),
            )

            output_files.extend(self._write_mmg_filtered_outputs(diffexp_dir, filtered_mmg, actual_combinations))
            total_mmgs = int(filtered_mmg["summary"]["Total_DEGs"].sum()) if "summary" in filtered_mmg else 0

            self.logger.info(
                "MMG differential expression completed successfully: %d filtered MMG(s)",
                total_mmgs,
            )
            return {
                "summary": {
                    "tool": self.diffexp_tool,
                    "total_mmgs": len(combined_results),
                    "filtered_mmgs": total_mmgs,
                    "comparisons": len(actual_combinations),
                },
                "output_files": output_files,
            }

        except Exception as e:
            self.logger.error(f"MMG differential expression failed: {e}")
            return None

    def _write_mmg_filtered_outputs(
        self,
        diffexp_dir: Path,
        filtered_mmg: Dict[str, Any],
        comparisons: List[str],
    ) -> List[str]:
        """Write filtered MMG Excel files and annotation-ready gene-list files."""
        output_files: List[str] = []

        def _write_sheets(sheet_data: Dict[str, pd.DataFrame], filename: str) -> str:
            path = diffexp_dir / filename
            with pd.ExcelWriter(path) as writer:
                wrote_sheet = False
                for comparison, df in sheet_data.items():
                    if df is not None and not df.empty:
                        df.to_excel(writer, sheet_name=comparison[:31], index=False)
                        wrote_sheet = True
                if not wrote_sheet:
                    pd.DataFrame({"Message": ["No filtered MMGs found"]}).to_excel(
                        writer,
                        sheet_name="No_MMGs",
                        index=False,
                    )
            self.logger.info(f"Saved {filename} to: {path}")
            return str(path)

        output_files.append(_write_sheets(filtered_mmg.get("filtered", {}), "Filtered_MMGs.xlsx"))
        output_files.append(_write_sheets(filtered_mmg.get("filteredup", {}), "Filtered_upMMGs.xlsx"))
        output_files.append(_write_sheets(filtered_mmg.get("filtereddown", {}), "Filtered_downMMGs.xlsx"))

        summary_file = diffexp_dir / "Filtered_MMGs_summary.xlsx"
        filtered_mmg["summary"].to_excel(summary_file, index=False)
        output_files.append(str(summary_file))
        self.logger.info(f"Saved filtered MMG summary to: {summary_file}")

        diff_genes_dir = diffexp_dir / "diff_genes"
        diff_genes_dir.mkdir(parents=True, exist_ok=True)
        for comparison in comparisons:
            df = filtered_mmg.get("filtered", {}).get(comparison)
            if df is None or df.empty or "Gene" not in df.columns:
                self.logger.warning(f"No filtered MMGs for comparison {comparison}; skipping MMG gene-list creation")
                continue

            genes = df["Gene"].dropna().astype(str).str.replace("gene:", "", regex=False).str.split("-").explode().str.strip()
            genes = genes[genes.astype(bool)].drop_duplicates().str.upper()
            if genes.empty:
                continue

            gene_file = diff_genes_dir / f"{comparison}_mmg.txt"
            genes.to_csv(gene_file, sep="\t", index=False, header=False)
            output_files.append(str(gene_file))
            self.logger.info(f"Created MMG gene file: {gene_file} ({len(genes)} genes)")

        return output_files

    def validate_stage_results(self, stage_name: str, results: Dict) -> bool:
        """
        Validate that stage result files/directories actually exist.

        Args:
            stage_name: Name of the stage
            results: Results dictionary to validate

        Returns:
            bool: True if all results exist and are valid
        """
        if not results:
            self.logger.warning(f"No results to validate for stage: {stage_name}")
            return False

        all_valid = True
        metadata_keys = {
            "job_id",
            "slurm_job_id",
            "dependency",
            "status",
            "tool",
            "command",
        }

        for sample_name, result_path in results.items():
            if isinstance(result_path, (list, tuple)):
                # Handle multiple files (e.g., paired-end trimming results)
                for path in result_path:
                    if not Path(path).exists():
                        self.logger.warning(f"Missing result file for {stage_name} {sample_name}: {path}")
                        all_valid = False
            elif isinstance(result_path, dict):
                # Handle complex result structures (e.g., alignment results with multiple files)
                for file_type, path in result_path.items():
                    if file_type in metadata_keys:
                        continue
                    if isinstance(path, str) and not Path(path).exists():
                        self.logger.warning(f"Missing {file_type} file for {stage_name} {sample_name}: {path}")
                        all_valid = False
            elif isinstance(result_path, str):
                # Handle single file/directory results
                if not Path(result_path).exists():
                    self.logger.warning(f"Missing result for {stage_name} {sample_name}: {result_path}")
                    all_valid = False

        if all_valid:
            self.logger.info(f"All {stage_name} results validated successfully")
        else:
            self.logger.warning(f"Some {stage_name} results are missing or invalid")

        return all_valid

    def get_pipeline_status(self) -> dict:
        """
        Get the current pipeline status.

        Returns:
            dict: Pipeline status information
        """
        if not self.checkpoint_manager:
            return {"status": "not_initialized"}

        return self.checkpoint_manager.get_pipeline_status()

    def _report_config(self) -> Dict[str, Any]:
        """Collect user-facing configuration values for comprehensive reports."""
        keys = [
            "threads",
            "memory",
            "local_jobs",
            "resume_policy",
            "dryrun",
            "force_paired",
            "force_single_end",
            "species",
            "organism_type",
            "source",
            "add_gene_names",
            "quality_tool",
            "quality_trim",
            "skip_quality",
            "trimming_tool",
            "skip_trim",
            "alignment_tool",
            "skip_alignment",
            "alignment_stats",
            "alignment_stats_source",
            "quant_method",
            "quantification_tool",
            "skip_quantification",
            "mmg_min_count",
            "mmg_percent_sample",
            "mmg_feature",
            "mmg_min_overlap",
            "mmg_fraction_overlap",
            "mmg_include_ambiguous_unique",
            "mmg_collapse_contained_groups",
            "normalize_counts",
            "normalization_method",
            "skip_normalization_plots",
            "run_clustering",
            "cluster_target",
            "cluster_method",
            "cluster_count",
            "cluster_metric",
            "cluster_linkage",
            "cluster_top_variable",
            "cluster_scale",
            "run_coexpression",
            "coexpression_tool",
            "diffexp_tool",
            "diffexp_normalization",
            "diffexp_abundance",
            "diffexp_dispersion",
            "diffexp_test",
            "skip_diffexp",
            "fdr_threshold",
            "log2fc_threshold",
            "pvalue_threshold",
            "subset",
            "pca_plot",
            "tsne_plot",
            "volcano_plot",
            "ma_plot",
            "deg_heatmap",
            "heatmap_top_genes",
            "venn",
            "upset",
            "venn_comparisons",
            "venn_label",
            "gene_ontology",
            "kegg_pathway",
            "skip_functional_annotation",
            "go_pvalue_threshold",
            "kegg_pvalue_threshold",
            "slurm",
            "slurm_partition",
            "slurm_account",
            "slurm_time",
            "slurm_email",
            "slurm_qos",
            "slurm_array_max_parallel",
            "slurm_cpus_per_task",
            "slurm_memory_per_task",
            "slurm_wait_timeout_hours",
            "config_file",
            "param_dir",
        ]
        config = {key: getattr(self, key, None) for key in keys}
        config["run_multimapped_groups"] = self.enable_multimapped_groups
        return config

    def run_report(self) -> bool:
        """Generate comprehensive analysis reports from the run directory."""
        return self.downstream.run_report()

    def run_visualization(self) -> bool:
        """
        Run visualization stage.

        Returns:
            True if successful, False otherwise
        """
        return self.visualization.run_visualization()

    def _find_annotation_deg_files(self, diffexp_dir: Path) -> List[Path]:
        """Find annotation-ready DEG gene-list files from the differential stage."""
        diff_genes_dir = diffexp_dir / "diff_genes"
        search_dirs = [diff_genes_dir, diffexp_dir]
        deg_files: List[Path] = []

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for file_path in sorted(search_dir.glob("*.txt")):
                # Functional annotation should use the complete filtered DEG
                # list. Up/down-specific lists are available for users, but
                # running all three by default creates duplicated enrichment.
                if file_path.stem.endswith(("_up", "_down")):
                    continue
                if file_path.is_file() and (self.dryrun or file_path.stat().st_size > 0):
                    deg_files.append(file_path)

            if deg_files:
                self.logger.info(
                    "Found %d annotation DEG gene-list file(s) in %s",
                    len(deg_files),
                    search_dir,
                )
                break

        return deg_files

    def run_gene_ontology(self) -> bool:
        """
        Run Gene Ontology enrichment analysis on DEG files.

        Returns:
            True if successful, False otherwise
        """
        return self.annotation.run_gene_ontology()

    def run_pathway_enrichment(self) -> bool:
        """
        Run KEGG pathway enrichment analysis on DEG files.

        Returns:
            True if successful, False otherwise
        """
        return self.annotation.run_pathway_enrichment()

    def run_annotation(self) -> bool:
        """
        Run functional annotation stages enabled by the user.

        Returns:
            True if all requested annotation stages complete successfully, False otherwise
        """
        return self.annotation.run_annotation()

    def run(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting PySeqRNA pipeline execution")

            # Setup pipeline
            if not self.setup():
                return False

            # Show pipeline status
            self._log_pipeline_status_for_current_run()

            pipeline_stages = get_pipeline_stages(self)

            # Determine which stages to run based on resume point
            stages_to_run = []
            if self.resume == "all":
                stages_to_run = pipeline_stages
            else:
                # Find the resume point and run from there
                resume_found = False
                for stage_name, stage_func in pipeline_stages:
                    if stage_name == self.resume:
                        resume_found = True
                    if resume_found:
                        stages_to_run.append((stage_name, stage_func))

                if not resume_found:
                    self.logger.error(f"Invalid resume point: {self.resume}")
                    return False

            # Run pipeline stages
            for stage_name, stage_func in stages_to_run:
                self.logger.info(f"Running stage: {stage_name}")

                # Handle special cases for stages that depend on optional analyses or loaded state.
                if stage_name == "bam_preparation":
                    if self.skip_alignment:
                        self.logger.info("Skipping BAM preparation because alignment is disabled")
                        continue
                    if not self.alignment_results and not self._load_alignment_results_from_checkpoint():
                        self.logger.warning("Skipping BAM preparation as alignment results are missing")
                        continue

                elif stage_name == "alignment_stats":
                    if not self.alignment_stats:
                        self.logger.info("Skipping alignment statistics as requested")
                        continue
                    if not self._load_prepared_bams_from_checkpoint():
                        if self._load_alignment_results_from_checkpoint():
                            if not self.run_bam_preparation():
                                return False
                        else:
                            self.logger.warning("Skipping alignment statistics as alignment results are missing")
                            continue
                    if not self.alignment_results:
                        self.logger.warning("Skipping alignment statistics as alignment results are missing")
                        continue

                elif stage_name == "multimapped_groups":
                    if not self.enable_multimapped_groups:
                        self.logger.info("Skipping multimapped groups analysis as requested")
                        continue
                    elif not self._load_prepared_bams_from_checkpoint():
                        if self._load_alignment_results_from_checkpoint():
                            if not self.run_bam_preparation():
                                return False
                        else:
                            self.logger.warning("Skipping multimapped groups analysis as alignment results are missing")
                            continue
                    if not self.alignment_results:
                        self.logger.warning("Skipping multimapped groups analysis as alignment results are missing")
                        continue

                elif stage_name == "quantification":
                    if self.skip_quantification:
                        self.logger.info("Skipping quantification as requested")
                        continue
                    elif not self._load_prepared_bams_from_checkpoint():
                        if self._load_alignment_results_from_checkpoint():
                            if not self.run_bam_preparation():
                                return False
                        else:
                            self.logger.warning("Skipping quantification as alignment results are missing")
                            continue
                    if not self.alignment_results:
                        self.logger.warning("Skipping quantification as alignment results are missing")
                        continue

                elif stage_name == "normalization":
                    if not self.normalize_counts:
                        self.logger.info("Skipping normalization as requested (--skip-normalization)")
                        continue
                    elif self.quantification_results is None:
                        self.logger.warning("Skipping normalization as quantification results are missing")
                        continue

                elif stage_name == "sample_clustering":
                    if not self.run_clustering:
                        self.logger.info("Skipping sample clustering as requested")
                        continue
                    elif not self.normalize_counts:
                        self.logger.warning("Skipping sample clustering as normalization is disabled")
                        continue

                elif stage_name == "coexpression":
                    if not self.enable_coexpression:
                        self.logger.info("Skipping co-expression analysis as requested")
                        continue
                    elif not self.normalize_counts:
                        self.logger.warning("Skipping co-expression analysis as normalization is disabled")
                        continue

                elif stage_name == "differential":
                    if self.skip_diffexp:
                        self.logger.info("Skipping differential expression as requested")
                        continue
                    elif self.quantification_results is None:
                        # Try to load quantification results from checkpoint when resuming
                        if self.checkpoint_manager.is_stage_complete("quantification"):
                            quantification_metadata = self.checkpoint_manager.get_stage_metadata("quantification")
                            if quantification_metadata and "output_file" in quantification_metadata:
                                self.quantification_results = quantification_metadata["output_file"]
                                self.logger.info("Loaded quantification results from checkpoint for differential expression")
                            else:
                                self.logger.warning("Skipping differential expression as quantification results are missing")
                                continue
                        else:
                            self.logger.warning("Skipping differential expression as quantification results are missing")
                            continue

                elif stage_name == "annotation":
                    if self.skip_functional_annotation:
                        self.logger.info("Skipping functional annotation as requested")
                        continue
                    elif not self.gene_ontology and not self.kegg_pathway:
                        self.logger.info("No functional annotation analyses requested")
                        continue
                    elif not self.species:
                        self.logger.warning("Skipping functional annotation: species not specified")
                        continue
                    elif not self.checkpoint_manager.is_stage_complete("differential"):
                        recovered = self._discover_differential_results()
                        if recovered:
                            self.differential_expression_results = recovered["output_file"]
                            self.checkpoint_manager.mark_stage_complete(
                                "differential", metadata=recovered, dry_run=self.dryrun
                            )
                            self.logger.info("Recovered differential expression results from existing files")
                        else:
                            self.logger.warning("Skipping functional annotation: differential expression outputs not found")
                            continue

                # Execute the stage
                if not stage_func():
                    return False

            # Generate execution report (dry-run or actual execution)
            if self.dryrun:
                report_path = Path(self.output_dir) / "dry_run_report.txt"
                self.dry_run_manager.generate_dry_run_report(str(report_path))
                self.logger.info("Dry-run report generated")
            else:
                self.logger.info(
                    "Skipping legacy execution_report.txt; run status is recorded in checkpoint and final reports"
                )

            self._write_run_record(success=True)

            if not self.run_report():
                return False

            self.logger.info("PySeqRNA pipeline completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            try:
                self._write_run_record(success=False, error=str(e))
            except Exception as record_error:
                self.logger.warning(f"Failed to write run record after pipeline error: {record_error}")
            return False
        finally:
            self.logger.close_logger()

    def _detect_paired_end_data(self) -> bool:
        """
        Detect if the data is paired-end with validation and user confirmation.

        Handles three modes:
        1. --paired: Force paired-end mode, validate data is actually paired-end
        2. --single-end: Force single-end mode even if data appears paired-end
        3. Auto-detect: Detect paired-end and confirm with user (unless --no-confirm)

        Returns:
            bool: True if paired-end data should be used, False for single-end

        Raises:
            ValueError: If validation fails (e.g., --paired but data is single-end)
        """
        # Return cached result if available
        if self._paired_end_cache is not None:
            return self._paired_end_cache

        if not self.sample_dict:
            self.logger.warning("No sample data available for paired-end detection")
            self._paired_end_cache = False
            return self._paired_end_cache

        # Check if data appears to be paired-end
        appears_paired = self._check_if_data_appears_paired()

        # Handle forced modes first
        if self.force_paired:
            if not appears_paired:
                # Warn user and ask for confirmation
                if self.dryrun:
                    self.logger.warning("User specified --paired but data appears to be single-end (dry-run mode)")
                    self._paired_end_cache = True
                    return self._paired_end_cache
                elif not self._can_prompt_user():
                    raise ValueError(
                        "User specified paired-end mode, but input files appear single-end. "
                        "Non-interactive runs cannot ask for confirmation; fix the sample sheet "
                        "or disable paired-end mode."
                    )
                else:
                    result = self._confirm_forced_paired_with_single_data()
                    self._paired_end_cache = result
                    return result
            self.logger.info("Using paired-end mode (user specified --paired)")
            self._paired_end_cache = True
            return self._paired_end_cache

        if self.force_single_end:
            if appears_paired:
                # Warn user and ask for confirmation
                if self.dryrun:
                    self.logger.warning("User specified --single-end but data appears to be paired-end (dry-run mode)")
                    self._paired_end_cache = False
                    return self._paired_end_cache
                elif not self._can_prompt_user():
                    self.logger.warning(
                        "User specified single-end mode but data appears paired-end; "
                        "continuing with single-end mode in non-interactive run"
                    )
                    self._paired_end_cache = False
                    return self._paired_end_cache
                else:
                    result = self._confirm_forced_single_with_paired_data()
                    self._paired_end_cache = result
                    return result
            self.logger.info("Using single-end mode (user specified --single-end)")
            self._paired_end_cache = False
            return self._paired_end_cache

        # Auto-detection mode (no flags specified)
        if appears_paired:
            if self.dryrun:
                self.logger.info("Auto-detected paired-end data (dry-run mode)")
                self._paired_end_cache = True
                return self._paired_end_cache
            elif not self._can_prompt_user():
                self.logger.info("Auto-detected paired-end data; using paired-end mode in non-interactive run")
                self._paired_end_cache = True
                return self._paired_end_cache
            else:
                # Always ask user to confirm when paired data is detected
                result = self._confirm_paired_end_with_user()
                self._paired_end_cache = result
                return result
        else:
            self.logger.info("Auto-detected single-end data")
            self._paired_end_cache = False
            return self._paired_end_cache

    def _check_if_data_appears_paired(self) -> bool:
        """
        Check if the data appears to be paired-end based on file patterns.

        Returns:
            bool: True if data appears to be paired-end, False otherwise
        """
        paired_samples = 0
        total_samples = len(self.sample_dict)

        for sample_id, sample_info in self.sample_dict.items():
            if len(sample_info) >= 4:
                file1 = sample_info[2] if len(sample_info) > 2 else ""
                file2 = sample_info[3] if len(sample_info) > 3 else ""

                # Check for common paired-end patterns
                r1_patterns = ["_r1", "_1", ".r1", "_R1", "_1.", ".R1"]
                r2_patterns = ["_r2", "_2", ".r2", "_R2", "_2.", ".R2"]

                file1_lower = file1.lower()
                file2_lower = file2.lower()

                has_r1 = any(pattern.lower() in file1_lower for pattern in r1_patterns)
                has_r2 = any(pattern.lower() in file2_lower for pattern in r2_patterns)

                if has_r1 and has_r2:
                    paired_samples += 1
                    self.logger.debug(f"Sample {sample_id} appears paired-end: {file1} & {file2}")

        # Consider it paired-end if majority of samples appear paired
        is_paired = paired_samples > (total_samples * 0.5)

        if is_paired:
            self.logger.debug(f"Found {paired_samples}/{total_samples} samples with paired-end patterns")

        return is_paired

    def _confirm_paired_end_with_user(self) -> bool:
        """
        Ask user to confirm if they want to use paired-end mode for detected paired data.

        Returns:
            bool: True if user confirms paired-end, False for single-end
        """
        try:
            from ..utils.file_manager import colored_output

            print(colored_output("\nPAIRED-END DATA DETECTED", "yellow"))
            print(
                colored_output(
                    "Your data appears to contain paired-end reads (R1/R2 files).",
                    "white",
                )
            )
            print(colored_output("How would you like to proceed?\n", "white"))

            print(colored_output("Options:", "cyan"))
            print(
                colored_output(
                    "  [P] Paired-end mode (recommended) - Use both R1 and R2 files",
                    "white",
                )
            )
            print(colored_output("  [S] Single-end mode - Use only R1 files", "white"))
            print(colored_output("  [A] Abort - Exit to review data", "white"))

            while True:
                try:
                    choice = input(colored_output("\nYour choice [P/S/A]: ", "green")).strip().upper()

                    if choice in ["P", "PAIRED", "PAIRED-END"]:
                        self.logger.info("User confirmed paired-end mode")
                        return True
                    elif choice in ["S", "SINGLE", "SINGLE-END"]:
                        self.logger.info("User selected single-end mode for paired data")
                        return False
                    elif choice in ["A", "ABORT", "EXIT", "Q", "QUIT"]:
                        self.logger.info("User chose to abort for data review")
                        raise ValueError("Pipeline aborted by user for data review")
                    else:
                        print(colored_output("Invalid choice. Please enter P, S, or A.", "red"))

                except KeyboardInterrupt:
                    print(colored_output("\n\nPipeline interrupted by user.", "red"))
                    raise ValueError("Pipeline interrupted by user")

        except ImportError:
            # Fallback if colored_output is not available
            print("\nPAIRED-END DATA DETECTED")
            print("Your data appears to contain paired-end reads (R1/R2 files).")

            while True:
                try:
                    choice = input("Use paired-end mode? [Y/n]: ").strip().lower()
                    if choice in ["", "y", "yes"]:
                        return True
                    elif choice in ["n", "no"]:
                        return False
                    else:
                        print("Please enter Y or N.")
                except KeyboardInterrupt:
                    raise ValueError("Pipeline interrupted by user")

    def _confirm_forced_paired_with_single_data(self) -> bool:
        """
        Ask user to confirm when they specified --paired but data appears single-end.

        Returns:
            bool: True to proceed with paired-end, False to abort
        """
        try:
            from ..utils.file_manager import colored_output

            print(colored_output("\nWARNING: DATA TYPE MISMATCH", "red"))
            print(
                colored_output(
                    "You specified --paired but your data appears to be single-end.",
                    "white",
                )
            )
            print(colored_output("This might cause errors during processing.\n", "white"))

            print(colored_output("Options:", "cyan"))
            print(colored_output("  [C] Continue with paired-end mode (may cause errors)", "white"))
            print(
                colored_output(
                    "  [A] Abort - Exit to review data and remove --paired flag",
                    "white",
                )
            )

            while True:
                try:
                    choice = input(colored_output("\nYour choice [C/A]: ", "green")).strip().upper()

                    if choice in ["C", "CONTINUE"]:
                        self.logger.warning("User chose to continue with --paired despite single-end data")
                        return True
                    elif choice in ["A", "ABORT", "EXIT", "Q", "QUIT"]:
                        self.logger.info("User chose to abort due to data type mismatch")
                        raise ValueError("Pipeline aborted due to --paired flag with single-end data")
                    else:
                        print(colored_output("Invalid choice. Please enter C or A.", "red"))

                except KeyboardInterrupt:
                    print(colored_output("\n\nPipeline interrupted by user.", "red"))
                    raise ValueError("Pipeline interrupted by user")

        except ImportError:
            # Fallback if colored_output is not available
            print("\nWARNING: DATA TYPE MISMATCH")
            print("You specified --paired but your data appears to be single-end.")

            while True:
                try:
                    choice = input("Continue with paired-end mode anyway? [y/N]: ").strip().lower()
                    if choice in ["y", "yes"]:
                        return True
                    elif choice in ["", "n", "no"]:
                        raise ValueError("Pipeline aborted due to data type mismatch")
                    else:
                        print("Please enter Y or N.")
                except KeyboardInterrupt:
                    raise ValueError("Pipeline interrupted by user")

    def _confirm_forced_single_with_paired_data(self) -> bool:
        """
        Ask user to confirm when they specified --single-end but data appears paired-end.

        Returns:
            bool: False to proceed with single-end, True to abort
        """
        try:
            from ..utils.file_manager import colored_output

            print(colored_output("\nWARNING: DATA TYPE MISMATCH", "yellow"))
            print(
                colored_output(
                    "You specified --single-end but your data appears to be paired-end.",
                    "white",
                )
            )
            print(
                colored_output(
                    "You might be missing out on better analysis with paired-end mode.\n",
                    "white",
                )
            )

            print(colored_output("Options:", "cyan"))
            print(colored_output("  [C] Continue with single-end mode (only use R1 files)", "white"))
            print(
                colored_output(
                    "  [A] Abort - Exit to review data and remove --single-end flag",
                    "white",
                )
            )

            while True:
                try:
                    choice = input(colored_output("\nYour choice [C/A]: ", "green")).strip().upper()

                    if choice in ["C", "CONTINUE"]:
                        self.logger.warning("User chose to continue with --single-end despite paired-end data")
                        return False  # Continue with single-end mode
                    elif choice in ["A", "ABORT", "EXIT", "Q", "QUIT"]:
                        self.logger.info("User chose to abort due to data type mismatch")
                        raise ValueError("Pipeline aborted due to --single-end flag with paired-end data")
                    else:
                        print(colored_output("Invalid choice. Please enter C or A.", "red"))

                except KeyboardInterrupt:
                    print(colored_output("\n\nPipeline interrupted by user.", "red"))
                    raise ValueError("Pipeline interrupted by user")
        except ImportError:
            # Fallback if colored_output is not available
            print("\nWARNING: DATA TYPE MISMATCH")
            print("You specified --single-end but your data appears to be paired-end.")

            while True:
                try:
                    choice = input("Continue with single-end mode anyway? [C/A]: ").strip().upper()
                    if choice in ["C", "CONTINUE"]:
                        return False
                    elif choice in ["A", "ABORT", "EXIT", "Q", "QUIT"]:
                        raise ValueError("Pipeline aborted due to data type mismatch")
                    else:
                        print("Please enter C or A.")
                except KeyboardInterrupt:
                    raise ValueError("Pipeline interrupted by user")

    def _clean_and_create_directory(self, dir_path: Path, stage_name: str):
        """
        Clean existing directory and create fresh one for resume operations.

        Args:
            dir_path: Directory path to clean and create
            stage_name: Name of the stage for logging
        """
        if dir_path.exists() and not self.dryrun:
            import shutil

            self.logger.info(f"Removing existing {stage_name} directory for fresh start: {dir_path}")
            shutil.rmtree(dir_path)

        if not self.dryrun:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created {stage_name} directory: {dir_path}")
        else:
            self.logger.info(f"Would create {stage_name} directory: {dir_path}")

    def wait_for_slurm_jobs(self, job_ids: Dict[str, str], tool_name: str) -> None:
        """
        Wait for submitted SLURM jobs to complete.

        Args:
            job_ids: Dictionary of sample name to SLURM job ID
            tool_name: Name of the tool/stage for logging
        """
        logs_dir = Path(self.output_dir) / "logs"
        if tool_name == "coexpression":
            logs_dir = Path(self.output_dir) / "5.Coexpression" / "logs"

        if not self.dryrun:
            logs_dir.mkdir(parents=True, exist_ok=True)

        check_interval = int(getattr(self, "slurm_check_interval", 60) or 60)
        initial_delay = int(getattr(self, "slurm_initial_delay", 10) or 10)

        timeout_hours = getattr(self, "slurm_wait_timeout_hours", 72.0)
        max_wait_seconds = int(float(timeout_hours) * 3600) if timeout_hours else None

        self.command_executor._wait_for_slurm_jobs(
            job_ids=job_ids,
            logs_dir=logs_dir,
            tool_name=tool_name,
            check_interval=check_interval,
            initial_delay=initial_delay,
            max_wait_seconds=max_wait_seconds,
        )
