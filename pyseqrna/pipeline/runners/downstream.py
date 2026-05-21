#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Downstream Analysis Runner

This module coordinates downstream analysis tasks following differential expression and
normalization. It runs sample similarity clustering, gene co-expression clustering,
and generates multi-format summary reports.

Features:
    - Gene co-expression clustering using PyCoexpression
    - Sample similarity clustering with hierarchical or k-means algorithms
    - Verification of input normalization matrices before analytical executions
    - Detailed checkpointing to save execution configurations and statistics
    - Automated report generation in HTML, Markdown, and JSON formats

Configuration:
    - Configured via Pipeline context properties (enable_coexpression, coexpression_tool, coexpression_tightness, coexpression_k_values, coexpression_outlier, coexpression_cluster_size, coexpression_replicates, coexpression_preprocessing, run_clustering, cluster_target, cluster_method, cluster_count, cluster_metric, cluster_linkage, cluster_top_variable, cluster_no_log, cluster_scale, cluster_no_heatmap, cluster_cmap, skip_report, report_formats, report_title).

Dependencies:
    - Python packages: pathlib, datetime
    - Python modules: pyseqrna.modules.clustering, pyseqrna.modules.coexpression, pyseqrna.modules.reporting

Classes / Functions / Exceptions:
    - DownstreamRunner: Coordinates downstream RNA-seq pipeline steps including co-expression analysis, sample clustering, and multi-format report generation.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

from datetime import datetime

from ...modules.clustering import ClusteringAnalyzer
from ...modules.coexpression import PyCoexpression
from ...modules.reporting import ReportGenerator, ReportGenerationError


class DownstreamRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_coexpression_analysis(self) -> bool:
        """Run gene co-expression analysis from normalized counts."""
        if not self.ctx.enable_coexpression:
            self.ctx.logger.info("Skipping co-expression analysis as requested")
            return True

        try:
            if self.ctx.coexpression_tool != "pycoexpression":
                self.ctx.logger.error(f"Unsupported co-expression tool: {self.ctx.coexpression_tool}")
                return False

            coexpression_dir = Path(self.ctx.output_dir) / "5.Coexpression"
            self.ctx._clean_and_create_directory(coexpression_dir, "co-expression analysis")

            normalized_file = (
                Path(self.ctx.output_dir)
                / "4.Normalization"
                / f"{self.ctx.normalization_method.upper()}_normalized_counts.xlsx"
            )
            if not normalized_file.exists() and not self.ctx.dryrun:
                self.ctx.logger.error(f"Normalized counts file not found for co-expression analysis: {normalized_file}")
                return False

            self.ctx.logger.info("Running built-in gene co-expression analysis (PyCoexpression)")
            runner = PyCoexpression(
                matrix_file=str(normalized_file),
                out_dir=str(coexpression_dir),
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
            )
            result = runner.run(
                tightness=self.ctx.coexpression_tightness,
                k_values=self.ctx.coexpression_k_values,
                outlier=self.ctx.coexpression_outlier,
                cluster_size=self.ctx.coexpression_cluster_size,
                replicates=self.ctx.coexpression_replicates,
                preprocessing=self.ctx.coexpression_preprocessing,
            )

            metadata = {
                "tool": self.ctx.coexpression_tool,
                "input_file": str(normalized_file),
                "output_directory": str(coexpression_dir),
                "command": result.get("command"),
                "execution_time": datetime.now().isoformat(),
            }
            self.ctx.checkpoint_manager.mark_stage_complete("coexpression", metadata=metadata, dry_run=self.ctx.dryrun)
            self.ctx.logger.info("Co-expression analysis completed successfully")
            return True

        except Exception as e:
            self.ctx.logger.error(f"Co-expression analysis failed: {e}")
            return False

    def run_sample_clustering(self) -> bool:
        """Run sample similarity clustering from normalized counts."""
        if not self.ctx.run_clustering:
            self.ctx.logger.info("Skipping sample clustering as requested")
            return True

        try:
            clustering_dir = Path(self.ctx.output_dir) / "5.Clustering"
            self.ctx._clean_and_create_directory(clustering_dir, "sample clustering")

            normalized_file = (
                Path(self.ctx.output_dir)
                / "4.Normalization"
                / f"{self.ctx.normalization_method.upper()}_normalized_counts.xlsx"
            )
            if not normalized_file.exists() and not self.ctx.dryrun:
                self.ctx.logger.error(f"Normalized counts file not found for clustering: {normalized_file}")
                return False

            self.ctx.logger.info("Running sample similarity clustering")
            analyzer = ClusteringAnalyzer(
                matrix_file=str(normalized_file),
                out_dir=str(clustering_dir),
                gene_column="Gene",
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
            )
            result = analyzer.run(
                cluster_target=self.ctx.cluster_target,
                method=self.ctx.cluster_method,
                n_clusters=self.ctx.cluster_count,
                metric=self.ctx.cluster_metric,
                linkage_method=self.ctx.cluster_linkage,
                top_variable=None if self.ctx.cluster_top_variable == 0 else self.ctx.cluster_top_variable,
                log_transform=not self.ctx.cluster_no_log,
                scale=self.ctx.cluster_scale,
                heatmap=not self.ctx.cluster_no_heatmap,
                color_map=self.ctx.cluster_cmap,
                prefix="sample_clustering",
            )

            outputs = result.get("outputs", {})
            metadata = {
                "input_file": str(normalized_file),
                "output_directory": str(clustering_dir),
                "outputs": outputs,
                "execution_time": datetime.now().isoformat(),
            }
            self.ctx.checkpoint_manager.mark_stage_complete("sample_clustering", metadata=metadata, dry_run=self.ctx.dryrun)
            self.ctx.logger.info("Sample clustering completed successfully")
            return True

        except Exception as e:
            self.ctx.logger.error(f"Sample clustering failed: {e}")
            return False

    def run_report(self) -> bool:
        """Generate comprehensive analysis reports from the run directory."""
        if self.ctx.skip_report:
            self.ctx.logger.info("Skipping comprehensive report generation as requested")
            return True

        try:
            report_formats = ReportGenerator.parse_formats(self.ctx.report_formats)
            self.ctx.logger.info(f"Generating comprehensive report formats: {', '.join(report_formats)}")
            generator = ReportGenerator(
                output_dir=self.ctx.output_dir,
                title=self.ctx.report_title,
                input_file=self.ctx.input_file,
                samples_path=self.ctx.samples_path,
                reference_genome=self.ctx.reference_genome,
                feature_file=self.ctx.feature_file,
                config=self.ctx._report_config(),
                logger=self.ctx.logger.logger,
            )
            written = generator.generate(formats=report_formats)
            for fmt, report_path in written.items():
                self.ctx.logger.info(f"Report saved ({fmt}): {report_path}")
            return True
        except ReportGenerationError as exc:
            self.ctx.logger.error(f"Report generation failed: {exc}")
            return False
        except Exception as exc:
            self.ctx.logger.error(f"Unexpected report generation failure: {exc}")
            return False
