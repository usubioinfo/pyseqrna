#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Visualization Runner

This module coordinates data visualization steps as part of the pipeline execution.
It identifies appropriate input data (normalized counts, differential expression, and multi-mapped group expression)
and generates PCA, t-SNE, volcano, MA, heatmap, Venn, and UpSet plots.

Features:
    - Dynamic discovery of normalization count matrices and differential expression outputs
    - Dimensionality reduction plots (PCA and t-SNE) from normalized counts
    - Statistical diagnostic plots (volcano and MA plots) from differential expression results
    - High-dimensional expression clustering via DEG heatmaps
    - DEG intersection plotting through Venn diagrams and UpSet plots
    - Detailed checkpointing tracking which plots were generated and their output directories

Configuration:
    - Configured via Pipeline context properties (normalization_results, differential_expression_results, sample_dict, log2fc_threshold, fdr_threshold, pca_plot, tsne_plot, volcano_plot, ma_plot, deg_heatmap, heatmap_top_genes, venn, venn_comparisons, venn_label, upset).

Dependencies:
    - Python packages: pathlib
    - Python modules: pyseqrna.modules.visualization

Classes / Functions / Exceptions:
    - VisualizationRunner: Coordinates plotting and visualization steps including PCA, t-SNE, volcano, MA, DEG heatmaps, Venn diagrams, and UpSet plots.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

from ...modules.visualization import Visualization


class VisualizationRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_visualization(self) -> bool:
        """
        Run visualization stage.

        Returns:
            True if successful, False otherwise
        """
        self.ctx.logger.info("Running visualization stage")

        try:
            # Create visualization directory
            viz_dir = Path(self.ctx.output_dir) / "5.Visualization"
            self.ctx._clean_and_create_directory(viz_dir, "visualization")

            # Initialize Visualization module
            visualizer = Visualization(
                outdir=str(viz_dir),
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
            )

            # 1. Find Normalized Counts File
            norm_counts_file = None
            if isinstance(self.ctx.normalization_results, dict) and "normalized_counts_file" in self.ctx.normalization_results:
                norm_counts_file = self.ctx.normalization_results["normalized_counts_file"]
            elif isinstance(self.ctx.normalization_results, (str, Path)):
                norm_counts_file = str(self.ctx.normalization_results)
            else:
                # Check standard locations
                possible_files = [
                    Path(self.ctx.output_dir)
                    / "4.Normalization"
                    / f"{self.ctx.normalization_method.upper()}_normalized_counts.xlsx",
                    Path(self.ctx.output_dir)
                    / "4.Normalization"
                    / f"{self.ctx.normalization_method.upper()}_normalized_counts.csv",
                    Path(self.ctx.output_dir) / "3.Quantification" / "TMM_counts.csv",
                    Path(self.ctx.output_dir) / "3.Quantification" / "TPM_counts.csv",
                    Path(self.ctx.output_dir) / "3.Quantification" / "RPKM_counts.csv",
                    Path(self.ctx.output_dir) / "3.Quantification" / "FPKM_counts.csv",
                    Path(self.ctx.output_dir) / "3.Quantification" / "normalized_counts.csv",
                ]
                for p in possible_files:
                    if p.exists():
                        norm_counts_file = str(p)
                        break

            # 2. Find Differential Expression Results File
            de_results_file = None
            if isinstance(self.ctx.differential_expression_results, (str, Path)):
                de_results_file = str(self.ctx.differential_expression_results)
            elif (
                isinstance(self.ctx.differential_expression_results, dict)
                and "output_files" in self.ctx.differential_expression_results
            ):
                for f in self.ctx.differential_expression_results["output_files"]:
                    if "All_gene_expression.xlsx" in str(f):
                        de_results_file = str(f)
                        break

            if not de_results_file:
                de_dir = Path(self.ctx.output_dir) / "4.Differential_Expression"
                de_results_file = str(de_dir / "All_gene_expression.xlsx")

            diffexp_dir = Path(self.ctx.output_dir) / "4.Differential_Expression"
            quantification_dir = Path(self.ctx.output_dir) / "3.Quantification"
            mmg_de_results_file = None
            filtered_mmg_file = None
            mmg_counts_file = None
            if self.ctx.enable_multimapped_groups:
                candidate_mmg_de = diffexp_dir / "All_MMG_expression.xlsx"
                candidate_filtered_mmg = diffexp_dir / "Filtered_MMGs.xlsx"
                candidate_mmg_counts = quantification_dir / "Raw_MMGcounts.xlsx"
                mmg_de_results_file = str(candidate_mmg_de) if candidate_mmg_de.exists() else None
                filtered_mmg_file = str(candidate_filtered_mmg) if candidate_filtered_mmg.exists() else None
                mmg_counts_file = str(candidate_mmg_counts) if candidate_mmg_counts.exists() else None

            venn_comparisons = None
            if self.ctx.venn_comparisons:
                venn_comparisons = [item.strip() for item in str(self.ctx.venn_comparisons).split(",") if item.strip()] or None

            # Run visualization
            visualizer.run(
                norm_counts_file=norm_counts_file,
                de_results_file=de_results_file,
                filtered_deg_file=(
                    str(diffexp_dir / "Filtered_DEGs.xlsx") if (diffexp_dir / "Filtered_DEGs.xlsx").exists() else None
                ),
                mmg_de_results_file=mmg_de_results_file,
                mmg_counts_file=mmg_counts_file,
                filtered_mmg_file=filtered_mmg_file,
                sample_dict=self.ctx.sample_dict,
                log2fc_threshold=self.ctx.log2fc_threshold,
                fdr_threshold=self.ctx.fdr_threshold,
                pca_plot=self.ctx.pca_plot,
                tsne_plot=self.ctx.tsne_plot,
                volcano_plot=self.ctx.volcano_plot,
                ma_plot=self.ctx.ma_plot,
                deg_heatmap=self.ctx.deg_heatmap,
                heatmap_top_genes=self.ctx.heatmap_top_genes,
                venn=self.ctx.venn,
                venn_comparisons=venn_comparisons,
                venn_label=self.ctx.venn_label,
                upset=self.ctx.upset,
            )

            self.ctx.logger.info(f"Created visualization directory: {viz_dir}")

            # Mark stage as complete
            self.ctx.checkpoint_manager.mark_stage_complete(
                "visualization",
                metadata={
                    "output_directory": str(viz_dir),
                    "pca_plot": self.ctx.pca_plot,
                    "tsne_plot": self.ctx.tsne_plot,
                    "volcano_plot": self.ctx.volcano_plot,
                    "ma_plot": self.ctx.ma_plot,
                    "deg_heatmap": self.ctx.deg_heatmap,
                    "heatmap_top_genes": self.ctx.heatmap_top_genes,
                    "venn": self.ctx.venn,
                    "upset": self.ctx.upset,
                    "venn_comparisons": venn_comparisons,
                    "venn_label": self.ctx.venn_label,
                },
                dry_run=self.ctx.dryrun,
            )

            return True

        except Exception as e:
            self.ctx.logger.error(f"Visualization stage failed: {e}")
            import traceback

            self.ctx.logger.error(traceback.format_exc())
            return False
