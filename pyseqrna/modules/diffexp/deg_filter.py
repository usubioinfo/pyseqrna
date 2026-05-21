#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEG Filter Module

This module provides functionality to filter differential expression results
based on statistical significance (FDR/p-value) and fold change thresholds.
It supports grouping of results and exports publication-ready DEG summary plots
representing up-regulated and down-regulated genes.

Features:
    - Filters wide-format differential expression dataframes based on FDR and fold change
    - Supports experimental designs with or without biological replicates
    - Generates up-regulated and down-regulated gene subsets for multiple comparisons
    - Exports publication-quality horizontal bar charts of DEG distributions
    - Computes summary tables of DEG counts across comparisons
    - Handles standard genes as well as Multimapped Gene Groups (MMGs)

Configuration:
    Configured via constructor arguments:
    - fdr_threshold: Statistical significance cutoff (FDR/padj)
    - fold_threshold: Fold change threshold (internally converted to log2)
    - has_replicates: Boolean indicating if biological replicates are present
    - mmg: Boolean indicating if the input data represents Multimapped Gene Groups
    - extra_columns: Boolean indicating if annotation columns (Name/Description) are present
    - logger: Optional custom logger instance

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - pyseqrna.modules.diffexp.base.DifferentialExpressionError

Classes / Functions / Exceptions:
    - DEGFilter (Class): Filter differential expression results based on FDR and fold change thresholds and generate summary plots.

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

from .base import DifferentialExpressionError


class DEGFilter:
    """
    Filter differential expression results based on FDR and fold change thresholds.

    The filter operates on the wide result table produced by the supported
    differential-expression analyzers.
    """

    def __init__(
        self,
        fdr_threshold: float = 0.05,
        fold_threshold: float = 2.0,
        has_replicates: bool = True,
        mmg: bool = False,
        extra_columns: bool = False,
        logger=None,
    ):
        """
        Initialize DEGFilter.

        Args:
            fdr_threshold: False Discovery Rate threshold for filtering
            fold_threshold: Fold change threshold (will be log2 transformed)
            has_replicates: Whether the data has biological replicates
            mmg: Whether data is from multimapped gene groups
            extra_columns: Whether to expect extra annotation columns
            logger: Logger instance
        """
        self.fdr_threshold = fdr_threshold
        self.fold_threshold = fold_threshold
        self.log2_fold_threshold = np.log2(fold_threshold)
        self.has_replicates = has_replicates
        self.mmg = mmg
        self.extra_columns = extra_columns
        self.logger = logger

        if self.logger:
            self.logger.info(f"DEGFilter initialized: FDR={fdr_threshold}, Fold={fold_threshold}, Replicates={has_replicates}")

    def filter_degs(
        self,
        deg_df: pd.DataFrame,
        compare_list: List[str],
        create_plot: bool = True,
        plot_figsize: Tuple[int, int] = (10, 6),
        plot_text_size: int = 14,
        save_plot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Filter differential expression results based on FDR and fold change thresholds.

        Args:
            deg_df: Wide DataFrame containing all differential expression results
            compare_list: List of comparisons to filter
            create_plot: Whether to create a summary plot
            plot_figsize: Figure size for the plot
            plot_text_size: Text size for plot labels
            save_plot_path: Path to save the plot (if None, plot won't be saved)

        Returns:
            Dictionary containing filtered results, summary, and plot
        """
        try:
            summary_records = []
            DEGs = {}
            Ups = {}
            Downs = {}

            deg_df_indexed = deg_df.copy()

            if self.extra_columns:
                deg_df_indexed = deg_df_indexed.set_index(["Gene", "Name", "Description"])
            elif self.mmg:
                deg_df_indexed = deg_df_indexed.set_index(["MMG", "Gene"])
            else:
                deg_df_indexed = deg_df_indexed.set_index("Gene")

            for c in compare_list:
                if self.logger:
                    self.logger.info(f"Filtering DEGs for comparison: {c}")

                dk = deg_df_indexed.filter(regex=c, axis=1)

                # Define column names based on tool
                if f"FDR({c})" in dk.columns:
                    # DESeq2 format
                    fdr_col = f"FDR({c})"
                    lfc_col = f"logFC({c})"
                elif f"padj({c})" in dk.columns:
                    # Also DESeq2 format with different naming
                    fdr_col = f"padj({c})"
                    lfc_col = f"log2FoldChange({c})"
                else:
                    # edgeR format - look for any padj column for this comparison
                    fdr_cols = [col for col in dk.columns if "padj" in col and c in col]
                    lfc_cols = [col for col in dk.columns if "log2FoldChange" in col and c in col]

                    if fdr_cols and lfc_cols:
                        fdr_col = fdr_cols[0]
                        lfc_col = lfc_cols[0]
                    else:
                        if self.logger:
                            self.logger.warning(f"Could not find FDR/logFC columns for comparison {c}")
                        empty = deg_df_indexed.iloc[0:0].reset_index()
                        DEGs[c] = empty
                        Ups[c] = empty
                        Downs[c] = empty
                        summary_records.append(
                            {
                                "Comparisons": c,
                                "Total_DEGs": 0,
                                "Up_DEGs": 0,
                                "Down_DEGs": 0,
                            }
                        )
                        continue

                if self.has_replicates:
                    fdr_filtered = dk[dk[fdr_col] <= self.fdr_threshold].dropna()

                    upDF = fdr_filtered[fdr_filtered[lfc_col] >= self.log2_fold_threshold]
                    downDF = fdr_filtered[fdr_filtered[lfc_col] <= -self.log2_fold_threshold]
                else:
                    upDF = dk[dk[lfc_col] >= self.log2_fold_threshold]
                    downDF = dk[dk[lfc_col] <= -self.log2_fold_threshold]

                upDF = upDF.reset_index()
                downDF = downDF.reset_index()
                final = pd.concat([upDF, downDF], axis=0)
                DEGs[c] = final
                Ups[c] = upDF
                Downs[c] = downDF
                summary_records.append(
                    {
                        "Comparisons": c,
                        "Total_DEGs": len(final),
                        "Up_DEGs": len(upDF),
                        "Down_DEGs": len(downDF),
                    }
                )

                if self.logger:
                    self.logger.info(f"  {c}: {len(upDF)} up, {len(downDF)} down, {len(final)} total DEGs")

            summary = pd.DataFrame(
                summary_records,
                columns=["Comparisons", "Total_DEGs", "Up_DEGs", "Down_DEGs"],
            )

            fig = None
            if create_plot:
                fig = self._create_deg_plot(summary, plot_figsize, plot_text_size, save_plot_path)

            return {
                "summary": summary,
                "filtered": DEGs,
                "filteredup": Ups,
                "filtereddown": Downs,
                "plot": fig,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"DEG filtering failed: {str(e)}")
            raise DifferentialExpressionError(f"DEG filtering failed: {str(e)}")

    def _create_deg_plot(
        self,
        summary: pd.DataFrame,
        figsize: Tuple[int, int],
        text_size: int,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a publication-ready DEG summary plot.

        Args:
            summary: Summary DataFrame with comparison counts
            figsize: Figure size
            text_size: Text size for labels
            save_path: Path to save plot

        Returns:
            matplotlib Figure object
        """
        try:
            style_context = {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 12,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
            palette = {
                "up": "#0072B2",
                "down": "#D55E00",
                "zero": "#666666",
                "grid": "#D9D9D9",
                "text": "#222222",
            }

            plot_df = summary.copy()
            if plot_df.empty:
                plot_df = pd.DataFrame(
                    [
                        {
                            "Comparisons": "No comparisons",
                            "Total_DEGs": 0,
                            "Up_DEGs": 0,
                            "Down_DEGs": 0,
                        }
                    ]
                )

            plot_df["Total_DEGs"] = pd.to_numeric(plot_df["Total_DEGs"], errors="coerce").fillna(0).astype(int)
            plot_df["Up_DEGs"] = pd.to_numeric(plot_df["Up_DEGs"], errors="coerce").fillna(0).astype(int)
            plot_df["Down_DEGs"] = pd.to_numeric(plot_df["Down_DEGs"], errors="coerce").fillna(0).astype(int)
            plot_df = plot_df.sort_values(["Total_DEGs", "Comparisons"], ascending=[True, True])
            plot_df["Down_plot"] = -plot_df["Down_DEGs"]
            labels = plot_df["Comparisons"].astype(str).values.tolist()

            fig_height = max(5.5, min(24.0, len(labels) * 0.55 + 3.0))
            fig_width = max(float(figsize[0]), 10.5)
            text_size = max(8, min(int(text_size), 12))

            updata = plot_df["Up_DEGs"].values.tolist()
            downdata = plot_df["Down_plot"].values.tolist()
            y_positions = np.arange(len(plot_df))

            with plt.rc_context(style_context):
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
                ax.barh(
                    y_positions,
                    updata,
                    height=0.68,
                    color=palette["up"],
                    label="Up-regulated",
                    edgecolor="white",
                    linewidth=0.6,
                )
                ax.barh(
                    y_positions,
                    downdata,
                    height=0.68,
                    color=palette["down"],
                    label="Down-regulated",
                    edgecolor="white",
                    linewidth=0.6,
                )
                ax.axvline(0, color=palette["zero"], linewidth=1.1)
                ax.grid(
                    axis="x",
                    color=palette["grid"],
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.75,
                )

                max_count = int(max(plot_df["Up_DEGs"].max(), plot_df["Down_DEGs"].max(), 1))
                x_limit = max(max_count * 1.38, 1)
                ax.set_xlim(-x_limit, x_limit)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(labels, fontsize=text_size)
                ax.set_ylim(-0.7, len(plot_df) - 0.3)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(int(x)):,}" if x else "0"))

                xlabel = "Number of MMGs" if self.mmg else "Number of genes"
                ax.set_xlabel(xlabel, fontsize=text_size, fontweight="bold", labelpad=12)
                ax.set_ylabel("Comparison", fontsize=text_size, fontweight="bold", labelpad=12)
                ax.tick_params(axis="x", labelsize=text_size)

                label_offset = max(x_limit * 0.035, 0.25)
                if len(plot_df) <= 60:
                    for y, up_count, down_count in zip(y_positions, plot_df["Up_DEGs"], plot_df["Down_DEGs"]):
                        if up_count > 0:
                            ax.text(
                                up_count + label_offset,
                                y,
                                f"{up_count:,}",
                                va="center",
                                ha="left",
                                fontsize=8,
                                color=palette["text"],
                            )
                        if down_count > 0:
                            ax.text(
                                -(down_count + label_offset),
                                y,
                                f"{down_count:,}",
                                va="center",
                                ha="right",
                                fontsize=8,
                                color=palette["text"],
                            )

                if int(plot_df["Total_DEGs"].sum()) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No filtered DEGs at the selected thresholds",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=11,
                        color=palette["zero"],
                        bbox={
                            "boxstyle": "round,pad=0.35",
                            "facecolor": "white",
                            "edgecolor": "#CCCCCC",
                        },
                    )

                threshold_label = (
                    f"Fold >= {self.fold_threshold:g}, FDR <= {self.fdr_threshold:g}"
                    if self.has_replicates
                    else f"Fold >= {self.fold_threshold:g}"
                )
                ax.set_title(
                    "Differential expression summary",
                    fontsize=13,
                    fontweight="bold",
                    pad=28,
                )
                ax.text(
                    0.5,
                    1.015,
                    threshold_label,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=palette["zero"],
                )
                ax.legend(
                    ncol=2,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.06),
                    frameon=False,
                    fontsize=9,
                    handlelength=1.6,
                    columnspacing=1.6,
                )
                sns.despine(ax=ax, left=False, bottom=False)
                fig.tight_layout(rect=(0, 0, 1, 0.94))

            # Save plot if path provided
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                pdf_path = str(save_path)
                if not pdf_path.lower().endswith(".pdf"):
                    pdf_path = pdf_path.rsplit(".", 1)[0] + ".pdf" if "." in pdf_path else pdf_path + ".pdf"
                fig.savefig(pdf_path, bbox_inches="tight")
                if self.logger:
                    self.logger.info(f"DEG plot saved to: {save_path}")

            return fig

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create DEG plot: {str(e)}")
            raise DifferentialExpressionError(f"Failed to create DEG plot: {str(e)}")

    # File saving functionality moved to base class - this method removed
