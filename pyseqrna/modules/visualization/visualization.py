#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Visualization Module

This module provides functionality for generating various visualizations
for RNA-seq analysis results, including PCA, t-SNE, Volcano plots, MA plots,
heatmaps, Venn diagrams, and UpSet-style intersection summaries.

Features:
    - Custom volcano plots and MA plots with automatic labeling of top significant genes
    - PCA and t-SNE ordination visualization for sample clustering QC
    - High-resolution Venn diagrams (2-way, 3-way, and 4-way) and UpSet intersection plots
    - Normalized publication-style plotting themes using seaborn and matplotlib
    - Subfolder organization and double-format export (PNG and vector PDF)
    - Resource-efficient dry-run execution mode

Configuration:
    The module can be configured with:
    - Output directory for plot files (outdir)
    - Custom color mapping parameters

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - adjustText

Classes:
    Visualization - Class for generating RNA-seq visualizations.

Exceptions:
    VisualizationError - Custom exception for visualization errors.

:Created: May 20, 2021
:Updated: May 5, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.patches as patches
from adjustText import adjust_text
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import re

# Import utility modules
from pyseqrna.utils import LogManager, FileManager, DryRunManager


class VisualizationError(Exception):
    """Custom exception for visualization errors."""

    pass


class Visualization:
    """
    Class for generating RNA-seq visualizations.
    """

    def __init__(
        self,
        outdir: str = ".",
        logger: Optional[Any] = None,
        dryrun: bool = False,
        dry_run_manager: Optional[DryRunManager] = None,
    ):
        """
        Initializes the Visualization class.

        Args:
            outdir: Output directory for saving plots.
            logger: Logger instance for logging messages.
            dryrun: Whether to perform a dry run.
            dry_run_manager: Manager for dry run operations.
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.get_logger(__name__)
        else:
            self.logger = logger

        self.outdir = outdir
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager

        # Initialize FileManager
        self.file_manager = FileManager(logger=self.logger)

        self.defaultColors = [
            [92 / 255.0, 192 / 255.0, 98 / 255.0, 0.5],
            [90 / 255.0, 155 / 255.0, 212 / 255.0, 0.5],
            [241 / 255.0, 90 / 255.0, 96 / 255.0, 0.4],
            [255 / 255.0, 255 / 255.0, 102 / 255.0, 0.3],
            [255 / 255.0, 117 / 255.0, 0 / 255.0, 0.3],
        ]
        self.signal_colors = {
            "down": "#CC3311",
            "neutral": "#B3B3B3",
            "up": "#009988",
        }

        # Ensure output directory exists (if not dryrun)
        if not self.dryrun:
            os.makedirs(self.outdir, exist_ok=True)

    def _apply_publication_style(self) -> None:
        """Apply a consistent, publication-friendly plotting style."""
        sns.set_theme(
            style="whitegrid",
            context="paper",
            palette="colorblind",
            rc={
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#333333",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
                "grid.color": "#D9D9D9",
                "grid.linestyle": "--",
                "grid.linewidth": 0.7,
                "grid.alpha": 0.5,
                "font.size": 10,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.frameon": False,
            },
        )

    def _style_axis(self, ax: plt.Axes) -> None:
        """Standardize axis styling across plots."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4A4A4A")
        ax.spines["bottom"].set_color("#4A4A4A")
        ax.tick_params(colors="#333333")
        ax.set_facecolor("white")

    def _save_figure(self, fig: plt.Figure, output_file: str, tight_layout: bool = True) -> None:
        """Save and close a figure consistently."""
        os.makedirs(os.path.dirname(output_file) or self.outdir, exist_ok=True)
        if tight_layout:
            fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        fig.savefig(os.path.splitext(output_file)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)

    def _safe_plot_name(self, value: str) -> str:
        """Return a filesystem-safe plot stem while keeping names readable."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")

    def _identifier_column(self, df: pd.DataFrame) -> Optional[str]:
        """Prefer MMG identifiers when present, otherwise gene identifiers."""
        if "MMG" in df.columns:
            return "MMG"
        if "Gene" in df.columns:
            return "Gene"
        return None

    def _plot_output_file(self, subdir: str, filename: str) -> str:
        """Build a plot output path and create its subdirectory."""
        plot_dir = os.path.join(self.outdir, subdir)
        if not self.dryrun:
            os.makedirs(plot_dir, exist_ok=True)
        return os.path.join(plot_dir, filename)

    def _insert_ellipse(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        width: float,
        height: float,
        angle: float,
        color: Tuple[float, float, float, float],
    ) -> None:
        """Draw one ellipse for a Venn diagram."""
        ellipse = patches.Ellipse(
            xy=(x, y),
            width=width,
            height=height,
            angle=angle,
            linewidth=2,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(ellipse)

    def _insert_text(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        text: str,
        fontsize: int = 12,
        color: str = "black",
        ha: str = "center",
        va: str = "center",
        fontweight: Union[str, int] = 600,
    ) -> None:
        """Place text in a Venn diagram."""
        ax.text(
            x,
            y,
            text,
            horizontalalignment=ha,
            verticalalignment=va,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
        )

    def _generate_gene_collection(self, data: List[pd.Series]) -> Dict[str, set]:
        """Return exact-intersection gene sets for every Venn region."""
        gene_sets = [set(pd.Series(values).dropna().astype(str)) for values in data]
        all_genes = set(chain(*gene_sets)) if gene_sets else set()
        collections: Dict[str, set] = {}

        for index in range(1, 2 ** len(gene_sets)):
            key = bin(index).split("0b")[-1].zfill(len(gene_sets))
            value = set(all_genes)
            for set_index, genes in enumerate(gene_sets):
                if key[set_index] == "1":
                    value &= genes
                else:
                    value -= genes
            collections[key] = value

        return collections

    def _read_venn_sets(
        self,
        deg_file: Union[str, Path],
        comparisons: List[str],
        fold: float = 2,
    ) -> Tuple[List[pd.Series], List[pd.Series], List[pd.Series]]:
        """Read all/up/down DEG sets for selected comparisons."""
        all_sets: List[pd.Series] = []
        up_sets: List[pd.Series] = []
        down_sets: List[pd.Series] = []
        threshold = np.log2(fold)

        for comparison in comparisons:
            df = pd.read_excel(deg_file, sheet_name=comparison)
            id_col = self._identifier_column(df)
            if id_col is None:
                raise VisualizationError(f"Gene/MMG identifier column not found in sheet {comparison}")

            logfc_col = f"logFC({comparison})"
            if logfc_col not in df.columns:
                logfc_candidates = [col for col in df.columns if str(col).lower() == "logfc"]
                if not logfc_candidates:
                    raise VisualizationError(f"logFC column not found in sheet {comparison}")
                logfc_col = logfc_candidates[0]

            logfc = pd.to_numeric(df[logfc_col], errors="coerce")
            genes = df[id_col].dropna().astype(str)
            all_sets.append(genes)
            up_sets.append(df.loc[logfc >= threshold, id_col].dropna().astype(str))
            down_sets.append(df.loc[logfc <= -threshold, id_col].dropna().astype(str))

        return all_sets, up_sets, down_sets

    def _draw_venn(
        self,
        all_sets: List[pd.Series],
        up_sets: List[pd.Series],
        down_sets: List[pd.Series],
        comparisons: List[str],
        deg_label: str,
        fontsize: int,
        figsize: Tuple[int, int],
        dpi: int,
    ) -> plt.Figure:
        """Draw a 2-, 3-, or 4-way Venn diagram from prepared gene sets."""
        n_comparisons = len(comparisons)
        if n_comparisons not in {2, 3, 4}:
            raise VisualizationError("Venn diagrams require 2 to 4 comparisons")

        labels_total = self._generate_gene_collection(all_sets)
        labels_up = self._generate_gene_collection(up_sets)
        labels_down = self._generate_gene_collection(down_sets)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_axis_off()
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_xlim(left=0.0, right=1.0)

        colors = [self.defaultColors[i] for i in range(n_comparisons)]
        if n_comparisons == 4:
            ellipse_points = [
                (0.350, 0.400, 0.72, 0.45, 140.0),
                (0.450, 0.500, 0.72, 0.45, 140.0),
                (0.544, 0.500, 0.72, 0.45, 40.0),
                (0.644, 0.400, 0.72, 0.45, 40.0),
            ]
            data_points = [
                (0.85, 0.42),
                (0.66, 0.72),
                (0.77, 0.59),
                (0.32, 0.72),
                (0.69, 0.30),
                (0.50, 0.66),
                (0.65, 0.50),
                (0.14, 0.42),
                (0.50, 0.17),
                (0.30, 0.30),
                (0.38, 0.25),
                (0.23, 0.59),
                (0.63, 0.26),
                (0.35, 0.50),
                (0.50, 0.38),
            ]
            label_keys = [
                "0001",
                "0010",
                "0011",
                "0100",
                "0101",
                "0110",
                "0111",
                "1000",
                "1001",
                "1010",
                "1011",
                "1100",
                "1101",
                "1110",
                "1111",
            ]
            legend_points = [
                (0.13, 0.18, "right", "center"),
                (0.18, 0.83, "right", "bottom"),
                (0.82, 0.83, "left", "bottom"),
                (0.87, 0.18, "left", "top"),
            ]
        elif n_comparisons == 3:
            ellipse_points = [
                (0.333, 0.633, 0.5, 0.5, 0.0),
                (0.666, 0.633, 0.5, 0.5, 0.0),
                (0.500, 0.310, 0.5, 0.5, 0.0),
            ]
            data_points = [
                (0.50, 0.27),
                (0.73, 0.65),
                (0.61, 0.46),
                (0.27, 0.65),
                (0.39, 0.46),
                (0.50, 0.65),
                (0.50, 0.51),
            ]
            label_keys = ["001", "010", "011", "100", "101", "110", "111"]
            legend_points = [
                (0.15, 0.87, "right", "bottom"),
                (0.85, 0.87, "left", "bottom"),
                (0.50, 0.02, "center", "top"),
            ]
        else:
            ellipse_points = [
                (0.375, 0.3, 0.5, 0.5, 0.0),
                (0.625, 0.3, 0.5, 0.5, 0.0),
            ]
            data_points = [(0.74, 0.30), (0.26, 0.30), (0.50, 0.30)]
            label_keys = ["01", "10", "11"]
            legend_points = [
                (0.20, 0.56, "right", "bottom"),
                (0.80, 0.56, "left", "bottom"),
            ]

        for ellipse, color in zip(ellipse_points, colors):
            self._insert_ellipse(ax, *ellipse, color)

        for (x_coord, y_coord), key in zip(data_points, label_keys):
            if deg_label == "total":
                self._insert_text(
                    ax,
                    x_coord,
                    y_coord,
                    str(len(labels_total.get(key, set()))),
                    fontsize=fontsize,
                    color="#005AB5",
                )
            else:
                self._insert_text(
                    ax,
                    x_coord,
                    y_coord,
                    str(len(labels_up.get(key, set()))),
                    fontsize=fontsize,
                    color="#005AB5",
                )
                self._insert_text(
                    ax,
                    x_coord,
                    y_coord - 0.035,
                    str(len(labels_down.get(key, set()))),
                    fontsize=fontsize,
                    color="#DC3220",
                )

        for comparison, (x_coord, y_coord, ha, va) in zip(comparisons, legend_points):
            self._insert_text(ax, x_coord, y_coord, comparison, fontsize=fontsize, ha=ha, va=va)

        if deg_label != "total":
            self._insert_text(
                ax,
                0.02,
                0.97,
                "Up",
                fontsize=max(fontsize - 2, 8),
                color="#005AB5",
                ha="left",
            )
            self._insert_text(
                ax,
                0.02,
                0.92,
                "Down",
                fontsize=max(fontsize - 2, 8),
                color="#DC3220",
                ha="left",
            )

        fig.tight_layout()
        return fig

    def _comparison_batches(
        self,
        available_comparisons: List[str],
        requested_comparisons: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Return Venn comparison groups, using requested groups or chunks of four."""
        if requested_comparisons:
            selected = [comp for comp in requested_comparisons if comp in available_comparisons]
            return [selected] if 2 <= len(selected) <= 4 else []

        return [
            available_comparisons[index : index + 4]
            for index in range(0, len(available_comparisons), 4)
            if len(available_comparisons[index : index + 4]) >= 2
        ]

    def plot_venn(
        self,
        deg_file: Union[str, Path],
        comparisons: Optional[List[str]] = None,
        fold: float = 2,
        deg_label: str = "updown",
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 300,
        prefix: str = "",
    ) -> List[str]:
        """Create Venn diagrams for filtered DEG sheets."""
        if self.logger:
            self.logger.info("Starting Venn plot generation.")

        deg_file = Path(deg_file)
        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate Venn plots from {deg_file}")
            return []
        if not deg_file.exists():
            if self.logger:
                self.logger.warning(f"Filtered DEG file not found for Venn plots: {deg_file}")
            return []

        try:
            available = pd.ExcelFile(deg_file).sheet_names
            batches = self._comparison_batches(available, comparisons)
            if not batches:
                if self.logger:
                    self.logger.warning("Venn plots require 2 to 4 available comparisons.")
                return []

            output_files = []
            for batch_index, batch in enumerate(batches, start=1):
                all_sets, up_sets, down_sets = self._read_venn_sets(deg_file, batch, fold=fold)
                fig = self._draw_venn(
                    all_sets=all_sets,
                    up_sets=up_sets,
                    down_sets=down_sets,
                    comparisons=batch,
                    deg_label=deg_label,
                    fontsize=fontsize,
                    figsize=figsize,
                    dpi=dpi,
                )
                batch_name = "_".join(self._safe_plot_name(comp) for comp in batch)
                output_file = self._plot_output_file(
                    "Venn_Plots",
                    f"{prefix}Venn_{batch_index}_{batch_name}.png",
                )
                self._save_figure(fig, output_file)
                output_files.append(output_file)
                if self.logger:
                    self.logger.info(f"Venn plot saved to {output_file}")

            return output_files

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating Venn plots: {e}")
            return []

    def plot_upset(
        self,
        deg_file: Union[str, Path],
        fold: float = 2,
        max_intersections: int = 30,
        prefix: str = "",
    ) -> Optional[str]:
        """Create a matrix-style UpSet summary for DEG intersections."""
        if self.logger:
            self.logger.info("Starting UpSet-style intersection summary generation.")

        deg_file = Path(deg_file)
        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate UpSet-style plot from {deg_file}")
            return None
        if not deg_file.exists():
            return None

        try:
            self._apply_publication_style()
            comparisons = pd.ExcelFile(deg_file).sheet_names
            if len(comparisons) < 2:
                return None

            gene_membership: Dict[str, set] = {}
            set_sizes = {comparison: 0 for comparison in comparisons}
            id_label = "DEGs"
            for comparison in comparisons:
                df = pd.read_excel(
                    deg_file,
                    sheet_name=comparison,
                    usecols=lambda col: col in {"MMG", "Gene"},
                )
                id_col = self._identifier_column(df)
                if id_col is None:
                    continue
                id_label = "MMGs" if id_col == "MMG" else "DEGs"
                genes = set(df[id_col].dropna().astype(str))
                set_sizes[comparison] = len(genes)
                for gene in genes:
                    gene_membership.setdefault(gene, set()).add(comparison)

            intersection_counts: Dict[Tuple[str, ...], int] = {}
            for members in gene_membership.values():
                if not members:
                    continue
                key = tuple(sorted(members))
                intersection_counts[key] = intersection_counts.get(key, 0) + 1

            if not intersection_counts:
                return None

            top_items = sorted(intersection_counts.items(), key=lambda item: item[1], reverse=True)[:max_intersections]
            counts = np.array([count for _, count in top_items])
            active_comparisons = sorted(
                set(chain.from_iterable(key for key, _ in top_items)),
                key=lambda item: set_sizes.get(item, 0),
                reverse=True,
            )

            if not active_comparisons:
                return None

            fig_width = min(22, max(11, len(top_items) * 0.48 + 4))
            fig_height = min(18, max(8, len(active_comparisons) * 0.34 + 4))
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
            grid = fig.add_gridspec(
                nrows=2,
                ncols=2,
                width_ratios=(1.8, max(5, len(top_items) * 0.36)),
                height_ratios=(2.4, max(3, len(active_comparisons) * 0.28)),
                hspace=0.05,
                wspace=0.04,
            )
            ax_empty = fig.add_subplot(grid[0, 0])
            ax_bar = fig.add_subplot(grid[0, 1])
            ax_sets = fig.add_subplot(grid[1, 0])
            ax_matrix = fig.add_subplot(grid[1, 1], sharex=ax_bar)
            ax_empty.axis("off")

            x_positions = np.arange(len(top_items))
            bar_color = "#E69F00"
            active_color = "#0072B2"
            inactive_color = "#D7D7D7"
            line_color = "#4D4D4D"

            ax_bar.bar(x_positions, counts, color=bar_color, width=0.75)
            ax_bar.set_ylabel("Intersection size")
            ax_bar.set_title(f"Top {id_label} Intersections (top {len(top_items)})", pad=12)
            ax_bar.set_xticks([])
            self._style_axis(ax_bar)
            ax_bar.spines["bottom"].set_visible(False)
            for x_coord, count in zip(x_positions, counts):
                ax_bar.text(
                    x_coord,
                    count,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                )

            y_positions = np.arange(len(active_comparisons))
            set_values = [set_sizes.get(comparison, 0) for comparison in active_comparisons]
            ax_sets.barh(y_positions, set_values, color="#56B4E9", height=0.62)
            ax_sets.set_yticks(y_positions)
            ax_sets.set_yticklabels(active_comparisons, fontsize=8)
            ax_sets.invert_xaxis()
            ax_sets.invert_yaxis()
            ax_sets.set_xlabel(f"{id_label} per comparison")
            self._style_axis(ax_sets)
            ax_sets.grid(axis="x", alpha=0.25)
            ax_sets.grid(axis="y", visible=False)

            for x_coord, (members, _) in zip(x_positions, top_items):
                member_set = set(members)
                active_rows = []
                for y_coord, comparison in zip(y_positions, active_comparisons):
                    is_active = comparison in member_set
                    ax_matrix.scatter(
                        x_coord,
                        y_coord,
                        s=68 if is_active else 42,
                        facecolor=active_color if is_active else inactive_color,
                        edgecolor=active_color if is_active else "white",
                        linewidth=0.8,
                        zorder=3 if is_active else 2,
                    )
                    if is_active:
                        active_rows.append(y_coord)
                if len(active_rows) > 1:
                    ax_matrix.plot(
                        [x_coord, x_coord],
                        [min(active_rows), max(active_rows)],
                        color=line_color,
                        linewidth=1.1,
                        zorder=1,
                    )

            ax_matrix.set_yticks(y_positions)
            ax_matrix.set_yticklabels([])
            ax_matrix.invert_yaxis()
            ax_matrix.set_xlim(-0.7, len(top_items) - 0.3)
            ax_matrix.set_xlabel(f"Exact {id_label} set intersections")
            ax_matrix.tick_params(axis="x", bottom=False, labelbottom=False)
            ax_matrix.grid(axis="y", color="#ECECEC", linestyle="-", linewidth=0.8)
            ax_matrix.grid(axis="x", visible=False)
            self._style_axis(ax_matrix)
            ax_matrix.spines["left"].set_visible(False)

            output_file = self._plot_output_file("Venn_Plots", f"{prefix}UpSet_intersections.png")
            self._save_figure(fig, output_file, tight_layout=False)
            if self.logger:
                self.logger.info(f"UpSet-style plot saved to {output_file}")
            return output_file

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating UpSet-style plot: {e}")
            return None

    def _prepare_gene_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy indexed by MMG or gene identifiers when available."""
        prepared = df.copy()
        if "MMG" in prepared.columns:
            prepared = prepared.set_index("MMG")
        elif "Gene" in prepared.columns:
            prepared = prepared.set_index("Gene")
        return prepared

    def _build_sample_annotations(self, sample_names: List[str], targets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Infer per-sample condition labels for ordination plots."""
        annotations = pd.DataFrame({"Sample": sample_names})
        annotations["Condition"] = annotations["Sample"].str.split(r"[_\-.]", n=1).str[0]

        if targets is None or targets.empty:
            return annotations

        target_df = targets.copy()
        sample_column = None
        for column in ("sample", "Sample", "samples", "sample_id"):
            if column in target_df.columns:
                sample_column = column
                break

        if sample_column and "condition" in target_df.columns:
            condition_map = dict(zip(target_df[sample_column], target_df["condition"]))
            mapped = annotations["Sample"].map(condition_map)
            annotations["Condition"] = mapped.fillna(annotations["Condition"])
        elif "condition" in target_df.columns and len(target_df) == len(annotations):
            annotations["Condition"] = target_df["condition"].astype(str).values

        return annotations

    def load_data(self, data: Union[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Load data from file path or return DataFrame if already loaded.

        Args:
            data: DataFrame or file path string.

        Returns:
            Loaded pandas DataFrame or None if there's an issue.
        """
        if data is None:
            if self.logger:
                self.logger.error("Data input is None.")
            return None

        if isinstance(data, str):
            try:
                if not os.path.exists(data):
                    if self.logger:
                        self.logger.error(f"File not found: {data}")
                    return None

                file_ext = os.path.splitext(data)[1].lower()
                if file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(data)
                elif file_ext == ".csv":
                    df = pd.read_csv(data)
                elif file_ext in [".tsv", ".txt"]:
                    df = pd.read_csv(data, sep="\t")
                else:
                    if self.logger:
                        self.logger.error("Unsupported file format. Please provide a .xlsx, .csv, .tsv, or .txt file.")
                    return None
                return df
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error reading file {data}: {e}")
                return None
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            if self.logger:
                self.logger.error("Input should be a pandas DataFrame or a file path string.")
            return None

    def plot_volcano(
        self,
        degDF: Union[str, pd.DataFrame],
        comp: str,
        FOLD: float = 2,
        pValue: float = 0.05,
        color: Tuple[str, str, str] = ("red", "grey", "green"),
        dim: Tuple[int, int] = (8, 5),
        font: int = 14,
        dotsize: int = 10,
        markerType: str = "o",
        alpha: float = 0.5,
        prefix: str = "",
    ):
        """
        Plots a Volcano plot.
        """
        if self.logger:
            self.logger.info(f"Starting Volcano plot generation for {comp}.")

        output_file = self._plot_output_file(
            "Volcano_Plots",
            f"{prefix}{self._safe_plot_name(comp)}_volcano.png",
        )

        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate Volcano plot for {comp} and save to {output_file}")
            return

        degDF = self.load_data(degDF)
        if degDF is None:
            return

        PVAL = f"pvalue({comp})"
        LFC = f"logFC({comp})"

        # Check if columns exist
        if PVAL not in degDF.columns or LFC not in degDF.columns:
            if self.logger:
                self.logger.warning(f"Columns {PVAL} or {LFC} not found in DataFrame. Skipping Volcano plot for {comp}.")
            return

        _x = r"$ log_{2}(Fold Change)$"
        _y = r"$ -log_{10}(P-value)$"

        # Filter differential expression results based on comparison
        dk = degDF.copy()

        try:
            self._apply_publication_style()
            final = dk[dk[PVAL] <= 1.0].copy()  # Filter valid p-values
            final = final[final[PVAL] > 0].copy()

            # Initialize color column
            plot_colors = (
                self.signal_colors["down"],
                self.signal_colors["neutral"],
                self.signal_colors["up"],
            )
            final["colorADD"] = plot_colors[1]  # Default to grey (not significant)

            # Assign colors
            final.loc[(final[LFC] >= np.log2(FOLD)) & (final[PVAL] <= pValue), "colorADD"] = plot_colors[2]
            final.loc[(final[LFC] <= -np.log2(FOLD)) & (final[PVAL] <= pValue), "colorADD"] = plot_colors[0]

            final["log(10)_pvalue"] = -(np.log10(final[PVAL]))

            # Map colors to numerical values for scatter plot
            color_map_dict = {plot_colors[0]: 0, plot_colors[1]: 1, plot_colors[2]: 2}
            final["color_num"] = final["colorADD"].map(color_map_dict)

            # Create custom colormap
            cmap = ListedColormap([plot_colors[0], plot_colors[1], plot_colors[2]])

            # Create the plot
            fig, ax = plt.subplots(figsize=dim, dpi=300)

            # Scatter plot
            ax.scatter(
                final[LFC],
                final["log(10)_pvalue"],
                c=final["color_num"],
                cmap=cmap,
                alpha=alpha,
                s=dotsize,
                marker=markerType,
                vmin=0,
                vmax=2,
                edgecolors="none",
                rasterized=True,
            )

            # Create custom legend
            legend_labels = [
                f"Down-regulated ({(final['colorADD'] == plot_colors[0]).sum()})",
                f"Not significant ({(final['colorADD'] == plot_colors[1]).sum()})",
                f"Up-regulated ({(final['colorADD'] == plot_colors[2]).sum()})",
            ]
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=markerType,
                    color="w",
                    label=label,
                    markerfacecolor=col,
                    markersize=8,
                )
                for label, col in zip(legend_labels, plot_colors)
            ]

            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.24),
                ncol=len(legend_labels),
                frameon=False,
                columnspacing=1.4,
                handletextpad=0.5,
            )

            ax.set_xlabel(_x, fontsize=font, fontweight="bold")
            ax.set_ylabel(_y, fontsize=font, fontweight="bold")
            ax.axvline(np.log2(FOLD), color="#7A7A7A", linestyle=":", linewidth=1)
            ax.axvline(-np.log2(FOLD), color="#7A7A7A", linestyle=":", linewidth=1)
            ax.axhline(-np.log10(pValue), color="#7A7A7A", linestyle=":", linewidth=1)
            ax.set_title(f"Volcano Plot: {comp}", pad=28)

            self._style_axis(ax)
            self._save_figure(fig, output_file)
            if self.logger:
                self.logger.info(f"Volcano plot saved to {output_file}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"An error occurred during Volcano plot generation: {e}")

    def plot_ma(
        self,
        degDF: Union[str, pd.DataFrame],
        countDF: Union[str, pd.DataFrame],
        comp: str,
        FOLD: float = 2,
        FDR: float = 0.05,
        color: Tuple[str, str, str] = ("red", "grey", "green"),
        font: int = 14,
        dim: Tuple[int, int] = (8, 5),
        dotsize: int = 8,
        markerType: str = "o",
        alpha: float = 0.5,
        prefix: str = "",
    ):
        """
        Plots a MA plot.
        """
        if self.logger:
            self.logger.info(f"Starting MA plot generation for {comp}.")

        output_file = self._plot_output_file(
            "MA_Plots",
            f"{prefix}{self._safe_plot_name(comp)}_ma.png",
        )

        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate MA plot for {comp} and save to {output_file}")
            return

        degDF = self.load_data(degDF)
        countDF = self.load_data(countDF)

        if degDF is None or countDF is None:
            return

        LFC = f"logFC({comp})"
        PVAL = f"FDR({comp})"

        if LFC not in degDF.columns or PVAL not in degDF.columns:
            if self.logger:
                self.logger.warning(f"Columns {LFC} or {PVAL} not found. Skipping MA plot.")
            return

        _y = r"$ log_{2}(Fold Change)$"
        _x = r"$ log_{2}(Mean Count)$"

        try:
            self._apply_publication_style()
            if color == ("red", "grey", "green"):
                color = (
                    self.signal_colors["down"],
                    self.signal_colors["neutral"],
                    self.signal_colors["up"],
                )

            id_col = "MMG" if "MMG" in degDF.columns and "MMG" in countDF.columns else "Gene"
            if id_col in degDF.columns:
                degDF = degDF.set_index(id_col)
            if id_col in countDF.columns:
                countDF = countDF.set_index(id_col)

            countDF = countDF.apply(pd.to_numeric, errors="coerce")

            # Prepare counts for the comparison groups
            # Assuming comp format is "ConditionA-ConditionB"
            conditions = comp.split("-")
            if len(conditions) != 2:
                if self.logger:
                    self.logger.warning(
                        f"Comparison format '{comp}' not supported for MA plot mean count calculation. Expected 'CondA-CondB'."
                    )
                # Fallback: Use overall mean if specific columns can't be identified easily
                cdf_mean = countDF.mean(axis=1, numeric_only=True)
                counts = pd.DataFrame({"mean": cdf_mean})
            else:
                # Filter columns based on conditions
                # This assumes column names contain the condition string
                cdf1 = countDF.filter(regex=conditions[0], axis=1)
                cdf2 = countDF.filter(regex=conditions[1], axis=1)

                if cdf1.empty or cdf2.empty:
                    if self.logger:
                        self.logger.warning(
                            f"Could not find columns for conditions {conditions} in count matrix. Using overall mean."
                        )
                    cdf_mean = countDF.mean(axis=1, numeric_only=True)
                    counts = pd.DataFrame({"mean": cdf_mean})
                else:
                    cdf_mean1 = cdf1.mean(axis=1, numeric_only=True)
                    cdf_mean2 = cdf2.mean(axis=1, numeric_only=True)
                    counts = pd.concat([cdf_mean1, cdf_mean2], axis=1)
                    counts.columns = ["first", "second"]
                    counts["mean"] = counts.mean(axis=1, numeric_only=True)

            # Merge DE results with counts
            # Assuming index matches (Gene IDs)
            # Make sure indices align
            common_genes = degDF.index.intersection(counts.index)
            if len(common_genes) == 0:
                if self.logger:
                    self.logger.warning("No common genes found for MA plot.")
                return
            final = pd.concat([degDF.loc[common_genes], counts.loc[common_genes]], axis=1)

            # Color assignment
            final["colorADD"] = color[1]
            final.loc[(final[LFC] >= np.log2(FOLD)) & (final[PVAL] <= FDR), "colorADD"] = color[2]
            final.loc[(final[LFC] <= -np.log2(FOLD)) & (final[PVAL] <= FDR), "colorADD"] = color[0]

            # Calculate log2 mean
            # Avoid log(0)
            final = final[final["mean"] > 0].copy()
            final["log2_Mean"] = np.log2(final["mean"])

            # Map colors
            color_map_dict = {color[0]: 0, color[1]: 1, color[2]: 2}
            final["color_num"] = final["colorADD"].map(color_map_dict)
            cmap = ListedColormap([color[0], color[1], color[2]])

            # Plot
            fig, ax = plt.subplots(figsize=dim, dpi=300)
            ax.scatter(
                final["log2_Mean"],
                final[LFC],
                c=final["color_num"],
                cmap=cmap,
                alpha=alpha,
                s=dotsize,
                marker=markerType,
                vmin=0,
                vmax=2,
                edgecolors="none",
                rasterized=True,
            )

            legend_labels = [
                f"Significant down ({(final['colorADD'] == color[0]).sum()})",
                f"Not significant ({(final['colorADD'] == color[1]).sum()})",
                f"Significant up ({(final['colorADD'] == color[2]).sum()})",
            ]
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=markerType,
                    color="w",
                    label=label,
                    markerfacecolor=col,
                    markersize=10,
                )
                for label, col in zip(legend_labels, color)
            ]

            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.24),
                ncol=len(legend_labels),
                frameon=False,
                columnspacing=1.4,
                handletextpad=0.5,
            )

            plt.axhline(y=0, color="#7d7d7d", linestyle="--")
            ax.set_xlabel(_x, fontsize=font, fontweight="bold")
            ax.set_ylabel(_y, fontsize=font, fontweight="bold")
            ax.axhline(np.log2(FOLD), color="#7A7A7A", linestyle=":", linewidth=1)
            ax.axhline(-np.log2(FOLD), color="#7A7A7A", linestyle=":", linewidth=1)
            ax.set_title(f"MA Plot: {comp}", pad=28)
            self._style_axis(ax)

            self._save_figure(fig, output_file)
            if self.logger:
                self.logger.info(f"MA plot saved to {output_file}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"An error occurred during MA plot generation: {e}")

    def plot_heatmap(
        self,
        degDF: Union[str, pd.DataFrame],
        combinations: List[str],
        num: int = 50,
        figdim: Tuple[int, int] = (15, 10),
        color_map: str = "tab20",
        annot: bool = False,
        scale: bool = True,
        rowclus: bool = True,
        colclus: bool = True,
        zscore: Optional[int] = None,
        xlabel: bool = True,
        ylabel: bool = True,
        tickfont: Tuple[int, int] = (10, 10),
        theme: Optional[str] = None,
        prefix: str = "",
    ):
        """
        Plot a clustered heatmap for the top genes across all comparisons.
        """
        if self.logger:
            self.logger.info("Starting heatmap generation.")

        output_file_clustered = self._plot_output_file(
            "Heatmaps",
            f"{prefix}Top_{num}_genes_heatmap_clustered.png",
        )
        output_file_normal = self._plot_output_file(
            "Heatmaps",
            f"{prefix}Top_{num}_genes_heatmap.png",
        )

        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate Heatmap and save to {output_file_clustered} or {output_file_normal}")
            return

        degDF = self.load_data(degDF)
        if degDF is None or not combinations:
            return

        try:
            self._apply_publication_style()

            feature_label = "MMGs" if "MMG" in degDF.columns else "Genes"
            deg_indexed = self._prepare_gene_index(degDF)
            logfc_data = pd.DataFrame(index=deg_indexed.index)

            for combination in combinations:
                col_name = f"logFC({combination})"
                if col_name in deg_indexed.columns:
                    logfc_data[combination] = pd.to_numeric(deg_indexed[col_name], errors="coerce")
                else:
                    if self.logger:
                        self.logger.warning(f"Column {col_name} not found.")

            if logfc_data.empty:
                if self.logger:
                    self.logger.warning("No genes found for heatmap.")
                return

            logfc_data = logfc_data.dropna(how="all")
            if logfc_data.empty:
                if self.logger:
                    self.logger.warning("No finite values found for heatmap.")
                return

            gene_scores = logfc_data.abs().max(axis=1).sort_values(ascending=False)
            top_genes = gene_scores.head(max(1, int(num))).index
            heatmap_data = logfc_data.loc[top_genes]
            valid_combinations = heatmap_data.columns.tolist()

            if zscore is None and scale:
                row_std = heatmap_data.std(axis=1).replace(0, np.nan)
                heatmap_data = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0).div(row_std, axis=0).fillna(0)

            if heatmap_data.empty or heatmap_data.shape[0] < 2:
                if self.logger:
                    self.logger.warning("Not enough genes for heatmap clustering.")
                return

            if np.nanmax(np.abs(heatmap_data.to_numpy(dtype=float))) == 0:
                if self.logger:
                    self.logger.warning("Heatmap values are all zero; skipping DEG heatmap.")
                return

            # Dynamic figure size
            longest_label = max((len(str(c)) for c in valid_combinations), default=8)
            fig_width = min(30, max(12, len(valid_combinations) * 0.65 + longest_label * 0.08))
            fig_height = min(22, max(8, heatmap_data.shape[0] * 0.18 + 3))
            figdim = (fig_width, fig_height)
            cmap = color_map if color_map != "tab20" else "vlag"
            row_cluster = rowclus and heatmap_data.shape[0] > 1
            col_cluster = colclus and heatmap_data.shape[1] > 1

            # Plotting
            if row_cluster or col_cluster:
                g = sns.clustermap(
                    heatmap_data,
                    row_cluster=row_cluster,
                    col_cluster=col_cluster,
                    cmap=cmap,
                    annot=annot,
                    cbar=scale,
                    z_score=zscore,
                    xticklabels=xlabel,
                    yticklabels=ylabel,
                    figsize=figdim,
                    linewidths=0.5,
                    linecolor="white",
                    center=0,
                    dendrogram_ratio=(0.15, 0.08),
                    cbar_kws={"label": "Z-scored logFC" if scale else "logFC"},
                )
                g.ax_heatmap.set_xlabel("Comparisons")
                g.ax_heatmap.set_ylabel(feature_label)
                g.ax_heatmap.tick_params(axis="x", labelrotation=60, labelsize=8)
                for label in g.ax_heatmap.get_xticklabels():
                    label.set_ha("right")
                g.ax_heatmap.tick_params(axis="y", labelsize=7)
                g.fig.suptitle(
                    f"Top Differentially Expressed {feature_label} (n={len(heatmap_data)})",
                    y=1.02,
                )
                g.savefig(output_file_clustered, dpi=300, bbox_inches="tight")
                g.savefig(
                    os.path.splitext(output_file_clustered)[0] + ".pdf",
                    bbox_inches="tight",
                )
                plt.close(g.fig)
                if self.logger:
                    self.logger.info(f"Heatmap saved to {output_file_clustered}")
            else:
                fig, ax = plt.subplots(figsize=figdim, dpi=300)
                sns.heatmap(
                    heatmap_data,
                    cmap=cmap,
                    annot=annot,
                    cbar=scale,
                    linewidths=0.5,
                    linecolor="white",
                    center=0,
                    cbar_kws={"label": "Z-scored logFC" if scale else "logFC"},
                    ax=ax,
                )
                ax.set_xlabel("Comparisons")
                ax.set_ylabel(feature_label)
                ax.tick_params(axis="x", labelrotation=60, labelsize=8)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                ax.tick_params(axis="y", labelsize=7)
                ax.set_title(
                    f"Top Differentially Expressed {feature_label} (n={len(heatmap_data)})",
                    pad=16,
                )
                self._save_figure(fig, output_file_normal)
                if self.logger:
                    self.logger.info(f"Heatmap saved to {output_file_normal}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating heatmap: {e}")

    def pca_plot(
        self,
        ncountdf: Union[str, pd.DataFrame],
        legends: bool = False,
        fontsize: int = 14,
        figsize: Tuple[int, int] = (12, 12),
        dpi: int = 300,
        color_palette: str = "husl",
        prefix: str = "",
    ):
        """
        Plots a PCA plot based on normalized counts.
        """
        if self.logger:
            self.logger.info("Starting PCA plot.")

        output_file = self._plot_output_file("Sample_Plots", f"{prefix}PCA_plot.png")

        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate PCA plot and save to {output_file}")
            return

        ncountdf = self.load_data(ncountdf)
        if ncountdf is None:
            return

        # Ensure Gene column is handled
        if "Gene" in ncountdf.columns:
            ncountdf.set_index("Gene", inplace=True)

        # Transpose: PCA needs samples as rows, genes as columns
        ndf = ncountdf.T

        try:
            self._apply_publication_style()
            # Standardize and PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(StandardScaler().fit_transform(ndf))

            pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"], index=ndf.index)
            pca_df = pca_df.join(self._build_sample_annotations(pca_df.index.tolist()).set_index("Sample"))

            # Calculate variance explained
            explained_variance = pca.explained_variance_ratio_
            pc1_var = round(explained_variance[0] * 100, 2)
            pc2_var = round(explained_variance[1] * 100, 2)

            # Plot
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            unique_conditions = pca_df["Condition"].unique().tolist()
            palette = dict(
                zip(
                    unique_conditions,
                    sns.color_palette(color_palette, n_colors=len(unique_conditions)),
                )
            )

            for sample, row in pca_df.iterrows():
                ax.scatter(
                    row["PC1"],
                    row["PC2"],
                    color=palette[row["Condition"]],
                    label=row["Condition"] if legends else sample,
                    s=110,
                    edgecolors="white",
                    linewidths=0.6,
                )

            # Add labels
            texts = [
                ax.text(row["PC1"], row["PC2"], sample, fontsize=max(fontsize - 3, 8)) for sample, row in pca_df.iterrows()
            ]
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5))

            if legends:
                handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=condition,
                        markerfacecolor=palette[condition],
                        markersize=8,
                    )
                    for condition in unique_conditions
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    title="Condition",
                )

            ax.set_xlabel(f"Principal Component 1 ({pc1_var}%)", fontsize=fontsize)
            ax.set_ylabel(f"Principal Component 2 ({pc2_var}%)", fontsize=fontsize)
            ax.set_title("Principal Component Analysis", pad=16)
            self._style_axis(ax)

            self._save_figure(fig, output_file)
            if self.logger:
                self.logger.info(f"PCA plot saved to {output_file}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to perform PCA: {e}")

    def plot_tsne(
        self,
        count_file: Union[str, pd.DataFrame],
        targets: pd.DataFrame = None,
        prefix: str = "",
    ):
        """
        Plots t-SNE based on counts.
        """
        if self.logger:
            self.logger.info("Starting t-SNE plot.")

        output_file = self._plot_output_file("Sample_Plots", f"{prefix}t-SNE_plot.png")

        if self.dryrun:
            self.logger.info(f"DRY RUN: Would generate t-SNE plot and save to {output_file}")
            return

        count_df = self.load_data(count_file)
        if count_df is None:
            return

        if "Gene" in count_df.columns:
            count_df.set_index("Gene", inplace=True)

        try:
            self._apply_publication_style()
            # Log transform
            normalized_counts = np.log1p(count_df)
            sample_count = normalized_counts.shape[1]
            if sample_count < 3:
                raise VisualizationError("t-SNE requires at least three samples.")

            # t-SNE
            perplexity = max(2, min(30, sample_count - 1))
            if perplexity >= sample_count:
                perplexity = sample_count - 1
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_results = tsne.fit_transform(normalized_counts.T)

            tsne_df = pd.DataFrame(tsne_results, columns=["t-SNE1", "t-SNE2"])
            tsne_df["Sample"] = count_df.columns

            # Add conditions if available
            annotations = self._build_sample_annotations(tsne_df["Sample"].tolist(), targets)
            tsne_df = tsne_df.merge(annotations, on="Sample", how="left", suffixes=("", "_ann"))
            if "Condition_ann" in tsne_df.columns:
                tsne_df["Condition"] = tsne_df["Condition_ann"]
                tsne_df = tsne_df.drop(columns=["Condition_ann"])

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            sns.scatterplot(
                x="t-SNE1",
                y="t-SNE2",
                hue="Condition",
                style="Condition",
                data=tsne_df,
                palette="colorblind",
                s=120,
                ax=ax,
            )
            texts = [ax.text(row["t-SNE1"], row["t-SNE2"], row["Sample"], fontsize=8) for _, row in tsne_df.iterrows()]
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5))
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Condition")
            ax.set_title("t-SNE of Samples", pad=16)
            self._style_axis(ax)

            self._save_figure(fig, output_file)
            if self.logger:
                self.logger.info(f"t-SNE plot saved to {output_file}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in t-SNE plot: {e}")

    def run(
        self,
        norm_counts_file: Optional[str] = None,
        de_results_file: Optional[str] = None,
        filtered_deg_file: Optional[str] = None,
        mmg_de_results_file: Optional[str] = None,
        mmg_counts_file: Optional[str] = None,
        filtered_mmg_file: Optional[str] = None,
        sample_dict: Optional[Dict] = None,
        log2fc_threshold: float = 1.0,
        fdr_threshold: float = 0.05,
        pca_plot: bool = True,
        tsne_plot: bool = True,
        volcano_plot: bool = True,
        ma_plot: bool = True,
        deg_heatmap: bool = True,
        heatmap_top_genes: int = 50,
        venn: bool = True,
        venn_comparisons: Optional[List[str]] = None,
        venn_label: str = "updown",
        upset: bool = True,
    ) -> Dict[str, str]:
        """
        Run the visualization pipeline.

        Args:
            norm_counts_file: Path to normalized counts file.
            de_results_file: Path to differential expression results file.
            filtered_deg_file: Path to filtered DEG workbook.
            mmg_de_results_file: Path to MMG differential expression results.
            mmg_counts_file: Path to raw MMG count matrix for MMG MA plots.
            filtered_mmg_file: Path to filtered MMG workbook for Venn/UpSet plots.
            sample_dict: Dictionary of sample information.
            log2fc_threshold: Log2 Fold Change threshold.
            fdr_threshold: FDR threshold.
            pca_plot: Whether to generate PCA plots from normalized counts.
            tsne_plot: Whether to generate t-SNE plots from normalized counts.
            volcano_plot: Whether to generate volcano plots.
            ma_plot: Whether to generate MA plots.
            deg_heatmap: Whether to generate DEG heatmap.
            heatmap_top_genes: Number of top genes to include in DEG heatmap.

        Returns:
            Dictionary of output files (or empty if dryrun).
        """
        self.logger.info("Running visualization module")

        results = {}

        # 1. PCA and t-SNE Plots
        if norm_counts_file and os.path.exists(norm_counts_file):
            if pca_plot:
                self.logger.info(f"Generating PCA plot using {norm_counts_file}")
                self.pca_plot(ncountdf=norm_counts_file, legends=True, prefix="All_Samples_")

            if tsne_plot and sample_dict:
                try:
                    targets_data = []
                    for sample, info in sample_dict.items():
                        # sample_dict format: {sample: [rep, condition, ...]}
                        targets_data.append({"sample": sample, "condition": info[1]})
                    targets_df = pd.DataFrame(targets_data)

                    self.plot_tsne(
                        count_file=norm_counts_file,
                        targets=targets_df,
                        prefix="All_Samples_",
                    )
                except Exception as e:
                    self.logger.warning(f"Skipping t-SNE: {e}")
            elif tsne_plot:
                self.logger.info("Sample metadata not available; skipping t-SNE plot.")
        else:
            if self.dryrun:
                self.logger.info("DRYRUN: Would generate PCA/t-SNE plots when normalized counts are produced.")
            else:
                self.logger.warning("Normalized counts file not found or invalid. Skipping PCA and t-SNE.")

        # 2. Differential Expression Plots
        if de_results_file and os.path.exists(de_results_file):
            self.logger.info(f"Generating DE plots using {de_results_file}")

            try:
                de_df = pd.read_excel(de_results_file)

                # Identify comparisons
                comparisons = set()
                for col in de_df.columns:
                    match = re.search(r"logFC\((.+)\)", col)
                    if match:
                        comparisons.add(match.group(1))

                comparisons = list(comparisons)
                self.logger.info(f"Found comparisons: {comparisons}")

                for comp in comparisons:
                    if volcano_plot:
                        self.plot_volcano(
                            degDF=de_df,
                            comp=comp,
                            FOLD=2**log2fc_threshold,
                            pValue=fdr_threshold,
                            prefix="",
                        )

                    if ma_plot and norm_counts_file:
                        self.plot_ma(
                            degDF=de_df,
                            countDF=norm_counts_file,
                            comp=comp,
                            FOLD=2**log2fc_threshold,
                            FDR=fdr_threshold,
                            prefix="",
                        )

                if deg_heatmap and comparisons:
                    self.plot_heatmap(
                        degDF=de_df,
                        combinations=comparisons,
                        num=heatmap_top_genes,
                        prefix="All_",
                    )

                if venn:
                    if filtered_deg_file is None:
                        candidate = Path(de_results_file).with_name("Filtered_DEGs.xlsx")
                        filtered_deg_file = str(candidate) if candidate.exists() else None
                    if filtered_deg_file:
                        self.plot_venn(
                            deg_file=filtered_deg_file,
                            comparisons=venn_comparisons,
                            fold=2**log2fc_threshold,
                            deg_label=venn_label,
                        )
                        if upset:
                            self.plot_upset(
                                deg_file=filtered_deg_file,
                                fold=2**log2fc_threshold,
                            )
                    else:
                        self.logger.info("Filtered DEG file not found; skipping Venn and UpSet-style plots.")

            except Exception as e:
                self.logger.error(f"Error processing DE results: {e}")
        else:
            if self.dryrun:
                self.logger.info("DRYRUN: Would generate DE plots when differential expression results are produced.")
            else:
                self.logger.warning("DE results file not found or invalid. Skipping DE plots.")

        # 3. Multimapped gene-group plots
        if mmg_de_results_file and os.path.exists(mmg_de_results_file):
            self.logger.info(f"Generating MMG DE plots using {mmg_de_results_file}")
            try:
                mmg_df = pd.read_excel(mmg_de_results_file)
                mmg_comparisons = []
                for col in mmg_df.columns:
                    match = re.search(r"logFC\((.+)\)", col)
                    if match:
                        mmg_comparisons.append(match.group(1))
                mmg_comparisons = sorted(set(mmg_comparisons))
                self.logger.info(f"Found MMG comparisons: {mmg_comparisons}")

                for comp in mmg_comparisons:
                    if volcano_plot:
                        self.plot_volcano(
                            degDF=mmg_df,
                            comp=comp,
                            FOLD=2**log2fc_threshold,
                            pValue=fdr_threshold,
                            prefix="MMG_",
                        )
                    if ma_plot and mmg_counts_file and os.path.exists(mmg_counts_file):
                        self.plot_ma(
                            degDF=mmg_df,
                            countDF=mmg_counts_file,
                            comp=comp,
                            FOLD=2**log2fc_threshold,
                            FDR=fdr_threshold,
                            prefix="MMG_",
                        )

                if deg_heatmap and mmg_comparisons:
                    self.plot_heatmap(
                        degDF=mmg_df,
                        combinations=mmg_comparisons,
                        num=heatmap_top_genes,
                        prefix="MMG_",
                    )

                if venn:
                    if filtered_mmg_file is None:
                        candidate = Path(mmg_de_results_file).with_name("Filtered_MMGs.xlsx")
                        filtered_mmg_file = str(candidate) if candidate.exists() else None
                    if filtered_mmg_file:
                        self.plot_venn(
                            deg_file=filtered_mmg_file,
                            comparisons=venn_comparisons,
                            fold=2**log2fc_threshold,
                            deg_label=venn_label,
                            prefix="MMG_",
                        )
                        if upset:
                            self.plot_upset(
                                deg_file=filtered_mmg_file,
                                fold=2**log2fc_threshold,
                                prefix="MMG_",
                            )
                    else:
                        self.logger.info("Filtered MMG file not found; skipping MMG Venn and UpSet-style plots.")
            except Exception as e:
                self.logger.error(f"Error processing MMG DE results: {e}")
        elif mmg_de_results_file and self.dryrun:
            self.logger.info("DRYRUN: Would generate MMG DE plots when MMG differential results are produced.")

        return results
