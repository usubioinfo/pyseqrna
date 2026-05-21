#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gene Ontology Enrichment Module

This module provides Gene Ontology (GO) enrichment analysis for differentially
expressed genes with support for both ENSEMBL and NCBI gene identifiers.

Features:
    - Gene Ontology (GO) enrichment analysis for biological processes, molecular functions, and cellular components
    - Support for both ENSEMBL and NCBI gene identifiers
    - Multi-test correction using Benjamini-Hochberg (FDR) or Bonferroni methods
    - Rich visualizations including barplots, dotplots, and combined category plots
    - Flexible background definition for enrichment statistics
    - BioMart-based automatic annotation retrieval

Configuration:
    The module can be configured with:
    - Species identifier and organism type (plants/animals)
    - Key type (ensembl/ncbi) and taxonomy ID (taxid)
    - Path to GFF file for custom gene mapping

Dependencies:
    - numpy
    - pandas
    - scipy
    - matplotlib
    - requests

Classes:
    GeneOntologyPlotter - Plotting utilities for Gene Ontology enrichment analysis.
    GeneOntology - Gene Ontology enrichment analysis for differentially expressed genes.

Exceptions:
    GeneOntologyError - Custom exception for Gene Ontology-related errors.

:Created: January 5, 2022
:Updated: March 10, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import json
import math
import os
from typing import List, Optional, Any, Tuple
from xml.etree import ElementTree
from io import StringIO, TextIOWrapper
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from future.utils import native_str
from scipy.stats import rankdata

# Import utility modules
from ...utils import LogManager, FileManager
from ...utils.dry_run_manager import DryRunManager

HTTP_TIMEOUT_SECONDS = 60


class GeneOntologyError(Exception):
    """Custom exception for Gene Ontology-related errors."""

    pass


class GeneOntologyPlotter:
    """
    Plotting utilities for Gene Ontology enrichment analysis.

    This class provides various visualization methods for GO enrichment results
    including dotplots and barplots with support for different ontology categories.

    Attributes:
        logger: Logger instance for tracking operations
        dryrun: Whether to perform dry run (skip actual plot generation)
    """

    def __init__(self, logger, dryrun=False):
        """
        Initialize the GO plotter.

        Args:
            logger: Logger instance for tracking operations
            dryrun: Whether to perform dry run (skip actual plot generation)
        """
        self.logger = logger
        self.dryrun = dryrun

    def _prepare_plot_data(
        self,
        df: pd.DataFrame,
        term_col: str,
        nrows: int,
        sort_by: str = "Counts",
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        ontology: Optional[str] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return a plotting-safe copy with valid labels and numeric columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input df must be a pandas DataFrame.")

        plot_df = df.copy()
        if ontology is not None:
            plot_df = plot_df[plot_df["Ontology"] == ontology].copy()

        required = {term_col, sort_by}
        if color_by:
            required.add("Pvalues" if color_by == "logPvalues" else color_by)
        if size_by:
            required.add(size_by)
        missing = [col for col in required if col not in plot_df.columns]
        if missing:
            raise ValueError(f"Missing required GO plot column(s): {', '.join(missing)}")

        numeric_cols = {"Counts", "Pvalues", "FDR", sort_by}
        if color_by and color_by != "logPvalues":
            numeric_cols.add(color_by)
        if size_by:
            numeric_cols.add(size_by)
        for col in numeric_cols.intersection(plot_df.columns):
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        if color_by == "logPvalues":
            with np.errstate(divide="ignore", invalid="ignore"):
                plot_df["logPvalues"] = -np.log10(plot_df["Pvalues"])
            plot_df["logPvalues"] = plot_df["logPvalues"].replace([np.inf, -np.inf], np.nan)

        plot_df = plot_df[plot_df[term_col].notna()].copy()
        plot_df[term_col] = plot_df[term_col].astype(str).str.strip()
        plot_df = plot_df[(plot_df[term_col] != "") & (plot_df[term_col].str.lower() != "nan")]

        drop_cols = [sort_by]
        if color_by:
            drop_cols.append(color_by)
        if size_by:
            drop_cols.append(size_by)
        plot_df = plot_df.dropna(subset=list(dict.fromkeys(drop_cols)))
        plot_df = plot_df[plot_df[sort_by] > 0]

        if plot_df.empty:
            label = f" for ontology {ontology}" if ontology else ""
            self.logger.warning(f"No valid GO rows available for plotting{label}.")
            return plot_df

        return plot_df.sort_values(sort_by, ascending=ascending).head(nrows)

    @staticmethod
    def _normalise(values: pd.Series) -> pd.Series:
        max_value = pd.to_numeric(values, errors="coerce").max()
        if pd.isna(max_value) or max_value <= 0:
            return pd.Series(np.full(len(values), 0.5), index=values.index)
        return values / max_value

    @staticmethod
    def _truncate_labels(labels: pd.Series, width: int = 40) -> List[str]:
        return [label[: width - 3] + "..." if len(label) > width else label for label in labels.astype(str)]

    def barplotGO(self, df=None, nrows=20, colorBy="logPvalues", outdir=".", prefix="pyseqrna"):
        """
        Create a barplot for Gene Ontology enrichment.

        :param df: DataFrame containing Gene Ontology enrichment data.
        :param nrows: Number of rows to plot (default: 20).
        :param colorBy: Variable to color the bars ('logPvalues' or 'FDR').
        :return: A matplotlib figure containing the barplot.
        """
        if self.dryrun:
            self.logger.info(f"DRYRUN: Would create barplot GO and save to {os.path.join(outdir, f'{prefix}_barplot.png')}")
            return

        self.logger.info("Creating barplot GO.")

        # Validate input
        if colorBy not in ["logPvalues", "FDR"]:
            self.logger.error("Invalid value for colorBy: %s", colorBy)
            raise ValueError("Invalid value for colorBy. Use 'logPvalues' or 'FDR'.")

        df = self._prepare_plot_data(df, "GO Term", nrows, sort_by="Counts", color_by=colorBy)
        if df.empty:
            return
        counts = df["Counts"].values
        terms = df["GO Term"]

        # Normalize colors for the bar plot
        data_color_normalized = self._normalise(df[colorBy])
        actual_nrows = len(df)
        fsize = max(5, math.ceil(actual_nrows / 2))  # Ensure minimum figure height

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)
        colors = plt.cm.tab20c(data_color_normalized)

        # Create horizontal bar plot
        ax.barh(range(actual_nrows), counts, color=colors)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers

        # Set axis properties
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds((0, actual_nrows))
        ax.spines["left"].set_position(("outward", 8))
        ax.spines["bottom"].set_position(("outward", 5))
        ax.margins(y=0)  # Remove space between y-axis and bars

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Counts", fontsize=12, fontweight="bold")
        plt.ylabel("GO Description", fontsize=12, fontweight="bold")

        # Create color bar
        sm = ScalarMappable(cmap="tab20", norm=plt.Normalize(0, max(float(df[colorBy].max()), 1e-12)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=0.02, aspect=10)
        cbar.ax.set_title(colorBy, pad=20, fontweight="bold")

        # Set y-tick labels with wrapping
        truncated_labels = self._truncate_labels(terms)
        ax.set_yticks(range(len(truncated_labels)))  # Set y-ticks to match actual number of rows
        ax.set_yticklabels(truncated_labels, fontsize=12)

        # Set y-limits to ensure they reflect the number of terms plotted
        ax.set_ylim(-1, len(truncated_labels))  # Use -1 to adjust for zero-based index

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}_barplot.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Barplot GO created successfully and saved at {os.path.join(outdir, f'{prefix}_barplot.png')}")

        return

    def barplotTopFromEachOntology(self, df, colorBy="logPvalues", nrows=10, outdir=".", prefix="pyseqrna"):
        """
        Create a bar plot for the top GO terms from each ontology category (BP, MF, CC)
        with gradient colors based on a specified column.

        :param df: DataFrame containing Gene Ontology enrichment data.
        :param colorBy: Column name to use for gradient coloring (default: 'logPvalues').
        :param nrows: Number of rows to plot for each category (default: 10).
        :return: List of matplotlib figures containing the bar plots for each category.
        """
        if self.dryrun:
            self.logger.info(f"DRYRUN: Would create barplot for each ontology and save to {outdir}")
            return

        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input df must be a pandas DataFrame.")
            raise TypeError("Input df must be a pandas DataFrame.")

        # Validate colorBy column
        if colorBy != "logPvalues" and colorBy not in df.columns:
            self.logger.error(f"colorBy column '{colorBy}' not found in DataFrame.")
            raise ValueError(f"colorBy column '{colorBy}' not found in DataFrame.")

        # Define distinct colors for each ontology category (base colors)

        # Filter and plot for each ontology category
        categories = ["BP", "MF", "CC"]

        for category in categories:
            category_df = self._prepare_plot_data(
                df,
                "GO Term",
                nrows,
                sort_by="Counts",
                color_by=colorBy,
                ontology=category,
            )
            if category_df.empty:
                continue
            category_df = category_df.sort_values(by="Counts", ascending=True)  # Sort to have highest on top

            counts = category_df["Counts"].values
            terms = category_df["GO Term"]

            # Prepare gradient colors based on the specified colorBy column
            gradient_values = category_df[colorBy].values  # Values to create gradient
            norm = plt.Normalize(float(np.nanmin(gradient_values)), float(np.nanmax(gradient_values)))
            colors = plt.cm.tab20c(norm(gradient_values))  # Using viridis colormap for gradients

            # Determine the number of rows to use for the plot height
            actual_nrows = len(category_df)  # Actual number of rows available
            fsize = max(5, math.ceil(actual_nrows / 2))  # Adjust figure height based on actual nrows

            # Create the plot for the current category
            fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)

            # Create horizontal bar plot with gradient colors
            ax.barh(range(actual_nrows), counts, color=colors)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers

            # Set axis properties
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_bounds((0, actual_nrows))
            ax.spines["left"].set_position(("outward", 8))
            ax.spines["bottom"].set_position(("outward", 5))

            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel("Counts", fontsize=12, fontweight="bold")
            plt.ylabel("GO Description", fontsize=12, fontweight="bold")

            # Set y-limits to ensure they reflect the number of terms
            ax.set_ylim(-1, actual_nrows)  # -1 to adjust for the zero-based index

            # Truncate y-tick labels if they exceed 40 characters
            truncated_labels = self._truncate_labels(terms)
            ax.set_yticks(range(actual_nrows))  # Set y-ticks to match actual number of rows
            ax.set_yticklabels(truncated_labels, fontsize=12)

            # Add a title for each category
            ax.set_title(
                f"Top {min(nrows, len(category_df))} GO Terms for {category} Category",
                fontsize=14,
                fontweight="bold",
            )

            # Create a colorbar to represent the colorBy variable
            sm = plt.cm.ScalarMappable(cmap="tab20c", norm=norm)
            sm.set_array([])  # Only needed for older versions of matplotlib
            cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=0.2, aspect=10)
            cbar.set_label(colorBy, fontsize=12, fontweight="bold")

            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"{prefix}_{category}_barplot.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Log the creation of the plot for the current category
            self.logger.info(
                f"Created bar plot for {category} category with {actual_nrows} terms and saved at {os.path.join(outdir, f'{prefix}_{category}_barplot.png')}"
            )

        return

    def barplotCombinedTop(self, df, nrows=10, outdir=".", prefix="pyseqrna"):
        """
        Create a combined bar plot for the top GO terms from each ontology category (BP, MF, CC)
        with separate colors for each ontology and category labels on the right.

        :param df: DataFrame containing Gene Ontology enrichment data.
        :param nrows: Number of rows to plot for each category (default: 10).
        :return: Matplotlib figure containing the combined bar plot.
        """
        if self.dryrun:
            self.logger.info(
                f"DRYRUN: Would create combined barplot and save to {os.path.join(outdir, f'{prefix}_combined_top_barplot.png')}"
            )
            return

        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input df must be a pandas DataFrame.")
            raise TypeError("Input df must be a pandas DataFrame.")

        # Define distinct colors for each ontology category
        category_colors = {"BP": "blue", "MF": "green", "CC": "red"}

        # Prepare DataFrame for combined plot
        combined_data = []

        for category in category_colors.keys():
            category_df = self._prepare_plot_data(df, "GO Term", nrows, sort_by="Counts", ontology=category)
            if category_df.empty:
                continue
            combined_data.append(category_df)

        if not combined_data:
            self.logger.warning("No valid GO rows available for combined barplot.")
            return

        # Concatenate all the top results into a single DataFrame
        combined_df = pd.concat(combined_data).reset_index(drop=True)
        if combined_df.empty:
            self.logger.warning("No valid GO rows available for combined barplot.")
            return
        combined_df["GO Term"] = combined_df["GO Term"].astype(str)

        # Calculate the figure height based on the number of rows
        fsize = max(6, nrows * 1)  # Adjust factor as necessary for aesthetic
        fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)

        # Create the bar plot with distinct colors for each ontology
        ax.barh(
            range(len(combined_df)),
            combined_df["Counts"],
            color=[category_colors[ont] for ont in combined_df["Ontology"]],
        )

        # Set axis properties
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Counts", fontsize=12, fontweight="bold")
        plt.ylabel("GO Description", fontsize=12, fontweight="bold")

        # Truncate y-tick labels if they exceed 40 characters
        truncated_labels = self._truncate_labels(combined_df["GO Term"])
        ax.set_yticks(range(len(truncated_labels)))  # Set y-ticks to match actual number of rows
        ax.set_yticklabels(truncated_labels, fontsize=12)

        # Set y-limits to ensure they reflect the number of terms plotted
        ax.set_ylim(-1, len(truncated_labels))  # Use -1 to adjust for zero-based index

        # Add a title for the combined plot
        ax.set_title(
            f"Combined Top {nrows} GO Terms from Each Ontology Category",
            fontsize=14,
            fontweight="bold",
        )

        # Add labels on the right side for each category
        y_offset = 0.2  # Vertical offset for placing text labels
        for i, (index, row) in enumerate(combined_df.iterrows()):
            category = row["Ontology"]
            ax.text(
                combined_df["Counts"].max() + 10,
                i - y_offset,
                category,
                color=category_colors[category],
                fontsize=12,
                fontweight="bold",
            )

        # Draw a vertical line to separate the bars and the category labels
        ax.axvline(x=combined_df["Counts"].max() + 9, color="gray", linestyle="--")

        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"{prefix}_combined_top_barplot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        # Log the creation of the combined plot
        self.logger.info(
            f"Created combined bar plot for top {nrows} GO terms from each ontology and saved at {os.path.join(outdir, f'{prefix}_combined_top_barplot.png')}"
        )

        return

    def dotplotGO(
        self,
        df=None,
        nrows=20,
        colorBy="logPvalues",
        sizeBy="Counts",
        outdir=".",
        prefix="pyseqrna",
    ):
        """
        Create a dotplot for Gene Ontology enrichment.

        :param df: DataFrame containing Gene Ontology enrichment data.
        :param nrows: Number of rows to plot (default: 20).
        :param colorBy: Variable to color the dots ('logPvalues' or 'FDR').
        :param sizeBy: Variable to scale the dot sizes ('Counts').
        :return: A matplotlib figure containing the dotplot.
        """
        if self.dryrun:
            self.logger.info(f"DRYRUN: Would create dotplot GO and save to {os.path.join(outdir, f'{prefix}_dotplot.png')}")
            return

        self.logger.info("Creating dotplot GO.")

        # Validate input
        if colorBy not in ["logPvalues", "FDR"]:
            self.logger.error("Invalid value for colorBy: %s", colorBy)
            raise ValueError("Invalid value for colorBy. Use 'logPvalues' or 'FDR'.")

        df = self._prepare_plot_data(df, "GO Term", nrows, sort_by=sizeBy, color_by=colorBy, size_by=sizeBy)
        if df.empty:
            return
        counts = df[sizeBy].values
        terms = df["GO Term"]
        actual_nrows = len(df)

        # Normalize colors and sizes for the dot plot
        data_color_normalized = self._normalise(df[colorBy])
        data_size_normalized = self._normalise(df[sizeBy]) * 300  # Scaling the size of dots

        fsize = max(5, math.ceil(actual_nrows / 2))  # Ensure minimum figure height

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)
        ax.scatter(
            counts,
            range(actual_nrows),
            s=data_size_normalized,
            c=data_color_normalized,
            cmap="tab20",
        )

        # Set axis properties
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds((0, actual_nrows))
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))
        ax.margins(y=0)  # Remove space between y-axis and bars

        plt.xticks(fontsize=10)
        plt.yticks(range(actual_nrows), terms, fontsize=10)
        plt.xlabel("Counts", fontsize=12, fontweight="bold")
        plt.ylabel("GO Description", fontsize=12, fontweight="bold")

        # Create color bar
        sm = ScalarMappable(cmap="tab20", norm=plt.Normalize(0, max(float(df[colorBy].max()), 1e-12)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=0.02, aspect=10)
        cbar.ax.set_title(colorBy, pad=20, fontweight="bold")

        # Set y-tick labels with wrapping
        truncated_labels = self._truncate_labels(terms)
        ax.set_yticklabels(truncated_labels, fontsize=12)

        # Set y-limits to ensure they reflect the number of terms plotted
        ax.set_ylim(-1, len(truncated_labels))  # Use -1 to adjust for zero-based index

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}_dotplot.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Dotplot GO created successfully and saved at {os.path.join(outdir, f'{prefix}_dotplot.png')}")
        return

    def dotplotCombinedTop(
        self,
        df,
        nrows=10,
        colorBy="logPvalues",
        sizeBy="Counts",
        outdir=".",
        prefix="pyseqrna",
    ):
        """
        Create a combined dot plot for the top GO terms from each ontology category (BP, MF, CC)
        with separate colors for each ontology and category labels on the right.

        :param df: DataFrame containing Gene Ontology enrichment data.
        :param nrows: Number of rows to plot for each category (default: 10).
        :param colorBy: Variable to color the dots ('logPvalues' or 'FDR').
        :param sizeBy: Variable to scale the dot sizes ('Counts').
        :return: Matplotlib figure containing the combined dot plot.
        """
        if self.dryrun:
            self.logger.info(
                f"DRYRUN: Would create combined dotplot and save to {os.path.join(outdir, f'{prefix}_combined_top_dotplot.png')}"
            )
            return

        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input df must be a pandas DataFrame.")
            raise TypeError("Input df must be a pandas DataFrame.")

        # Define distinct colors for each ontology category
        category_colors = {"BP": "blue", "MF": "green", "CC": "red"}

        # Prepare DataFrame for combined plot
        combined_data = []

        for category in category_colors.keys():
            category_df = self._prepare_plot_data(
                df,
                "GO Term",
                nrows,
                sort_by=sizeBy,
                color_by=colorBy,
                size_by=sizeBy,
                ontology=category,
            )
            if category_df.empty:
                continue
            combined_data.append(category_df)

        if not combined_data:
            self.logger.warning("No valid GO rows available for combined dotplot.")
            return

        # Concatenate all the top results into a single DataFrame
        combined_df = pd.concat(combined_data).reset_index(drop=True)

        # Normalize colors and sizes for the dot plot
        data_size_normalized = self._normalise(combined_df[sizeBy]) * 300  # Scale for dot sizes

        # Calculate the figure height based on the number of rows
        fsize = max(6, nrows * 1)  # Adjust factor as necessary for aesthetic
        fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)

        # Create the dot plot with distinct colors for each ontology
        for point_index, (index, row) in enumerate(combined_df.iterrows()):
            category = row["Ontology"]
            ax.scatter(
                row[sizeBy],
                point_index,
                s=data_size_normalized.loc[index],
                color=category_colors[category],
                alpha=0.6,
                edgecolor="black",
            )

        # Set axis properties
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))
        plt.xticks(fontsize=10)
        plt.yticks(
            range(len(combined_df["GO Term"])),
            combined_df["GO Term"].astype(str),
            fontsize=10,
        )
        plt.xlabel("Counts", fontsize=12, fontweight="bold")
        plt.ylabel("GO Description", fontsize=12, fontweight="bold")

        # Truncate y-tick labels if they exceed 40 characters
        truncated_labels = self._truncate_labels(combined_df["GO Term"])
        ax.set_yticks(range(len(truncated_labels)))  # Set y-ticks to match actual number of rows
        ax.set_yticklabels(truncated_labels, fontsize=12)

        # Set y-limits to ensure they reflect the number of terms plotted
        ax.set_ylim(-1, len(truncated_labels))  # Use -1 to adjust for zero-based index

        # Add labels on the right side for each category
        y_offset = 0.2  # Vertical offset for placing text labels
        for i, (index, row) in enumerate(combined_df.iterrows()):
            category = row["Ontology"]
            ax.text(
                combined_df["Counts"].max() + 10,
                i - y_offset,
                category,
                color=category_colors[category],
                fontsize=12,
                fontweight="bold",
            )

        # Draw a vertical line to separate the bars and the category labels
        ax.axvline(x=combined_df["Counts"].max() + 9, color="gray", linestyle="--")
        # Add a title for the combined plot
        ax.set_title(
            f"Combined Top {nrows} GO Terms from Each Ontology Category",
            fontsize=14,
            fontweight="bold",
        )

        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"{prefix}_combined_top_dotplot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        # Log the creation of the combined plot
        self.logger.info(
            f"Created combined dot plot for top {nrows} GO terms from each ontology and saved at {os.path.join(outdir, f'{prefix}_combined_top_dotplot.png')}"
        )

        return

    def dotplotTopFromEachOntology(self, df, colorBy="logPvalues", nrows=10, outdir=".", prefix="pyseqrna"):
        """
        Create a dot plot for the top GO terms from each ontology category (BP, MF, CC)
        with gradient colors based on a specified column.

        :param df: DataFrame containing Gene Ontology enrichment data.
        :param colorBy: Column name to use for gradient coloring (default: 'logPvalues').
        :param nrows: Number of rows to plot for each category (default: 10).
        :return: List of matplotlib figures containing the dot plots for each category.
        """
        if self.dryrun:
            self.logger.info(f"DRYRUN: Would create dotplot for each ontology and save to {outdir}")
            return

        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input df must be a pandas DataFrame.")
            raise TypeError("Input df must be a pandas DataFrame.")

        # Validate colorBy column
        if colorBy != "logPvalues" and colorBy not in df.columns:
            self.logger.error(f"colorBy column '{colorBy}' not found in DataFrame.")
            raise ValueError(f"colorBy column '{colorBy}' not found in DataFrame.")

        # Define distinct colors for each ontology category (base colors)

        # Create a list to store the figures for each category
        figures = []

        # Filter and plot for each ontology category
        categories = ["BP", "MF", "CC"]

        for category in categories:
            category_df = self._prepare_plot_data(
                df,
                "GO Term",
                nrows,
                sort_by="Counts",
                color_by=colorBy,
                ontology=category,
            )
            if category_df.empty:
                continue
            category_df = category_df.sort_values(by="Counts", ascending=True)  # Sort to have highest on top

            counts = category_df["Counts"].values
            terms = category_df["GO Term"]

            # Prepare gradient colors based on the specified colorBy column
            gradient_values = category_df[colorBy].values  # Values to create gradient
            norm = plt.Normalize(float(np.nanmin(gradient_values)), float(np.nanmax(gradient_values)))
            colors = plt.cm.tab20c(norm(gradient_values))  # Using tab20c colormap for gradients

            # Determine the number of rows to use for the plot height
            actual_nrows = len(category_df)  # Actual number of rows available
            fsize = max(5, math.ceil(actual_nrows / 2))  # Adjust figure height based on actual nrows

            # Create the plot for the current category
            fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)

            # Create dot plot with gradient colors
            ax.scatter(counts, range(actual_nrows), color=colors, s=100)  # s is the size of the dots
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers

            # Set axis properties
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_bounds((0, actual_nrows))
            ax.spines["left"].set_position(("outward", 8))
            ax.spines["bottom"].set_position(("outward", 5))

            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel("Counts", fontsize=12, fontweight="bold")
            plt.ylabel("GO Description", fontsize=12, fontweight="bold")

            # Set y-limits to ensure they reflect the number of terms
            ax.set_ylim(-1, actual_nrows)  # -1 to adjust for the zero-based index

            # Truncate y-tick labels if they exceed 40 characters
            truncated_labels = self._truncate_labels(terms)
            ax.set_yticks(range(actual_nrows))  # Set y-ticks to match actual number of rows
            ax.set_yticklabels(truncated_labels, fontsize=12)

            # Add a title for each category
            ax.set_title(
                f"Top {min(nrows, len(category_df))} GO Terms for {category} Category",
                fontsize=14,
                fontweight="bold",
            )

            # Create a colorbar to represent the colorBy variable
            sm = plt.cm.ScalarMappable(cmap="tab20c", norm=norm)
            sm.set_array([])  # Only needed for older versions of matplotlib
            cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=0.2, aspect=10)
            cbar.set_label(colorBy, fontsize=12, fontweight="bold")

            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"{prefix}_{category}_dotplot.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Log the creation of the plot for the current category
            self.logger.info(
                f"Created dot plot for {category} category with {actual_nrows} terms saved at {os.path.join(outdir, f'{prefix}_{category}_dotplot.png')}"
            )

            figures.append(fig)

        return


class GeneOntology:
    """
    Gene Ontology enrichment analysis for differentially expressed genes.

    This class provides comprehensive GO enrichment analysis with support for
    both ENSEMBL and NCBI gene identifiers, fetching GO annotations from
    BioMart or custom APIs.

    Attributes:
        species (str): Species identifier (e.g., 'athaliana' for Arabidopsis)
        organism_type (str): Organism type - 'plants' or 'animals'
        key_type (str): Gene ID type - 'ensembl' or 'ncbi'
        taxid (Optional[str]): Taxonomy ID for NCBI gene IDs
        gff (Optional[str]): Path to GFF/GTF file for gene mapping
        logger: Logger instance for tracking operations
        df (pd.DataFrame): GO annotation data
        background_count (int): Total number of genes with GO annotations
    """

    def __init__(
        self,
        species: str,
        organism_type: str = "plants",
        key_type: str = "ensembl",
        taxid: Optional[str] = None,
        gff: Optional[str] = None,
        dryrun: bool = False,
        logger: Optional[Any] = None,
        dry_run_manager: Optional[DryRunManager] = None,
    ):
        """
        Initialize the Gene Ontology enrichment analyzer.

        Args:
            species: Species identifier (e.g., 'athaliana' for Arabidopsis thaliana)
            organism_type: Type of organism - 'plants' or 'animals'
            key_type: Gene ID type - 'ensembl' or 'ncbi'
            taxid: Taxonomy ID (required if key_type is 'ncbi')
            gff: Path to GFF/GTF annotation file (required if key_type is 'ncbi')
            dryrun: Whether to perform a dry run (no actual file operations)
            logger: Logger instance for tracking operations
            dry_run_manager: Dry run manager instance

        Raises:
            GeneOntologyError: If required parameters are missing or invalid
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.logger
        else:
            self.logger = logger

        # Initialize utilities
        self.file_manager = FileManager(logger=self.logger)

        # Initialize dry-run manager
        if dry_run_manager is not None:
            self.dry_run_manager = dry_run_manager
        else:
            self.dry_run_manager = DryRunManager(enabled=dryrun, logger=self.logger)

        # Store dry run flag
        self.dryrun = dryrun

        # Store configuration
        self.species = species
        self.organism_type = organism_type.lower()
        self.key_type = key_type.lower()
        self.taxid = taxid
        self.gff = gff

        # Validate inputs
        self._validate_inputs()

        # Initialize GO data
        self.df: Optional[pd.DataFrame] = None
        self.background_count: int = 0
        self.idmapping: Optional[pd.DataFrame] = None

        # Load GO annotations
        self._load_go_annotations()

        self.logger.info(f"Gene Ontology initialized for {species} with {self.background_count} annotated genes")

    def _validate_inputs(self) -> None:
        """
        Validate input parameters.

        Raises:
            GeneOntologyError: If required parameters are missing or invalid
        """
        if not self.species:
            raise GeneOntologyError("Species identifier is required")

        if self.organism_type not in ["plants", "animals"]:
            raise GeneOntologyError(f"Invalid organism type: {self.organism_type}. Must be 'plants' or 'animals'")

        if self.key_type not in ["ensembl", "ncbi"]:
            raise GeneOntologyError(f"Invalid key type: {self.key_type}. Must be 'ensembl' or 'ncbi'")

        if self.key_type == "ncbi":
            if not self.taxid:
                raise GeneOntologyError("Taxonomy ID is required for NCBI gene IDs")
            if not self.gff:
                raise GeneOntologyError("GFF/GTF file is required for NCBI gene IDs")
            if not self.file_manager.verify_files_exist(self.gff):
                raise GeneOntologyError(f"GFF/GTF file not found: {self.gff}")

    def _load_go_annotations(self) -> None:
        """Load GO annotations from appropriate source based on key_type."""
        try:
            if self.key_type == "ensembl":
                self.logger.info("Fetching Gene Ontology from BioMart")
                go_data = self._query_biomart(self.species, self.organism_type)
                self.logger.info("Processing Gene Ontology data from BioMart")
                self.df, self.background_count = self._preprocess_biomart(go_data)

            elif self.key_type == "ncbi":
                self.logger.info("Fetching Gene Ontology from pyseqrna API")
                self.df, self.background_count = self._parse_go_ncbi(self.taxid)
                self.idmapping = self._parse_gff(self.gff)

        except Exception as e:
            raise GeneOntologyError(f"Failed to load GO annotations: {str(e)}")

    def _get_request(self, url: str, **params) -> requests.Response:
        """
        Make HTTP GET request with error handling.

        Args:
            url: URL to request
            **params: Query parameters

        Returns:
            Response object

        Raises:
            GeneOntologyError: If request fails
        """
        try:
            if params:
                response = requests.get(url, params=params, stream=True, timeout=HTTP_TIMEOUT_SECONDS)
            else:
                response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise GeneOntologyError(f"HTTP request failed: {str(e)}")

    def _add_attribute_node(self, root: ElementTree.Element, attr: str) -> None:
        """
        Add attribute node to XML element for BioMart query.

        Args:
            root: XML root element
            attr: Attribute name to add
        """
        attr_el = ElementTree.SubElement(root, "Attribute")
        attr_el.set("name", attr)

    def get_supported_organisms(self, organism_type: str = "plants") -> pd.DataFrame:
        """
        Get list of supported organisms from Ensembl/BioMart.

        Args:
            organism_type: Type of organism - 'plants' or 'animals'

        Returns:
            DataFrame with organism codes and names

        Raises:
            GeneOntologyError: If organism list cannot be fetched
        """
        try:
            if organism_type == "plants":
                url = "https://plants.ensembl.org/biomart/martservice?type=datasets&requestid=biomaRt&mart=plants_mart"
            else:
                url = "https://www.ensembl.org/biomart/martservice?type=datasets&requestid=biomaRt&mart=ENSEMBL_MART_ENSEMBL"

            resp = urlopen(url)
            handle = TextIOWrapper(resp, encoding="UTF-8")
            handle.url = resp.url

            df = pd.read_csv(
                handle,
                sep="\t",
                names=[
                    "Table",
                    "Code",
                    "Organism",
                    "A",
                    "Assembly",
                    "B",
                    "C",
                    "Default",
                    "Date",
                ],
            )

            return df[["Code", "Organism"]]

        except Exception as e:
            raise GeneOntologyError(f"Failed to fetch organism list: {str(e)}")

    def _query_biomart(self, species: str, organism_type: str) -> pd.DataFrame:
        """
        Query BioMart for GO annotations.

        Args:
            species: Species identifier
            organism_type: Type of organism - 'plants' or 'animals'

        Returns:
            DataFrame with GO annotations

        Raises:
            GeneOntologyError: If query fails
        """
        try:
            # Determine BioMart configuration
            if organism_type == "animals":
                uri = "https://ensembl.org/biomart/martservice"
                scheme = "default"
                fspecies = f"{species}_gene_ensembl"
            else:  # plants
                uri = "https://plants.ensembl.org/biomart/martservice"
                scheme = "plants_mart"
                fspecies = f"{species}_eg_gene"

            # Build XML query
            root = ElementTree.Element("Query")
            root.set("virtualSchemaName", scheme)
            root.set("formatter", "TSV")
            root.set("header", "1")
            root.set("uniqueRows", native_str(int(True)))
            root.set("datasetConfigVersion", "0.6")

            dataset = ElementTree.SubElement(root, "Dataset")
            dataset.set("name", fspecies)
            dataset.set("interface", "default")

            # Add attributes to query
            attributes = [
                "ensembl_gene_id",
                "ensembl_transcript_id",
                "go_id",
                "name_1006",
                "namespace_1003",
                "definition_1006",
            ]
            for attr in attributes:
                self._add_attribute_node(dataset, attr)

            # Execute query
            response = self._get_request(uri, query=ElementTree.tostring(root))
            result = pd.read_csv(StringIO(response.text), sep="\t")
            result.columns = [
                "Gene",
                "Transcript",
                "GO_ID",
                "GO_term",
                "GO_ontology",
                "GO_def",
            ]

            self.logger.debug(f"Retrieved {len(result)} GO annotation records from BioMart")
            return result

        except Exception as e:
            raise GeneOntologyError(f"BioMart query failed: {str(e)}")

    def _preprocess_biomart(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Preprocess BioMart GO annotation data.

        Args:
            data: Raw BioMart data

        Returns:
            Tuple of (processed DataFrame, background gene count)
        """
        # Filter rows with GO annotations
        df = data[data["GO_ID"].notna()].copy()

        # Calculate background gene count
        genes = df["Gene"].unique()
        bg_count = len(genes)

        self.logger.debug(f"Found {bg_count} unique genes with GO annotations")

        # Group genes by GO term
        gene_id_dict = {}
        go_info_dict = {}

        for _, row in df.iterrows():
            go_id = row["GO_ID"]
            gene = row["Gene"]

            # Store gene IDs for each GO term
            if go_id not in gene_id_dict:
                gene_id_dict[go_id] = []
            gene_id_dict[go_id].append(gene)

            # Store GO term information
            if go_id not in go_info_dict:
                # Map ontology names to abbreviations
                ontology = row["GO_ontology"]
                if ontology == "cellular_component":
                    ontology = "CC"
                elif ontology == "molecular_function":
                    ontology = "MF"
                elif ontology == "biological_process":
                    ontology = "BP"

                go_info_dict[go_id] = {
                    "ID": go_id,
                    "Term": row["GO_term"],
                    "Ontology": ontology,
                }

        # Build final DataFrame
        final_data = []
        for go_id, info in go_info_dict.items():
            genes = [str(g).upper() for g in gene_id_dict[go_id] if pd.notna(g)]
            genes = list(set(genes))  # Remove duplicates

            final_data.append([info["ID"], info["Term"], info["Ontology"], genes, len(genes)])

        final_df = pd.DataFrame(final_data, columns=["ID", "Term", "Ontology", "Gene", "Gene_length"])

        self.logger.debug(f"Processed into {len(final_df)} unique GO terms")
        return final_df, bg_count

    def _parse_go_ncbi(self, taxid: str) -> Tuple[pd.DataFrame, int]:
        """
        Parse GO annotations for NCBI gene IDs from API.

        Args:
            taxid: Taxonomy ID

        Returns:
            Tuple of (processed DataFrame, background gene count)

        Raises:
            GeneOntologyError: If API request fails
        """
        try:
            url = f"http://bioinfo.usu.edu/pyseqrnautils/api/go/?taxid={taxid}"
            self.logger.debug(f"Fetching GO data from: {url}")

            response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()

            df = pd.DataFrame(json.loads(response.text))

            # Calculate background gene count
            genes = df["entrez"].unique()
            bg_count = len(genes)

            self.logger.debug(f"Found {bg_count} unique genes with GO annotations")

            # Group genes by GO term
            gene_id_dict = {}
            go_info_dict = {}

            for _, row in df.iterrows():
                go_id = row[3]  # GO ID column
                gene = row[2]  # Gene ID column

                if go_id not in gene_id_dict:
                    gene_id_dict[go_id] = []
                gene_id_dict[go_id].append(gene)

                if go_id not in go_info_dict:
                    go_info_dict[go_id] = {
                        "ID": go_id,
                        "Term": row[5],  # GO term
                        "Ontology": row[7],  # Ontology type
                    }

            # Build final DataFrame
            final_data = []
            for go_id, info in go_info_dict.items():
                genes = gene_id_dict[go_id]
                final_data.append([info["ID"], info["Term"], info["Ontology"], genes, len(genes)])

            final_df = pd.DataFrame(final_data, columns=["ID", "Term", "Ontology", "Gene", "Gene_length"])

            return final_df, bg_count

        except Exception as e:
            raise GeneOntologyError(f"Failed to parse NCBI GO data: {str(e)}")

    def _parse_gff(self, gff_file: str) -> pd.DataFrame:
        """
        Parse GFF/GTF file to extract gene ID mappings.

        Args:
            gff_file: Path to GFF/GTF file

        Returns:
            DataFrame with gene ID mappings

        Raises:
            GeneOntologyError: If GFF parsing fails
        """
        try:
            self.logger.info(f"Parsing GFF file: {gff_file}")

            # Parse GFF file to extract gene and entrez ID mappings
            gene_mappings = []

            with open(gff_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) < 9:
                        continue

                    attributes = parts[8]

                    # Extract gene ID and entrez ID from attributes
                    gene_id = None
                    entrez_id = None

                    for attr in attributes.split(";"):
                        attr = attr.strip()
                        if attr.startswith("gene_id"):
                            gene_id = attr.split("=")[1].strip('"')
                        elif attr.startswith("db_xref"):
                            if "GeneID:" in attr:
                                entrez_id = attr.split("GeneID:")[1].split(",")[0].split('"')[0]

                    if gene_id and entrez_id:
                        gene_mappings.append({"Gene": gene_id, "entrez": entrez_id})

            df = pd.DataFrame(gene_mappings).drop_duplicates()
            self.logger.debug(f"Extracted {len(df)} gene ID mappings from GFF")

            return df

        except Exception as e:
            raise GeneOntologyError(f"Failed to parse GFF file: {str(e)}")

    def _calculate_fdr(self, pvalues: List[float]) -> List[float]:
        """
        Calculate False Discovery Rate using Benjamini-Hochberg method.

        Args:
            pvalues: List of p-values

        Returns:
            List of adjusted p-values (FDR)
        """
        if len(pvalues) == 0:
            return []

        p_vals = pd.Series(pvalues)
        ranked_p_values = rankdata(p_vals)
        fdr = p_vals * len(p_vals) / ranked_p_values
        fdr[fdr > 1] = 1

        return fdr.tolist()

    def _read_gene_list(self, file: str) -> pd.DataFrame:
        """Read DEG gene-list files with or without a Gene header."""
        try:
            table = pd.read_csv(file, comment="#")
        except pd.errors.EmptyDataError:
            raise GeneOntologyError(f"Gene list file is empty: {file}")

        if "Gene" not in table.columns:
            table = pd.read_csv(file, comment="#", header=None, names=["Gene"])

        table = table[["Gene"]].copy()
        table["Gene"] = table["Gene"].astype(str).str.strip()
        table = table[(table["Gene"] != "") & (table["Gene"].str.lower() != "nan")]
        table = table.drop_duplicates()

        if table.empty:
            raise GeneOntologyError(f"No valid gene IDs found in {file}")

        return table

    def enrichGO(
        self,
        file=None,
        pvalueCutoff=0.05,
        plot=True,
        plotType="all",
        nrows=20,
        outdir=".",
        colorBy="logPvalues",
    ):
        """
        Perform Gene Ontology enrichment of DEGs.

        :param file: Differentially expressed genes in a sample.
        :param pvalueCutoff: P-value cutoff for enrichment. Default is 0.05.
        :param plot: True if a plot is needed. Default is True.
        :param plotType: Gene Ontology enrichment visualization on dotplot/barplot. Default is dotplot.
        :param nrows: Number of rows to plot. Default to 20 rows.
        :param colorBy: Color dot on plots with logPvalues / FDR. Defaults to 'logPvalues'.
        :returns: A dictionary with Gene Ontology enrichment results and optional plots.
        """

        self.logger.info(f"Performing GO enrichment analysis on {file}")

        # Get file basename and create sample-specific output directory
        sample = os.path.basename(file).split(".")[0]
        sample_outdir = os.path.join(outdir, sample)
        self.file_manager.create_directory(sample_outdir)

        df_goList = self.df[["ID", "Gene"]].values.tolist()
        count = self.df[["ID", "Gene_length"]].values.tolist()
        df_List = self.df[["ID", "Term", "Ontology"]].values.tolist()

        go_dict = {}
        for value in df_goList:
            go_dict[value[0]] = str(value[1]).upper()

        go_count = {}
        for c in count:
            go_count[c[0]] = c[1]

        KOdescription = {}
        for line in df_List:
            KOdescription[line[0]] = [line[1], line[2]]

        get_gene_ids_from_user = dict()
        gene_GO_count = dict()
        get_user_id_count_for_GO = dict()
        user_provided_uniq_ids = dict()
        user_genecount = []

        for item in go_dict:
            get_gene_ids_from_user[item] = []
            gene_GO_count[item] = go_count[item]
            get_user_id_count_for_GO[item] = 0

        bg_gene_count = self.background_count

        if self.key_type == "ncbi":
            ufile = self._read_gene_list(file)
            id_intermediate = ufile.merge(self.idmapping, on="Gene").drop_duplicates()
            read_id_file = id_intermediate["entrez"].values.tolist()
            for gene_id in read_id_file:
                gene_id = str(gene_id).strip().upper()
                user_provided_uniq_ids[gene_id] = 0

        if self.key_type == "ensembl":
            ufile = self._read_gene_list(file)
            for gene_id in ufile["Gene"]:
                gene_id = str(gene_id).strip().upper()
                user_provided_uniq_ids[gene_id] = 0

        anot_count = 0
        for k1 in go_dict:
            for k2 in user_provided_uniq_ids:
                if k2 in go_dict[k1]:
                    get_gene_ids_from_user[k1].append(k2)
                    get_user_id_count_for_GO[k1] += 1
                    anot_count += 1
                    if k2 not in user_genecount:
                        user_genecount.append(k2)

        pvalues = []
        enrichment_result = []
        mapped_query_ids = len(user_genecount)

        for k in get_user_id_count_for_GO:
            gene_in_category = get_user_id_count_for_GO[k]
            mapped_query_ids - gene_in_category
            gene_GO_count[k] - gene_in_category
            gene_ids = get_gene_ids_from_user[k]
            gID = ""

            for g in gene_ids:
                gID += g + ","

            gID = gID.rsplit(",", 1)[0]
            pvalue = stats.hypergeom.sf(gene_in_category - 1, bg_gene_count, gene_GO_count[k], mapped_query_ids)

            if gene_in_category > 0:
                enrichment_result.append(
                    [
                        k,
                        KOdescription[k][0],
                        KOdescription[k][1],
                        f"{gene_in_category}/{mapped_query_ids}",
                        f"{go_count[k]}/{bg_gene_count}",
                        pvalue,
                        len(gene_ids),
                        gID,
                    ]
                )

        end = pd.DataFrame(enrichment_result)

        if end.shape[0] > 1:
            end.columns = [
                "GO ID",
                "GO Term",
                "Ontology",
                "GeneRatio",
                "BgRatio",
                "Pvalues",
                "Counts",
                "Genes",
            ]
            end = end[end["Pvalues"] <= pvalueCutoff]
            if end.empty:
                self.logger.warning("No Gene Ontology results passed the p-value cutoff.")
                return "No Gene Ontology results."
            pvalues = end["Pvalues"].values.tolist()
            fdr = list(self._calculate_fdr(pvalues))
            end.insert(7, "FDR", fdr)

            if self.key_type == "ncbi":
                results = end  # Would need change_ids function
            elif self.key_type == "ensembl":
                results = end

            nrows = min(nrows, results.shape[0])

            if plot:
                plotter = GeneOntologyPlotter(logger=self.logger, dryrun=self.dryrun)
                try:
                    if plotType == "dotplot":
                        plotter.dotplotGO(results, nrows, colorBy, outdir=sample_outdir, prefix=sample)
                        plotter.dotplotCombinedTop(results, nrows, outdir=sample_outdir, prefix=sample)
                        plotter.dotplotTopFromEachOntology(results, colorBy, nrows, outdir=sample_outdir, prefix=sample)

                    elif plotType == "barplot":
                        plotter.barplotGO(results, nrows, colorBy, outdir=sample_outdir, prefix=sample)
                        plotter.barplotCombinedTop(results, nrows, outdir=sample_outdir, prefix=sample)
                        plotter.barplotTopFromEachOntology(results, colorBy, nrows, outdir=sample_outdir, prefix=sample)

                    elif plotType == "all":
                        plotter.barplotGO(results, nrows, colorBy, outdir=sample_outdir, prefix=sample)
                        plotter.barplotCombinedTop(results, nrows, outdir=sample_outdir, prefix=sample)
                        plotter.barplotTopFromEachOntology(results, colorBy, nrows, outdir=sample_outdir, prefix=sample)

                        plotter.dotplotGO(results, nrows, colorBy, outdir=sample_outdir, prefix=sample)
                        plotter.dotplotCombinedTop(results, nrows, colorBy, outdir=sample_outdir, prefix=sample)
                        plotter.dotplotTopFromEachOntology(results, colorBy, nrows, outdir=sample_outdir, prefix=sample)
                except Exception as plot_error:
                    self.logger.warning(
                        f"GO plotting failed for {sample}; enrichment tables will still be saved: {plot_error}"
                    )

            self.logger.info("GO enrichment analysis completed successfully.")

            # Save results to CSV file with separate sheets for each ontology
            if self.dryrun:
                self.logger.info(f"DRYRUN: Would save GO enrichment results to {sample_outdir}")
                self.dry_run_manager.record_command(
                    "gene_ontology_save",
                    f"Save GO enrichment results for {sample}",
                    f"results.to_csv('{os.path.join(sample_outdir, f'{sample}_GO_all.csv')}', index=False)",
                )
            else:
                results_file = os.path.join(sample_outdir, f"{sample}_GO_all.csv")
                results.to_csv(results_file, index=False)
                self.logger.info(f"GO enrichment results saved to {results_file}")

                # Save separate files for each ontology category
                categories = ["BP", "MF", "CC"]
                for category in categories:
                    category_df = results[results["Ontology"] == category]
                    if not category_df.empty:
                        category_file = os.path.join(sample_outdir, f"{sample}_GO_{category}.csv")
                        category_df.to_csv(category_file, index=False)
                        self.logger.info(f"GO {category} results saved to {category_file}")

            return

        self.logger.warning("No Gene Ontology results found.")
        return "No Gene Ontology results."
