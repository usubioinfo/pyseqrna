#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KEGG Pathway Enrichment Module

This module provides KEGG pathway enrichment analysis for differentially
expressed genes with support for both ENSEMBL and NCBI gene identifiers.

Features:
    - KEGG Pathway enrichment analysis for differentially expressed genes (DEGs)
    - Support for both ENSEMBL and NCBI gene identifiers
    - Parsing of GFF/GTF annotation files for NCBI-to-GeneID mapping
    - Dynamic querying of the KEGG REST API
    - Automatic FDR correction of hyper-geometric p-values
    - Visualization of enriched pathways via barplots and dotplots

Configuration:
    The module can be configured with:
    - Species identifier (e.g., 'ath')
    - Organism type ('plants' or 'animals')
    - Key type ('ensembl' or 'ncbi')
    - Path to GFF file for custom mapping

Dependencies:
    - numpy
    - pandas
    - scipy
    - matplotlib
    - requests

Classes:
    PathwayPlotter - Plotting utilities for KEGG Pathway enrichment analysis.
    Pathway - This class is for KEGG Pathway enrichment analysis.

Exceptions:
    PathwayError - Custom exception for pathway-related errors.

:Created: January 10, 2022
:Updated: March 10, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import io
import os
import re
import math
from typing import List, Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.stats import rankdata

# Import utility modules
from ...utils import LogManager, FileManager
from ...utils.dry_run_manager import DryRunManager

HTTP_TIMEOUT_SECONDS = 60


class PathwayError(Exception):
    """Custom exception for pathway-related errors."""

    pass


class PathwayPlotter:
    """
    Plotting utilities for KEGG Pathway enrichment analysis.

    This class provides visualization methods for pathway enrichment results
    including dotplots and barplots.

    Attributes:
        logger: Logger instance for tracking operations
        dryrun: Whether to perform dry run (skip actual plot generation)
    """

    def __init__(self, logger, dryrun=False):
        """
        Initialize the Pathway plotter.

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
        color_by: str = "logPvalues",
        size_by: Optional[str] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return a plotting-safe copy with valid labels and numeric columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input df must be a pandas DataFrame.")

        plot_df = df.copy()
        required = {
            term_col,
            sort_by,
            "Pvalues" if color_by == "logPvalues" else color_by,
        }
        if size_by:
            required.add(size_by)
        missing = [col for col in required if col not in plot_df.columns]
        if missing:
            raise ValueError(f"Missing required KEGG plot column(s): {', '.join(missing)}")

        numeric_cols = {"Counts", "Pvalues", "FDR", sort_by}
        if color_by != "logPvalues":
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

        drop_cols = [sort_by, color_by]
        if size_by:
            drop_cols.append(size_by)
        plot_df = plot_df.dropna(subset=list(dict.fromkeys(drop_cols)))
        plot_df = plot_df[plot_df[sort_by] > 0]

        if plot_df.empty:
            self.logger.warning("No valid KEGG rows available for plotting.")
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

    def dotplotKEGG(
        self,
        df=None,
        nrows=20,
        colorBy="logPvalues",
        sizeBy="Counts",
        outdir=".",
        prefix="pyseqrna",
    ):
        """
        Create a dotplot for KEGG enrichment.

        :param df: DataFrame containing KEGG enrichment data.
        :param nrows: Number of rows to plot (default: 20).
        :param colorBy: Variable to color the dots ('logPvalues' or 'FDR').
        :param sizeBy: Variable to scale the dot sizes ('Counts').
        :param outdir: Directory to save the plot.
        :param prefix: Prefix for the saved plot file name.
        :return: None
        """
        if self.dryrun:
            self.logger.info(f"DRYRUN: Would create dotplot KEGG and save to {os.path.join(outdir, f'{prefix}_dotplot.png')}")
            return

        self.logger.info("Creating dotplot KEGG.")

        # Validate input
        if colorBy not in ["logPvalues", "FDR"]:
            self.logger.error("Invalid value for colorBy: %s", colorBy)
            raise ValueError("Invalid value for colorBy. Use 'logPvalues' or 'FDR'.")

        df = self._prepare_plot_data(df, "Description", nrows, sort_by=sizeBy, color_by=colorBy, size_by=sizeBy)
        if df.empty:
            return
        counts = df[sizeBy].values
        terms = df["Description"]
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
        # plt.ylabel("KEGG Description", fontsize=12, fontweight='bold')

        # Create color bar
        sm = ScalarMappable(cmap="tab20", norm=plt.Normalize(0, max(float(df[colorBy].max()), 1e-12)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=0.02, aspect=10)
        cbar.ax.set_title(colorBy, pad=20, fontweight="bold")

        # Set y-tick labels with wrapping
        truncated_labels = self._truncate_labels(terms)
        ax.set_yticks(range(len(truncated_labels)))
        ax.set_yticklabels(truncated_labels, fontsize=12)

        # Set y-limits to ensure they reflect the number of terms plotted
        ax.set_ylim(-1, len(truncated_labels))  # Use -1 to adjust for zero-based index

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}_dotplot.png"), dpi=300, bbox_inches="tight")

        plt.close()
        self.logger.info(f"Dotplot KEGG created successfully and saved at {os.path.join(outdir, f'{prefix}_dotplot.png')}")

        return

    def barplotKEGG(self, df=None, nrows=20, colorBy="logPvalues", outdir=".", prefix="pyseqrna"):
        """
        Create a barplot for KEGG pathway enrichment.

        :param df: DataFrame containing KEGG enrichment data.
        :param nrows: Number of rows to plot (default: 20).
        :param colorBy: Variable to color the bars ('logPvalues' or 'FDR').
        :param outdir: Directory to save the plot.
        :param prefix: Prefix for the saved plot file name.
        :return: A matplotlib figure containing the barplot.
        """
        if self.dryrun:
            self.logger.info(f"DRYRUN: Would create barplot KEGG and save to {os.path.join(outdir, f'{prefix}_barplot.png')}")
            return

        self.logger.info("Creating barplot KEGG.")

        # Validate input
        if colorBy not in ["logPvalues", "FDR"]:
            self.logger.error("Invalid value for colorBy: %s", colorBy)
            raise ValueError("Invalid value for colorBy. Use 'logPvalues' or 'FDR'.")

        df = self._prepare_plot_data(df, "Description", nrows, sort_by="Counts", color_by=colorBy)
        if df.empty:
            return
        df = df.sort_values("Counts", ascending=True)
        counts = df["Counts"].values
        terms = df["Description"]
        actual_nrows = len(df)

        # Normalize colors and sizes for the dot plot
        data_color_normalized = self._normalise(df[colorBy])
        fsize = max(5, math.ceil(actual_nrows / 2))  # Ensure minimum figure height

        fig, ax = plt.subplots(figsize=(10, fsize), dpi=300)
        colors = plt.cm.tab20(data_color_normalized)

        ax.barh(range(actual_nrows), counts, color=colors)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))
        ax.margins(y=0)  # Remove space between y-axis and bars

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Counts", fontsize=12, fontweight="bold")
        plt.ylabel("KEGG Description", fontsize=12, fontweight="bold")
        # plt.title(f'KEGG Pathway Barplot: {title}', fontsize=16)

        # Create color bar
        sm = ScalarMappable(cmap="tab20", norm=plt.Normalize(0, df[colorBy].max()))
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
        plt.close()
        self.logger.info(f"Barplot KEGG created successfully and saved at {os.path.join(outdir, f'{prefix}_barplot.png')}")

        return


class Pathway:
    """
    This class is for KEGG Pathway enrichment analysis.

    :param species: Species name (e.g., 'athaliana' for Arabidopsis thaliana).
    :param type: Species is from plants or animals.
    :param keyType: Gene source, either 'NCBI' or 'ENSEMBL'. Default is 'ENSEMBL'.
    :param gff: Gene feature file (if keyType is 'NCBI').
    :param logger: Logger instance for logging information.
    """

    def __init__(
        self,
        species=None,
        type=None,
        keyType="ensembl",
        gff=None,
        dryrun=False,
        logger=None,
        dry_run_manager=None,
    ):
        """
        Initialize the KEGG Pathway enrichment analyzer.

        Args:
            species: Species identifier (e.g., 'ath' for Arabidopsis thaliana)
            type: Organism type - 'plants' or 'animals'
            keyType: Gene ID type - 'ensembl' or 'ncbi'
            gff: Path to GFF/GTF annotation file (required if keyType is 'ncbi')
            dryrun: Whether to perform a dry run (no actual file operations)
            logger: Logger instance for tracking operations
            dry_run_manager: Dry run manager instance
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
        self.type = type
        self.keyType = keyType.lower()
        self.gff = gff

        # Initialize pathway data
        self.df: Optional[pd.DataFrame] = None
        self.background_count: int = 0
        self.idmapping: Optional[pd.DataFrame] = None

        # Load pathway annotations
        self._initialize_pathway_data()

    def _initialize_pathway_data(self):
        """Load pathway annotations from appropriate source based on keyType."""
        if self.keyType == "ncbi":
            self.logger.info("Fetching pathways from KEGG API")
            self.df, self.background_count = self._kegg_list()
            if self.gff:
                self.idmapping = self._parse_gff(self.gff)
        elif self.keyType == "ensembl":
            self.logger.info("Fetching pathways from pyseqrna API")
            self.df, self.background_count = self._get_pathways()
        else:
            self.logger.error("Unsupported keyType: %s", self.keyType)
            raise PathwayError("Unsupported keyType. Use 'ncbi' or 'ensembl'.")

    def _q(self, op, arg1, arg2=None, arg3=None):
        """Query KEGG REST API."""
        URL = f"https://rest.kegg.jp/{op}/{arg1}"
        if arg2:
            URL += f"/{arg2}"
            if arg3:
                URL += f"/{arg3}"
        self.logger.debug(f"Querying KEGG API: {URL}")
        resp = urlopen(URL)
        return io.TextIOWrapper(resp, encoding="UTF-8")

    def _kegg_list(self):
        """Fetch pathway list from KEGG API for NCBI gene IDs."""
        self.logger.debug(f"Retrieving pathways for species: {self.species}")
        resp = self._q("list", "pathway", self.species)
        pathways = {}
        bg_genes = []

        for r in resp:
            pathway_id, name = r.split("\t")
            pathway_id = pathway_id.split(":")[1]
            genes, desc = self._get_pathway_genes(pathway_id)
            bg_genes.extend(genes)
            pathways[pathway_id] = [pathway_id, desc, genes, len(genes)]

        df = pd.DataFrame.from_dict(pathways, orient="index", columns=["ID", "Term", "Gene", "Gene_length"])
        bg_count = len(np.unique(bg_genes))

        self.logger.info(f"Retrieved {len(pathways)} pathways.")
        return df, bg_count

    def _get_pathway_genes(self, pathway_id):
        """Get genes for a specific pathway."""
        resp = self._q("get", pathway_id)
        genes = []
        desc = ""
        parse = None

        for line in resp:
            line = line.strip()
            if not line.startswith("/"):
                if not line.startswith(" "):
                    first_word = line.split(" ")[0]
                    if first_word.isupper() and first_word.isalpha():
                        parse = first_word

                    if parse == "NAME":
                        nad = line.replace(parse, "").strip()
                        desc = nad.split(" - ")[0]

                    if parse == "GENE":
                        gened = line.replace(parse, "").strip().split(" ")[0]
                        genes.append(gened)

        return genes, desc

    def _get_pathways(self):
        """Fetch pathway list from pyseqrna API for ENSEMBL gene IDs."""
        try:
            self.logger.debug(f"Fetching pathways from pyseqrna API for species: {self.species}")
            r = requests.get(
                f"https://bioinfo.usu.edu/api-pyseqrna/list/pathways/{self.species}",
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            r.raise_for_status()
            m = re.sub("<[^<]+?>", "", r.text)
            df = pd.read_csv(io.StringIO(m), sep="\t", names=["Species", "Gene", "ID", "Term"])

            # Group genes by pathway
            pathway_genes = {}
            pathway_terms = {}
            all_genes = []

            for _, row in df.iterrows():
                pathway_id = row["ID"]
                gene = str(row["Gene"]).upper()
                term = row["Term"]

                if pathway_id not in pathway_genes:
                    pathway_genes[pathway_id] = []
                    pathway_terms[pathway_id] = term

                pathway_genes[pathway_id].append(gene)
                all_genes.append(gene)

            # Build pathway DataFrame
            pathway_data = []
            for pathway_id, genes in pathway_genes.items():
                pathway_data.append([pathway_id, pathway_terms[pathway_id], genes, len(genes)])

            pathway_df = pd.DataFrame(pathway_data, columns=["ID", "Term", "Gene", "Gene_length"])

            bg_count = len(np.unique(all_genes))

            self.logger.info(f"Retrieved {len(pathway_df)} pathways for {bg_count} genes from pyseqrna API.")
            return pathway_df, bg_count

        except Exception as e:
            self.logger.error(f"Failed to fetch organism data: {e}")
            raise PathwayError(f"Failed to fetch pathways: {str(e)}")

    def _parse_gff(self, gff_file: str) -> pd.DataFrame:
        """Parse GFF/GTF file to extract gene ID mappings."""
        try:
            self.logger.info(f"Parsing GFF file: {gff_file}")

            gene_mappings = []

            with open(gff_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) < 9:
                        continue

                    attributes = parts[8]

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
            raise PathwayError(f"Failed to parse GFF file: {str(e)}")

    def _fdr_calc(self, x):
        """
        Calculate the False Discovery Rate (FDR) from a list or numpy array of p-values.

        :param x: List or numpy array of p-values.
        :return: A pandas Series containing FDR-adjusted p-values.
        """
        self.logger.info("Starting FDR calculation.")

        if len(x) == 0:
            self.logger.warning("Input p-values list is empty; returning empty Series.")
            return pd.Series(dtype=float)

        p_vals = pd.Series(x)
        ranked_p_values = rankdata(p_vals)
        fdr = p_vals * len(p_vals) / ranked_p_values
        fdr = fdr.clip(upper=1)

        self.logger.info("FDR calculation completed.")
        return fdr

    def enrichKEGG(
        self,
        file,
        pvalueCutoff=0.05,
        plot=True,
        plotType="all",
        nrows=20,
        outdir=".",
        colorBy="logPvalues",
    ):
        """
        Perform KEGG pathway enrichment analysis of differentially expressed genes (DEGs).

        :param file: Path to the file containing differentially expressed genes.
        :param pvalueCutoff: P-value cutoff for enrichment. Default is 0.05.
        :param plot: Whether to generate a plot. Default is True.
        :param plotType: Type of plot for KEGG pathway enrichment ('dotplot', 'barplot', or 'all'). Default is 'all'.
        :param nrows: Number of rows to display in the plot. Default is 20.
        :param outdir: Output directory to save results and plots.
        :param colorBy: Variable to color the plot ('logPvalues' or 'FDR'). Default is 'logPvalues'.
        :returns: A dictionary containing KEGG pathway enrichment results and the plot, if requested.
        """
        self.logger.info(f"Performing KEGG enrichment analysis on {file}")

        user_df = self._read_gene_list(file)
        kegg_list = self.df.set_index("ID")["Gene"].to_dict()
        kegg_count = self.df.set_index("ID")["Gene_length"].to_dict()
        kegg_description = self.df.set_index("ID")["Term"].to_dict()

        read_id_file = self._get_user_gene_ids(user_df)
        user_genecount = set()
        user_gene_ids = {k: [] for k in kegg_list.keys()}
        userID_count_kegg = {k: 0 for k in kegg_list.keys()}

        for kegg_id in kegg_list:
            for gene_id in read_id_file:
                if gene_id in kegg_list[kegg_id]:
                    user_gene_ids[kegg_id].append(gene_id)
                    userID_count_kegg[kegg_id] += 1
                    user_genecount.add(gene_id)

        return self._calculate_enrichment(
            file,
            outdir,
            userID_count_kegg,
            user_gene_ids,
            user_genecount,
            kegg_count,
            kegg_description,
            pvalueCutoff,
            nrows,
            plot,
            plotType,
            colorBy,
        )

    def _get_user_gene_ids(self, user_df):
        """Retrieve unique gene IDs from the user input based on keyType."""
        if self.keyType == "ncbi":
            id_intermediate = user_df.merge(self.idmapping, on="Gene").drop_duplicates()
            return id_intermediate["entrez"].str.upper().unique()
        elif self.keyType == "ensembl":
            return user_df["Gene"].str.upper().unique()
        else:
            self.logger.error("Unsupported key type: %s", self.keyType)
            raise PathwayError("Unsupported key type. Use 'ncbi' or 'ensembl'.")

    def _read_gene_list(self, file: str) -> pd.DataFrame:
        """Read DEG gene-list files with or without a Gene header."""
        try:
            table = pd.read_csv(file, comment="#")
        except pd.errors.EmptyDataError:
            raise PathwayError(f"Gene list file is empty: {file}")

        if "Gene" not in table.columns:
            table = pd.read_csv(file, comment="#", header=None, names=["Gene"])

        table = table[["Gene"]].copy()
        table["Gene"] = table["Gene"].astype(str).str.strip()
        table = table[(table["Gene"] != "") & (table["Gene"].str.lower() != "nan")]
        table = table.drop_duplicates()

        if table.empty:
            raise PathwayError(f"No valid gene IDs found in {file}")

        return table

    def _calculate_enrichment(
        self,
        file,
        outdir,
        userID_count_kegg,
        user_gene_ids,
        user_genecount,
        kegg_count,
        kegg_description,
        pvalueCutoff,
        nrows,
        plot,
        plotType,
        colorBy,
    ):
        """Calculate the enrichment results."""
        bg_gene_count = self.background_count
        mapped_user_ids = len(user_genecount)

        enrichment_results = []
        for kegg_id in userID_count_kegg:
            gene_in_pathway = userID_count_kegg[kegg_id]
            total_genes_in_pathway = kegg_count[kegg_id]
            pvalue = stats.hypergeom.sf(
                gene_in_pathway - 1,
                bg_gene_count,
                total_genes_in_pathway,
                mapped_user_ids,
            )

            if gene_in_pathway > 0:
                enrichment_results.append(
                    [
                        kegg_id,
                        kegg_description[kegg_id],
                        f"{gene_in_pathway}/{mapped_user_ids}",
                        f"{total_genes_in_pathway}/{bg_gene_count}",
                        pvalue,
                        len(user_gene_ids[kegg_id]),
                        ",".join(user_gene_ids[kegg_id]),
                    ]
                )

        results_df = pd.DataFrame(
            enrichment_results,
            columns=[
                "Pathway_ID",
                "Description",
                "GeneRatio",
                "BgRatio",
                "Pvalues",
                "Counts",
                "Genes",
            ],
        )
        results_df = results_df[results_df["Pvalues"] <= pvalueCutoff]

        sample = os.path.splitext(os.path.basename(file))[0]

        if not results_df.empty:
            fdr_values = self._fdr_calc(results_df["Pvalues"].values)
            results_df.insert(5, "FDR", fdr_values)
            nrows = min(nrows, results_df.shape[0])

            if plot:
                plotter = PathwayPlotter(logger=self.logger, dryrun=self.dryrun)
                try:
                    if plotType == "dotplot":
                        plotter.dotplotKEGG(results_df, nrows, colorBy, outdir=outdir, prefix=sample)
                    elif plotType == "barplot":
                        plotter.barplotKEGG(results_df, nrows, colorBy, outdir=outdir, prefix=sample)
                    elif plotType == "all":
                        plotter.dotplotKEGG(results_df, nrows, colorBy, outdir=outdir, prefix=sample)
                        plotter.barplotKEGG(results_df, nrows, colorBy, outdir=outdir, prefix=sample)
                except Exception as plot_error:
                    self.logger.warning(
                        f"KEGG plotting failed for {sample}; enrichment table will still be saved: {plot_error}"
                    )

            # Save results
            if self.dryrun:
                self.logger.info(f"DRYRUN: Would save KEGG enrichment results to {outdir}")
                self.dry_run_manager.record_command(
                    "kegg_pathway_save",
                    f"Save KEGG enrichment results for {sample}",
                    f"results_df.to_csv('{os.path.join(outdir, f'{sample}_KEGG.csv')}', index=False)",
                )
            else:
                results_df.to_csv(os.path.join(outdir, f"{sample}_KEGG.csv"), index=False)
                self.logger.info(f"KEGG enrichment results saved to {os.path.join(outdir, f'{sample}_KEGG.csv')}")

            return {"result": results_df}

        self.logger.warning("No pathways found with the specified p-value cutoff.")
        return "No Pathways results."
