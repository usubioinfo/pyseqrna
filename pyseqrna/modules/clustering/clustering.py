#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Expression Clustering and Co-expression Analyzer Module

This module provides the user-facing tools for clustering expression matrices. It features a
hierarchical and KMeans clustering implementation (ClusteringAnalyzer) that generates dendrograms,
cluster heatmaps, and membership tables, as well as a wrapper (ClustRunner) for running the
external clust command-line co-expression analysis tool.

Features:
    - Robust matrix preprocessing (top-variable selection, log-transform, mean-filtering, row/column scaling)
    - Support for both hierarchical and KMeans clustering on genes, samples, or both
    - Automated creation of high-quality Seaborn expression heatmaps with clustering dendrograms
    - Automated creation of standalone hierarchical cluster tree dendrogram plots
    - Wrapper for executing the external python package clust command-line tool with options
    - Output of sample/gene cluster assignment TSV lists

Configuration:
    Configured via parameters passed to the constructors and method arguments (such as
    matrix_file, out_dir, gene_column, cluster_target, method, linkage_method, color_map).

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scipy (cluster.hierarchy, spatial.distance)
    - scikit-learn (KMeans, StandardScaler)
    - External command-line tool: clust (optional, managed by ClustRunner)

Classes / Functions / Exceptions:
    - ClusteringError: Custom exception for clustering errors.
    - ClusteringAnalyzer: Cluster genes and/or samples from an expression matrix.
    - ClustRunner: Run the external Clust co-expression clustering tool.

:Created: May 20, 2021
:Updated: April 5, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from typing import Any, Dict, Optional
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusteringError(Exception):
    """Custom exception for clustering errors."""


class ClusteringAnalyzer:
    """Cluster genes and/or samples from an expression matrix."""

    def __init__(
        self,
        matrix_file: str,
        out_dir: str = ".",
        gene_column: str = "Gene",
        logger: Optional[Any] = None,
        dryrun: bool = False,
        dry_run_manager: Optional[Any] = None,
    ):
        self.matrix_file = matrix_file
        self.out_dir = Path(out_dir)
        self.gene_column = gene_column
        self.logger = logger
        self.dryrun = dryrun
        self.dry_run_manager = dry_run_manager

        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, level: str, message: str, *args) -> None:
        """Log when logger exists, otherwise stay quiet."""
        if self.logger:
            getattr(self.logger, level)(message, *args)

    def load_matrix(self) -> pd.DataFrame:
        """Load expression matrix from Excel/CSV/TSV/TXT."""
        path = Path(self.matrix_file)
        if not path.exists():
            raise ClusteringError(f"Expression matrix not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".tsv", ".txt"}:
            df = pd.read_csv(path, sep="\t")
        else:
            raise ClusteringError(f"Unsupported expression matrix format: {suffix}")

        if self.gene_column not in df.columns:
            raise ClusteringError(f"Gene column '{self.gene_column}' not found in expression matrix")

        self._log(
            "info",
            "Loaded expression matrix: %d genes, %d samples",
            df.shape[0],
            df.shape[1] - 1,
        )
        return df

    def prepare_matrix(
        self,
        df: pd.DataFrame,
        top_variable: Optional[int] = None,
        min_mean: float = 0.0,
        log_transform: bool = True,
        scale: str = "row",
    ) -> pd.DataFrame:
        """Prepare numeric expression matrix for clustering."""
        expression = df.set_index(self.gene_column)
        expression = expression.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        expression = expression.fillna(0.0)

        if min_mean > 0:
            expression = expression[expression.mean(axis=1) >= min_mean]

        if log_transform:
            expression = np.log2(expression + 1.0)

        if top_variable and top_variable > 0 and top_variable < len(expression):
            variances = expression.var(axis=1).sort_values(ascending=False)
            expression = expression.loc[variances.head(top_variable).index]

        if expression.empty:
            raise ClusteringError("No genes left after filtering")

        if scale == "row":
            row_std = expression.std(axis=1).replace(0, np.nan)
            expression = expression.sub(expression.mean(axis=1), axis=0).div(row_std, axis=0).fillna(0)
        elif scale == "column":
            scaled = StandardScaler().fit_transform(expression.values)
            expression = pd.DataFrame(scaled, index=expression.index, columns=expression.columns)
        elif scale != "none":
            raise ClusteringError(f"Unsupported scale mode: {scale}")

        self._log(
            "info",
            "Prepared clustering matrix: %d genes, %d samples",
            expression.shape[0],
            expression.shape[1],
        )
        return expression

    def run(
        self,
        cluster_target: str = "both",
        method: str = "hierarchical",
        n_clusters: int = 6,
        metric: str = "euclidean",
        linkage_method: str = "average",
        top_variable: Optional[int] = 1000,
        min_mean: float = 0.0,
        log_transform: bool = True,
        scale: str = "row",
        heatmap: bool = True,
        color_map: str = "vlag",
        prefix: str = "clustering",
    ) -> Dict[str, Any]:
        """Run clustering and write assignment files/heatmap."""
        if self.dryrun:
            outputs = self._expected_outputs(cluster_target, heatmap, prefix)
            for output in outputs.values():
                self._log("info", "DRYRUN: Would create clustering output: %s", output)
            return {"outputs": outputs}

        df = self.load_matrix()
        matrix = self.prepare_matrix(
            df,
            top_variable=top_variable,
            min_mean=min_mean,
            log_transform=log_transform,
            scale=scale,
        )

        outputs: Dict[str, str] = {}
        if cluster_target in {"genes", "both"}:
            outputs["gene_clusters"] = self._cluster_axis(
                matrix,
                axis="genes",
                method=method,
                n_clusters=n_clusters,
                metric=metric,
                linkage_method=linkage_method,
                prefix=prefix,
            )
        if cluster_target in {"samples", "both"}:
            outputs["sample_clusters"] = self._cluster_axis(
                matrix,
                axis="samples",
                method=method,
                n_clusters=min(n_clusters, matrix.shape[1]),
                metric=metric,
                linkage_method=linkage_method,
                prefix=prefix,
            )
        if heatmap:
            outputs["heatmap"] = self._plot_heatmap(
                matrix,
                row_cluster=cluster_target in {"genes", "both"},
                col_cluster=cluster_target in {"samples", "both"},
                color_map=color_map,
                prefix=prefix,
            )

        return {
            "matrix_shape": matrix.shape,
            "outputs": outputs,
        }

    def _cluster_axis(
        self,
        matrix: pd.DataFrame,
        axis: str,
        method: str,
        n_clusters: int,
        metric: str,
        linkage_method: str,
        prefix: str,
    ) -> str:
        """Cluster rows or columns and save assignments."""
        if axis == "genes":
            data = matrix.values
            labels = matrix.index.astype(str).tolist()
            label_column = "Gene"
        else:
            data = matrix.T.values
            labels = matrix.columns.astype(str).tolist()
            label_column = "Sample"

        n_clusters = max(1, min(n_clusters, len(labels)))
        if method == "hierarchical":
            if len(labels) == 1:
                clusters = np.array([1])
            else:
                distances = pdist(data, metric=metric)
                linkage_matrix = linkage(distances, method=linkage_method)
                clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
                self._plot_dendrogram(linkage_matrix, labels, axis, prefix)
        elif method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            clusters = model.fit_predict(data) + 1
        else:
            raise ClusteringError(f"Unsupported clustering method: {method}")

        output_file = self.out_dir / f"{prefix}_{axis}_clusters.tsv"
        pd.DataFrame({label_column: labels, "Cluster": clusters}).to_csv(output_file, sep="\t", index=False)
        self._log("info", "Saved %s cluster assignments to: %s", axis, output_file)
        return str(output_file)

    def _plot_dendrogram(self, linkage_matrix: np.ndarray, labels: list[str], axis: str, prefix: str) -> str:
        """Create dendrogram/tree plot for hierarchical clustering."""
        output_file = self.out_dir / f"{prefix}_{axis}_dendrogram.png"
        label_count = len(labels)
        width = min(max(10, label_count * 0.25), 40)
        height = 8 if axis == "samples" else min(max(8, label_count * 0.04), 40)
        fig, ax = plt.subplots(figsize=(width, height), dpi=300)
        dendrogram(
            linkage_matrix,
            labels=labels if label_count <= 150 else None,
            leaf_rotation=90 if axis == "samples" else 0,
            ax=ax,
            color_threshold=None,
        )
        ax.set_title(f"{axis.capitalize()} clustering dendrogram")
        ax.set_ylabel("Distance")
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self._log("info", "Saved %s dendrogram to: %s", axis, output_file)
        return str(output_file)

    def _plot_heatmap(
        self,
        matrix: pd.DataFrame,
        row_cluster: bool,
        col_cluster: bool,
        color_map: str,
        prefix: str,
    ) -> str:
        """Create clustered heatmap."""
        output_file = self.out_dir / f"{prefix}_cluster_heatmap.png"
        sns.set_theme(style="white", context="paper")
        height = min(max(8, matrix.shape[0] * 0.02), 40)
        width = min(max(8, matrix.shape[1] * 0.6), 30)
        grid = sns.clustermap(
            matrix,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            cmap=color_map,
            center=0,
            figsize=(width, height),
            xticklabels=True,
            yticklabels=matrix.shape[0] <= 120,
            dendrogram_ratio=(0.12, 0.08),
            cbar_kws={"label": "Scaled expression"},
        )
        grid.ax_heatmap.set_xlabel("Samples")
        grid.ax_heatmap.set_ylabel("Genes")
        grid.fig.suptitle("Expression Clustering", y=1.02)
        grid.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(grid.fig)
        self._log("info", "Saved clustering heatmap to: %s", output_file)
        return str(output_file)

    def _expected_outputs(self, cluster_target: str, heatmap: bool, prefix: str) -> Dict[str, str]:
        """Return expected output paths for dry-run mode."""
        outputs = {}
        if cluster_target in {"genes", "both"}:
            outputs["gene_clusters"] = str(self.out_dir / f"{prefix}_genes_clusters.tsv")
        if cluster_target in {"samples", "both"}:
            outputs["sample_clusters"] = str(self.out_dir / f"{prefix}_samples_clusters.tsv")
        if heatmap:
            outputs["heatmap"] = str(self.out_dir / f"{prefix}_cluster_heatmap.png")
        return outputs


class ClustRunner:
    """Run the external Clust co-expression clustering tool."""

    def __init__(
        self,
        matrix_file: str,
        out_dir: str = ".",
        executable: str = "clust",
        logger: Optional[Any] = None,
        dryrun: bool = False,
    ):
        self.matrix_file = matrix_file
        self.out_dir = Path(out_dir)
        self.executable = executable
        self.logger = logger
        self.dryrun = dryrun
        if not self.dryrun:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, level: str, message: str, *args) -> None:
        if self.logger:
            getattr(self.logger, level)(message, *args)

    def run(
        self,
        cluster_tightness: float = 1.0,
        normalisation: int = 1000,
        k_values: str = "4",
        outlier: float = 3.0,
        cluster_size: int = 11,
        replicates: bool = False,
        delimiter: Optional[str] = None,
        preprocessing: bool = False,
        extra_args: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run external Clust or report the planned command in dry-run mode."""
        executable_path = shutil.which(self.executable) or self.executable
        if not self.dryrun and shutil.which(self.executable) is None:
            raise ClusteringError(f"Clust executable not found in PATH: {self.executable}")

        command = [
            executable_path,
            str(self.matrix_file),
            "-o",
            str(self.out_dir),
            "-t",
            str(cluster_tightness),
            "-n",
            str(normalisation),
            "-K",
            *str(k_values).split(),
            "-s",
            str(outlier),
            "-cs",
            str(cluster_size),
        ]
        if replicates:
            command.append("-r")
        if delimiter:
            command.extend(["-d", delimiter])
        if preprocessing:
            command.append("-p")
        if extra_args:
            command.extend(extra_args.split())

        command_text = " ".join(command)
        if self.dryrun:
            self._log("info", "DRYRUN: Would run Clust command: %s", command_text)
            return {"command": command_text, "out_dir": str(self.out_dir)}

        self._log("info", "Running Clust command: %s", command_text)
        completed = subprocess.run(command, cwd=str(self.out_dir), capture_output=True, text=True, check=False)
        if completed.stdout:
            self._log("info", completed.stdout)
        if completed.stderr:
            self._log("warning", completed.stderr)
        if completed.returncode != 0:
            raise ClusteringError(f"Clust failed with exit code {completed.returncode}")

        return {"command": command_text, "out_dir": str(self.out_dir)}
