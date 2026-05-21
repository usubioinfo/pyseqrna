#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Built-in PyCoexpression Clustering Tool

This module implements a native Python-based gene co-expression clustering algorithm that
groups genes with similar profiles across samples/conditions. It provides automatic selection
of the number of clusters (K), replicate collapsing, row Z-score normalization,
flat-profile filtering, and outlier detection based on correlation and distance.

Features:
    - Automatic selection of the optimal cluster count (K) using KMeans and Silhouette scores
    - Row Z-score normalization and filtering of flat/non-responsive gene profiles
    - Flexible replicate collapsing based on prefix naming patterns
    - Outlier detection within clusters using Pearson correlation and Euclidean distance thresholds
    - Automated creation of high-quality cluster profile plots with centroids
    - Exports detailed cluster membership TSV files and unclustered lists

Configuration:
    Configured via parameters passed to the run method (tightness, k_values, outlier,
    cluster_size, replicates, preprocessing) and via the constructor arguments.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - scipy
    - scikit-learn (KMeans, silhouette_score)
    - pyseqrna.modules.coexpression.base (BaseCoexpression, CoexpressionError)

Classes / Functions / Exceptions:
    - PyCoexpression: Built-in gene co-expression clustering using K-Means and Silhouette scores.

:Created: May 21, 2026
:Updated: May 21, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import BaseCoexpression, CoexpressionError


class PyCoexpression(BaseCoexpression):
    """Built-in gene co-expression clustering using K-Means and Silhouette scores."""

    def __init__(
        self,
        matrix_file: str,
        out_dir: str = ".",
        logger: Optional[Any] = None,
        dryrun: bool = False,
        dry_run_manager: Optional[Any] = None,
    ):
        super().__init__(
            matrix_file=matrix_file,
            out_dir=out_dir,
            logger=logger,
            dryrun=dryrun,
            dry_run_manager=dry_run_manager,
        )

    def _load_matrix(self) -> pd.DataFrame:
        """Load expression matrix from Excel, CSV, or TSV."""
        path = Path(self.matrix_file)
        if not path.exists():
            raise CoexpressionError(f"Expression matrix not found: {path}")

        suffix = path.suffix.lower()
        try:
            if suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            elif suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix in {".tsv", ".txt"}:
                df = pd.read_csv(path, sep="\t")
            else:
                raise CoexpressionError(f"Unsupported matrix file format: {suffix}")
        except Exception as exc:
            raise CoexpressionError(f"Failed to read expression matrix {path}: {exc}") from exc

        if df.empty:
            raise CoexpressionError(f"Expression matrix is empty: {path}")

        # Assume first column contains Gene IDs
        gene_col = df.columns[0]
        df = df.rename(columns={gene_col: "Gene"})
        return df

    def _collapse_replicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse replicate columns by averaging them based on common prefix patterns."""
        cols = df.columns
        group_map = {}
        for col in cols:
            col_str = str(col)
            # Match common replicate suffix patterns:
            # 1. Sample_R1, Sample-R2, Sample_r3
            m1 = re.match(r"^(.*?)[_-][Rr]?\d+$", col_str)
            if m1:
                group_map[col] = m1.group(1)
                continue
            # 2. Sample_1, Sample-2
            m2 = re.match(r"^(.*?)[_-]\d+$", col_str)
            if m2:
                group_map[col] = m2.group(1)
                continue
            # 3. Sample1, Sample2 (trailing digits)
            m3 = re.match(r"^(.*?)\d+$", col_str)
            if m3:
                group_map[col] = m3.group(1)
                continue
            # 4. Sample_A, Sample-B (separator followed by trailing letter)
            m4 = re.match(r"^(.*?)[_-][a-zA-Z]$", col_str)
            if m4:
                group_map[col] = m4.group(1)
                continue
            # 5. M1A, M1B (trailing single letter preceded by digit)
            m5 = re.match(r"^([a-zA-Z0-9_-]*\d)[a-zA-Z]$", col_str)
            if m5:
                group_map[col] = m5.group(1)
                continue
            group_map[col] = col_str

        self._log("info", "Collapsing replicates using grouping: %s", str(group_map))

        # Collapse columns (transpose, group by index, average, transpose back)
        collapsed = df.T.groupby(group_map).mean().T

        # Preserve the original condition order
        unique_groups = []
        for col in cols:
            g = group_map[col]
            if g not in unique_groups:
                unique_groups.append(g)
        collapsed = collapsed[unique_groups]
        return collapsed

    def _plot_cluster_profile(
        self,
        cluster_id: int,
        cluster_data: pd.DataFrame,
        centroid: pd.Series,
        output_path: Path,
    ) -> None:
        """Plot and save a high-quality co-expression cluster profile plot."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        samples = cluster_data.columns

        # Plot individual gene lines
        for _, row in cluster_data.iterrows():
            ax.plot(samples, row.values, color="#1f77b4", alpha=0.15, linewidth=1.2)

        # Plot cluster mean profile (centroid)
        ax.plot(
            samples,
            centroid.values,
            color="#d62728",
            linewidth=3.2,
            label="Centroid",
        )

        ax.set_title(
            f"Cluster {cluster_id} (N = {len(cluster_data)} genes)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Samples / Conditions", fontsize=12, labelpad=10)
        ax.set_ylabel("Z-score Normalized Expression", fontsize=12, labelpad=10)

        if len(samples) > 5:
            plt.xticks(rotation=45, ha="right")

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(
        self,
        tightness: float = 1.0,
        k_values: str = "4",
        outlier: float = 3.0,
        cluster_size: int = 11,
        replicates: bool = False,
        preprocessing: bool = False,
    ) -> Dict[str, Any]:
        """Run the built-in PyCoexpression clustering tool."""
        # 1. Handle dry-run mode
        if self.dryrun:
            self._log("info", "DRYRUN: Running built-in PyCoexpression clustering on matrix: %s", self.matrix_file)
            return {"command": "pycoexpression_native_run", "out_dir": str(self.out_dir)}

        # 2. Load count matrix
        df_raw = self._load_matrix()
        genes = df_raw["Gene"].values
        expression_df = df_raw.drop(columns=["Gene"])

        # Convert everything to numeric, fill NaNs
        expression_df = expression_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # 3. Optional log preprocessing
        if preprocessing:
            expression_df = np.log2(expression_df + 1.0)

        # 4. Collapse replicates if requested
        if replicates:
            expression_df = self._collapse_replicates(expression_df)

        # 5. Row Z-score normalization and flat-profile filtering
        row_stds = expression_df.std(axis=1)
        # Identify non-flat genes (variation SD >= 0.1)
        non_flat_mask = row_stds >= 0.1
        flat_count = np.sum(~non_flat_mask)
        self._log("info", "Filtered out %d flat/non-responsive genes (SD < 0.1)", flat_count)

        if np.sum(non_flat_mask) < 2:
            raise CoexpressionError("Not enough non-flat genes available for clustering.")

        filtered_expr = expression_df[non_flat_mask]
        filtered_genes = genes[non_flat_mask]

        # Calculate Z-score
        mean_profiles = filtered_expr.mean(axis=1)
        std_profiles = filtered_expr.std(axis=1).replace(0, 1.0)
        scaled_values = filtered_expr.sub(mean_profiles, axis=0).div(std_profiles, axis=0).values
        scaled_df = pd.DataFrame(scaled_values, index=filtered_genes, columns=expression_df.columns)

        # 6. Parse K values and select optimal K
        try:
            parts = re.split(r"[\s,]+", str(k_values).strip())
            k_range = [int(p) for p in parts if p.isdigit()]
        except Exception:
            k_range = []

        if not k_range:
            k_range = [2, 3, 4, 5, 6, 7, 8]

        # Estimate optimal K if range is provided
        if len(k_range) > 1:
            best_k = k_range[0]
            best_score = -1.0
            n_samples = scaled_df.shape[0]

            # Subsample up to 2000 genes for fast silhouette evaluation
            if n_samples > 2000:
                np.random.seed(42)
                sub_indices = np.random.choice(n_samples, 2000, replace=False)
                sub_data = scaled_df.iloc[sub_indices].values
            else:
                sub_data = scaled_df.values

            for k in k_range:
                if k >= n_samples or k < 2:
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sub_data)
                score = silhouette_score(sub_data, labels)
                self._log("info", "Silhouette score for K=%d: %.4f", k, score)
                if score > best_score:
                    best_score = score
                    best_k = k
            k_opt = best_k
            self._log("info", "Selected optimal K = %d with Silhouette score %.4f", k_opt, best_score)
        else:
            k_opt = k_range[0]
            self._log("info", "Using specified K = %d", k_opt)

        # 7. Run KMeans clustering
        kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
        initial_labels = kmeans.fit_predict(scaled_df.values) + 1  # 1-indexed clusters

        # 8. Outlier filtering (distance and correlation)
        filtered_assignments = []
        min_corr = 0.5 + 0.2 * tightness  # default tightness=1.0 -> min_corr=0.7

        for c in range(1, k_opt + 1):
            cluster_mask = initial_labels == c
            c_genes = filtered_genes[cluster_mask]
            c_data = scaled_df.loc[c_genes]

            if len(c_genes) == 0:
                continue

            centroid = c_data.mean(axis=0)

            # Compute correlation and distance for each gene in cluster
            corrs = []
            dists = []
            for gene, row in c_data.iterrows():
                r, _ = stats.pearsonr(row.values, centroid.values)
                corrs.append(r)
                d = np.linalg.norm(row.values - centroid.values)
                dists.append(d)

            dists = np.array(dists)
            mean_d = dists.mean()
            std_d = dists.std()
            dist_threshold = mean_d + outlier * std_d if std_d > 1e-5 else float("inf")

            for idx, gene in enumerate(c_genes):
                r = corrs[idx]
                d = dists[idx]
                is_outlier = (r < min_corr) or (d > dist_threshold)

                filtered_assignments.append(
                    {
                        "Gene": gene,
                        "Cluster": -1 if is_outlier else c,
                        "Correlation": r,
                        "Distance": d,
                    }
                )

        # 9. Enforce minimum cluster size constraint
        from collections import Counter

        sizes = Counter(item["Cluster"] for item in filtered_assignments if item["Cluster"] != -1)
        valid_clusters = {c for c, size in sizes.items() if size >= cluster_size}

        # Save final assignments, marking discarded cluster genes as unclustered (-1)
        final_assignments = []
        for item in filtered_assignments:
            c = item["Cluster"]
            if c != -1 and c not in valid_clusters:
                final_assignments.append(
                    {
                        "Gene": item["Gene"],
                        "Cluster": -1,
                        "Correlation": item["Correlation"],
                        "Distance": item["Distance"],
                    }
                )
            else:
                final_assignments.append(item)

        # 10. Write results to files
        df_assign = pd.DataFrame(final_assignments)
        # Create a full map from all input genes to final clusters
        full_assign_map = {g: -1 for g in genes}
        for item in final_assignments:
            full_assign_map[item["Gene"]] = item["Cluster"]

        # Master assignment file
        master_df = pd.DataFrame(
            {
                "Gene": genes,
                "Cluster": [full_assign_map[g] for g in genes],
            }
        )
        master_file = self.out_dir / "coexpression_clusters.tsv"
        master_df.to_csv(master_file, sep="\t", index=False)

        # Write cluster files and generate profile plots
        for c in sorted(valid_clusters):
            c_genes = [item["Gene"] for item in final_assignments if item["Cluster"] == c]
            c_data = scaled_df.loc[c_genes]
            centroid = c_data.mean(axis=0)

            # Gene list file
            c_file = self.out_dir / f"Cluster_{c}.tsv"
            pd.DataFrame({"Gene": c_genes}).to_csv(c_file, sep="\t", index=False)

            # Plot profile
            plot_file = self.out_dir / f"Cluster_{c}.png"
            self._plot_cluster_profile(c, c_data, centroid, plot_file)

        # Write unclustered genes list
        unclustered_genes = [g for g in genes if full_assign_map[g] == -1]
        unclustered_file = self.out_dir / "Unclustered_genes.tsv"
        pd.DataFrame({"Gene": unclustered_genes}).to_csv(unclustered_file, sep="\t", index=False)

        self._log("info", "Finished built-in co-expression clustering. Created %d valid clusters.", len(valid_clusters))
        return {"command": "pycoexpression_native_run", "out_dir": str(self.out_dir)}
