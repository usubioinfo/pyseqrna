"""
Base-mean Wald backend for pydiffexpress.

Classes:
    - BaseMeanWaldBackend: Run the base-mean/Wald component path and export snapshot files

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..datasets.dataset import ExpressionDataset
from ..results.contrasts import ContrastAnalyzer


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


class BaseMeanWaldBackend:
    """Run the base-mean/Wald component path and export snapshot files."""

    name = "base_mean_wald"

    def run_snapshot(
        self,
        counts: pd.DataFrame,
        samples: pd.DataFrame,
        outdir: Path,
        gene_column: str = "Gene",
        comparisons: list[str] | None = None,
        sf_type: str = "ratio",
        **kwargs,
    ) -> None:
        del kwargs
        comparisons = comparisons or ["GA-GB", "GA-GC", "GB-GC"]
        outdir.mkdir(parents=True, exist_ok=True)

        dataset = ExpressionDataset(
            counts=counts,
            sample_metadata=samples,
            gene_column=gene_column,
            design_column="condition",
        )
        dataset.diffexpress(sf_type=sf_type, quiet=True)
        adata = dataset._adata

        size_factors = pd.DataFrame(
            {
                "sample": adata.obs_names.astype(str),
                "size_factor": adata.obs["size_factors"].astype(float).values,
            }
        )
        _save_frame(size_factors, outdir / "size_factors.tsv")

        normalized = pd.DataFrame(
            dataset.get_normalized_counts(),
            index=adata.obs_names.astype(str),
            columns=adata.var_names.astype(str),
        ).reset_index(names="sample")
        _save_frame(normalized, outdir / "normalized_counts.tsv")

        base_stats = pd.DataFrame(
            {
                "Gene": adata.var_names.astype(str),
                "baseMean": dataset.get_base_means(),
                "baseVariance": dataset.get_base_variances(),
                "dispersion": adata.var["dispersion"].astype(float).values,
                "all_zero": adata.var.get("all_zero", pd.Series(False, index=adata.var_names)).values,
            }
        )
        _save_frame(base_stats, outdir / "gene_stats.tsv")

        design_columns = pd.DataFrame({"design_column": list(adata.uns.get("design_columns", []))})
        _save_frame(design_columns, outdir / "design_columns.tsv")

        contrast_analyzer = ContrastAnalyzer()
        contrast_dir = outdir / "contrasts"
        contrast_dir.mkdir(exist_ok=True)
        for comparison in comparisons:
            result_df = contrast_analyzer.extract_contrast(
                adata,
                contrast=comparison,
                name=comparison,
                lfc_threshold=0.0,
                alpha=1.0,
            ).reset_index()
            _save_frame(result_df, contrast_dir / f"{comparison}.tsv")
