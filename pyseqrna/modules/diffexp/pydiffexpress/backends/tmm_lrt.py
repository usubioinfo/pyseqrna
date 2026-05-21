"""
TMM/LRT backend for pydiffexpress.

Classes:
    - TMMLRTBackend: Run the TMM/average-log-CPM/LRT component path and export snapshot files

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.dataset import ExpressionDataset
from ..tmm_lrt.dispersion import TagwiseDispersionEstimator
from ..tmm_lrt.hypothesis_testing import LRTContrastTester
from ..tmm_lrt.normalization import TMMFactorNormalizer, ave_log_cpm
from ..tmm_lrt.results import export_lrt_contrast


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _compute_group_logfc(
    counts: np.ndarray,
    effective_lib_sizes: np.ndarray,
    conditions: np.ndarray,
    comparison: str,
    prior_count: float = 0.125,
) -> np.ndarray:
    numerator, denominator = comparison.split("-", 1)
    numer_idx = conditions == numerator
    denom_idx = conditions == denominator
    cpm = counts / effective_lib_sizes[:, None] * 1e6
    numer_mean = np.mean(cpm[numer_idx], axis=0)
    denom_mean = np.mean(cpm[denom_idx], axis=0)
    return np.log2((numer_mean + prior_count) / (denom_mean + prior_count))


class TMMLRTBackend:
    """Run the TMM/average-log-CPM/LRT component path and export snapshot files."""

    name = "tmm_lrt"

    def run_snapshot(
        self,
        counts: pd.DataFrame,
        samples: pd.DataFrame,
        outdir: Path,
        gene_column: str = "Gene",
        comparisons: list[str] | None = None,
        sf_type: str = "tmm",
        dispersion: str = "tagwise",
        **kwargs,
    ) -> None:
        del sf_type  # this component path uses TMM by construction
        del kwargs
        dispersion = str(dispersion).strip().lower()
        if dispersion not in {"common", "trended", "tagwise"}:
            raise ValueError(f"Unsupported dispersion component for this backend: {dispersion}")
        comparisons = comparisons or ["GA-GB", "GA-GC", "GB-GC"]
        outdir.mkdir(parents=True, exist_ok=True)

        dataset = ExpressionDataset(
            counts=counts,
            sample_metadata=samples,
            gene_column=gene_column,
            design_column="condition",
        )
        tmm_normalizer = TMMFactorNormalizer()
        tmm_normalizer.fit(dataset)

        adata = dataset._adata
        lib_sizes = np.asarray(dataset.counts.sum(axis=1), dtype=float)
        norm_factors = np.asarray(tmm_normalizer.size_factors, dtype=float)
        effective_lib_sizes = lib_sizes * norm_factors

        normalization = pd.DataFrame(
            {
                "sample": adata.obs_names.astype(str),
                "lib_size": lib_sizes,
                "norm_factor": norm_factors,
                "effective_lib_size": effective_lib_sizes,
            }
        )
        _save_frame(normalization, outdir / "normalization_factors.tsv")

        ave_log_cpm_values = ave_log_cpm(
            np.asarray(dataset.counts, dtype=float),
            effective_lib_sizes,
            prior_count=2.0,
            dispersion=0.114988669246773,
        )

        # Seed from the existing NB fit, then replace with the separate
        # abundance-aware dispersion profile.
        dataset.diffexpress(sf_type="tmm", test="LRT", quiet=True)
        seed_dispersions = np.asarray(adata.var["dispersion"], dtype=float)

        dispersion_estimator = TagwiseDispersionEstimator()
        dispersion_df = dispersion_estimator.fit(
            dataset=dataset,
            ave_log_cpm=ave_log_cpm_values,
            seed_dispersions=seed_dispersions,
        )
        adata.var["common_dispersion"] = dispersion_df["common_dispersion"].to_numpy(dtype=float)
        adata.var["trended_dispersion"] = dispersion_df["trended_dispersion"].to_numpy(dtype=float)
        adata.var["tagwise_dispersion"] = dispersion_df["tagwise_dispersion"].to_numpy(dtype=float)
        selected_dispersion_column = f"{dispersion}_dispersion"
        adata.var["dispersion"] = adata.var[selected_dispersion_column].to_numpy(dtype=float)

        gene_stats = pd.DataFrame(
            {
                "Gene": adata.var_names.astype(str),
                "AveLogCPM": ave_log_cpm_values,
                "common_dispersion": np.asarray(adata.var["common_dispersion"], dtype=float),
                "trended_dispersion": np.asarray(adata.var["trended_dispersion"], dtype=float),
                "tagwise_dispersion": np.asarray(adata.var["tagwise_dispersion"], dtype=float),
                "dispersion": np.asarray(adata.var["dispersion"], dtype=float),
            }
        )
        _save_frame(gene_stats, outdir / "gene_stats.tsv")

        design_columns = pd.DataFrame({"design_column": list(adata.uns.get("design_columns", []))})
        _save_frame(design_columns, outdir / "design_columns.tsv")

        conditions = dataset.sample_metadata["condition"].astype(str).to_numpy()
        raw_counts = np.asarray(dataset.counts, dtype=float)
        contrast_tester = LRTContrastTester()
        fit_results = contrast_tester.fit_full_model(
            counts=raw_counts,
            effective_lib_sizes=effective_lib_sizes,
            dispersions=np.asarray(adata.var[selected_dispersion_column], dtype=float),
            conditions=conditions,
        )

        contrast_dir = outdir / "contrasts"
        contrast_dir.mkdir(exist_ok=True)
        for comparison in comparisons:
            contrast_results = contrast_tester.score_contrast(fit_results, comparison)
            result_df = export_lrt_contrast(
                genes=adata.var_names.astype(str),
                logfc=_compute_group_logfc(
                    counts=raw_counts,
                    effective_lib_sizes=effective_lib_sizes,
                    conditions=conditions,
                    comparison=comparison,
                ),
                logcpm=ave_log_cpm_values,
                lr=contrast_results["LR"],
                pvalue=contrast_results["pvalue"],
            )
            _save_frame(result_df, contrast_dir / f"{comparison}.tsv")
