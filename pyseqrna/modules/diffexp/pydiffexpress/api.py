"""
Standalone public API for modular differential-expression experiments.

Functions:
    - available_normalizations: Return public normalization options
    - available_abundance: Return public abundance-summary options
    - available_dispersions: Return public dispersion options
    - available_tests: Return public hypothesis-test options
    - run_analysis: Run a standalone analysis using public component names
    - available_backends: Backward-compatible helper for the current concrete implementations
    - run_backend: Backward-compatible direct backend runner

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backends import get_backend


def available_normalizations() -> list[str]:
    """Return public normalization options."""
    return ["median_ratio", "poscounts", "iterate", "tmm"]


def available_abundance() -> list[str]:
    """Return public abundance-summary options."""
    return ["base_mean", "ave_log_cpm"]


def available_dispersions() -> list[str]:
    """Return public dispersion options."""
    return ["map", "common", "trended", "tagwise"]


def available_tests() -> list[str]:
    """Return public hypothesis-test options."""
    return ["wald", "lrt"]


def _resolve_backend(
    normalization: str,
    abundance: str,
    dispersion: str,
    test: str,
) -> tuple[str, str]:
    normalization = str(normalization).strip().lower()
    abundance = str(abundance).strip().lower()
    dispersion = str(dispersion).strip().lower()
    test = str(test).strip().lower()

    if (
        normalization in {"median_ratio", "poscounts", "iterate"}
        and abundance == "base_mean"
        and dispersion == "map"
        and test == "wald"
    ):
        return "base_mean_wald", normalization

    if (
        normalization == "tmm"
        and abundance == "ave_log_cpm"
        and dispersion in {"common", "trended", "tagwise"}
        and test == "lrt"
    ):
        return "tmm_lrt", normalization

    raise ValueError(
        "Unsupported component combination: "
        f"normalization={normalization}, abundance={abundance}, "
        f"dispersion={dispersion}, test={test}"
    )


def run_analysis(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    outdir: str | Path,
    gene_column: str = "Gene",
    comparisons: list[str] | None = None,
    normalization: str = "median_ratio",
    abundance: str = "base_mean",
    dispersion: str = "map",
    test: str = "wald",
) -> None:
    """Run a standalone analysis using public component names."""
    backend_name, sf_type = _resolve_backend(
        normalization=normalization,
        abundance=abundance,
        dispersion=dispersion,
        test=test,
    )
    backend_impl = get_backend(backend_name)
    backend_impl.run_snapshot(
        counts=counts,
        samples=samples,
        outdir=Path(outdir),
        gene_column=gene_column,
        comparisons=comparisons,
        sf_type=sf_type,
        dispersion=dispersion,
        test=test,
    )


def available_backends() -> list[str]:
    """Backward-compatible helper for the current concrete implementations."""
    return ["base_mean_wald", "tmm_lrt"]


def run_backend(
    backend: str,
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    outdir: str | Path,
    gene_column: str = "Gene",
    comparisons: list[str] | None = None,
    sf_type: str = "ratio",
) -> None:
    """Backward-compatible direct backend runner."""
    backend_impl = get_backend(backend)
    backend_impl.run_snapshot(
        counts=counts,
        samples=samples,
        outdir=Path(outdir),
        gene_column=gene_column,
        comparisons=comparisons,
        sf_type=sf_type,
    )
