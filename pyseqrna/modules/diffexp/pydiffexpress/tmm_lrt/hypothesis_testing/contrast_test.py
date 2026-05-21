"""
Standalone contrast testing for the TMM/LRT component path.

Classes:
    - LRTContrastTester: Fit a full NB GLM and derive contrast-specific test statistics

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2


class LRTContrastTester:
    """Fit a full NB GLM and derive contrast-specific test statistics."""

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter

    def _fit_glm(
        self,
        y: np.ndarray,
        design: np.ndarray,
        alpha: float,
        offset: np.ndarray,
    ):
        """Fit a fixed-dispersion NB GLM with a stable fallback order."""
        model = sm.GLM(
            y,
            design,
            family=sm.families.NegativeBinomial(alpha=alpha),
            offset=offset,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for method in ("irls", "newton"):
                try:
                    result = model.fit(method=method, maxiter=self.max_iter, disp=0)
                    if np.all(np.isfinite(result.params)) and np.isfinite(result.llf):
                        return result
                except Exception:
                    continue
        raise RuntimeError("GLM fit failed")

    @staticmethod
    def make_condition_design(conditions: np.ndarray) -> tuple[np.ndarray, list[str]]:
        levels = sorted(np.unique(conditions))
        design = np.column_stack([(conditions == level).astype(float) for level in levels])
        return design, levels

    @staticmethod
    def make_contrast_vector(levels: list[str], comparison: str) -> np.ndarray:
        numerator, denominator = comparison.split("-", 1)
        contrast = np.zeros(len(levels), dtype=float)
        contrast[levels.index(numerator)] = 1.0
        contrast[levels.index(denominator)] = -1.0
        return contrast

    def fit_full_model(
        self,
        counts: np.ndarray,
        effective_lib_sizes: np.ndarray,
        dispersions: np.ndarray,
        conditions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        design, levels = self.make_condition_design(conditions)
        offset = np.log(np.asarray(effective_lib_sizes, dtype=float))
        counts = np.asarray(counts, dtype=float)
        dispersions = np.asarray(dispersions, dtype=float)

        n_genes = counts.shape[1]
        n_coef = design.shape[1]
        beta = np.full((n_genes, n_coef), np.nan, dtype=float)
        covariance = np.full((n_genes, n_coef, n_coef), np.nan, dtype=float)
        llf = np.full(n_genes, np.nan, dtype=float)

        for gene_idx in range(n_genes):
            alpha = float(dispersions[gene_idx])
            if not np.isfinite(alpha) or alpha <= 0:
                continue

            try:
                result = self._fit_glm(
                    y=counts[:, gene_idx],
                    design=design,
                    alpha=alpha,
                    offset=offset,
                )
                beta[gene_idx, :] = result.params
                covariance[gene_idx, :, :] = result.cov_params()
                llf[gene_idx] = result.llf
            except Exception:
                continue

        return {
            "levels": np.array(levels, dtype=object),
            "design_full": design,
            "beta": beta,
            "covariance": covariance,
            "counts": counts,
            "offset": offset,
            "dispersions": dispersions,
            "llf_full": llf,
        }

    def score_contrast(
        self,
        fit_results: dict[str, np.ndarray],
        comparison: str,
    ) -> dict[str, np.ndarray]:
        levels = list(fit_results["levels"])
        beta = fit_results["beta"]
        covariance = fit_results["covariance"]
        contrast = self.make_contrast_vector(levels, comparison)
        reduced_design = self._make_reduced_design(levels=levels, comparison=comparison, fit_results=fit_results)

        counts = fit_results["counts"]
        offset = fit_results["offset"]
        dispersions = fit_results["dispersions"]
        llf_full = fit_results["llf_full"]

        n_genes = beta.shape[0]
        lfc = np.full(n_genes, np.nan, dtype=float)
        stat = np.full(n_genes, np.nan, dtype=float)
        pvalue = np.full(n_genes, np.nan, dtype=float)

        for gene_idx in range(n_genes):
            beta_i = beta[gene_idx]
            cov_i = covariance[gene_idx]
            if np.all(np.isfinite(beta_i)) and np.all(np.isfinite(cov_i)):
                estimate = float(contrast @ beta_i)
                lfc[gene_idx] = estimate / np.log(2.0)

            full_llf = llf_full[gene_idx]
            alpha = float(dispersions[gene_idx])
            if not np.isfinite(full_llf) or not np.isfinite(alpha) or alpha <= 0:
                continue

            try:
                result_reduced = self._fit_glm(
                    y=counts[:, gene_idx],
                    design=reduced_design,
                    alpha=alpha,
                    offset=offset,
                )
                lr = max(0.0, 2.0 * (full_llf - result_reduced.llf))
                stat[gene_idx] = lr
                pvalue[gene_idx] = chi2.sf(lr, df=1)
            except Exception:
                continue

        return {
            "logFC_model": lfc,
            "LR": stat,
            "pvalue": pvalue,
        }

    def _make_reduced_design(
        self,
        levels: list[str],
        comparison: str,
        fit_results: dict[str, np.ndarray],
    ) -> np.ndarray:
        numerator, denominator = comparison.split("-", 1)
        design_full = fit_results["design_full"]
        sample_level_index = np.argmax(design_full, axis=1)
        sample_conditions = np.array([levels[idx] for idx in sample_level_index], dtype=object)

        tested = np.isin(sample_conditions, [numerator, denominator]).astype(float)
        columns = [tested]
        for level in levels:
            if level not in {numerator, denominator}:
                columns.append((sample_conditions == level).astype(float))
        return np.column_stack(columns)
