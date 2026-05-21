"""
Iterative normalization strategy for gene expression data.

This module implements an iterative size factor estimator. Unlike
the median-ratio methods, it alternates between refitting dispersions with an
intercept-only design and optimizing sample size factors under a negative
binomial likelihood objective.

Classes:
    - IterativeNormalizer: Iterative normalization strategy

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .base import BaseNormalizer
from ..datasets.dataset import ExpressionDataset
from ..dispersion import (
    GeneWiseDispersionEstimator,
    MAPDispersionEstimator,
    TrendDispersionEstimator,
)

logger = logging.getLogger(__name__)


class IterativeNormalizer(BaseNormalizer):
    """
    Iterative normalization strategy.

    This temporarily reduces the design to an intercept, refits dispersions
    using the current size factors, and then optimizes new size factors by
    maximizing the negative binomial likelihood.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-4,
        quantile: float = 0.05,
        min_mu: float = 0.5,
        min_disp: float = 1e-8,
        quiet: bool = True,
    ):
        super().__init__(name="IterativeNormalizer")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.quantile = quantile
        self.min_mu = min_mu
        self.min_disp = min_disp
        self.quiet = quiet
        self.iterations_converged = None

    def _estimate_factors(self, dataset: ExpressionDataset, **kwargs) -> np.ndarray:
        max_iterations = kwargs.get("max_iterations", self.max_iterations)
        tolerance = kwargs.get("tolerance", self.tolerance)
        quantile = kwargs.get("quantile", self.quantile)
        min_mu = kwargs.get("min_mu", self.min_mu)
        min_disp = kwargs.get("min_disp", self.min_disp)
        quiet = kwargs.get("quiet", self.quiet)

        counts = np.asarray(dataset.counts, dtype=float)
        n_samples, n_genes = counts.shape

        # Use genes with at least one non-zero count.
        non_zero_genes = counts.sum(axis=0) > 0
        if not np.any(non_zero_genes):
            raise ValueError("Iterative normalization requires at least one non-zero gene")

        original_design = dataset._adata.obsm.get("design", None)
        original_design_columns = dataset._adata.uns.get("design_columns", None)

        intercept_design = np.ones((n_samples, 1), dtype=float)
        dataset._adata.obsm["design"] = intercept_design
        dataset._adata.uns["design_columns"] = ["Intercept"]

        try:
            size_factors = np.ones(n_samples, dtype=float)

            for iteration in range(max_iterations):
                old_size_factors = size_factors.copy()
                self._update_dataset_scaling(dataset, counts, old_size_factors)

                disp_data = dataset._adata.copy()

                gene_wise = GeneWiseDispersionEstimator(
                    min_disp=min_disp,
                    min_mu=min_mu,
                    quiet=True,
                    n_iter=1,
                )
                gene_wise.fit(disp_data, intercept_design)
                gene_wise.update_data(disp_data)

                trend = TrendDispersionEstimator(
                    fit_type="mean",
                    min_disp=min_disp,
                    quiet=True,
                )
                trend.fit(disp_data, intercept_design)
                trend.update_data(disp_data)

                map_estimator = MAPDispersionEstimator(
                    min_disp=min_disp,
                    quiet=True,
                )
                map_estimator.fit(disp_data, intercept_design)
                map_estimator.update_data(disp_data)

                mu_hat = np.asarray(disp_data.layers["mu"], dtype=float)[:, non_zero_genes]
                dispersions = np.asarray(disp_data.var["dispersion"], dtype=float)[non_zero_genes]
                counts_nz = counts[:, non_zero_genes]

                def objective(log_sf: np.ndarray) -> float:
                    candidate_sf = np.exp(log_sf - np.mean(log_sf))
                    mu = mu_hat / old_size_factors[:, None] * candidate_sf[:, None]
                    log_likelihood = self._gene_negative_binomial_loglik(
                        counts=counts_nz,
                        mu=np.clip(mu, min_mu, None),
                        alpha=np.clip(dispersions, min_disp, None),
                    )
                    cutoff = np.quantile(log_likelihood, quantile)
                    return -np.sum(log_likelihood[log_likelihood >= cutoff])

                result = minimize(
                    objective,
                    np.log(old_size_factors),
                    method="L-BFGS-B",
                )

                if result.success:
                    size_factors = np.exp(result.x - np.mean(result.x))
                else:
                    size_factors = old_size_factors
                    if not quiet:
                        logger.warning("Iterative size factor optimization failed to converge")
                    break

                if iteration > 1 and np.sum((np.log(old_size_factors) - np.log(size_factors)) ** 2) < tolerance:
                    self.iterations_converged = iteration + 1
                    break
            else:
                self.iterations_converged = max_iterations

            if self.iterations_converged is None:
                self.iterations_converged = max_iterations

            self._update_dataset_scaling(dataset, counts, size_factors)
            return size_factors

        finally:
            if original_design is None:
                dataset._adata.obsm.pop("design", None)
            else:
                dataset._adata.obsm["design"] = original_design

            if original_design_columns is None:
                dataset._adata.uns.pop("design_columns", None)
            else:
                dataset._adata.uns["design_columns"] = original_design_columns

    @staticmethod
    def _gene_negative_binomial_loglik(
        counts: np.ndarray,
        mu: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        size = 1.0 / alpha

        log_prob = (
            gammaln(counts + size[None, :])
            - gammaln(size[None, :])
            - gammaln(counts + 1.0)
            + size[None, :] * (np.log(size)[None, :] - np.log(size[None, :] + mu))
            + counts * (np.log(mu) - np.log(size[None, :] + mu))
        )
        return np.sum(log_prob, axis=0)

    def _update_dataset_scaling(
        self,
        dataset: ExpressionDataset,
        counts: np.ndarray,
        size_factors: np.ndarray,
    ) -> None:
        normalized_counts = counts / size_factors[:, None]
        self.normalized_counts = normalized_counts
        self.base_means = np.mean(normalized_counts, axis=0)
        self.base_variances = np.var(normalized_counts, axis=0, ddof=1)

        dataset.size_factors = size_factors
        dataset._adata.obs["size_factors"] = size_factors
        dataset._adata.var["base_mean"] = self.base_means
        dataset._adata.var["base_variance"] = self.base_variances
        dataset._adata.layers["normalized_counts"] = normalized_counts

    def _get_summary(self) -> dict:
        summary = super()._get_summary()
        if self.iterations_converged is not None:
            summary["iteration_info"] = {
                "iterations_converged": self.iterations_converged,
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
                "converged": self.iterations_converged < self.max_iterations,
            }
        return summary
