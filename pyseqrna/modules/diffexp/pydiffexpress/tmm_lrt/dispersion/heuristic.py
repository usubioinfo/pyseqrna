"""
Abundance-aware common/trended/tagwise dispersion estimation.

Classes:
    - TagwiseDispersionEstimator: Native common/trended/tagwise dispersion estimator

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...datasets.dataset import ExpressionDataset


class TagwiseDispersionEstimator:
    """
    Native common/trended/tagwise dispersion estimator.

    The estimator uses a seed dispersion profile from the existing sandbox NB
    fits, then remaps it into an abundance-aware shape:
    - common dispersion from the high-expression tail
    - logistic abundance trend over AveLogCPM
    - log-scale shrinkage from the seed dispersion toward the abundance trend
    """

    def __init__(
        self,
        high_expression_threshold: float = 6.0,
        high_expression_quantile: float = 0.75,
        trend_floor_multiplier: float = 0.66,
        trend_amplitude_multiplier: float = 14.0,
        trend_slope: float = 1.0,
        trend_center: float = 0.0,
        shrink_slope: float = 1.25,
        shrink_center: float = 6.0,
        min_dispersion: float = 1e-8,
    ):
        self.high_expression_threshold = high_expression_threshold
        self.high_expression_quantile = high_expression_quantile
        self.trend_floor_multiplier = trend_floor_multiplier
        self.trend_amplitude_multiplier = trend_amplitude_multiplier
        self.trend_slope = trend_slope
        self.trend_center = trend_center
        self.shrink_slope = shrink_slope
        self.shrink_center = shrink_center
        self.min_dispersion = min_dispersion

    def fit(
        self,
        dataset: ExpressionDataset,
        ave_log_cpm: np.ndarray,
        seed_dispersions: np.ndarray,
    ) -> pd.DataFrame:
        ave_log_cpm = np.asarray(ave_log_cpm, dtype=float)
        seed_dispersions = np.asarray(seed_dispersions, dtype=float)

        if ave_log_cpm.shape[0] != dataset.n_genes:
            raise ValueError("AveLogCPM length must match number of genes")
        if seed_dispersions.shape[0] != dataset.n_genes:
            raise ValueError("seed_dispersions length must match number of genes")

        common = self._estimate_common_dispersion(ave_log_cpm, seed_dispersions)
        trended = self._estimate_trended_dispersion(ave_log_cpm, common)
        tagwise = self._estimate_tagwise_dispersion(ave_log_cpm, seed_dispersions, trended)

        return pd.DataFrame(
            {
                "Gene": dataset.var_names.astype(str),
                "common_dispersion": np.full(dataset.n_genes, common, dtype=float),
                "trended_dispersion": trended,
                "tagwise_dispersion": tagwise,
            }
        )

    def _estimate_common_dispersion(
        self,
        ave_log_cpm: np.ndarray,
        seed_dispersions: np.ndarray,
    ) -> float:
        high_expression = np.isfinite(seed_dispersions) & (ave_log_cpm > self.high_expression_threshold)
        if not np.any(high_expression):
            finite = np.isfinite(seed_dispersions)
            if not np.any(finite):
                return 0.1
            return float(np.nanmedian(seed_dispersions[finite]))

        common = float(np.nanquantile(seed_dispersions[high_expression], self.high_expression_quantile))
        return max(common, self.min_dispersion)

    def _estimate_trended_dispersion(
        self,
        ave_log_cpm: np.ndarray,
        common_dispersion: float,
    ) -> np.ndarray:
        trend = self.trend_floor_multiplier * common_dispersion + self.trend_amplitude_multiplier * common_dispersion / (
            1.0 + np.exp(self.trend_slope * (ave_log_cpm - self.trend_center))
        )
        return np.clip(trend, self.min_dispersion, None)

    def _estimate_tagwise_dispersion(
        self,
        ave_log_cpm: np.ndarray,
        seed_dispersions: np.ndarray,
        trended_dispersion: np.ndarray,
    ) -> np.ndarray:
        seed = np.where(np.isfinite(seed_dispersions), seed_dispersions, trended_dispersion)
        seed = np.clip(seed, self.min_dispersion, None)
        trend = np.clip(trended_dispersion, self.min_dispersion, None)

        # Weight on the raw seed rises with expression; low-count genes are
        # pulled strongly to the abundance trend.
        seed_weight = 1.0 / (1.0 + np.exp(-self.shrink_slope * (ave_log_cpm - self.shrink_center)))
        tagwise = np.exp((1.0 - seed_weight) * np.log(trend) + seed_weight * np.log(seed))
        return np.clip(tagwise, self.min_dispersion, None)
