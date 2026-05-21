"""
TMM (Trimmed Mean of M-values) normalization strategy for gene expression data.

This module implements TMM normalization, which computes normalization factors
based on the trimmed mean of M-values between pairs of samples.

Classes:
    - TMMNormalizer: TMM (Trimmed Mean of M-values) normalization strategy

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np

from .base import BaseNormalizer
from ..datasets.dataset import ExpressionDataset


class TMMNormalizer(BaseNormalizer):
    """
    TMM (Trimmed Mean of M-values) normalization strategy.

    This normalizer computes factors based on the trimmed mean of M-values
    between pairs of samples.
    """

    def __init__(
        self,
        trim: float = 0.3,
        log_ratio_trim: float = 0.3,
        sum_trim: float = 0.05,
        do_weighting: bool = True,
        acutoff: float = -1e10,
    ):
        """
        Initialize the TMM normalizer.

        Parameters
        ----------
        trim : float
            Fraction of data to trim from each end (0.3 = 30% from each end).
        log_ratio_trim : float
            Fraction of data to trim from each end of log ratios.
        sum_trim : float
            Fraction of data to trim from each end of sum counts.
        do_weighting : bool
            Whether to use inverse variance weighting.
        acutoff : float
            Cutoff for A-values (log2 average expression).
        """
        super().__init__(name="TMMNormalizer")
        self.trim = trim
        self.log_ratio_trim = log_ratio_trim
        self.sum_trim = sum_trim
        self.do_weighting = do_weighting
        self.acutoff = acutoff

    def fit(self, dataset: ExpressionDataset, **kwargs) -> "TMMNormalizer":
        """
        Fit the TMM normalizer to the dataset.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to normalize.
        **kwargs
            Additional parameters.

        Returns
        -------
        TMMNormalizer
            Self for method chaining.
        """
        # Estimate TMM normalization factors
        self.normalization_factors = self._estimate_factors(dataset, **kwargs)

        # For TMM, size factors are the same as normalization factors
        self.size_factors = self.normalization_factors.copy()

        # Store results
        self.results = {
            "size_factors": self.size_factors.copy(),
            "normalization_factors": self.normalization_factors.copy(),
            "parameters": kwargs,
        }

        # Update dataset with size factors (which are the same as TMM factors)
        dataset.size_factors = self.size_factors
        dataset._adata.obs["size_factors"] = self.size_factors
        dataset._adata.obs["normalization_factors"] = self.normalization_factors

        self.fitted = True
        return self

    def _estimate_factors(self, dataset: ExpressionDataset, **kwargs) -> np.ndarray:
        """
        Estimate normalization factors using TMM method.

        Parameters
        ----------
        dataset : ExpressionDataset
            The dataset to normalize.
        **kwargs
            Additional parameters.

        Returns
        -------
        np.ndarray
            Normalization factors for each sample.
        """
        counts = dataset.counts  # samples x genes
        n_samples = counts.shape[0]

        # Calculate library sizes
        lib_sizes = counts.sum(axis=1)  # sum over genes, per sample

        # Select reference sample using the median 75th percentile rule.
        f75_factors = [np.percentile(counts[i, :], 75) / lib_sizes[i] for i in range(n_samples)]
        mean_f75 = np.mean(f75_factors)
        reference_sample = np.argmin(np.abs(f75_factors - mean_f75))

        # Calculate TMM factors for all samples
        tmm_factors = np.ones(n_samples)
        for i in range(n_samples):
            if i == reference_sample:
                continue
            obs_counts = counts[i, :]
            ref_counts = counts[reference_sample, :]
            tmm_factors[i] = self._calc_factor_tmm(obs_counts, ref_counts, lib_sizes[i], lib_sizes[reference_sample])

        # Apply scaling factor to make geometric mean = 1
        scaling_factor = np.exp(np.mean(np.log(tmm_factors)))
        tmm_factors = tmm_factors / scaling_factor

        return tmm_factors

    def _calc_factor_tmm(
        self,
        obs_counts: np.ndarray,
        ref_counts: np.ndarray,
        libsize_obs: float,
        libsize_ref: float,
    ) -> float:
        """
        Calculate TMM normalization factor between two samples.

        Parameters
        ----------
        obs_counts : np.ndarray
            Counts for observation sample.
        ref_counts : np.ndarray
            Counts for reference sample.
        libsize_obs : float
            Library size for observation sample.
        libsize_ref : float
            Library size for reference sample.

        Returns
        -------
        float
            Normalization factor.
        """
        obs = obs_counts.astype(float)
        ref = ref_counts.astype(float)
        nO = libsize_obs
        nR = libsize_ref

        # Suppress divide-by-zero and invalid warnings for this block
        with np.errstate(divide="ignore", invalid="ignore"):
            logR = np.log2((obs / nO) / (ref / nR))
            absE = (np.log2(obs / nO) + np.log2(ref / nR)) / 2
            v = (nO - obs) / nO / obs + (nR - ref) / nR / ref

        fin = np.isfinite(logR) & np.isfinite(absE) & (absE > self.acutoff)
        logR = logR[fin]
        absE = absE[fin]
        v = v[fin]

        if len(logR) == 0:
            return 1.0
        if np.max(np.abs(logR)) < 1e-6:
            return 1.0

        n = len(logR)
        loL = int(np.floor(n * self.log_ratio_trim)) + 1
        hiL = n + 1 - loL
        loS = int(np.floor(n * self.sum_trim)) + 1
        hiS = n + 1 - loS

        rank_logR = np.argsort(np.argsort(logR)) + 1
        rank_absE = np.argsort(np.argsort(absE)) + 1
        keep = (rank_logR >= loL) & (rank_logR <= hiL) & (rank_absE >= loS) & (rank_absE <= hiS)

        if self.do_weighting:
            if np.sum(1 / v[keep]) > 0:
                f = np.sum(logR[keep] / v[keep]) / np.sum(1 / v[keep])
            else:
                f = np.mean(logR[keep])
        else:
            f = np.mean(logR[keep])

        if np.isnan(f):
            f = 0.0

        return 2**f

    def _calc_weights(self, ref_counts: np.ndarray, obs_counts: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Calculate inverse variance weights for TMM.

        Parameters
        ----------
        ref_counts : np.ndarray
            Counts for reference sample.
        obs_counts : np.ndarray
            Counts for observation sample.
        a : np.ndarray
            A-values (log2 average expression).

        Returns
        -------
        np.ndarray
            Weights for each gene.
        """
        # Calculate variance based on negative binomial assumption
        # This is a simplified variance estimate.

        # Use Poisson variance as approximation
        var_ref = ref_counts
        var_obs = obs_counts

        # Calculate variance of log ratio
        var_log_ratio = var_ref / (ref_counts**2) + var_obs / (obs_counts**2)

        # Convert to variance of M-value
        var_m = var_log_ratio / (np.log(2) ** 2)

        # Calculate weights (inverse variance)
        weights = 1.0 / (var_m + 1e-8)  # Add small constant to avoid division by zero

        return weights

    def _get_summary(self) -> dict:
        """Get summary of normalization parameters."""
        return {
            "method": "TMM",
            "trim": self.trim,
            "log_ratio_trim": self.log_ratio_trim,
            "sum_trim": self.sum_trim,
            "do_weighting": self.do_weighting,
            "acutoff": self.acutoff,
        }
