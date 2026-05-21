#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wald test analyzer for differential expression analysis.

This module provides a Wald test implementation for negative-binomial GLMs.

Classes:
    - WaldTestAnalyzer: Wald test analyzer for differential expression analysis

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from scipy.stats import norm
from anndata import AnnData
from .base import BaseHypothesisTestAnalyzer
from ..dispersion.glm import NegativeBinomialGLM

logger = logging.getLogger(__name__)


class WaldTestAnalyzer(BaseHypothesisTestAnalyzer):
    """
    Wald test analyzer for differential expression analysis.

    This class implements Wald tests using the native negative-binomial GLM.
    """

    def __init__(
        self,
        beta_tol: float = 1e-8,
        max_iter: int = 100,
        quiet: bool = False,
        min_mu: float = 0.5,
        **kwargs,
    ):
        """
        Initialize WaldTestAnalyzer.

        Parameters
        ----------
        beta_tol : float, default=1e-8
            Convergence tolerance for GLM fitting
        max_iter : int, default=100
            Maximum number of iterations for GLM fitting
        quiet : bool, default=False
            Whether to suppress progress messages
        min_mu : float, default=0.5
            Minimum value for fitted means
        **kwargs
            Additional arguments passed to GLM
        """
        super().__init__(**kwargs)
        self.beta_tol = beta_tol
        self.max_iter = max_iter
        self.quiet = quiet
        self.min_mu = min_mu
        self.glm_kwargs = kwargs
        self.results = {}
        self.fitted = False

    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "WaldTestAnalyzer":
        """
        Fit the Wald test using a negative-binomial GLM.

        Parameters
        ----------
        data : AnnData
            AnnData object containing counts and metadata
        design_matrix : array-like, optional
            Design matrix for the model

        Returns
        -------
        WaldTestAnalyzer
            Self for method chaining
        """
        self._validate_data(data)

        if not self.quiet:
            logger.info("Running negative-binomial Wald test...")

        # Get data in the correct format
        counts = data.X.toarray() if hasattr(data.X, "toarray") else data.X
        size_factors = data.obs["size_factors"].values[:, None]  # (N, 1)
        size_factors_matrix = np.broadcast_to(size_factors, (size_factors.shape[0], counts.shape[1]))  # (N, M)
        dispersions = data.var["dispersion"].values

        # Use provided design matrix or get from data
        if design_matrix is None:
            design_matrix = data.obsm["design"]

        # Convert design matrix to numpy array if needed
        if hasattr(design_matrix, "values"):
            design_matrix = np.array(design_matrix)

        n_samples, n_genes = counts.shape
        n_coefficients = design_matrix.shape[1]

        # Filter out all-zero genes
        all_zero = data.var.get("all_zero", np.all(counts == 0, axis=0))
        valid_genes = ~all_zero

        counts = counts[:, valid_genes]
        dispersions = dispersions[valid_genes]
        size_factors_matrix = size_factors_matrix[:, valid_genes]

        # Filter invalid dispersions
        valid_disp = ~(np.isnan(dispersions) | np.isinf(dispersions) | (dispersions <= 0))
        counts = counts[:, valid_disp]
        dispersions = dispersions[valid_disp]
        size_factors_matrix = size_factors_matrix[:, valid_disp]

        n_genes_valid = counts.shape[1]
        if not self.quiet:
            logger.info(
                "Fitting GLM for %s/%s genes (filtered zero and invalid dispersions)...",
                n_genes_valid,
                n_genes,
            )

        # Convert lambda to natural log scale.
        lambda_nat_log_scale = np.repeat(1e-6, design_matrix.shape[1]) / (np.log(2) ** 2)

        # Fit GLM using the native implementation.
        glm_results = NegativeBinomialGLM.fit_negative_binomial_glm(
            y=counts,
            x=design_matrix,
            nf=size_factors_matrix,
            alpha_hat=dispersions,
            lambda_=lambda_nat_log_scale,
            use_weights=False,
            use_qr=False,  # Use standard matrix inversion
            minmu=self.min_mu,
            tol=self.beta_tol,
            maxit=self.max_iter,
            **self.glm_kwargs,
        )

        return self._process_glm_results(glm_results, n_genes, valid_genes, valid_disp, n_coefficients, data)

    def _process_glm_results(self, glm_results, n_genes, valid_genes, valid_disp, n_coefficients, data):
        """Process GLM results and compute Wald statistics."""
        beta_mat = glm_results["beta_mat"]
        beta_var_mat = glm_results["beta_var_mat"]
        converged = glm_results["iter"] < self.max_iter
        iterations = glm_results["iter"]

        # Convert beta coefficients to log2 scale.
        beta_mat_log2 = np.log2(np.exp(1)) * beta_mat
        beta_se_log2 = np.log2(np.exp(1)) * np.sqrt(np.maximum(beta_var_mat, 0))

        # Initialize result arrays
        beta_coefficients = np.full((n_genes, n_coefficients), np.nan)
        beta_se = np.full((n_genes, n_coefficients), np.nan)
        p_values = np.full((n_genes, n_coefficients), np.nan)
        wald_statistics = np.full((n_genes, n_coefficients), np.nan)
        converged_mask = np.full(n_genes, False)
        iteration_counts = np.zeros(n_genes, dtype=int)

        # Store variance matrix in AnnData for later use
        # Create full-size variance matrix (including genes with invalid dispersions)
        full_beta_var_mat = np.full((n_genes, n_coefficients), np.nan)
        full_beta_mat = np.full((n_genes, n_coefficients), np.nan)
        full_beta_cov_mat = np.full((n_genes, n_coefficients, n_coefficients), np.nan)  # Full covariance matrix

        valid_gene_indices = np.where(valid_genes)[0][valid_disp]
        full_beta_var_mat[valid_gene_indices] = beta_var_mat
        full_beta_mat[valid_gene_indices] = beta_mat
        full_beta_cov_mat[valid_gene_indices] = glm_results["beta_cov_mat"]  # Store full covariance matrix

        # Calculate Wald statistics for each coefficient
        for i, gene_idx in enumerate(valid_gene_indices):
            for j in range(n_coefficients):
                if not np.isnan(beta_mat_log2[i, j]) and not np.isnan(beta_se_log2[i, j]) and beta_se_log2[i, j] > 0:
                    wald_statistics[gene_idx, j] = beta_mat_log2[i, j] / beta_se_log2[i, j]
                    p_values[gene_idx, j] = 2 * (1 - norm.cdf(abs(wald_statistics[gene_idx, j])))
                else:
                    wald_statistics[gene_idx, j] = np.nan
                    p_values[gene_idx, j] = np.nan

            beta_coefficients[gene_idx] = beta_mat_log2[i]
            beta_se[gene_idx] = beta_se_log2[i]
            converged_mask[gene_idx] = converged[i]
            iteration_counts[gene_idx] = iterations[i]

        # Store in AnnData uns (unstructured data) for flexible storage
        data.uns["beta_var_mat"] = full_beta_var_mat

        # Also store the beta coefficients matrix
        data.uns["beta_mat"] = full_beta_mat

        # Store results
        self.results = {
            "beta_coefficients": beta_coefficients,
            "beta_se": beta_se,
            "p_values": p_values,
            "wald_statistics": wald_statistics,
            "beta_var_mat": full_beta_var_mat,
            "beta_mat": full_beta_mat,
            "beta_cov_mat": full_beta_cov_mat,  # Add full covariance matrix
            "converged": converged_mask,
            "iterations": iteration_counts,
            "metadata": {
                "n_genes": n_genes,
                "n_coefficients": n_coefficients,
                "valid_genes": np.sum(valid_genes),
                "valid_dispersions": np.sum(valid_disp),
            },
        }

        self.fitted = True
        return self

    def update_data(self, data: AnnData) -> None:
        """
        Update the AnnData object with test results.

        Parameters
        ----------
        data : AnnData
            AnnData object to update
        """
        if not self.results:
            raise ValueError("No results available. Run fit() first.")

        # Store the full results in data.uns for use by contrast analyzer
        data.uns["wald_results"] = self.results

        # Store matrices used by downstream contrast extraction.
        data.uns["beta_var_mat"] = self.results["beta_var_mat"]
        data.uns["beta_mat"] = self.results["beta_mat"]

        if not self.quiet:
            logger.info("Results stored in data.uns['wald_results']")

    def get_coefficient_results(self, coefficient_idx: int = 1) -> pd.DataFrame:
        """
        Get results for a specific coefficient.

        Parameters
        ----------
        coefficient_idx : int, default=1
            Index of the coefficient to extract (0=intercept, 1=first contrast, etc.)

        Returns
        -------
        pd.DataFrame
            DataFrame with results for the specified coefficient
        """
        if not self.results:
            raise ValueError("No results available. Run fit() first.")

        results_df = pd.DataFrame(
            {
                "logFC": self.results["beta_coefficients"][:, coefficient_idx],
                "lfcSE": self.results["beta_se"][:, coefficient_idx],
                "stat": self.results["wald_statistics"][:, coefficient_idx],
                "pvalue": self.results["p_values"][:, coefficient_idx],
            }
        )

        return results_df

    def _validate_data(self, data: AnnData) -> None:
        """Validate that required data is present in AnnData object."""
        if "dispersion" not in data.var:
            raise ValueError("Dispersion estimates not found. Run dispersion estimation first.")
        if "size_factors" not in data.obs:
            raise ValueError("Size factors not found. Run size factor estimation first.")
        if "design" not in data.obsm:
            raise ValueError("Design matrix not found. Create design matrix first.")
