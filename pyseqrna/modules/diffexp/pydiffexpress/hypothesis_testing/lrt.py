"""
Likelihood Ratio Test (LRT) for differential expression analysis.

This module implements the likelihood ratio test, which compares the fit of a full model
against a reduced model to test for significance of specific terms in the model.

Classes:
    - LRTAnalyzer: Likelihood Ratio Test (LRT) analyzer for differential expression

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List, Union
from anndata import AnnData
import warnings
from scipy.stats import chi2

from .base import BaseHypothesisTestAnalyzer
from ..utils.design_matrix import create_design_matrix

logger = logging.getLogger(__name__)


class LRTAnalyzer(BaseHypothesisTestAnalyzer):
    """
    Likelihood Ratio Test (LRT) analyzer for differential expression.
    """

    def __init__(
        self,
        reduced_formula: str = "~1",
        full_formula: Optional[str] = None,
        beta_tol: float = 1e-8,
        max_iter: int = 100,
        quiet: bool = False,
    ):
        super().__init__()
        self.reduced_formula = reduced_formula
        self.full_formula = full_formula
        self.beta_tol = beta_tol
        self.max_iter = max_iter
        self.quiet = quiet
        self._results = None
        self.fitted = False

    def fit(self, data: AnnData, design_matrix: Optional[np.ndarray] = None) -> "LRTAnalyzer":
        """
        Fit the LRT model.

        Parameters
        ----------
        data : AnnData
            Annotated data object
        design_matrix : Optional[np.ndarray]
            Design matrix. If None, will be created from data.

        Returns
        -------
        self : LRTAnalyzer
            Fitted analyzer
        """
        self._data = data

        # Get or create design matrix
        if design_matrix is not None:
            full_design = design_matrix
        else:
            full_design = self._build_design_from_formula(data, self.full_formula)

        # Store design column names for contrast analysis
        if "design_columns" in data.uns:
            self._design_columns = data.uns["design_columns"]
        else:
            # Try to get column names from design matrix stored in obsm
            if hasattr(data, "obsm") and "design" in data.obsm:
                design_obj = data.obsm["design"]
                if hasattr(design_obj, "columns"):
                    self._design_columns = list(design_obj.columns)
                else:
                    self._design_columns = [f"coef_{i}" for i in range(full_design.shape[1])]
            else:
                self._design_columns = [f"coef_{i}" for i in range(full_design.shape[1])]

        # Create reduced design matrix
        reduced_design = self._construct_reduced_design(data, full_design)

        # Get data
        counts = data.X  # samples x genes
        size_factors = data.obs["size_factors"].values
        dispersions = data.var["dispersion"].values

        # Get valid genes (non-zero counts and finite dispersions)
        valid_genes = (counts.sum(axis=0) > 0) & np.isfinite(dispersions) & (dispersions > 0)

        if not self.quiet:
            logger.info("Running LRT for %s/%s genes...", np.sum(valid_genes), len(valid_genes))

        # Convert to numpy arrays and ensure correct shapes
        size_factors_array = np.asarray(size_factors)
        np.broadcast_to(size_factors_array.reshape(-1, 1), (counts.shape[0], np.sum(valid_genes)))

        # Fit full model using statsmodels (like the working implementation)
        if not self.quiet:
            logger.info("Fitting full model...")

        import statsmodels.api as sm

        # Add prior count before fitting.
        prior_count = 0.125
        counts_adj = counts[:, valid_genes] + prior_count

        # Calculate offsets
        offsets = np.log(size_factors_array)

        # Fit models using statsmodels GLM
        valid_gene_indices = np.where(valid_genes)[0]
        lr_stat = np.zeros(len(valid_gene_indices))
        deviance_full = np.zeros(len(valid_gene_indices))
        deviance_reduced = np.zeros(len(valid_gene_indices))
        df_full = np.zeros(len(valid_gene_indices))
        df_reduced = np.zeros(len(valid_gene_indices))
        beta_coeffs = np.full((len(valid_gene_indices), full_design.shape[1]), np.nan)

        for i, gene_idx in enumerate(valid_gene_indices):
            try:
                # Fit full model
                model_full = sm.GLM(
                    counts_adj[:, i],
                    full_design,
                    family=sm.families.NegativeBinomial(alpha=dispersions[gene_idx]),
                    offset=offsets,
                )
                res_full = model_full.fit(method="newton", maxiter=100)

                # Fit reduced model
                model_reduced = sm.GLM(
                    counts_adj[:, i],
                    reduced_design,
                    family=sm.families.NegativeBinomial(alpha=dispersions[gene_idx]),
                    offset=offsets,
                )
                res_reduced = model_reduced.fit(method="newton", maxiter=100)

                # Calculate LRT statistic
                lr_stat[i] = res_reduced.deviance - res_full.deviance
                deviance_full[i] = res_full.deviance
                deviance_reduced[i] = res_reduced.deviance
                df_full[i] = res_full.df_resid
                df_reduced[i] = res_reduced.df_resid
                beta_coeffs[i, :] = res_full.params

            except Exception as e:
                if not self.quiet:
                    logger.warning("LRT failed for gene %s: %s", gene_idx, e)
                lr_stat[i] = np.nan
                deviance_full[i] = np.nan
                deviance_reduced[i] = np.nan
                df_full[i] = np.nan
                df_reduced[i] = np.nan
                beta_coeffs[i, :] = np.nan

        # Clip negative values for numerical stability
        if np.any(lr_stat < 0):
            warnings.warn(f"{np.sum(lr_stat < 0)} negative LR statistics were clipped to 0 (numerical stability).")
            lr_stat = np.clip(lr_stat, 0, None)

        # Calculate degrees of freedom (like the working implementation)
        df_test_vector = df_reduced - df_full

        # p-values
        p_values = chi2.sf(lr_stat, df=df_test_vector)

        n_genes = data.n_vars
        all_lr_stat = np.full(n_genes, np.nan)
        all_p_values = np.full(n_genes, np.nan)
        all_full_deviance = np.full(n_genes, np.nan)
        all_reduced_deviance = np.full(n_genes, np.nan)
        all_df_test = np.full(n_genes, np.nan)

        all_lr_stat[valid_gene_indices] = lr_stat
        all_p_values[valid_gene_indices] = p_values
        all_full_deviance[valid_gene_indices] = deviance_full
        all_reduced_deviance[valid_gene_indices] = deviance_reduced
        all_df_test[valid_gene_indices] = df_test_vector

        all_beta_coeffs = np.full((n_genes, full_design.shape[1]), np.nan)
        all_beta_coeffs[valid_gene_indices, :] = beta_coeffs

        self._results = {
            "beta_coefficients": all_beta_coeffs,
            "p_values": all_p_values,
            "wald_statistics": all_lr_stat,
            "deviance": all_full_deviance,
            "reduced_deviance": all_reduced_deviance,
            "converged": np.full(n_genes, True),
            "iterations": np.full(n_genes, 1),
            "metadata": {
                "test": "LRT",
                "reduced_formula": self.reduced_formula,
                "df_test": all_df_test,  # Now stores the vector
                "fit_type": "parametric",
                "sf_type": "ratio",
            },
        }

        self.fitted = True
        return self

    def _construct_reduced_design(self, data: AnnData, full_design: np.ndarray) -> np.ndarray:
        """
        Construct reduced design matrix by removing condition terms.

        For a design matrix with intercept + condition terms, we remove the condition terms
        to create an intercept-only model.
        """
        if full_design.shape[1] > 1:
            # Keep only the intercept column (first column)
            return full_design[:, :1]
        else:
            raise ValueError("Cannot construct reduced model: full model has only intercept")

    def _build_design_from_formula(self, data: AnnData, formula: str) -> np.ndarray:
        sample_metadata = data.obs.copy()
        design_matrix, design_columns = create_design_matrix(sample_metadata=sample_metadata, design_formula=formula)

        # Store design column names for contrast analysis
        self._design_columns = design_columns

        return design_matrix

    def get_results(self, contrast: Optional[Union[str, List[str], np.ndarray]] = None) -> pd.DataFrame:
        """
        Get LRT results as a DataFrame.

        Parameters
        ----------
        contrast : Optional[Union[str, List[str], np.ndarray]]
            Contrast specification. If None, returns overall LRT results.
            If specified, performs LRT for the specific contrast.

        Returns
        -------
        pd.DataFrame
            DataFrame with LRT results including:
            - lr_stat: Likelihood ratio statistic
            - p_values: P-values
            - df_test: Degrees of freedom for the test
            - beta_coefficients: Model coefficients (logFC)
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before getting results")

        if contrast is None:
            # Create results DataFrame with all available information
            results_dict = {
                "lr_stat": self._results["wald_statistics"],  # LRT statistics
                "p_values": self._results["p_values"],
                "df_test": self._results["metadata"]["df_test"],
            }

            # Add beta coefficients (logFC) if available
            if "beta_coefficients" in self._results:
                beta_coeffs = self._results["beta_coefficients"]
                # Add each coefficient as a separate column
                for i in range(beta_coeffs.shape[1]):
                    results_dict[f"beta_{i}"] = beta_coeffs[:, i]

            results = pd.DataFrame(results_dict)

            if hasattr(self, "_data") and self._data is not None:
                results.index = self._data.var_names
            return results
        else:
            return self._get_contrast_results(contrast)

    def _get_contrast_results(self, contrast: Union[str, List[str], np.ndarray]) -> pd.DataFrame:
        """
        Get LRT results for a specific contrast.

        LRT is a global test (full vs reduced model), so p-values are from
        the overall test. The logFC for the requested contrast is extracted
        from the beta coefficients.
        """
        beta_coeffs = self._results.get("beta_coefficients")
        design_cols = getattr(self, "_design_columns", [])

        # Resolve contrast to a coefficient index
        logfc = np.zeros(len(self._results["p_values"]))
        if beta_coeffs is not None:
            if isinstance(contrast, str) and contrast in design_cols:
                idx = design_cols.index(contrast)
                logfc = beta_coeffs[:, idx]
            elif isinstance(contrast, np.ndarray) and len(contrast) == beta_coeffs.shape[1]:
                logfc = np.dot(beta_coeffs, contrast)
            elif isinstance(contrast, list) and len(contrast) == beta_coeffs.shape[1]:
                logfc = np.dot(beta_coeffs, np.array(contrast, dtype=float))

        results = pd.DataFrame(
            {
                "logFC": logfc,
                "lr_stat": self._results["wald_statistics"],
                "p_values": self._results["p_values"],
                "df_test": self._results["metadata"]["df_test"],
            }
        )
        if hasattr(self, "_data") and self._data is not None:
            results.index = self._data.var_names
        return results

    def get_test_statistics(self) -> Dict[str, np.ndarray]:
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before getting test statistics")
        return {
            "lr_stat": self._results["wald_statistics"],
            "p_values": self._results["p_values"],
            "df_test": self._results["metadata"]["df_test"],
            "full_deviance": self._results["deviance"],
            "reduced_deviance": self._results["reduced_deviance"],
        }

    def get_model_comparison(self) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before getting model comparison")
        return {
            "full_model_params": self._results["beta_coefficients"].shape[1],
            "reduced_model_params": self._results["beta_coefficients"].shape[1] - self._results["metadata"]["df_test"],
            "df_test": self._results["metadata"]["df_test"],
            "reduced_formula": self._results["metadata"]["reduced_formula"],
        }

    def update_data(self, data: AnnData) -> None:
        """
        Update AnnData object with LRT results.

        Parameters
        ----------
        data : AnnData
            Annotated data object to update
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before updating data")

        # Store design column names for contrast analysis
        if hasattr(self, "_design_columns"):
            data.uns["design_columns"] = self._design_columns

        # Store results in AnnData
        if "analysis_results" not in data.uns:
            data.uns["analysis_results"] = {}

        # Update with LRT results
        data.uns["analysis_results"].update(self._results)

        # Store coefficient-level results for downstream inspection.
        if "beta_coefficients" in self._results:
            beta_coeffs = self._results["beta_coefficients"]
            if hasattr(self, "_design_columns") and len(self._design_columns) == beta_coeffs.shape[1]:
                for i, col_name in enumerate(self._design_columns):
                    data.var[col_name] = beta_coeffs[:, i]

        if "p_values" in self._results:
            data.var["LRT_p_value"] = self._results["p_values"]

        if "wald_statistics" in self._results:
            data.var["LRT_statistic"] = self._results["wald_statistics"]

        if "metadata" in self._results and "df_test" in self._results["metadata"]:
            data.var["LRT_df"] = self._results["metadata"]["df_test"]
