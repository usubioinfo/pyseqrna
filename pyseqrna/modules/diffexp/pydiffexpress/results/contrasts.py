"""
Contrast analysis for differential expression results.

This module provides tools for extracting results for specific contrasts
from differential expression analysis.

Classes:
    - ContrastAnalyzer: Analyzer for extracting results for specific contrasts

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import List, Optional, Union, Tuple
import logging
import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


class ContrastAnalyzer:
    """
    Analyzer for extracting results for specific contrasts.

    This class provides methods to extract differential expression results
    for specific contrasts from a fitted differential expression analysis.
    """

    def __init__(self):
        """Initialize the contrast analyzer."""
        pass

    def extract_contrast(
        self,
        data: AnnData,
        contrast: Union[str, List[str], np.ndarray],
        name: Optional[str] = None,
        lfc_threshold: float = 0.0,
        alpha: float = 1.0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract results for a specific contrast.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results
        contrast : Union[str, List[str], np.ndarray]
            Contrast specification. Can be:
            - String: Column name in design matrix
            - List of strings: Multiple column names
            - numpy array: Custom contrast vector
        name : Optional[str]
            Name for the contrast (used in output)
        lfc_threshold : float
            Log fold change threshold for filtering
        alpha : float
            Significance level for filtering
        **kwargs
            Additional parameters

        Returns
        -------
        pd.DataFrame
            Results table for the specified contrast
        """
        # Check if analysis results exist
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        data.uns["analysis_results"]

        # Get coefficient names from design matrix
        if "design" in data.obsm:
            design_cols = self._get_design_column_names(data)
        else:
            raise ValueError("No design matrix found")

        # Handle different contrast specifications
        if isinstance(contrast, str):
            # Single column name or contrast specification
            if "_vs_" in contrast:
                # This is a contrast specification like "condition_M1_vs_V1"
                if contrast.startswith("condition_"):
                    # Standard format like "condition_M1_vs_V1"
                    contrast_idx = self._get_contrast_index(contrast, design_cols)

                    if contrast_idx == -1:
                        # This is a computed contrast like "condition_M1_vs_V1"
                        # Parse the contrast name to get the individual contrasts
                        parts = contrast.split("_vs_")
                        factor_part = parts[0]
                        denom_level = parts[1]
                        factor_parts = factor_part.split("_")
                        factor_name = "_".join(factor_parts[:-1])
                        num_level = factor_parts[-1]

                        # Create contrast vector: num_contrast - denom_contrast
                        contrast_vector = np.zeros(len(design_cols))
                        num_contrast = f"{factor_name}_{num_level}_vs_A1"
                        denom_contrast = f"{factor_name}_{denom_level}_vs_A1"

                        num_idx = design_cols.index(num_contrast)
                        denom_idx = design_cols.index(denom_contrast)

                        contrast_vector[num_idx] = 1
                        contrast_vector[denom_idx] = -1
                    else:
                        # Direct contrast
                        contrast_vector = np.zeros(len(design_cols))
                        contrast_vector[contrast_idx] = 1.0
                else:
                    # This shouldn't happen with current logic, but handle it
                    contrast_idx = self._get_contrast_index(contrast, design_cols)
                    contrast_vector = np.zeros(len(design_cols))
                    contrast_vector[contrast_idx] = 1.0
            elif "-" in contrast:
                # Simple format like "A1-V1" or "M1-V1"
                # Parse as "numerator-denominator" format
                parts = contrast.split("-")
                if len(parts) == 2:
                    num_level = parts[0]
                    denom_level = parts[1]

                    # Create contrast vector
                    contrast_vector = np.zeros(len(design_cols))

                    # Dynamically detect the reference level and factor name
                    ref_level = self._find_reference_level(design_cols, "condition")
                    if ref_level is None:
                        # Fallback: try to find any reference level
                        ref_level = self._find_any_reference_level(design_cols)

                    if ref_level is None:
                        raise ValueError(f"Could not determine reference level from design matrix columns: {design_cols}")

                    # Check if this is a comparison against the reference level
                    if denom_level == ref_level:
                        # Direct comparison against reference level
                        direct_contrast = f"condition_{num_level}_vs_{ref_level}"
                        if direct_contrast in design_cols:
                            contrast_idx = design_cols.index(direct_contrast)
                            contrast_vector[contrast_idx] = 1.0
                        else:
                            raise ValueError(f"Contrast '{direct_contrast}' not found in design matrix columns: {design_cols}")

                    elif num_level == ref_level:
                        # Comparison of reference level against another level
                        # This is the negative of the other contrast
                        direct_contrast = f"condition_{denom_level}_vs_{ref_level}"
                        if direct_contrast in design_cols:
                            contrast_idx = design_cols.index(direct_contrast)
                            contrast_vector[contrast_idx] = -1.0
                        else:
                            raise ValueError(f"Contrast '{direct_contrast}' not found in design matrix columns: {design_cols}")

                    else:
                        # Comparison between two non-reference levels
                        # Create contrast vector: num_contrast - denom_contrast
                        num_contrast = f"condition_{num_level}_vs_{ref_level}"
                        denom_contrast = f"condition_{denom_level}_vs_{ref_level}"

                        # Check if both contrasts exist in design matrix
                        if num_contrast in design_cols and denom_contrast in design_cols:
                            num_idx = design_cols.index(num_contrast)
                            denom_idx = design_cols.index(denom_contrast)

                            contrast_vector[num_idx] = 1
                            contrast_vector[denom_idx] = -1
                        else:
                            raise ValueError(
                                f"Contrasts '{num_contrast}' and/or '{denom_contrast}' not found in design matrix columns: {design_cols}"
                            )
                else:
                    raise ValueError(f"Invalid contrast format '{contrast}'. Expected format like 'A1-V1' or 'M1-V1'")
            else:
                # Direct column name
                contrast_idx = self._get_contrast_index(contrast, design_cols)
                contrast_vector = np.zeros(len(design_cols))
                contrast_vector[contrast_idx] = 1.0

        elif isinstance(contrast, list):
            # List format like ["condition", "M1", "V1"] or ["condition", "M1", "A1"]
            if len(contrast) == 3:
                factor_name = contrast[0]
                num_level = contrast[1]
                denom_level = contrast[2]

                # Create contrast vector
                contrast_vector = np.zeros(len(design_cols))

                # Check if this is a comparison against the reference level
                # (e.g., ["condition", "M1", "A1"] should extract condition_M1_vs_A1 directly)
                if denom_level == "A1" or denom_level == "A1A" or denom_level == "A1B":
                    # Direct comparison against reference level
                    direct_contrast = f"{factor_name}_{num_level}_vs_A1"
                    if direct_contrast in design_cols:
                        contrast_idx = design_cols.index(direct_contrast)
                        contrast_vector[contrast_idx] = 1.0
                    else:
                        raise ValueError(f"Contrast '{direct_contrast}' not found in design matrix columns: {design_cols}")

                elif num_level == "A1" or num_level == "A1A" or num_level == "A1B":
                    # Comparison of reference level against another level
                    # This is the negative of the other contrast
                    direct_contrast = f"{factor_name}_{denom_level}_vs_A1"
                    if direct_contrast in design_cols:
                        contrast_idx = design_cols.index(direct_contrast)
                        contrast_vector[contrast_idx] = -1.0
                    else:
                        raise ValueError(f"Contrast '{direct_contrast}' not found in design matrix columns: {design_cols}")

                else:
                    # Comparison between two non-reference levels
                    # Create contrast vector: num_contrast - denom_contrast
                    num_contrast = f"{factor_name}_{num_level}_vs_A1"
                    denom_contrast = f"{factor_name}_{denom_level}_vs_A1"

                    # Check if both contrasts exist in design matrix
                    if num_contrast in design_cols and denom_contrast in design_cols:
                        num_idx = design_cols.index(num_contrast)
                        denom_idx = design_cols.index(denom_contrast)

                        contrast_vector[num_idx] = 1
                        contrast_vector[denom_idx] = -1
                    else:
                        raise ValueError(
                            f"Contrasts '{num_contrast}' and/or '{denom_contrast}' not found in design matrix columns: {design_cols}"
                        )
            else:
                # Multiple design-column names.
                contrast_vector = np.zeros(len(design_cols))
                for col in contrast:
                    idx = self._get_contrast_index(col, design_cols)
                    contrast_vector[idx] = 1.0

        elif isinstance(contrast, np.ndarray):
            # Custom contrast vector
            if len(contrast) != len(design_cols):
                raise ValueError(f"Contrast vector length {len(contrast)} must match design matrix columns {len(design_cols)}")
            contrast_vector = contrast

        else:
            raise ValueError("Invalid contrast specification")

        # Extract results for this contrast
        result_df = self._extract_contrast_results(data, contrast_vector, name)

        # Keep all-zero contrast rows stable across the relevant samples.
        self._apply_contrast_all_zero(result_df, data, contrast, contrast_vector)

        # No filtering - always return all genes
        return result_df

    def _resolve_contrast_groups(
        self,
        contrast: Union[str, List[str], np.ndarray],
    ) -> Optional[Tuple[str, str, str]]:
        """Resolve contrast to (factor_name, numerator_level, denominator_level)."""
        if isinstance(contrast, str) and "-" in contrast:
            parts = contrast.split("-")
            if len(parts) == 2:
                return ("condition", parts[0], parts[1])

        if isinstance(contrast, list) and len(contrast) == 3:
            return (str(contrast[0]), str(contrast[1]), str(contrast[2]))

        return None

    def _contrast_all_zero_character(
        self,
        data: AnnData,
        factor_name: str,
        numerator_level: str,
        denominator_level: str,
    ) -> np.ndarray:
        """Detect all-zero genes for character-style contrasts."""
        if factor_name not in data.obs.columns:
            return np.zeros(data.n_vars, dtype=bool)

        counts = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
        factor = data.obs[factor_name].astype(str)
        which_samples = factor.isin([str(numerator_level), str(denominator_level)]).values
        if not np.any(which_samples):
            return np.zeros(data.n_vars, dtype=bool)

        counts_sub = counts[which_samples, :]
        return np.sum(counts_sub == 0, axis=0) == counts_sub.shape[0]

    def _contrast_all_zero_numeric(
        self,
        data: AnnData,
        contrast_vector: np.ndarray,
    ) -> np.ndarray:
        """Detect all-zero genes for numeric contrast vectors."""
        design_matrix = data.obsm.get("design")
        if design_matrix is None:
            return np.zeros(data.n_vars, dtype=bool)

        design_matrix = np.asarray(design_matrix)
        if np.all(contrast_vector >= 0) or np.all(contrast_vector <= 0):
            return np.zeros(data.n_vars, dtype=bool)

        counts = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
        contrast_binary = np.where(contrast_vector == 0, 0, 1)
        which_samples = np.where(design_matrix @ contrast_binary == 0, 0, 1)
        zero_test = counts.T @ which_samples
        return zero_test == 0

    def _apply_contrast_all_zero(
        self,
        result_df: pd.DataFrame,
        data: AnnData,
        contrast: Union[str, List[str], np.ndarray],
        contrast_vector: np.ndarray,
    ) -> None:
        """Apply all-zero contrast handling."""
        resolved = self._resolve_contrast_groups(contrast)
        if resolved is not None:
            factor_name, numerator_level, denominator_level = resolved
            contrast_all_zero = self._contrast_all_zero_character(data, factor_name, numerator_level, denominator_level)
        else:
            contrast_all_zero = self._contrast_all_zero_numeric(data, contrast_vector)

        all_zero = data.var.get("all_zero", pd.Series(False, index=data.var_names))
        if hasattr(all_zero, "values"):
            all_zero = all_zero.values

        final_mask = contrast_all_zero & ~np.asarray(all_zero, dtype=bool)
        if not np.any(final_mask):
            return

        zero_genes = data.var_names[final_mask].astype(str)
        result_df.loc[zero_genes, "logFC"] = 0.0
        result_df.loc[zero_genes, "stat"] = 0.0
        result_df.loc[zero_genes, "pvalue"] = 1.0
        result_df.loc[zero_genes, "FDR"] = np.nan

    def _get_design_column_names(self, data: AnnData) -> List[str]:
        """Get column names from design matrix."""
        # Try to get column names from AnnData uns
        if "design_column_names" in data.uns:
            return data.uns["design_column_names"]
        elif "design_columns" in data.uns:
            return data.uns["design_columns"]

        # Try to get column names from design DataFrame
        if "design" in data.obsm and hasattr(data.obsm["design"], "columns"):
            return list(data.obsm["design"].columns)

        # Fallback to generic names
        design_matrix = data.obsm["design"]
        return [f"col_{i}" for i in range(design_matrix.shape[1])]

    def _get_contrast_index(self, contrast_name: str, design_cols: List[str]) -> int:
        """Get index of contrast column in design matrix."""
        try:
            return design_cols.index(contrast_name)
        except ValueError:
            # Try to parse as a contrast like "condition_M1_vs_V1"
            if "_vs_" in contrast_name:
                parts = contrast_name.split("_vs_")
                if len(parts) == 2:
                    factor_part = parts[0]
                    denom_level = parts[1]

                    # Find the factor name (everything before the last underscore)
                    factor_parts = factor_part.split("_")
                    if len(factor_parts) >= 2:
                        factor_name = "_".join(factor_parts[:-1])
                        num_level = factor_parts[-1]

                        # Look for the two individual contrasts
                        num_contrast = f"{factor_name}_{num_level}_vs_A1"
                        denom_contrast = f"{factor_name}_{denom_level}_vs_A1"

                        if num_contrast in design_cols and denom_contrast in design_cols:
                            # This is a valid contrast that can be computed
                            # Return a special value to indicate this is a computed contrast
                            return -1  # Special value for computed contrast

            raise ValueError(f"Contrast '{contrast_name}' not found in design matrix columns: {design_cols}")

    def _extract_contrast_results(self, data: AnnData, contrast_vector: np.ndarray, name: Optional[str]) -> pd.DataFrame:
        """Extract results for a specific contrast vector."""
        # Get analysis results
        results = data.uns["analysis_results"]

        # Detect test type
        test_type = results.get("metadata", {}).get("test", "Wald")

        if test_type == "LRT":
            # For LRT, use the pre-computed LRT statistics and p-values
            # LRT is a global test, so we return the overall LRT results for the contrast
            if "wald_statistics" in results:
                statistics = results["wald_statistics"]  # These are actually LRT statistics
            else:
                statistics = np.zeros(data.n_vars)

            if "p_values" in results:
                p_values = results["p_values"]  # These are LRT p-values
            else:
                p_values = np.ones(data.n_vars)

            # For LRT, we don't have standard errors for individual contrasts
            # since LRT tests the overall model comparison
            lfc_se = np.zeros(data.n_vars)
            lfc = np.zeros(data.n_vars)

        else:
            # Wald test - use pre-calculated results from our WaldTestAnalyzer
            # Our Wald test already calculates logFC and standard errors for each coefficient
            # We just need to extract the appropriate coefficient based on the contrast vector

            # Find which coefficient this contrast corresponds to
            # For simple contrasts like "condition_M1_vs_A1", this is just finding the coefficient index
            coefficient_idx = None
            for i, val in enumerate(contrast_vector):
                if abs(val) > 0.5:  # This coefficient is part of the contrast
                    if coefficient_idx is None:
                        coefficient_idx = i
                    else:
                        # This is a complex contrast involving multiple coefficients
                        # We'll need to calculate it from the individual coefficients
                        coefficient_idx = None
                        break

            if coefficient_idx is not None:
                # Simple contrast - extract the pre-calculated results for this coefficient
                if "beta_coefficients" in results and "beta_se" in results:
                    # Get the sign of the contrast vector for this coefficient
                    contrast_sign = contrast_vector[coefficient_idx]

                    # Apply the sign to the results
                    lfc = contrast_sign * results["beta_coefficients"][:, coefficient_idx]
                    lfc_se = abs(contrast_sign) * results["beta_se"][:, coefficient_idx]
                    statistics = contrast_sign * results["wald_statistics"][:, coefficient_idx]
                    p_values = results["p_values"][:, coefficient_idx]  # p-values don't change sign
                else:
                    # Fallback to zeros
                    lfc = np.zeros(data.n_vars)
                    lfc_se = np.zeros(data.n_vars)
                    statistics = np.zeros(data.n_vars)
                    p_values = np.ones(data.n_vars)
            else:
                # Complex contrast - calculate from individual coefficients
                # This is the case for contrasts like "condition_M1_vs_V1" which involves
                # subtracting two coefficients: M1_vs_A1 - V1_vs_A1

                if "beta_coefficients" in results and "beta_se" in results:
                    beta_coeffs = results["beta_coefficients"]
                    beta_se = results["beta_se"]

                    # Calculate log fold change for this contrast
                    lfc = np.dot(beta_coeffs, contrast_vector)

                    # Calculate standard error using the stored covariance matrix
                    if "beta_cov_mat" in results:
                        beta_cov_mat = results["beta_cov_mat"]  # Shape: (n_genes, n_coefficients, n_coefficients)
                        n_genes = beta_cov_mat.shape[0]
                        lfc_se = np.full(n_genes, np.nan)

                        for i in range(n_genes):
                            if not np.isnan(beta_cov_mat[i, 0, 0]):  # Check if gene has valid covariance matrix
                                # The covariance matrix is in natural log scale, but we need log2 scale
                                # Convert the covariance matrix to log2 scale
                                log2_factor = np.log2(np.exp(1))
                                cov_matrix_log2 = (log2_factor**2) * beta_cov_mat[i, :, :]

                                # Calculate contrast SE using the log2-scale covariance matrix
                                contrast_se_log2 = np.sqrt(contrast_vector.T @ cov_matrix_log2 @ contrast_vector)
                                lfc_se[i] = contrast_se_log2
                    else:
                        # Fallback to simplified formula if no covariance matrix available
                        lfc_se = np.sqrt(np.sum((contrast_vector**2)[None, :] * (beta_se**2), axis=1))

                    # Calculate Wald statistics only for genes with valid
                    # contrast estimates while preserving stable signs
                    # for all-zero/invalid genes, which should remain NA
                    # instead of being converted to stat=0, pvalue=1.
                    statistics = np.full(data.n_vars, np.nan)
                    valid_stats = np.isfinite(lfc) & np.isfinite(lfc_se) & (lfc_se > 1e-10)
                    statistics[valid_stats] = lfc[valid_stats] / lfc_se[valid_stats]

                    # Calculate p-values for valid genes only
                    from scipy.stats import norm

                    p_values = np.full(data.n_vars, np.nan)
                    p_values[valid_stats] = 2 * (1 - norm.cdf(np.abs(statistics[valid_stats])))
                    p_values[valid_stats] = np.clip(p_values[valid_stats], 0, 1)
                else:
                    # Fallback to zeros
                    lfc = np.zeros(data.n_vars)
                    lfc_se = np.zeros(data.n_vars)
                    statistics = np.zeros(data.n_vars)
                    p_values = np.ones(data.n_vars)

        # Create results DataFrame
        base_means = data.var["base_mean"] if "base_mean" in data.var.columns else np.zeros(data.n_vars)
        result_df = pd.DataFrame(
            {
                "Gene": data.var_names,
                "baseMean": base_means,
                "logFC": lfc,
                "lfcSE": lfc_se,
                "stat": statistics,
                "pvalue": p_values,
                "FDR": self._calculate_adjusted_pvalues(p_values, base_means),
            }
        )

        # Set gene names as index
        result_df.set_index("Gene", inplace=True)

        # Add contrast name if provided
        if name:
            result_df.name = name

        return result_df

    def _calculate_adjusted_pvalues(
        self,
        p_values: np.ndarray,
        base_means: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        n_thresholds: int = 20,
    ) -> np.ndarray:
        """
        Calculate adjusted p-values using Benjamini-Hochberg correction with independent filtering.

        Parameters
        ----------
        p_values : np.ndarray
            Raw p-values
        base_means : np.ndarray, optional
            Base mean expression values for independent filtering
        alpha : float
            FDR threshold for optimizing independent filtering
        n_thresholds : int
            Number of candidate thresholds to evaluate for filtering

        Returns
        -------
        np.ndarray
            Adjusted p-values (NaN for filtered-out genes)
        """
        # Import here to avoid dependency if not used elsewhere
        from statsmodels.stats.multitest import multipletests

        n = len(p_values)
        adjusted_pvalues = np.full(n, np.nan)

        # Mask valid p-values
        valid_mask = ~np.isnan(p_values)
        valid_p_values = p_values[valid_mask]

        if len(valid_p_values) == 0:
            return adjusted_pvalues

        # Default: no filtering
        filter_mask = np.ones_like(valid_p_values, dtype=bool)

        # Optional independent filtering.
        if base_means is not None:
            valid_base_means = base_means[valid_mask]
            thresholds = np.quantile(valid_base_means, np.linspace(0, 0.5, n_thresholds))
            best_filter_mask = filter_mask
            max_rejections = -1
            for thresh in thresholds:
                candidate_mask = valid_base_means >= thresh
                candidate_p = valid_p_values[candidate_mask]
                if len(candidate_p) == 0:
                    continue
                # Use multipletests for BH correction
                _, candidate_padj, _, _ = multipletests(candidate_p, method="fdr_bh")
                rejections = np.sum(candidate_padj <= alpha)
                if rejections > max_rejections:
                    max_rejections = rejections
                    best_filter_mask = candidate_mask
            filter_mask = best_filter_mask

        # Final BH correction for chosen filter
        filtered_p_values = valid_p_values[filter_mask]
        if len(filtered_p_values) > 0:
            _, filtered_adjusted, _, _ = multipletests(filtered_p_values, method="fdr_bh")
            valid_adjusted = np.full(len(valid_p_values), np.nan)
            valid_adjusted[filter_mask] = filtered_adjusted
            adjusted_pvalues[valid_mask] = valid_adjusted

        return adjusted_pvalues

    def _filter_results(self, result_df: pd.DataFrame, lfc_threshold: float, alpha: float) -> pd.DataFrame:
        """Filter results based on log fold change and significance thresholds."""
        # Apply log fold change threshold
        if lfc_threshold > 0:
            result_df = result_df[abs(result_df["logFC"]) >= lfc_threshold]

        # Apply significance threshold
        if alpha < 1.0:
            result_df = result_df[result_df["padj"] <= alpha]

        return result_df

    def get_significant_genes(
        self,
        data: AnnData,
        contrast: Union[str, List[str], np.ndarray],
        lfc_threshold: float = 1.0,
        alpha: float = 0.05,
        direction: str = "both",
    ) -> pd.DataFrame:
        """
        Get significant genes for a specific contrast.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results
        contrast : Union[str, List[str], np.ndarray]
            Contrast specification
        lfc_threshold : float
            Log fold change threshold for filtering
        alpha : float
            Significance level for filtering
        direction : str
            Direction of regulation: "up", "down", or "both"

        Returns
        -------
        pd.DataFrame
            Significant genes for the specified contrast
        """
        # Extract results for the contrast
        result_df = self.extract_contrast(data, contrast, lfc_threshold=lfc_threshold, alpha=alpha)

        # Filter by direction
        if direction == "up":
            result_df = result_df[result_df["logFC"] > lfc_threshold]
        elif direction == "down":
            result_df = result_df[result_df["logFC"] < -lfc_threshold]
        # For "both", no additional filtering needed

        return result_df

    def get_all_contrasts_results(
        self,
        data: AnnData,
        contrasts: Optional[List[Union[str, List[str], np.ndarray]]] = None,
        lfc_threshold: float = 0.0,
        alpha: float = 1.0,
        include_base_stats: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get results for all contrasts combined in one DataFrame.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results
        contrasts : Optional[List[Union[str, List[str], np.ndarray]]]
            List of contrasts to include. If None, will generate all possible contrasts.
        lfc_threshold : float
            Log fold change threshold for filtering
        alpha : float
            Significance level for filtering
        include_base_stats : bool
            Whether to include base statistics (baseMean, etc.) in the output
        **kwargs
            Additional parameters passed to extract_contrast

        Returns
        -------
        pd.DataFrame
            Combined results table with all contrasts
        """
        # Check if analysis results exist
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        # Get design columns to determine available contrasts
        if "design" in data.obsm:
            design_cols = self._get_design_column_names(data)
        else:
            raise ValueError("No design matrix found")

        # If no contrasts specified, generate all possible contrasts
        if contrasts is None:
            contrasts = self._generate_all_contrasts(design_cols)

        # Extract results for each contrast
        all_results = []

        for contrast in contrasts:
            try:
                # Get contrast name
                contrast_name = self._get_contrast_name(contrast, design_cols)

                # Extract results for this contrast
                result_df = self.extract_contrast(
                    data,
                    contrast,
                    name=contrast_name,
                    lfc_threshold=lfc_threshold,
                    alpha=alpha,
                    **kwargs,
                )

                # Select columns to include
                if include_base_stats:
                    # Include base statistics and contrast-specific results
                    if "baseMean" in result_df.columns:
                        result_for_concat = result_df[["baseMean", "logFC", "lfcSE", "stat", "pvalue", "FDR"]].copy()
                    else:
                        # For LRT results, baseMean might not be available
                        available_cols = ["logFC", "lfcSE", "stat", "pvalue", "FDR"]
                        available_cols = [col for col in available_cols if col in result_df.columns]
                        result_for_concat = result_df[available_cols].copy()
                else:
                    # Only include contrast-specific results
                    contrast_cols = ["logFC", "lfcSE", "stat", "pvalue", "FDR"]
                    contrast_cols = [col for col in contrast_cols if col in result_df.columns]
                    result_for_concat = result_df[contrast_cols].copy()

                # Reset index to avoid conflicts
                result_for_concat.reset_index(drop=True, inplace=True)

                # Add comparison names to column names
                result_for_concat.columns = [s + "(" + contrast_name + ")" for s in result_for_concat.columns]

                all_results.append(result_for_concat)

            except Exception as e:
                logger.warning("Failed to extract results for contrast %s: %s", contrast, e)
                continue

        if not all_results:
            raise ValueError("No valid contrasts could be processed")

        # Combine all results
        combined_results = pd.concat(all_results, axis=1)

        # Add gene names as index if available
        if hasattr(data, "var_names") and len(data.var_names) == len(combined_results):
            combined_results.index = data.var_names

        return combined_results

    def _generate_all_contrasts(self, design_cols: List[str]) -> List[str]:
        """
        Generate all possible contrasts from design columns.

        Parameters
        ----------
        design_cols : List[str]
            List of design column names

        Returns
        -------
        List[str]
            List of all possible contrast specifications
        """
        contrasts = []

        # Add direct contrasts (excluding intercept)
        for col in design_cols:
            if col != "Intercept":
                contrasts.append(col)

        # Generate computed contrasts (e.g., M1_vs_V1)
        factor_contrasts = {}

        for col in design_cols:
            if col != "Intercept" and "_vs_" in col:
                parts = col.split("_vs_")
                if len(parts) == 2:
                    factor_part = parts[0]
                    denom_level = parts[1]
                    factor_parts = factor_part.split("_")
                    factor_name = "_".join(factor_parts[:-1])
                    num_level = factor_parts[-1]

                    if factor_name not in factor_contrasts:
                        factor_contrasts[factor_name] = []
                    factor_contrasts[factor_name].append((num_level, denom_level))

        # Generate all pairwise contrasts within each factor
        for factor_name, levels in factor_contrasts.items():
            level_names = [level[0] for level in levels] + [levels[0][1]]  # Add denominator level
            level_names = list(set(level_names))  # Remove duplicates

            for i, level1 in enumerate(level_names):
                for level2 in level_names[i + 1 :]:
                    if level1 != level2:
                        contrast_name = f"{factor_name}_{level1}_vs_{level2}"
                        contrasts.append(contrast_name)

        return contrasts

    def _get_contrast_name(self, contrast: Union[str, List[str], np.ndarray], design_cols: List[str]) -> str:
        """
        Get a descriptive name for a contrast.

        Parameters
        ----------
        contrast : Union[str, List[str], np.ndarray]
            Contrast specification
        design_cols : List[str]
            List of design column names

        Returns
        -------
        str
            Descriptive name for the contrast
        """
        if isinstance(contrast, str):
            return contrast
        elif isinstance(contrast, list):
            return "_".join(contrast)
        elif isinstance(contrast, np.ndarray):
            # For array contrasts, create a descriptive name
            non_zero_idx = np.nonzero(contrast)[0]
            if len(non_zero_idx) == 1:
                return design_cols[non_zero_idx[0]]
            elif len(non_zero_idx) == 2:
                col1 = design_cols[non_zero_idx[0]]
                col2 = design_cols[non_zero_idx[1]]
                return f"{col1}_vs_{col2}"
            else:
                return f"contrast_{len(non_zero_idx)}_terms"
        else:
            return "unknown_contrast"

    def _find_reference_level(self, design_cols: List[str], factor_name: str) -> Optional[str]:
        """
        Find the reference level for a given factor from design matrix columns.

        Parameters
        ----------
        design_cols : List[str]
            List of design matrix column names
        factor_name : str
            Name of the factor (e.g., "condition")

        Returns
        -------
        Optional[str]
            Reference level name, or None if not found
        """
        # Look for columns that match the pattern "{factor_name}_{level}_vs_{ref_level}"
        pattern = f"{factor_name}_"
        matching_cols = [col for col in design_cols if col.startswith(pattern) and "_vs_" in col]

        if not matching_cols:
            return None

        # Extract all reference levels from the matching columns
        ref_levels = set()
        for col in matching_cols:
            if "_vs_" in col:
                ref_level = col.split("_vs_")[1]
                ref_levels.add(ref_level)

        # If we have multiple reference levels, this is unexpected
        # For now, return the first one (this could be improved with more sophisticated logic)
        if len(ref_levels) == 1:
            return list(ref_levels)[0]
        elif len(ref_levels) > 1:
            # Multiple reference levels found - this might indicate a complex design
            # For now, return the first one, but this could be improved
            return list(ref_levels)[0]
        else:
            return None

    def _find_any_reference_level(self, design_cols: List[str]) -> Optional[str]:
        """
        Find any reference level from design matrix columns.

        Parameters
        ----------
        design_cols : List[str]
            List of design matrix column names

        Returns
        -------
        Optional[str]
            Reference level name, or None if not found
        """
        # Look for any columns that match the pattern "{factor}_{level}_vs_{ref_level}"
        matching_cols = [col for col in design_cols if "_vs_" in col]

        if not matching_cols:
            return None

        # Extract all reference levels from the matching columns
        ref_levels = set()
        for col in matching_cols:
            if "_vs_" in col:
                ref_level = col.split("_vs_")[1]
                ref_levels.add(ref_level)

        # Return the first reference level found
        if ref_levels:
            return list(ref_levels)[0]
        else:
            return None
