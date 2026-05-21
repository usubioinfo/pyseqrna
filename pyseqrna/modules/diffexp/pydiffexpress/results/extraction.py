"""
Result extraction utilities for differential expression analysis.

This module provides utilities for extracting and formatting results
from differential expression analysis.

Classes:
    - ResultsExtractor: Utility class for extracting and formatting differential expression results

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from anndata import AnnData


class ResultsExtractor:
    """
    Utility class for extracting and formatting differential expression results.

    This class provides methods to extract various types of results
    from differential expression analysis and format them for output.
    """

    def __init__(self):
        """Initialize the results extractor."""
        pass

    def get_all_results(self, data: AnnData) -> pd.DataFrame:
        """
        Get all differential expression results as a DataFrame.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results

        Returns
        -------
        pd.DataFrame
            Complete results table with all genes
        """
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        results = data.uns["analysis_results"]

        # Create results DataFrame
        result_df = pd.DataFrame(
            {
                "Gene": data.var_names,
                "baseMean": data.var["base_mean"] if "base_mean" in data.var.columns else np.zeros(data.n_vars),
            }
        )

        # Add analysis results if available
        if "beta_coefficients" in results:
            # Add each coefficient as a separate column with proper names
            beta_coeffs = results["beta_coefficients"]
            design_cols = data.uns.get("design_columns", [f"coef_{i}" for i in range(beta_coeffs.shape[1])])

            for i in range(beta_coeffs.shape[1]):
                col_name = design_cols[i] if i < len(design_cols) else f"coef_{i}"
                result_df[col_name] = beta_coeffs[:, i]

        if "p_values" in results:
            p_values = results["p_values"]
            if len(p_values.shape) == 2:
                # 2D array - take the first column for now (intercept p-value)
                result_df["pvalue"] = p_values[:, 0]
                result_df["FDR"] = self._calculate_adjusted_pvalues(p_values[:, 0])
            else:
                # 1D array
                result_df["pvalue"] = p_values
                result_df["FDR"] = self._calculate_adjusted_pvalues(p_values)

        if "wald_statistics" in results:
            wald_stats = results["wald_statistics"]
            if len(wald_stats.shape) == 2:
                # 2D array - take the first column for now
                result_df["stat"] = wald_stats[:, 0]
            else:
                # 1D array
                result_df["stat"] = wald_stats

        # Set gene names as index
        result_df.set_index("Gene", inplace=True)

        return result_df

    def get_coefficients(self, data: AnnData) -> pd.DataFrame:
        """
        Get coefficient estimates from the analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results

        Returns
        -------
        pd.DataFrame
            Coefficient estimates for each gene
        """
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        results = data.uns["analysis_results"]

        if "beta_coefficients" not in results:
            raise ValueError("No coefficient estimates found in results")

        beta_coeffs = results["beta_coefficients"]

        # Create DataFrame with coefficients
        coef_df = pd.DataFrame(
            beta_coeffs,
            index=data.var_names,
            columns=[f"coef_{i}" for i in range(beta_coeffs.shape[1])],
        )

        return coef_df

    def get_p_values(self, data: AnnData) -> pd.DataFrame:
        """
        Get p-values from the analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results

        Returns
        -------
        pd.DataFrame
            P-values and adjusted p-values for each gene
        """
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        results = data.uns["analysis_results"]

        if "p_values" not in results:
            raise ValueError("No p-values found in results")

        p_values = results["p_values"]
        adjusted_p_values = self._calculate_adjusted_pvalues(p_values)

        pval_df = pd.DataFrame({"p_value": p_values, "padj": adjusted_p_values}, index=data.var_names)

        return pval_df

    def get_statistics(self, data: AnnData) -> pd.DataFrame:
        """
        Get test statistics from the analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results

        Returns
        -------
        pd.DataFrame
            Test statistics for each gene
        """
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        results = data.uns["analysis_results"]

        if "wald_statistics" not in results:
            raise ValueError("No test statistics found in results")

        stats_df = pd.DataFrame({"statistic": results["wald_statistics"]}, index=data.var_names)

        return stats_df

    def get_dispersion_estimates(self, data: AnnData) -> pd.DataFrame:
        """
        Get dispersion estimates from the analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data with dispersion estimates

        Returns
        -------
        pd.DataFrame
            Dispersion estimates for each gene
        """
        dispersion_cols = [col for col in data.var.columns if "dispersion" in col.lower()]

        if not dispersion_cols:
            raise ValueError("No dispersion estimates found in data")

        disp_df = data.var[dispersion_cols].copy()

        return disp_df

    def get_size_factors(self, data: AnnData) -> pd.Series:
        """
        Get size factors from the analysis.

        Parameters
        ----------
        data : AnnData
            Annotated data with size factors

        Returns
        -------
        pd.Series
            Size factors for each sample
        """
        if "size_factors" not in data.obs.columns:
            raise ValueError("No size factors found in data")

        return data.obs["size_factors"]

    def export_results(
        self,
        data: AnnData,
        filename: str,
        format: str = "csv",
        include_metadata: bool = True,
        contrast: Optional[str] = None,
    ) -> None:
        """
        Export results to a file.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results
        filename : str
            Output filename
        format : str
            Output format ("csv", "tsv", "excel")
        include_metadata : bool
            Whether to include metadata in the output
        contrast : Optional[str]
            Specific contrast to export. If None, exports all results.
        """
        # Get results for specific contrast or all results
        if contrast is not None:
            from ..results import ContrastAnalyzer

            analyzer = ContrastAnalyzer()
            result_df = analyzer.extract_contrast(data, contrast)
        else:
            result_df = self.get_all_results(data)

        # Collect metadata if requested
        metadata = {}
        if include_metadata and "analysis_results" in data.uns:
            metadata = data.uns["analysis_results"].get("metadata", {})

        # Export based on format
        if format.lower() == "csv":
            result_df.to_csv(filename)
        elif format.lower() == "tsv":
            result_df.to_csv(filename, sep="\t")
        elif format.lower() == "excel":
            if metadata:
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    result_df.to_excel(writer, sheet_name="Results")
                    meta_rows = [
                        {"Parameter": str(k), "Value": str(v)} for k, v in metadata.items() if not isinstance(v, np.ndarray)
                    ]
                    if meta_rows:
                        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Metadata", index=False)
            else:
                result_df.to_excel(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _calculate_adjusted_pvalues(self, p_values: np.ndarray) -> np.ndarray:
        """Calculate adjusted p-values using Benjamini-Hochberg correction."""
        # Convert to numpy array if it's a pandas Series
        if hasattr(p_values, "values"):
            p_values = p_values.values

        n = len(p_values)

        # Handle NaN values by filtering them out
        valid_mask = ~np.isnan(p_values)
        valid_p_values = p_values[valid_mask]

        if len(valid_p_values) == 0:
            # All values are NaN
            return np.full(n, np.nan)

        # Calculate adjusted p-values for valid values only
        sorted_indices = np.argsort(valid_p_values)
        adjusted_pvalues = np.full(n, np.nan)  # Initialize with NaN

        for i, idx in enumerate(sorted_indices):
            rank = i + 1
            original_idx = np.where(valid_mask)[0][idx]
            adjusted_pvalues[original_idx] = min(valid_p_values[idx] * len(valid_p_values) / rank, 1.0)

        return adjusted_pvalues

    def summary(self, data: AnnData) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.

        Parameters
        ----------
        data : AnnData
            Annotated data with differential expression results

        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if "analysis_results" not in data.uns:
            raise ValueError("No analysis results found. Run diffexpress() first.")

        results = data.uns["analysis_results"]

        summary = {
            "n_genes": data.n_vars,
            "n_samples": data.n_obs,
            "analysis_type": results.get("metadata", {}).get("test", "Unknown"),
            "fit_type": results.get("metadata", {}).get("fit_type", "Unknown"),
            "sf_type": results.get("metadata", {}).get("sf_type", "Unknown"),
        }

        # Add p-value summary if available
        if "p_values" in results:
            p_values = results["p_values"]
            summary.update(
                {
                    "n_significant_0.05": np.sum(p_values < 0.05),
                    "n_significant_0.01": np.sum(p_values < 0.01),
                    "n_significant_0.001": np.sum(p_values < 0.001),
                    "min_p_value": np.min(p_values),
                    "max_p_value": np.max(p_values),
                }
            )

        # Add coefficient summary if available
        if "beta_coefficients" in results:
            beta_coeffs = results["beta_coefficients"]
            summary.update(
                {
                    "n_coefficients": beta_coeffs.shape[1],
                    "min_coefficient": np.min(beta_coeffs),
                    "max_coefficient": np.max(beta_coeffs),
                }
            )

        return summary
