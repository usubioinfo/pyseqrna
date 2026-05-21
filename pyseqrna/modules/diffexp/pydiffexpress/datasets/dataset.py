#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expression dataset module for storing and managing gene expression data.

This module provides the ExpressionDataset class which serves as the central
data structure for storing gene expression counts, sample metadata, and
analysis results in a flexible and user-friendly manner.

This is part of the native pydiffexpress differential-expression implementation.

Classes:
    - ExpressionDataset: Central data structure for gene expression analysis

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


class ExpressionDataset:
    """
    Central data structure for gene expression analysis.

    The ExpressionDataset class provides a flexible interface for storing
    and managing gene expression data, sample metadata, and analysis results.
    It uses AnnData as the underlying storage but provides a more intuitive
    interface for differential expression analysis.
    """

    def __init__(
        self,
        counts: Union[str, pd.DataFrame, np.ndarray],
        sample_metadata: Optional[Union[str, pd.DataFrame]] = None,
        gene_metadata: Optional[Union[str, pd.DataFrame]] = None,
        sample_names: Optional[List[str]] = None,
        gene_names: Optional[List[str]] = None,
        auto_detect_orientation: bool = True,
        sample_id_column: Optional[str] = None,
        gene_column: str = "Gene",
        design_column: str = "condition",
        **kwargs,
    ):
        """
        Initialize the expression dataset.

        Parameters
        ----------
        counts : Union[str, pd.DataFrame, np.ndarray]
            Raw count data. Can be:
            - File path (string): Will be loaded based on file extension
            - pandas DataFrame: Used directly
            - numpy array: Converted to DataFrame
        sample_metadata : Optional[Union[str, pd.DataFrame]]
            Sample metadata. Can be file path or DataFrame.
        gene_metadata : Optional[Union[str, pd.DataFrame]]
            Gene metadata. Can be file path or DataFrame.
        sample_names : Optional[List[str]]
            Names for samples. If None, will use existing names or generate defaults.
        gene_names : Optional[List[str]]
            Names for genes. If None, will use existing names or generate defaults.
        auto_detect_orientation : bool
            Whether to automatically detect and fix data orientation.
        sample_id_column : Optional[str]
            Name of the column containing sample IDs in sample_metadata.
            If None, will use the index of sample_metadata.
        gene_column : str, default='Gene'
            Name of the column containing gene names in counts data.
            If this column exists, it will be set as the index.
        design_column : str, default='condition'
            Name of the column containing design/condition information in sample metadata.
        **kwargs
            Additional arguments passed to data loading functions.
        """
        # Import here to avoid circular imports
        from ..utils.data_loader import load_expression_data

        # Load data using the flexible loader
        counts_data, sample_meta, gene_meta = load_expression_data(
            counts,
            sample_metadata,
            gene_metadata,
            auto_detect_orientation,
            sample_id_column,
            gene_column=gene_column,
            design_column=design_column,
            **kwargs,
        )

        # Use loaded data
        if sample_metadata is None:
            sample_metadata = sample_meta
        if gene_metadata is None:
            gene_metadata = gene_meta

        # Convert counts to numpy array
        if isinstance(counts_data, pd.DataFrame):
            if sample_names is None:
                sample_names = counts_data.index.tolist()
            if gene_names is None:
                gene_names = counts_data.columns.tolist()
            counts = counts_data.values.astype(int)
        else:
            counts = np.asarray(counts_data, dtype=int)

        # Generate default names if not provided
        if sample_names is None:
            sample_names = [f"Sample_{i}" for i in range(counts.shape[0])]
        if gene_names is None:
            gene_names = [f"Gene_{i}" for i in range(counts.shape[1])]

        # Create sample metadata if not provided
        if sample_metadata is None:
            sample_metadata = pd.DataFrame(index=sample_names)
        else:
            # Use the sample metadata from data_loader (already properly indexed)
            sample_metadata = sample_meta

        # Create gene metadata if not provided
        if gene_metadata is None:
            gene_metadata = pd.DataFrame(index=gene_names)
        else:
            # Ensure gene metadata has correct index
            if not gene_metadata.index.equals(pd.Index(gene_names)):
                gene_metadata = gene_metadata.reindex(gene_names)

        # Create AnnData object
        self._adata = AnnData(X=counts, obs=sample_metadata, var=gene_metadata)

        # Set index names for proper CSV output (like inmoose)
        self._adata.var.index.name = "Gene"
        self._adata.obs.index.name = "Sample"

        # Initialize additional attributes
        self._size_factors = None
        self._normalization_factors = None
        self._design_matrix = None
        self._analysis_results = {}

        # Add categorical columns to obs if design column exists
        if design_column in self._adata.obs.columns and f"C({design_column})" not in self._adata.obs.columns:
            self._adata.obs[f"C({design_column})"] = self._adata.obs[design_column]

        # Create design matrix automatically if design column exists
        if design_column in self._adata.obs.columns:
            try:
                from ..utils.design_matrix import create_design_matrix

                # Use column-based approach instead of formulaic to ensure reference level encoding
                design_matrix, design_cols = create_design_matrix(self._adata.obs, design_column=design_column)
                self._adata.obsm["design"] = design_matrix
                self._adata.uns["design_columns"] = design_cols
                self._design_matrix = design_matrix
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Design matrix creation failed for column '{design_column}': {e}. "
                    "You may need to create the design matrix manually.",
                    stacklevel=2,
                )

    @property
    def counts(self) -> np.ndarray:
        """Get the raw count matrix."""
        return self._adata.X

    @property
    def sample_metadata(self) -> pd.DataFrame:
        """Get the sample metadata."""
        return self._adata.obs

    @property
    def gene_metadata(self) -> pd.DataFrame:
        """Get the gene metadata."""
        return self._adata.var

    @property
    def sample_names(self) -> List[str]:
        """Get the sample names."""
        return self._adata.obs_names.tolist()

    @property
    def gene_names(self) -> List[str]:
        """Get the gene names."""
        return self._adata.var_names.tolist()

    @property
    def n_samples(self) -> int:
        """Get the number of samples."""
        return self._adata.n_obs

    @property
    def n_obs(self) -> int:
        """Get the number of samples (like AnnData.n_obs)."""
        return self._adata.n_obs

    @property
    def n_genes(self) -> int:
        """Get the number of genes."""
        return self._adata.n_vars

    @property
    def shape(self) -> tuple:
        """Get the shape of the count matrix (samples, genes)."""
        return self._adata.shape

    @property
    def size_factors(self) -> Optional[np.ndarray]:
        """Get the size factors."""
        return self._size_factors

    @size_factors.setter
    def size_factors(self, factors: np.ndarray):
        """Set the size factors."""
        if factors is not None:
            factors = np.asarray(factors)
            if len(factors) != self.n_samples:
                raise ValueError(f"Size factors must have length {self.n_samples}")
            if not np.all(factors > 0):
                raise ValueError("Size factors must be positive")
        self._size_factors = factors

    @property
    def normalization_factors(self) -> Optional[np.ndarray]:
        """Get the normalization factors."""
        return self._normalization_factors

    @normalization_factors.setter
    def normalization_factors(self, factors: np.ndarray):
        """Set the normalization factors."""
        if factors is not None:
            factors = np.asarray(factors)
            if factors.shape != self.shape:
                raise ValueError(f"Normalization factors must have shape {self.shape}")
            if not np.all(factors > 0):
                raise ValueError("Normalization factors must be positive")
        self._normalization_factors = factors

    @property
    def design_matrix(self) -> Optional[np.ndarray]:
        """Get the design matrix."""
        return self._design_matrix

    @design_matrix.setter
    def design_matrix(self, matrix: np.ndarray):
        """Set the design matrix."""
        if matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape[0] != self.n_samples:
                raise ValueError(f"Design matrix must have {self.n_samples} rows")
        self._design_matrix = matrix

    @property
    def adata(self) -> AnnData:
        """Get the underlying AnnData object."""
        return self._adata

    @property
    def X(self) -> np.ndarray:
        """Get the data matrix (raw counts, like AnnData.X)."""
        return self._adata.X

    @property
    def obs(self) -> pd.DataFrame:
        """Get the sample metadata (like AnnData.obs)."""
        return self._adata.obs

    @property
    def var(self) -> pd.DataFrame:
        """Get the gene metadata (like AnnData.var)."""
        return self._adata.var

    @property
    def layers(self) -> Dict[str, np.ndarray]:
        """Get the layers (like AnnData.layers)."""
        return self._adata.layers

    @property
    def obsm(self) -> Dict[str, np.ndarray]:
        """Get the obsm (like AnnData.obsm)."""
        return self._adata.obsm

    @property
    def uns(self) -> Dict[str, Any]:
        """Get the uns (like AnnData.uns)."""
        return self._adata.uns

    @property
    def var_names(self) -> pd.Index:
        """Get the gene names (like AnnData.var_names)."""
        return self._adata.var_names

    @property
    def obs_names(self) -> pd.Index:
        """Get the sample names (like AnnData.obs_names)."""
        return self._adata.obs_names

    def get_normalized_counts(self) -> np.ndarray:
        """
        Get normalized counts using size factors or normalization factors.

        Returns
        -------
        np.ndarray
            Normalized count matrix.
        """
        # First check if normalized counts are already stored in AnnData
        if "normalized_counts" in self._adata.layers:
            return self._adata.layers["normalized_counts"]

        # Use raw counts from X
        counts = self._adata.X.copy()

        if self.normalization_factors is not None:
            # Use gene-specific normalization factors
            counts = counts / self.normalization_factors
        elif self.size_factors is not None:
            # Use sample-specific size factors
            counts = counts / self.size_factors.reshape(-1, 1)

        return counts

    def get_base_means(self) -> np.ndarray:
        """
        Get base means for each gene.

        Returns
        -------
        np.ndarray
            Base means for each gene.
        """
        if "base_mean" in self._adata.var.columns:
            return self._adata.var["base_mean"].values
        else:
            # Calculate on the fly
            normalized_counts = self.get_normalized_counts()
            return np.mean(normalized_counts, axis=0)

    def get_base_variances(self) -> np.ndarray:
        """
        Get base variances for each gene.

        Returns
        -------
        np.ndarray
            Base variances for each gene.
        """
        if "base_variance" in self._adata.var.columns:
            return self._adata.var["base_variance"].values
        else:
            # Calculate on the fly
            normalized_counts = self.get_normalized_counts()
            return np.var(normalized_counts, axis=0, ddof=1)

    def add_sample_metadata(self, column_name: str, values: Union[List, np.ndarray]):
        """
        Add a column to sample metadata.

        Parameters
        ----------
        column_name : str
            Name of the column to add.
        values : Union[List, np.ndarray]
            Values for the column.
        """
        if len(values) != self.n_samples:
            raise ValueError(f"Values must have length {self.n_samples}")
        self.sample_metadata[column_name] = values

    def add_gene_metadata(self, column_name: str, values: Union[List, np.ndarray]):
        """
        Add a column to gene metadata.

        Parameters
        ----------
        column_name : str
            Name of the column to add.
        values : Union[List, np.ndarray]
            Values for the column.
        """
        if len(values) != self.n_genes:
            raise ValueError(f"Values must have length {self.n_genes}")
        self.gene_metadata[column_name] = values

    def subset_samples(self, sample_indices: Union[List[int], List[str], np.ndarray]) -> "ExpressionDataset":
        """
        Create a subset of the dataset with selected samples.

        Parameters
        ----------
        sample_indices : Union[List[int], List[str], np.ndarray]
            Indices or names of samples to include.

        Returns
        -------
        ExpressionDataset
            Subset of the original dataset.
        """
        if isinstance(sample_indices[0], str):
            # Sample names provided
            mask = self.sample_metadata.index.isin(sample_indices)
        else:
            # Sample indices provided
            mask = np.zeros(self.n_samples, dtype=bool)
            mask[sample_indices] = True

        subset_adata = self._adata[mask].copy()

        # Create new dataset
        subset = ExpressionDataset(
            counts=subset_adata.X,
            sample_metadata=subset_adata.obs,
            gene_metadata=subset_adata.var,
            sample_names=subset_adata.obs_names.tolist(),
            gene_names=subset_adata.var_names.tolist(),
        )

        # Copy relevant attributes
        if self.size_factors is not None:
            subset.size_factors = self.size_factors[mask]
        if self.normalization_factors is not None:
            subset.normalization_factors = self.normalization_factors[mask]
        if self.design_matrix is not None:
            subset.design_matrix = self.design_matrix[mask]

        return subset

    def subset_genes(self, gene_indices: Union[List[int], List[str], np.ndarray]) -> "ExpressionDataset":
        """
        Create a subset of the dataset with selected genes.

        Parameters
        ----------
        gene_indices : Union[List[int], List[str], np.ndarray]
            Indices or names of genes to include.

        Returns
        -------
        ExpressionDataset
            Subset of the original dataset.
        """
        if isinstance(gene_indices[0], str):
            # Gene names provided
            mask = self.gene_metadata.index.isin(gene_indices)
        else:
            # Gene indices provided
            mask = np.zeros(self.n_genes, dtype=bool)
            mask[gene_indices] = True

        subset_adata = self._adata[:, mask].copy()

        # Create new dataset
        subset = ExpressionDataset(
            counts=subset_adata.X,
            sample_metadata=subset_adata.obs,
            gene_metadata=subset_adata.var,
            sample_names=subset_adata.obs_names.tolist(),
            gene_names=subset_adata.var_names.tolist(),
        )

        # Copy relevant attributes
        subset.size_factors = self.size_factors
        if self.normalization_factors is not None:
            subset.normalization_factors = self.normalization_factors[:, mask]
        subset.design_matrix = self.design_matrix

        return subset

    def copy(self) -> "ExpressionDataset":
        """Create a deep copy of the dataset."""
        copy_adata = self._adata.copy()

        copy_dataset = ExpressionDataset(
            counts=copy_adata.X,
            sample_metadata=copy_adata.obs,
            gene_metadata=copy_adata.var,
            sample_names=copy_adata.obs_names.tolist(),
            gene_names=copy_adata.var_names.tolist(),
        )

        # Copy additional attributes
        copy_dataset.size_factors = self.size_factors.copy() if self.size_factors is not None else None
        copy_dataset.normalization_factors = (
            self.normalization_factors.copy() if self.normalization_factors is not None else None
        )
        copy_dataset.design_matrix = self.design_matrix.copy() if self.design_matrix is not None else None

        return copy_dataset

    def __repr__(self) -> str:
        """String representation of the dataset (mimics inmoose AnnData)."""
        # Get obs columns
        obs_cols = list(self._adata.obs.columns)
        obs_str = f"obs: {', '.join([repr(col) for col in obs_cols])}" if obs_cols else "obs: (empty)"

        # Get var columns
        var_cols = list(self._adata.var.columns)
        var_str = f"var: {', '.join([repr(col) for col in var_cols])}" if var_cols else "var: (empty)"

        # Get obsm keys
        obsm_keys = list(self._adata.obsm.keys())
        obsm_str = f"obsm: {', '.join([repr(key) for key in obsm_keys])}" if obsm_keys else "obsm: (empty)"

        # Get layers keys
        layers_keys = list(self._adata.layers.keys())
        layers_str = f"layers: {', '.join([repr(key) for key in layers_keys])}" if layers_keys else "layers: (empty)"

        # Build the representation
        lines = [
            f"AnnData object with n_obs × n_vars = {self._adata.n_obs} × {self._adata.n_vars}",
            f"    {obs_str}",
            f"    {var_str}",
            f"    {obsm_str}",
            f"    {layers_str}",
        ]

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation of the dataset."""
        return self.__repr__()

    def estimate_size_factors(self, method: str = "median_ratio", **kwargs) -> "ExpressionDataset":
        """
        Estimate size factors for normalization.

        Parameters
        ----------
        method : str
            Normalization method ('median_ratio', 'poscounts', 'iterative')
        **kwargs
            Additional arguments passed to the normalizer

        Returns
        -------
        self : ExpressionDataset
            Dataset with size factors estimated
        """
        from ..normalization import create_normalizer

        # Create normalizer
        normalizer = create_normalizer(method, **kwargs)

        # Fit the normalizer
        normalizer.fit(self)

        # Get size factors
        size_factors = normalizer.size_factors

        # Store size factors
        self.size_factors = size_factors
        self._adata.obs["size_factors"] = size_factors

        # Get base statistics from normalizer if available
        if hasattr(normalizer, "base_means") and normalizer.base_means is not None:
            self._adata.var["base_mean"] = normalizer.base_means
        else:
            # Calculate base statistics manually
            normalized_counts = self.get_normalized_counts()
            self._adata.var["base_mean"] = np.mean(normalized_counts, axis=0)

        if hasattr(normalizer, "base_variances") and normalizer.base_variances is not None:
            self._adata.var["base_variance"] = normalizer.base_variances
        else:
            # Calculate base statistics manually
            normalized_counts = self.get_normalized_counts()
            self._adata.var["base_variance"] = np.var(normalized_counts, axis=0, ddof=1)

        # Set all_zero flag
        self._adata.var["all_zero"] = self._adata.var["base_mean"] == 0

        # Store raw counts in X and normalized counts in layers
        self._adata.X.copy()
        normalized_counts = self.get_normalized_counts()

        # Keep raw counts in X
        # self._adata.X = raw_counts  # Already contains raw counts

        # Store normalized counts in layers
        self._adata.layers["normalized_counts"] = normalized_counts

        # Remove counts from layers if it exists (we don't need it anymore)
        if "counts" in self._adata.layers:
            del self._adata.layers["counts"]

        return self

    def estimate_normalization(
        self,
        method: str = "median_ratio",
        design_formula: str = "~ condition",
        **kwargs,
    ) -> "ExpressionDataset":
        """
        Estimate normalization (size factors and base statistics).

        Parameters
        ----------
        method : str
            Normalization method ('median_ratio', 'poscounts', 'iterative')
        design_formula : str
            Design formula for normalization
        **kwargs
            Additional arguments passed to the normalizer

        Returns
        -------
        self : ExpressionDataset
            Dataset with normalization estimated
        """
        # Create design matrix if not exists
        if "design" not in self._adata.obsm:
            from ..utils.design_matrix import create_design_matrix

            # Use column-based approach for consistent reference level encoding
            design_matrix, design_cols = create_design_matrix(self._adata.obs, design_column="condition")
            self._adata.obsm["design"] = design_matrix
            self._adata.uns["design_columns"] = design_cols
            self._design_matrix = design_matrix

        # Estimate size factors
        self.estimate_size_factors(method, **kwargs)

        return self

    def normalize(
        self,
        method: str = "median_ratio",
        design_formula: str = "~ condition",
        **kwargs,
    ) -> "ExpressionDataset":
        """
        Apply normalization to the dataset.

        Parameters
        ----------
        method : str
            Normalization method ('median_ratio', 'poscounts', 'iterative')
        design_formula : str
            Design formula for normalization
        **kwargs
            Additional arguments passed to the normalizer

        Returns
        -------
        self : ExpressionDataset
            Normalized dataset
        """
        # Estimate normalization if not already done
        if self.size_factors is None:
            self.estimate_normalization(method, design_formula, **kwargs)

        return self

    def estimate_dispersions(
        self, method: str = "pipeline", design_formula: str = "~ condition", **kwargs
    ) -> "ExpressionDataset":
        """
        Estimate dispersions using the specified method.

        Parameters
        ----------
        method : str
            Dispersion estimation method ('gene_wise', 'trend', 'map', 'pipeline')
        design_formula : str
            Design formula for dispersion estimation
        **kwargs
            Additional arguments passed to the dispersion estimator

        Returns
        -------
        self : ExpressionDataset
            Dataset with dispersions estimated
        """
        from ..dispersion import create_dispersion_estimator

        # Ensure normalization is done first
        if self.size_factors is None:
            self.estimate_normalization(design_formula=design_formula)

        # Create design matrix if not exists
        if "design" not in self._adata.obsm:
            from ..utils.design_matrix import create_design_matrix

            # Use column-based approach for consistent reference level encoding
            design_matrix, design_cols = create_design_matrix(self._adata.obs, design_column="condition")
            self._adata.obsm["design"] = design_matrix
            self._adata.uns["design_columns"] = design_cols
            self._design_matrix = design_matrix

        # Create dispersion estimator
        estimator = create_dispersion_estimator(method, **kwargs)

        # Fit the estimator
        estimator.fit(self._adata, self._adata.obsm["design"])
        estimator.update_data(self._adata)

        return self

    def estimate_dispersions_gene_wise(self, design_formula: str = "~ condition", **kwargs) -> "ExpressionDataset":
        """Estimate gene-wise dispersions."""
        return self.estimate_dispersions("gene_wise", design_formula, **kwargs)

    def estimate_dispersions_trend(self, design_formula: str = "~ condition", **kwargs) -> "ExpressionDataset":
        """Estimate trend dispersions."""
        return self.estimate_dispersions("trend", design_formula, **kwargs)

    def estimate_dispersions_map(self, design_formula: str = "~ condition", **kwargs) -> "ExpressionDataset":
        """Estimate MAP dispersions."""
        return self.estimate_dispersions("map", design_formula, **kwargs)

    def diffexpress(
        self,
        test: str = "Wald",
        fit_type: str = "parametric",
        sf_type: str = "ratio",
        quiet: bool = False,
        min_mu: float = 0.5,
        contrast: Optional[Union[str, List[str], np.ndarray]] = None,
        **kwargs,
    ) -> "ExpressionDataset":
        """
        Run complete differential expression analysis pipeline.

        This method runs the full analysis pipeline including:
        1. Size factor estimation
        2. Dispersion estimation
        3. Statistical testing (Wald or LRT)

        Parameters
        ----------
        test : str, optional
            Statistical test to use ("Wald" or "LRT")
        fit_type : str, optional
            Type of dispersion fitting ("parametric", "local", "mean", "glmGamPoi")
        sf_type : str, optional
            Type of size factor estimation ("ratio", "poscounts", "iterate")
        quiet : bool, optional
            Whether to suppress progress messages
        min_mu : float, default=0.5
            Minimum value for fitted means
        **kwargs
            Additional parameters passed to the analyzer

        Returns
        -------
        self : ExpressionDataset
            Dataset with analysis results stored
        """
        # Step 1: Estimate size factors (if not already done)
        if self.size_factors is None:
            if not quiet:
                logger.info("Estimating size factors...")
            # Map common user-facing names to internal names.
            sf_type_map = {
                "ratio": "median_ratio",
                "poscounts": "poscounts",
                "iterate": "iterative",
                "tmm": "tmm",
            }
            method = sf_type_map.get(sf_type, sf_type)
            self.estimate_size_factors(method=method)

        # Step 2: Estimate dispersions (if not already done)
        if "dispersion" not in self._adata.var.columns:
            if not quiet:
                logger.info("Estimating dispersions...")

            # Use tagwise dispersion for TMM normalization
            if sf_type == "tmm":
                self.estimate_dispersions(method="tagwise", min_mu=min_mu, **kwargs)
            else:
                self.estimate_dispersions(fit_type=fit_type, min_mu=min_mu, **kwargs)

        # Step 3: Run statistical test
        if not quiet:
            logger.info("Running %s test...", test)

        if test == "Wald":
            from ..hypothesis_testing import WaldTestAnalyzer

            wald_analyzer = WaldTestAnalyzer(
                quiet=quiet,
                min_mu=min_mu,
                **{k: v for k, v in kwargs.items() if k in ["beta_tol", "max_iter"]},
            )
            wald_analyzer.fit(self._adata, self._adata.obsm.get("design"))
            wald_analyzer.update_data(self._adata)

            # Store results in the format expected by ContrastAnalyzer
            self._analysis_results = wald_analyzer.results
            self._adata.uns["analysis_results"] = self._analysis_results

        elif test == "LRT":
            if not quiet:
                logger.info("Running Likelihood Ratio Test (LRT)...")

            # Initialize LRT analyzer
            from ..hypothesis_testing import LRTAnalyzer

            lrt_analyzer = LRTAnalyzer(
                reduced_formula="~1",  # Default to intercept-only model
                quiet=quiet,
                **{k: v for k, v in locals().items() if k in ["beta_tol", "max_iter"]},
            )
            lrt_analyzer.fit(self._adata, self._adata.obsm.get("design"))
            lrt_analyzer.update_data(self._adata)

            # Store raw results dictionary for contrast analysis.
            self._analysis_results = lrt_analyzer._results
            self._adata.uns["analysis_results"] = self._analysis_results

        return self

    def get_results(
        self,
        contrast: Optional[Union[str, List[str], np.ndarray]] = None,
        lfc_threshold: float = 0.0,
        alpha: float = 1.0,
    ) -> pd.DataFrame:
        """
        Get differential expression results.

        Parameters
        ----------
        contrast : Optional[Union[str, List[str], np.ndarray]]
            Contrast specification. If None, returns all results.
        lfc_threshold : float
            Log fold change threshold for filtering
        alpha : float
            Significance level for filtering

        Returns
        -------
        pd.DataFrame
            Results table
        """
        from ..results import ContrastAnalyzer, ResultsExtractor

        if contrast is not None:
            # Get results for specific contrast
            analyzer = ContrastAnalyzer()
            return analyzer.extract_contrast(self._adata, contrast, lfc_threshold=lfc_threshold, alpha=alpha)
        else:
            # Get all results
            extractor = ResultsExtractor()
            return extractor.get_all_results(self._adata)

    def get_significant_genes(
        self,
        contrast: Union[str, List[str], np.ndarray],
        lfc_threshold: float = 1.0,
        alpha: float = 0.05,
        direction: str = "both",
    ) -> pd.DataFrame:
        """
        Get significant genes for a specific contrast.

        Parameters
        ----------
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
        from ..results.contrasts import ContrastAnalyzer

        analyzer = ContrastAnalyzer()
        return analyzer.get_significant_genes(self._adata, contrast, lfc_threshold, alpha, direction)

    def get_all_contrasts_results(
        self,
        contrasts: Optional[List[Union[str, List[str], np.ndarray]]] = None,
        lfc_threshold: float = 0.0,
        alpha: float = 1.0,
        include_base_stats: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get results for all contrasts combined in one DataFrame.

        This method combines results from all contrasts into a single DataFrame
        with properly labeled columns like "log2FoldChange(condition_V1_vs_A1)".

        Parameters
        ----------
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

        Examples
        --------
        >>> # Get all possible contrasts
        >>> all_results = eds.get_all_contrasts_results()

        >>> # Get specific contrasts only
        >>> specific_results = eds.get_all_contrasts_results([
        ...     "condition_V1_vs_A1",
        ...     "condition_M1_vs_A1",
        ...     "condition_M1_vs_V1"
        ... ])

        >>> # Get only contrast-specific results (no baseMean)
        >>> contrast_only = eds.get_all_contrasts_results(include_base_stats=False)
        """
        from ..results.contrasts import ContrastAnalyzer

        analyzer = ContrastAnalyzer()
        return analyzer.get_all_contrasts_results(self._adata, contrasts, lfc_threshold, alpha, include_base_stats, **kwargs)

    def export_results(
        self,
        filename: str,
        format: str = "csv",
        contrast: Optional[Union[str, List[str], np.ndarray]] = None,
        **kwargs,
    ) -> None:
        """
        Export results to a file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str
            Output format ("csv", "tsv", "excel")
        contrast : Optional[Union[str, List[str], np.ndarray]]
            Contrast specification. If None, exports all results.
        **kwargs
            Additional parameters passed to get_results()
        """

        # Get results
        if contrast is not None:
            results_df = self.get_results(contrast, **kwargs)
        else:
            results_df = self.get_results(**kwargs)

        # Export based on format
        if format.lower() == "csv":
            results_df.to_csv(filename)
        elif format.lower() == "tsv":
            results_df.to_csv(filename, sep="\t")
        elif format.lower() == "excel":
            results_df.to_excel(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.

        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        from ..results import ResultsExtractor

        extractor = ResultsExtractor()
        return extractor.summary(self._adata)
