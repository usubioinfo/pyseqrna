#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Input Processor Module

This module provides functionality to parse, validate, and process input files and
sample metadata for the PySeqRNA pipeline.

Features:
    - Support for reading multiple formats (.xlsx, .csv, .txt, .tsv) with custom delimiters
    - Extraction and validation of sample metadata (identifiers, replicates, fastq file paths)
    - Automatic detection and inference of single-end vs paired-end modes
    - Helper methods for generating factor combinations and experiment target matrices

Classes:
    - InputProcessor: Processes input configuration tables and sample information

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import logging


class InputProcessor:
    """Handles input file processing and sample information."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize InputProcessor.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for recording operations
        """
        self.logger = logger

    def read_dataframe(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read input file into DataFrame based on file extension.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to input file

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame

        Raises
        ------
        ValueError
            If file format is not supported
        """
        file_path = str(file_path)
        safe_file_path = str(file_path).replace("\n", "").replace("\r", "")
        self.logger.info(f"Reading input file: {safe_file_path}")

        try:
            if file_path.endswith(".xlsx"):
                return pd.read_excel(file_path, comment="#")
            elif file_path.endswith(".csv"):
                return pd.read_csv(file_path, comment="#")
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as handle:
                    non_comment_lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
                delimiter = "\t" if non_comment_lines and "\t" in non_comment_lines[0] else r"\s+"
                return pd.read_csv(file_path, sep=delimiter, comment="#", engine="python")
            elif file_path.endswith(".tsv"):
                return pd.read_csv(file_path, sep="\t", comment="#")
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            sanitized_error = str(e).replace("\n", " ").replace("\r", " ")
            self.logger.error(f"Failed to read file {file_path}: {sanitized_error}")
            raise

    def process_sample_file(
        self,
        file_path: Union[str, Path],
        samples_dir: Union[str, Path],
        paired: bool = False,
    ) -> Dict:
        """
        Process sample information file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to sample information file
        samples_dir : Union[str, Path]
            Directory containing sample files
        paired : bool
            Whether samples are paired-end

        Returns
        -------
        Dict
            Dictionary containing samples, combinations, and targets
        """
        self.logger.info(f"Processing sample file: {file_path}")
        self.logger.info(f"Sample directory: {samples_dir}")
        self.logger.info(f"Requested paired-end mode: {paired}")

        samples = {}
        factors = []
        sample_labels = {}

        try:
            df = self.read_dataframe(file_path)
            df.columns = [str(col).strip() for col in df.columns]

            sample_col = "SampleName" if "SampleName" in df.columns else df.columns[0]
            replication_col = "Replication" if "Replication" in df.columns else df.columns[1]
            identifier_col = "Identifier" if "Identifier" in df.columns else df.columns[2]
            file1_col = "File1" if "File1" in df.columns else ("File" if "File" in df.columns else df.columns[3])
            file2_col = "File2" if "File2" in df.columns else None

            inferred_paired = bool(
                file2_col and df[file2_col].notna().any() and df[file2_col].astype(str).str.strip().ne("").any()
            )
            use_paired = paired or inferred_paired
            self.logger.info(f"Effective paired-end mode: {use_paired}")

            for _, row in df.iterrows():
                sample_name = str(row[sample_col]).strip()
                sample_id = str(row[replication_col]).strip()
                sample_type = str(row[identifier_col]).strip()
                fastq_path = os.path.join(samples_dir, str(row[file1_col]).strip())
                sample_labels[sample_id] = sample_name

                if use_paired:
                    if not file2_col or pd.isna(row[file2_col]) or str(row[file2_col]).strip() == "":
                        raise ValueError(f"Missing File2 entry for paired-end sample: {sample_id}")
                    paired_path = os.path.join(samples_dir, str(row[file2_col]).strip())
                    samples[sample_id] = [
                        sample_name,
                        sample_type,
                        fastq_path,
                        paired_path,
                    ]
                    self.logger.debug(f"Added paired-end sample: {sample_id} ({fastq_path}, {paired_path})")
                else:
                    samples[sample_id] = [sample_name, sample_type, fastq_path]
                    self.logger.debug(f"Added single-end sample: {sample_id} ({fastq_path})")

                if sample_type not in factors:
                    factors.append(sample_type)

            combinations = self.create_combinations(factors)
            targets = self.create_targets(samples)

            self.logger.info(f"Processed {len(samples)} samples with {len(factors)} factors")
            self.logger.info(f"Created {len(combinations)} sample combinations")

            return {
                "samples": samples,
                "combinations": combinations,
                "targets": targets,
                "sample_labels": sample_labels,
                "paired": use_paired,
            }

        except Exception as e:
            self.logger.error(f"Error processing sample file: {str(e)}")
            raise

    def create_combinations(self, factors: List[str]) -> List[str]:
        """Create all possible sample combinations (one direction only)."""
        combinations = []
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                combinations.append(f"{factors[i]}-{factors[j]}")
        self.logger.debug(f"Created combinations: {combinations}")
        return combinations

    def create_targets(self, samples: Dict) -> pd.DataFrame:
        """Create targets DataFrame from samples."""
        sample_names = list(samples.keys())
        sample_types = [s[1] for s in samples.values()]
        targets = pd.DataFrame({"condition": sample_types}, index=sample_names)
        self.logger.debug(f"Created targets DataFrame with {len(sample_names)} samples")
        return targets
