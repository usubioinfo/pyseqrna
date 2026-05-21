#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTSeq Quantification Module

This module provides a wrapper around the htseq-count command-line tool, which is part of
the HTSeq Python package. It is designed to count aligned RNA-seq reads overlapping genomic
features in BAM files using union, intersection-strict, or intersection-nonempty modes.

Features:
    - Wrapper for the HTSeq package's htseq-count utility
    - Support for single-end and paired-end sequencing datasets
    - Flexible overlap counting modes: union, intersection-strict, and intersection-nonempty
    - Automatically handles GTF and GFF attribute/feature mapping defaults
    - Multi-sample parallel execution support with SLURM/local job dispatching
    - Post-processing to remove HTSeq metrics (e.g. __no_feature, __ambiguous) and export clean counts

Configuration:
    Configured via parameters passed to the constructor (forwarded to BaseQuantifier)
    and tool-specific configuration file htseq.ini (e.g., setting feature types,
    attributes, counting modes, and strandedness).

Dependencies:
    - HTSeq (htseq-count) command-line tool installed in system PATH
    - pandas
    - pyseqrna.modules.quantification.base (BaseQuantifier, QuantificationError)

Classes / Functions / Exceptions:
    - HTSeqQuantifier: Concrete implementation of BaseQuantifier using htseq-count for read quantification.

:Created: January 20, 2025
:Updated: February 25, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import shutil
from typing import Dict, List
import pandas as pd

from .base import BaseQuantifier, QuantificationError


class HTSeqQuantifier(BaseQuantifier):
    """
    HTSeq-count implementation for gene expression quantification.

    This class provides functionality to count reads mapping to genomic features
    using HTSeq-count.

    Attributes:
        tool_name (str): Name of the quantification tool ("htseq")
        executable_path (str): Path to htseq-count executable
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize HTSeqQuantifier.

        Args:
            *args: Positional arguments passed to BaseQuantifier
            **kwargs: Keyword arguments passed to BaseQuantifier
        """
        super().__init__(*args, **kwargs)
        self.executable_path = None
        self._find_executable()

    def _find_executable(self) -> None:
        """Find htseq-count executable in PATH."""
        # Try different possible names
        possible_names = ["htseq-count", "htseq_count"]
        for name in possible_names:
            self.executable_path = shutil.which(name)
            if self.executable_path:
                break

        if not self.executable_path:
            if self.dryrun:
                self.executable_path = "htseq-count"
                self.logger.warning("htseq-count executable not found in PATH; continuing in dry-run mode")
                return
            self.logger.error("htseq-count executable not found in system PATH")
            raise QuantificationError("htseq-count executable not found in system PATH")

    def check_tool_availability(self) -> bool:
        """
        Check if htseq-count is available.

        Returns:
            bool: True if htseq-count is available, False otherwise
        """
        return self.executable_path is not None

    def _detect_annotation_format(self) -> str:
        """
        Detect if the annotation file is GTF or GFF based on extension.
        Returns 'GTF' or 'GFF'.
        """
        ext = os.path.splitext(self.annotation_file)[1].lower()
        if ext in [".gtf"]:
            return "GTF"
        return "GFF"

    def _get_config_parameters(self) -> Dict[str, str]:
        """
        Get configuration parameters from config file, with robust GTF/GFF attribute handling.
        Returns:
            Dict: Configuration parameters from config file
        """
        config = self.load_config("htseq.ini")
        if not config:
            raise QuantificationError("Failed to load configuration file: htseq.ini")
        if "htseq-count" not in config:
            raise QuantificationError("Missing 'htseq-count' section in config file")
        htseq_config = config["htseq-count"]
        params = {}
        for key, value in htseq_config.items():
            if value and value != "NA":
                params[key] = value
        # Adjust attribute based on annotation file format
        annotation_format = self._detect_annotation_format()
        if annotation_format == "GTF":
            params["attribute"] = "-i gene_id"
        else:
            params["attribute"] = "-i ID"
        # Ensure other parameters are formatted correctly
        for param_key, flag in [
            ("format", "-f"),
            ("feature", "-t"),
            ("mode", "-m"),
            ("strand", "-s"),
            ("quality", "-a"),
        ]:
            if param_key in params and not params[param_key].startswith(flag):
                params[param_key] = f"{flag} {params[param_key]}"
        return params

    def _build_command(self, sample_id: str, bam_file: str) -> str:
        """
        Build HTSeq-count command for a single sample.

        Args:
            sample_id: Sample identifier
            bam_file: Path to BAM file

        Returns:
            str: Complete HTSeq-count command
        """
        if not self.executable_path:
            raise QuantificationError("htseq-count executable not found")

        # Get configuration parameters
        config_params = self._get_config_parameters()

        # Build output file path
        output_file = self.out_dir / f"{sample_id}_counts.txt"

        # Build parameter list from config
        param_parts = []
        for key, value in config_params.items():
            if value and value != "NA" and key != "additional":
                param_parts.append(value)

        # Add additional parameters if specified
        if config_params.get("additional"):
            param_parts.append(config_params["additional"])

        # Build command components
        cmd_parts = [
            self.executable_path,
            *param_parts,
            bam_file,
            self.annotation_file,
            ">",
            str(output_file),
        ]

        return " ".join(cmd_parts)

    def _process_output_files(self) -> pd.DataFrame:
        """
        Process and combine HTSeq-count output files into a single count matrix.
        Returns:
            DataFrame containing the final count matrix
        """
        try:
            count_data = None
            sample_columns = ["Gene"]
            for sample_id in self.bam_dict.keys():
                output_file = self.out_dir / f"{sample_id}_counts.txt"
                if not output_file.exists():
                    self.logger.warning(f"Output file not found for sample {sample_id}: {output_file}")
                    continue
                try:
                    df = pd.read_csv(output_file, sep="\t", header=None, names=["Gene", "Count"])
                    df_clean = df[~df["Gene"].str.startswith("__")].copy()
                    if len(df_clean) == 0:
                        self.logger.warning(f"No gene counts found in {output_file}")
                        continue
                    if count_data is None:
                        count_data = df_clean[["Gene"]].copy()
                    count_data[sample_id] = df_clean["Count"].values
                    sample_columns.append(sample_id)
                    if not self.dryrun:
                        output_file.unlink()
                    self.logger.debug(f"Processed counts for sample {sample_id}")
                except Exception as e:
                    self.logger.error(f"Error processing output file for {sample_id}: {str(e)}")
                    continue
            if count_data is None or len(count_data) == 0:
                self.logger.error("No valid count data found")
                return pd.DataFrame()
            count_data = self._clean_gene_column(count_data, gene_col="Gene")
            count_data.columns = sample_columns[: len(count_data.columns)]
            if not self.dryrun:
                output_excel = self.out_dir / "Raw_Counts.xlsx"
                count_data.to_excel(str(output_excel), index=False)
                self.logger.info(f"Count matrix saved to: {output_excel}")
            return count_data
        except Exception as e:
            self.logger.error(f"Error processing HTSeq-count output files: {str(e)}")
            return pd.DataFrame()

    def run(self) -> pd.DataFrame:
        """
        Run HTSeq-count quantification.

        Returns:
            DataFrame containing the count matrix

        Raises:
            QuantificationError: If quantification fails
        """
        self.logger.info("Starting HTSeq-count quantification")

        try:
            # Check tool availability
            if not self.check_tool_availability():
                raise QuantificationError("htseq-count is not available in PATH")

            # Create output directory
            if not self.dryrun:
                self.file_manager.create_subdirectory(str(self.out_dir), dry_run=False, preserve_existing=True)

            # Build commands for all samples
            commands = {}
            for sample_id, sample_info in self.bam_dict.items():
                try:
                    bam_file = self._extract_bam_path(sample_info)
                    command = self._build_command(sample_id, bam_file)
                    commands[sample_id] = command
                except Exception as e:
                    self.logger.error(f"Error building command for sample {sample_id}: {str(e)}")
                    continue

            if not commands:
                raise QuantificationError("No valid commands generated")

            # Execute commands
            self.execute_command(commands, str(self.out_dir), "htseq-count")

            if not self.dryrun:
                # Process output files and create count matrix
                count_matrix = self._process_output_files()

                if count_matrix.empty:
                    raise QuantificationError("No count data was generated")

                # Generate summary statistics
                stats = self.get_summary_stats(count_matrix)
                self.logger.info(
                    f"HTSeq-count completed: {stats.get('total_genes', 0)} genes, "
                    f"{stats.get('total_samples', 0)} samples, "
                    f"{stats.get('total_reads', 0)} total reads"
                )

                return count_matrix
            else:
                self.logger.info("DRYRUN: HTSeq-count quantification simulation completed")
                # Return mock DataFrame for dry run
                sample_names = ["Gene"] + list(self.bam_dict.keys())
                mock_data = [
                    ["gene1"] + [100] * len(self.bam_dict),
                    ["gene2"] + [200] * len(self.bam_dict),
                ]
                return pd.DataFrame(mock_data, columns=sample_names)

        except Exception as e:
            self.logger.error(f"HTSeq-count quantification failed: {str(e)}")
            raise QuantificationError(f"HTSeq-count quantification failed: {str(e)}")

    def get_counting_modes(self) -> List[str]:
        """
        Get available counting modes for HTSeq-count.

        Returns:
            List of available counting modes
        """
        return ["union", "intersection-strict", "intersection-nonempty"]

    def get_stranded_options(self) -> List[str]:
        """
        Get available stranded options for HTSeq-count.

        Returns:
            List of available stranded options
        """
        return ["yes", "no", "reverse"]

    def validate_parameters(self) -> bool:
        """
        Validate HTSeq-count parameters.

        Returns:
            bool: True if parameters are valid
        """
        try:
            # Check if annotation file exists and is readable
            if not self.file_manager.verify_files_exist(self.annotation_file):
                self.logger.error(f"Annotation file not found: {self.annotation_file}")
                return False

            # Check if BAM files exist
            for sample_id, sample_info in self.bam_dict.items():
                bam_file = self._extract_bam_path(sample_info)
                if not self.file_manager.verify_files_exist(bam_file):
                    self.logger.error(f"BAM file not found for {sample_id}: {bam_file}")
                    return False

            # Validate counting mode
            config_params = self._get_config_parameters()
            mode = config_params.get("mode", "-m union").replace("-m ", "")
            valid_modes = self.get_counting_modes()
            if mode not in valid_modes:
                self.logger.warning(f"Invalid counting mode: {mode}. Valid modes: {valid_modes}")

            # Validate strand option
            strand = config_params.get("strand", "-s no").replace("-s ", "")
            valid_strands = self.get_stranded_options()
            if strand not in valid_strands:
                self.logger.warning(f"Invalid strand option: {strand}. Valid options: {valid_strands}")

            return True

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {str(e)}")
            return False

    def get_summary_info(self, output_file: str) -> Dict[str, int]:
        """
        Extract summary information from HTSeq-count output.

        Args:
            output_file: Path to HTSeq-count output file

        Returns:
            Dict containing summary statistics
        """
        summary = {}

        try:
            if not os.path.exists(output_file):
                return summary

            with open(output_file, "r") as f:
                for line in f:
                    if line.startswith("__"):
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            key = parts[0].replace("__", "")
                            value = int(parts[1]) if parts[1].isdigit() else 0
                            summary[key] = value
        except Exception as e:
            self.logger.error(f"Error reading summary from {output_file}: {str(e)}")

        return summary
