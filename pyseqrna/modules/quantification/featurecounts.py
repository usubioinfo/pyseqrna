#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FeatureCounts Quantification Module

This module provides a wrapper around the featureCounts tool from the Subread package.
It is designed to count mapped RNA-seq reads against genomic features (e.g., genes, exons, transcripts)
defined in GFF/GTF annotation files, aggregating counts across multiple BAM files into a unified
expression matrix.

Features:
    - Wrapper for the Subread featureCounts command-line utility
    - Support for both single-end and paired-end RNA-seq alignment BAMs
    - Multi-threaded read summarization with customizable CPU core allocation
    - Automated configuration parsing from INI files with format-specific (GFF/GTF) attribute mapping
    - Batch execution mode that processes multiple BAM files simultaneously
    - Feature validation and format parsing of input annotation files

Configuration:
    Configured via parameters passed to the constructor (forwarded to BaseQuantifier)
    and tool-specific configuration file featureCount.ini (e.g., setting feature types,
    attributes, quality filters, and threads).

Dependencies:
    - Subread (featureCounts) command-line tool installed in system PATH
    - pandas
    - pyseqrna.modules.quantification.base (BaseQuantifier, QuantificationError)

Classes / Functions / Exceptions:
    - FeatureCountsQuantifier: Concrete implementation of BaseQuantifier using featureCounts for read quantification.

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


class FeatureCountsQuantifier(BaseQuantifier):
    """
    FeatureCounts implementation for gene expression quantification.

    This class provides functionality to count reads mapping to genomic features
    using featureCounts from the Subread package.

    Attributes:
        tool_name (str): Name of the quantification tool ("featurecounts")
        executable_path (str): Path to featureCounts executable
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize FeatureCountsQuantifier.

        Args:
            *args: Positional arguments passed to BaseQuantifier
            **kwargs: Keyword arguments passed to BaseQuantifier
        """
        super().__init__(*args, **kwargs)
        self.executable_path = None
        self._find_executable()

    def _find_executable(self) -> None:
        """Find featureCounts executable in PATH."""
        self.executable_path = shutil.which("featureCounts")
        if not self.executable_path:
            if self.dryrun:
                self.executable_path = "featureCounts"
                self.logger.warning("featureCounts executable not found in PATH; continuing in dry-run mode")
                return
            self.logger.error("featureCounts executable not found in system PATH")
            raise QuantificationError("featureCounts executable not found in system PATH")

    def check_tool_availability(self) -> bool:
        """
        Check if featureCounts is available.

        Returns:
            bool: True if featureCounts is available, False otherwise
        """
        return self.executable_path is not None

    def _get_config_parameters(self) -> Dict[str, str]:
        """
        Get configuration parameters from config file.

        Returns:
            Dict: Configuration parameters from config file
        """
        # Load configuration using the proper method like other modules
        config = self.load_config("featureCount.ini")

        if not config:
            raise QuantificationError("Failed to load configuration file: featureCount.ini")

        # Check if the featureCount section exists
        if "featureCount" not in config:
            raise QuantificationError("Missing 'featureCount' section in config file")

        # Get featureCount-specific config
        featurecount_config = config["featureCount"]

        # Build parameter dictionary from config
        params = {}

        # Process each config parameter
        for key, value in featurecount_config.items():
            if value and value != "NA":
                params[key] = value

        # Ensure we have essential parameters with reasonable defaults if missing
        # Pipeline/config-level CPU settings are authoritative. The tool config
        # can declare that featureCounts supports threads, but it should not
        # override the run-level --threads value.
        params["threads"] = f"-T {self.cpu_threads}"

        # Adjust format and attribute based on annotation file
        annotation_format = self._detect_annotation_format()
        if annotation_format == "GTF":
            if "attribute" not in params:
                params["attribute"] = "-g gene_id"  # GTF uses gene_id
            elif not params["attribute"].startswith("-g"):
                params["attribute"] = f"-g {params['attribute']}"
        else:
            if "attribute" not in params:
                params["attribute"] = "-g ID"  # GFF uses ID
            elif not params["attribute"].startswith("-g"):
                params["attribute"] = f"-g {params['attribute']}"

        # Ensure feature parameter is formatted correctly
        if "feature" in params and not params["feature"].startswith("-t"):
            params["feature"] = f"-t {params['feature']}"

        return params

    def _build_command(self, sample_id: str, bam_file: str) -> str:
        """
        Build featureCounts command for a single sample.

        Args:
            sample_id: Sample identifier
            bam_file: Path to BAM file

        Returns:
            str: Complete featureCounts command
        """
        if not self.executable_path:
            raise QuantificationError("featureCounts executable not found")

        # Get configuration parameters
        config_params = self._get_config_parameters()

        # Build output file path
        output_file = self.out_dir / f"{sample_id}_counts.txt"

        # Build parameter list from config
        param_parts = []
        for key, value in config_params.items():
            if value and value != "NA" and key != "additional":
                param_parts.append(value)

        # Add paired-end flag if needed
        if self.paired:
            param_parts.append("-p")

        # Build command components
        cmd_parts = [
            self.executable_path,
            "-a",
            self.annotation_file,
            "-o",
            str(output_file),
            *param_parts,
            bam_file,
        ]

        return " ".join(cmd_parts)

    def _process_output_files(self) -> pd.DataFrame:
        """
        Process and combine featureCounts output files into a single count matrix.

        Returns:
            DataFrame containing the final count matrix
        """
        try:
            count_data = []
            sample_names = ["Gene"]

            # Process each sample's output file
            for sample_id in self.bam_dict.keys():
                output_file = self.out_dir / f"{sample_id}_counts.txt"

                if not output_file.exists():
                    self.logger.warning(f"Output file not found for sample {sample_id}: {output_file}")
                    continue

                try:
                    # Read featureCounts output (skip comment lines)
                    df = pd.read_csv(output_file, sep="\t", comment="#", low_memory=False)

                    if len(df.columns) < 7:
                        self.logger.warning(f"Unexpected format in {output_file}")
                        continue

                    # FeatureCounts output columns: Geneid, Chr, Start, End, Strand, Length, Count
                    gene_col = df.columns[0]  # Usually 'Geneid'
                    count_col = df.columns[-1]  # Last column is the count

                    if len(count_data) == 0:
                        # First sample - initialize with gene names
                        count_data = df[[gene_col]].copy()
                        count_data.columns = ["Gene"]

                    # Add count data for this sample
                    count_data[sample_id] = df[count_col].values
                    sample_names.append(sample_id)

                    # Clean up temporary file
                    if not self.dryrun:
                        output_file.unlink()

                    self.logger.debug(f"Processed counts for sample {sample_id}")

                except Exception as e:
                    self.logger.error(f"Error processing output file for {sample_id}: {str(e)}")
                    continue

            if len(count_data) == 0:
                self.logger.error("No valid count data found")
                return pd.DataFrame()

            # Clean gene names (remove common prefixes)
            if "Gene" in count_data.columns:
                count_data["Gene"] = count_data["Gene"].str.replace("gene:", "")
                count_data["Gene"] = count_data["Gene"].str.replace("gene-", "")

            # Save final count matrix
            if not self.dryrun:
                output_excel = self.out_dir / "Raw_Counts.xlsx"
                count_data.to_excel(str(output_excel), index=False)
                self.logger.info(f"Count matrix saved to: {output_excel}")

            return count_data

        except Exception as e:
            self.logger.error(f"Error processing featureCounts output files: {str(e)}")
            return pd.DataFrame()

    def run(self) -> pd.DataFrame:
        self.logger.info("Starting featureCounts quantification (batch mode)")
        try:
            if not self.check_tool_availability():
                raise QuantificationError("featureCounts is not available in PATH")
            if not self.dryrun:
                self.out_dir.mkdir(parents=True, exist_ok=True)
            bam_files = []
            sample_names = []
            for sample_id, sample_info in self.bam_dict.items():
                bam_file = self._extract_bam_path(sample_info)
                bam_files.append(bam_file)
                sample_names.append(sample_id)
            config_params = self._get_config_parameters()
            param_parts = []
            for key, value in config_params.items():
                if value and value != "NA" and key != "additional":
                    param_parts.append(value)
            if self.paired:
                param_parts.append("-p")
            output_file = self.out_dir / "Counts.txt"
            cmd_parts = [
                self.executable_path,
                "-a",
                self.annotation_file,
                "-o",
                str(output_file),
                *param_parts,
                *bam_files,
            ]
            command = " ".join(cmd_parts)
            commands = {"featurecounts_batch": command}
            # Use execute_command for tracking and dry run manager integration
            self.execute_command(commands, str(self.out_dir), tool_name="featurecounts")
            if not self.dryrun:
                df = pd.read_csv(output_file, sep="\t", comment="#", low_memory=False)
                drop_cols = [c for c in ["Chr", "Start", "End", "Strand", "Length"] if c in df.columns]
                count_df = df.drop(columns=drop_cols)
                count_df.columns = ["Gene"] + sample_names
                count_df = self._clean_gene_column(count_df, gene_col="Gene")
                output_excel = self.out_dir / "Raw_Counts.xlsx"
                count_df.to_excel(str(output_excel), index=False)
                self.logger.info(f"Count matrix saved to: {output_excel}")
                os.remove(output_file)
                return count_df
            else:
                self.logger.info("DRYRUN: FeatureCounts quantification simulation completed")
                sample_names = ["Gene"] + sample_names
                mock_data = [
                    ["gene1"] + [100] * len(self.bam_dict),
                    ["gene2"] + [200] * len(self.bam_dict),
                ]
                return pd.DataFrame(mock_data, columns=sample_names)
        except Exception as e:
            self.logger.error(f"FeatureCounts quantification failed: {str(e)}")
            raise QuantificationError(f"FeatureCounts quantification failed: {str(e)}")

    def get_feature_types(self) -> List[str]:
        """
        Get available feature types from the annotation file.

        Returns:
            List of available feature types
        """
        feature_types = set()

        try:
            with open(self.annotation_file, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            feature_types.add(parts[2])
        except Exception as e:
            self.logger.error(f"Error reading annotation file: {str(e)}")

        return sorted(list(feature_types))

    def validate_parameters(self) -> bool:
        """
        Validate featureCounts parameters.

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

            # Check feature types
            feature_types = self.get_feature_types()
            if not feature_types:
                self.logger.warning("No feature types found in annotation file")
            else:
                self.logger.debug(f"Available feature types: {feature_types}")

            return True

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {str(e)}")
            return False
