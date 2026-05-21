#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flexbar Read Trimmer Module

This module provides functionality for trimming sequencing reads using Flexbar,
a flexible barcode and adapter removal tool for sequencing data.
It supports both single-end and paired-end reads with extensive configuration options.

Features:
    - Adapter and barcode removal through Flexbar
    - Support for both single-end and paired-end reads
    - Quality trimming with configurable thresholds
    - Multiple adapter detection modes
    - Configurable output formats
    - Threading support
    - Local and SLURM cluster execution
    - Comprehensive logging and error handling
    - Resource management optimization

Configuration:
    The trimmer can be configured using flexbar.ini with parameters:
    - adapter sequences
    - quality thresholds
    - barcode settings
    - Other Flexbar-specific parameters

Dependencies:
    - Flexbar: Must be installed and available in system PATH

:Created: May 20, 2021
:Updated: January 15, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import shutil
import subprocess
from typing import List, Union, Tuple
from pathlib import Path

from .base import ReadTrimmer, TrimmingError


class FlexbarTrimmer(ReadTrimmer):
    """
    Class for trimming reads using Flexbar.

    This class implements the Flexbar read trimming functionality, supporting
    both single-end and paired-end reads with resource management and
    comprehensive logging.
    """

    def _init_specific(self, **kwargs):
        """
        Initialize Flexbar-specific attributes.

        Args:
            **kwargs: Additional keyword arguments
        """
        # Ensure paired attribute exists (inherited from parent but add as a safeguard)
        if not hasattr(self, "paired"):
            self.paired = kwargs.get("paired", False)

        # Check for Flexbar executable
        flexbar_path = shutil.which("flexbar")
        if flexbar_path is None:
            if self.dryrun:
                self.logger.warning("Flexbar executable not found in PATH; continuing because dry-run mode is enabled")
                flexbar_path = "flexbar"
            else:
                self.logger.error("Flexbar executable not found in system PATH")
                raise TrimmingError("Flexbar executable not found in system PATH")

        self.executable_path = flexbar_path
        self.logger.debug(f"Found Flexbar at: {self.executable_path}")

    def _get_trimmer_command(self) -> str:
        """
        Get the command for Flexbar.

        Returns:
            str: Command name for Flexbar
        """
        return "flexbar"

    def get_version(self) -> str:
        """
        Get the version of Flexbar.

        Returns:
            str: Version string of Flexbar

        Raises:
            TrimmingError: If version check fails
        """
        try:
            result = subprocess.run(
                [self.executable_path, "--version"],
                shell=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Extract version from output
                for line in result.stdout.splitlines():
                    if "version" in line.lower():
                        return line.strip()

                # If no version line found, return first line
                return result.stdout.strip().split("\n")[0]
            else:
                return "Unknown (installed but version check failed)"
        except Exception as e:
            self.logger.error(f"Failed to get Flexbar version: {str(e)}")
            return "Unknown"

    def prepare_command(self, sample_name: str, reads: List[str]) -> str:
        """
        Prepare Flexbar command for the given sample.

        Args:
            sample_name: Name of the sample
            reads: List of read files [R1] or [R1, R2]

        Returns:
            str: Command string to run Flexbar

        Raises:
            TrimmingError: If command preparation fails
        """
        # Load configuration
        config = self.load_config("flexbar.ini")

        if not config:
            raise TrimmingError("Failed to load configuration file: flexbar.ini")

        # Check if the flexbar section exists
        if "flexbar" not in config:
            # Create a default config if section is missing
            config["flexbar"] = {}
            self.logger.warning("Missing 'flexbar' section in config, using default settings")

        # Get flexbar-specific config
        flexbar_config = config["flexbar"]

        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_results"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Using output directory: {output_dir}")

        # Prepare output file paths
        output_files = self.prepare_output_file(sample_name)

        # Build Flexbar arguments from config
        flexbar_args = []

        # Track if threads parameter is already included
        has_threads_param = False

        # Process other arguments
        for key, value in flexbar_config.items():
            if value and value.strip() != "NA":
                # Skip outdir parameter as we set it manually
                if key == "outdir" or "-t" in value or "--outdir" in value:
                    self.logger.debug(f"Skipping outdir parameter from config: {value}")
                    continue

                # Validate arguments against known Flexbar arguments
                arg_parts = value.split()
                arg_key = arg_parts[0]

                # Check if this is a threads parameter
                if "--threads" in arg_key or "-n" in arg_key:
                    has_threads_param = True
                    value = f"--threads {self.cpu_threads}"

                # Add the argument if it's valid
                flexbar_args.append(value)

        # Add threads parameter if not already included in config
        if not has_threads_param:
            flexbar_args.append(f"--threads {self.cpu_threads}")
            self.logger.debug(f"Adding threads parameter: --threads {self.cpu_threads}")

        # Build final command
        command_parts = [self.executable_path]

        # Add input file paths
        if self.paired:
            if len(reads) < 2:
                raise TrimmingError(f"Paired-end mode requires 2 read files, but only {len(reads)} provided for {sample_name}")
            command_parts.extend(["-r", reads[0], "-p", reads[1]])
        else:
            if len(reads) < 1:
                raise TrimmingError(f"No read file provided for {sample_name}")
            command_parts.extend(["-r", reads[0]])

        # Add output file path with correct naming for consistency
        output_files = self.prepare_output_file(sample_name)
        if self.paired:
            # For paired-end, use the first file as the base name
            output_file = output_files[0]
        else:
            output_file = output_files

        # Flexbar automatically appends .fastq.gz, so remove it from our filename
        output_file = str(output_file).replace(".fastq.gz", "")
        command_parts.extend(["-t", output_file])

        # Add other Flexbar arguments
        command_parts.extend(flexbar_args)

        # Join all command parts into a single command string
        cmd = " ".join(command_parts)
        # Sanitize command string to prevent log injection
        sanitized_cmd = cmd.replace("\n", " ").replace("\r", " ")
        self.logger.debug(f"Prepared Flexbar command for {sample_name}: {sanitized_cmd}")
        return cmd

    def validate_output_files(self, output_files: Union[str, Tuple[str, str]]) -> bool:
        """
        Validate that trimming output files exist and are not empty.

        Args:
            output_files: Output file path(s)

        Returns:
            True if output files exist and are valid
        """
        if self.dryrun:
            return True

        # Flexbar has a specific naming convention for output files
        if self.paired:
            if isinstance(output_files, tuple) and len(output_files) == 2:
                # For paired-end, check both R1 and R2 files
                r1_exists = self.file_manager.verify_files_exist(output_files[0])
                r2_exists = self.file_manager.verify_files_exist(output_files[1])

                if not r1_exists:
                    self.logger.debug(f"Output file R1 missing: {output_files[0]}")
                if not r2_exists:
                    self.logger.debug(f"Output file R2 missing: {output_files[1]}")

                return r1_exists and r2_exists
        else:
            if not isinstance(output_files, tuple):
                # For single-end, check the output file
                exists = self.file_manager.verify_files_exist(output_files)
                if not exists:
                    self.logger.debug(f"Output file missing: {output_files}")
                return exists

        return False

    def prepare_output_file(self, sample_name: str) -> str:
        """
        Override base method to return the actual Flexbar output file path.

        For consistency with Trim Galore, Flexbar should create output files in the
        flexbar_results subdirectory with sample-specific names.

        Args:
            sample_name: Name of the sample

        Returns:
            Path to the Flexbar output file for the sample
        """
        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_results"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        # For consistency with Trim Galore, use sample-specific naming
        # For single-end: {sample_name}_trimmed.fastq.gz
        # For paired-end: {sample_name}_R1_trimmed.fastq.gz and {sample_name}_R2_trimmed.fastq.gz
        if self.paired:
            # For paired-end, we need to handle both R1 and R2
            r1_file = output_dir / f"{sample_name}_R1_trimmed.fastq.gz"
            r2_file = output_dir / f"{sample_name}_R2_trimmed.fastq.gz"
            return (str(r1_file), str(r2_file))
        else:
            # For single-end
            output_file = output_dir / f"{sample_name}_trimmed.fastq.gz"
            return str(output_file)
