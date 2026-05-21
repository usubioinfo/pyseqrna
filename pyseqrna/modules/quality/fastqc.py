#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastQC Quality Control Module

This module provides functionality for quality control of high-throughput sequencing reads using FastQC.
It automates the analysis of sequence quality metrics for eukaryotic or prokaryotic libraries,
supporting both single-end and paired-end datasets, and generates HTML/ZIP report files.

Features:
    - High-throughput quality control analysis via FastQC
    - Support for single-end (SE) and paired-end (PE) read datasets
    - Intelligent CPU resource resolution for the `--threads` option via ResourceManager
    - Flexible job orchestration supporting local multiprocessing and SLURM workload manager execution
    - Dynamic verification checking for FastQC output files (.html, .zip, .txt)
    - Output directory isolation within pipeline directory structures

Configuration:
    The module reads parameters from the `fastqc.ini` file under the `[fastqc]` section.
    Tool paths and execution variables are set dynamically through constructor parameters.

Dependencies:
    - FastQC: Must be installed and available in the system PATH

Classes / Functions / Exceptions:
    - FastQCQualityControl: QualityControl subclass implementing FastQC execution methods.

:Created: May 20, 2021
:Updated: January 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import shutil
import subprocess
from typing import List
from pathlib import Path

from .base import QualityControl, QualityControlError


class FastQCQualityControl(QualityControl):
    """
    Class for quality control using FastQC.

    This class implements the FastQC quality control functionality, supporting
    both single-end and paired-end reads with resource management and
    comprehensive logging.
    """

    def __init__(self, **kwargs):
        """Initialize FastQC with tool-specific attributes."""
        self.name = "fastqc"

        super().__init__(**kwargs)

    def _init_specific(self, **kwargs):
        """
        Initialize FastQC-specific attributes.

        Args:
            **kwargs: Additional keyword arguments
        """
        # Ensure paired attribute exists (inherited from parent but add as a safeguard)
        if not hasattr(self, "paired"):
            self.paired = kwargs.get("paired", False)

        # Check for FastQC executable
        fastqc_path = shutil.which("fastqc")
        if fastqc_path is None:
            if self.dryrun:
                self.logger.warning("FastQC executable not found in PATH; continuing because dry-run mode is enabled")
                fastqc_path = "fastqc"
            else:
                self.logger.error("FastQC executable not found in system PATH")
                raise QualityControlError("FastQC executable not found in system PATH")

        self.executable_path = fastqc_path
        self.logger.debug(f"Found FastQC at: {self.executable_path}")

        # Load configuration once during initialization
        self.fastqc_config = self.load_config("fastqc.ini")
        if not self.fastqc_config:
            self.logger.warning("Failed to load configuration file: fastqc.ini, using defaults")
            self.fastqc_config = {}

        # Pre-resolve thread counts in config to avoid repeated ResourceManager calls
        self._resolve_config_threads()

        self.logger.debug("FastQC quality control initialization complete")

    def _resolve_config_threads(self):
        """Pre-resolve thread counts in config to avoid repeated ResourceManager calls."""
        fastqc_config = self.fastqc_config.get("fastqc", {})

        for key, value in list(fastqc_config.items()):
            if value and value.strip() != "NA":
                arg_parts = value.split()
                arg_key = arg_parts[0]

                # Check if this is a threads parameter
                if "--threads" in arg_key or "-t" in arg_key:
                    # Extract thread count from config
                    config_threads = None
                    if len(arg_parts) > 1:
                        try:
                            config_threads = int(arg_parts[1])
                        except (ValueError, IndexError):
                            self.logger.warning(f"Could not parse thread count from config: {value}")

                    # Use ResourceManager to determine thread count once
                    if config_threads is not None:
                        resolved_threads = self.resource_manager.resolve_threads(config_threads)
                        # Replace with the resolved thread count
                        fastqc_config[key] = f"--threads {resolved_threads}"
                    else:
                        # If thread count couldn't be parsed, use the auto-determined count
                        fastqc_config[key] = f"--threads {self.cpu_threads}"

    def _get_tool_command(self) -> str:
        """
        Get the command for FastQC.

        Returns:
            str: Command name for FastQC
        """
        return "fastqc"

    def get_version(self) -> str:
        """
        Get the version of FastQC.

        Returns:
            str: Version string of FastQC

        Raises:
            QualityControlError: If version check fails
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
            self.logger.error(f"Failed to get FastQC version: {str(e)}")
            return "Unknown"

    def prepare_command(self, sample_name: str, reads: List[str]) -> str:
        """
        Prepare FastQC command for the given sample.

        Args:
            sample_name: Name of the sample
            reads: List of read files [R1] or [R1, R2]

        Returns:
            str: Command string to run FastQC

        Raises:
            QualityControlError: If command preparation fails
        """
        # Use configuration loaded during initialization
        config = self.fastqc_config

        # Get fastqc-specific config
        fastqc_config = config.get("fastqc", {})

        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_results"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Using output directory: {output_dir}")

        # Build FastQC arguments from config
        fastqc_args = []

        # Track if threads parameter is already included
        has_threads_param = False

        # Process other arguments
        for key, value in fastqc_config.items():
            if value and value.strip() != "NA":
                # Validate arguments against known FastQC arguments
                arg_parts = value.split()
                arg_key = arg_parts[0]

                # Check if this is a threads parameter (already resolved during initialization)
                if "--threads" in arg_key or "-t" in arg_key:
                    has_threads_param = True
                    fastqc_args.append(f"--threads {self.cpu_threads}")
                    continue

                # Add the argument if it's valid
                fastqc_args.append(value)

        # Add threads parameter if not already included in config
        if not has_threads_param:
            fastqc_args.append(f"--threads {self.cpu_threads}")
            self.logger.debug(f"Adding threads parameter: --threads {self.cpu_threads}")

        # Build final command
        command_parts = [self.executable_path]

        # Add output directory
        command_parts.extend(["-o", str(output_dir)])

        # Add other FastQC arguments
        command_parts.extend(fastqc_args)

        # Add input file paths (FastQC can handle multiple files in one command)
        command_parts.extend(reads)

        # Join all command parts into a single command string
        cmd = " ".join(command_parts)
        self.logger.debug(f"Prepared FastQC command for {sample_name}: {cmd}")
        return cmd

    def validate_output_files(self, output_dir: str) -> bool:
        """
        Validate that FastQC output files exist and are not empty.

        Args:
            output_dir: Output directory path

        Returns:
            True if output files exist and are valid
        """
        if self.dryrun:
            return True

        # Check if output directory exists
        if not os.path.exists(output_dir):
            self.logger.debug(f"Output directory missing: {output_dir}")
            return False

        # Check if directory has any files
        files = os.listdir(output_dir)
        if not files:
            self.logger.debug(f"Output directory is empty: {output_dir}")
            return False

        # Check for FastQC-specific output files (HTML reports, zip files, etc.)
        fastqc_files = [f for f in files if f.endswith((".html", ".zip", ".txt"))]
        if not fastqc_files:
            self.logger.debug(f"No FastQC output files found in: {output_dir}")
            return False

        return True

    def prepare_output_file(self, sample_name: str) -> str:
        """
        Override base method to use correct directory structure.

        Args:
            sample_name: Name of the sample

        Returns:
            Output directory path for the sample
        """
        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_results"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir)
