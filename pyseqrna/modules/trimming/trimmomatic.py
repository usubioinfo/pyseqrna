#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trimmomatic Read Trimmer Module

This module provides functionality for trimming sequencing reads using Trimmomatic,
a Java-based tool for quality control and adapter trimming of Illumina NGS data.
It supports single-end and paired-end reads, automates Jar/binary execution detection,
and manages paired/unpaired read separation.

Features:
    - Illumina adapter and quality trimming using Trimmomatic
    - Support for single-end (SE) and paired-end (PE) sequencing read modes
    - Automatic Java runtime check and `trimmomatic.jar` lookup across common system paths
    - Multi-step trimming pipeline execution (e.g. SLIDINGWINDOW, ILLUMINACLIP, MINLEN)
    - Output segregation for paired-end runs (producing paired and unpaired fastq files)
    - Integration with SLURM cluster execution and multi-threaded local runs

Configuration:
    Configured via mode-specific INI configuration files: `trimmomaticPE.ini` for paired-end
    and `trimmomaticSE.ini` for single-end under the `[trimmomatic]` section. Standard parameters
    are set dynamically through constructor parameters.

Dependencies:
    - Trimmomatic: Must be installed (as command binary or `trimmomatic.jar`)
    - Java Runtime Environment (JRE): Required backend execution dependency

Classes / Functions / Exceptions:
    - TrimmomaticTrimmer: ReadTrimmer subclass implementing Trimmomatic execution methods.

:Created: May 20, 2021
:Updated: January 15, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import shutil
import subprocess
from typing import List, Union, Tuple
from pathlib import Path

from .base import ReadTrimmer, TrimmingError


class TrimmomaticTrimmer(ReadTrimmer):
    """
    Class for trimming reads using Trimmomatic.

    This class implements the Trimmomatic read trimming functionality, supporting
    both single-end and paired-end reads with resource management and
    comprehensive logging.
    """

    def _init_specific(self, **kwargs):
        """
        Initialize Trimmomatic-specific attributes.

        Args:
            **kwargs: Additional keyword arguments
        """
        # Ensure paired attribute exists (inherited from parent but add as a safeguard)
        if not hasattr(self, "paired"):
            self.paired = kwargs.get("paired", False)

        # Check for Trimmomatic executable
        trimmomatic_path = shutil.which("trimmomatic")
        if trimmomatic_path is None:
            # Try to find trimmomatic.jar
            java_home = os.environ.get("JAVA_HOME", "")
            if java_home:
                trimmomatic_jar = Path(java_home) / "bin" / "trimmomatic.jar"
                if trimmomatic_jar.exists():
                    trimmomatic_path = str(trimmomatic_jar)
                else:
                    # Try common locations
                    common_paths = [
                        "/usr/share/java/trimmomatic.jar",
                        "/opt/trimmomatic/trimmomatic.jar",
                        "/usr/local/share/trimmomatic/trimmomatic.jar",
                    ]
                    for path in common_paths:
                        if Path(path).exists():
                            trimmomatic_path = path
                            break

        if trimmomatic_path is None:
            if self.dryrun:
                self.logger.warning(
                    "Trimmomatic not found in PATH or common locations; continuing because dry-run mode is enabled"
                )
                trimmomatic_path = "trimmomatic"
            else:
                self.logger.error("Trimmomatic not found in PATH or common locations")
                raise TrimmingError("Trimmomatic not found. Please install Trimmomatic and ensure it's accessible")

        self.executable_path = trimmomatic_path
        self.logger.debug(f"Found Trimmomatic at: {self.executable_path}")

        # Check for Java dependency
        java_path = shutil.which("java")
        if java_path is None:
            self.logger.warning("Java not found in PATH - Trimmomatic requires this dependency")
        else:
            self.logger.debug(f"Found Java at: {java_path}")

    def _get_trimmer_command(self) -> str:
        """
        Get the command for Trimmomatic.

        Returns:
            str: Command name for Trimmomatic
        """
        return "trimmomatic"

    def get_version(self) -> str:
        """
        Get the version of Trimmomatic.

        Returns:
            str: Version string of Trimmomatic

        Raises:
            TrimmingError: If version check fails
        """
        try:
            if self.executable_path.endswith(".jar"):
                cmd = ["java", "-jar", self.executable_path, "-version"]
            else:
                cmd = [self.executable_path, "-version"]

            result = subprocess.run(cmd, shell=False, capture_output=True, text=True)

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
            self.logger.error(f"Failed to get Trimmomatic version: {str(e)}")
            return "Unknown"

    def prepare_command(self, sample_name: str, reads: List[str]) -> str:
        """
        Prepare Trimmomatic command for the given sample.

        Args:
            sample_name: Name of the sample
            reads: List of read files [R1] or [R1, R2]

        Returns:
            str: Command string to run Trimmomatic

        Raises:
            TrimmingError: If command preparation fails
        """
        # Load configuration based on paired-end or single-end mode
        config_file = "trimmomaticPE.ini" if self.paired else "trimmomaticSE.ini"
        config = self.load_config(config_file)

        if not config:
            raise TrimmingError(f"Failed to load configuration file: {config_file}")

        # Check if the trimmomatic section exists
        if "trimmomatic" not in config:
            # Create a default config if section is missing
            config["trimmomatic"] = {}
            self.logger.warning("Missing 'trimmomatic' section in config, using default settings")

        # Get trimmomatic-specific config
        trimmomatic_config = config["trimmomatic"]

        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_trimmed"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Using output directory: {output_dir}")

        # Prepare output file paths
        output_files = self.prepare_output_file(sample_name)

        # Build Trimmomatic arguments from config
        trimmomatic_args = []

        # Track if threads parameter is already included
        has_threads_param = False

        # Process other arguments
        for key, value in trimmomatic_config.items():
            if value and value.strip() != "NA":
                # Skip outdir parameter as we set it manually
                if key == "outdir" or "-t" in value or "--outdir" in value:
                    self.logger.debug(f"Skipping outdir parameter from config: {value}")
                    continue

                # Validate arguments against known Trimmomatic arguments
                arg_parts = value.split()
                arg_key = arg_parts[0]

                # Check if this is a threads parameter
                if "--threads" in arg_key or "-threads" in arg_key:
                    has_threads_param = True
                    value = f"--threads {self.cpu_threads}"

                # Add the argument if it's valid
                trimmomatic_args.append(value)

        # Add threads parameter if not already included in config
        if not has_threads_param:
            trimmomatic_args.append(f"--threads {self.cpu_threads}")
            self.logger.debug(f"Adding threads parameter: --threads {self.cpu_threads}")

        # Build final command
        if self.executable_path.endswith(".jar"):
            command_parts = ["java", "-jar", self.executable_path]
        else:
            command_parts = [self.executable_path]

        # Add PE/SE mode
        if self.paired:
            if len(reads) < 2:
                raise TrimmingError(f"Paired-end mode requires 2 read files, but only {len(reads)} provided for {sample_name}")
            command_parts.append("PE")
            command_parts.extend(reads)

            # Add output files for paired-end
            if isinstance(output_files, tuple) and len(output_files) == 2:
                r1_out, r2_out = output_files
                r1_unpaired = str(Path(output_dir) / f"{sample_name}_R1_unpaired.fastq.gz")
                r2_unpaired = str(Path(output_dir) / f"{sample_name}_R2_unpaired.fastq.gz")
                command_parts.extend([r1_out, r1_unpaired, r2_out, r2_unpaired])
        else:
            command_parts.append("SE")
            if len(reads) < 1:
                raise TrimmingError(f"No read file provided for {sample_name}")
            command_parts.append(reads[0])

            # Add output file for single-end
            if not isinstance(output_files, tuple):
                command_parts.append(output_files)

        # Add other Trimmomatic arguments
        command_parts.extend(trimmomatic_args)

        # Join all command parts into a single command string
        cmd = " ".join(command_parts)
        self.logger.debug(f"Prepared Trimmomatic command for {sample_name}: {cmd}")
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

        # Trimmomatic has a specific naming convention for output files
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
