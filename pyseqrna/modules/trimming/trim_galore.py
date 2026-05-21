#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trim Galore Read Trimmer Module

This module provides functionality for trimming sequencing reads using Trim Galore,
a wrapper tool around Cutadapt and FastQC. It automates quality control and adapter trimming
for eukaryotic and prokaryotic RNA-seq read libraries, supporting single-end and paired-end modes.

Features:
    - High-throughput adapter and quality trimming via Trim Galore / Cutadapt
    - Automatic single-end and paired-end sequencing reads validation and trimming
    - FastQC integration for pre- and post-trim quality control assessment
    - Intelligent CPU resource scaling using the `--cores` setting resolved via ResourceManager
    - Execution scheduling supporting SLURM cluster execution and multi-threaded local runs
    - Consistent output renaming using sample basenames (e.g. `_val_1.fq.gz` / `_val_2.fq.gz`)

Configuration:
    Configured via `trim_galore.ini` under the `[trim_galore]` section, allowing options
    for quality thresholds, adapter sequences, stringency, and read parameters. Tool paths
    and execution variables are passed using constructor parameters.

Dependencies:
    - Trim Galore: Must be installed and available in system PATH
    - Cutadapt: Required backend dependency for Trim Galore
    - FastQC: Optional dependency for automated quality reports

Classes / Functions / Exceptions:
    - TrimGaloreTrimmer: ReadTrimmer subclass implementing Trim Galore execution methods.

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


class TrimGaloreTrimmer(ReadTrimmer):
    """
    Trim Galore implementation of the ReadTrimmer interface.

    Trim Galore is a wrapper around Cutadapt and FastQC that provides
    automated quality and adapter trimming optimized for Illumina data.
    """

    def __init__(self, **kwargs):
        """Initialize TrimGalore trimmer with tool-specific attributes."""
        self.name = "trim_galore"
        self.has_fastqc = True  # Trim Galore has built-in FastQC support

        super().__init__(**kwargs)

    def _init_specific(self, **kwargs):
        """
        Perform TrimGalore-specific initialization after base initialization.

        Args:
            **kwargs: Additional keyword arguments
        """
        # Initialize has_fastqc attribute to False by default
        self.has_fastqc = False

        # Ensure paired attribute exists (inherited from parent but add as a safeguard)
        # This prevents errors like "'TrimGaloreTrimmer' object has no attribute 'paired'"
        if not hasattr(self, "paired"):
            self.paired = kwargs.get("paired", False)

        # Check for Trim Galore executable
        trim_galore_path = shutil.which("trim_galore")
        if trim_galore_path is None:
            if self.dryrun:
                self.logger.warning("Trim Galore executable not found in PATH; continuing because dry-run mode is enabled")
                trim_galore_path = "trim_galore"
            else:
                self.logger.error("Trim Galore executable not found in system PATH")
                raise TrimmingError("Trim Galore executable not found in system PATH")

        self.executable_path = trim_galore_path
        self.logger.debug(f"Found Trim Galore at: {self.executable_path}")

        # Check for Cutadapt dependency
        cutadapt_path = shutil.which("cutadapt")
        if cutadapt_path is None:
            self.logger.warning("Cutadapt not found in PATH - Trim Galore requires this dependency")
        else:
            self.logger.debug(f"Found Cutadapt at: {cutadapt_path}")

        # Check for FastQC (optional)
        fastqc_path = shutil.which("fastqc")
        if fastqc_path is None:
            self.logger.warning("FastQC not found in PATH - QC reports will not be generated")
        else:
            self.logger.debug(f"Found FastQC at: {fastqc_path}")
            self.has_fastqc = True

        # Load configuration once during initialization
        self.trim_galore_config = self.load_config("trim_galore.ini")
        if not self.trim_galore_config:
            raise TrimmingError("Failed to load configuration file: trim_galore.ini")

        # Pre-resolve thread counts in config to avoid repeated ResourceManager calls
        self._resolve_config_threads()

        self.logger.debug(f"TrimGalore trimmer initialization complete. FastQC available: {self.has_fastqc}")

    def _resolve_config_threads(self):
        """Pre-resolve thread counts in config to avoid repeated ResourceManager calls."""
        trim_galore_config = self.trim_galore_config.get("trim_galore", {})

        for key, value in list(trim_galore_config.items()):
            if value and value.strip() != "NA":
                arg_parts = value.split()
                arg_key = arg_parts[0]

                # Check if this is a cores parameter (Trim Galore uses --cores)
                if arg_key in {"--cores", "-cores"}:
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
                        trim_galore_config[key] = f"--cores {resolved_threads}"
                    else:
                        # If thread count couldn't be parsed, use the auto-determined count
                        trim_galore_config[key] = f"--cores {self.cpu_threads}"

                # Check older Trim Galore style thread parameters (using -j or --threads)
                elif "--threads" in arg_key or "-j" in arg_key:
                    self.logger.warning("Trim Galore now prefers --cores instead of --threads or -j")

                    # Extract thread count from config
                    config_threads = None
                    if len(arg_parts) > 1:
                        try:
                            config_threads = int(arg_parts[1])
                        except (ValueError, IndexError):
                            self.logger.warning(f"Could not parse thread count from config: {value}")

                    # Use ResourceManager to determine thread count once
                    resolved_threads = (
                        self.resource_manager.resolve_threads(config_threads) if config_threads else self.cpu_threads
                    )

                    # Replace with the recommended --cores parameter
                    trim_galore_config[key] = f"--cores {resolved_threads}"

    def _get_trimmer_command(self) -> str:
        """
        Get the command for Trim Galore.

        Returns:
            str: Command name for Trim Galore
        """
        return "trim_galore"

    def get_version(self) -> str:
        """
        Get the version of Trim Galore.

        Returns:
            str: Version string of Trim Galore

        Raises:
            TrimmingError: If version check fails
        """
        try:
            cmd = [self.executable_path, "--version"]
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
            self.logger.error(f"Failed to get Trim Galore version: {str(e)}")
            return "Unknown"

    def prepare_command(self, sample_name: str, reads: List[str]) -> str:
        """
        Prepare Trim Galore command for the given sample.

        Args:
            sample_name: Name of the sample
            reads: List of read files [R1] or [R1, R2]

        Returns:
            str: Command string to run Trim Galore

        Raises:
            TrimmingError: If command preparation fails
        """
        # Use configuration loaded during initialization
        config = self.trim_galore_config

        # Get trim_galore-specific config
        trim_galore_config = config["trim_galore"]

        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_results"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Using output directory: {output_dir}")

        # Prepare output file paths
        self.prepare_output_file(sample_name)

        # Build Trim Galore arguments from config
        trim_galore_args = []

        # Track if cores/threads parameter is already included
        has_cores_param = False

        # Process other arguments
        for key, value in trim_galore_config.items():
            if value and value.strip() != "NA":
                # Skip outdir parameter as we set it manually
                if key == "outdir" or "-o" in value or "--outdir" in value:
                    self.logger.debug(f"Skipping outdir parameter from config: {value}")
                    continue

                # Validate arguments against known Trim Galore arguments
                arg_parts = value.split()
                arg_key = arg_parts[0]

                # Check if this is a cores parameter (already resolved during initialization)
                if arg_key in {"--cores", "-cores"}:
                    has_cores_param = True
                    trim_galore_args.append(f"--cores {self.cpu_threads}")
                    continue

                # Check older Trim Galore style thread parameters (already resolved during initialization)
                elif "--threads" in arg_key or "-j" in arg_key:
                    has_cores_param = True
                    trim_galore_args.append(f"--cores {self.cpu_threads}")
                    continue

                # Add the argument if it's valid
                trim_galore_args.append(value)

        # Add cores parameter if not already included in config
        if not has_cores_param:
            trim_galore_args.append(f"--cores {self.cpu_threads}")
            self.logger.debug(f"Adding cores parameter: --cores {self.cpu_threads}")

        # Build final command
        command_parts = [self.executable_path]

        # Add output directory
        command_parts.extend(["-o", str(output_dir)])

        # Add basename to use sample name for output files
        command_parts.extend(["--basename", sample_name])

        # Add paired option if needed
        if self.paired:
            if len(reads) < 2:
                raise TrimmingError(f"Paired-end mode requires 2 read files, but only {len(reads)} provided for {sample_name}")
            command_parts.append("--paired")

        # Add FastQC option if available
        if self.has_fastqc and not any("--fastqc" in arg for arg in trim_galore_args):
            command_parts.append("--fastqc")

        # Add other Trim Galore arguments
        command_parts.extend(trim_galore_args)

        # Add input file paths
        if self.paired:
            command_parts.extend([reads[0], reads[1]])
        else:
            if len(reads) < 1:
                raise TrimmingError(f"No read file provided for {sample_name}")
            command_parts.append(reads[0])

        # Join all command parts into a single command string
        cmd = " ".join(command_parts)
        self.logger.debug(f"Prepared Trim Galore command for {sample_name}: {cmd}")
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
        return super().validate_output_files(output_files)

    def prepare_output_file(self, sample_name: str) -> str:
        """
        Override base method to return the actual trimmed file path.

        Args:
            sample_name: Name of the sample

        Returns:
            Path to the trimmed file for the sample
        """
        # Use the quality and trimming directory created by pipeline
        quality_trim_dir = Path(self.out_dir) / "1.Quality_and_trimming"
        output_dir = quality_trim_dir / f"{self.name}_results"

        # Create tool-specific directory if it doesn't exist
        if not self.dryrun:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Use sample name for trimmed file naming instead of original filename
        # This ensures consistent naming with sample replication names (M1A, M1B, etc.)

        # Trim Galore naming convention: adds "_trimmed" before the extension for single-end
        # For paired-end, it adds "_val_1" and "_val_2" before the extension
        if self.paired:
            # For paired-end, we need to handle both R1 and R2
            trimmed_file_r1 = output_dir / f"{sample_name}_val_1.fq.gz"
            trimmed_file_r2 = output_dir / f"{sample_name}_val_2.fq.gz"
            return (str(trimmed_file_r1), str(trimmed_file_r2))
        else:
            # For single-end
            trimmed_file = output_dir / f"{sample_name}_trimmed.fq.gz"

        return str(trimmed_file)

        return False
