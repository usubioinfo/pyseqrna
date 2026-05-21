#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Quantification Module

This module provides the abstract base class and custom exception for all quantification tools
in the pySeqRNA pipeline. It handles common inputs validation, CPU thread allocation strategies,
resource management, directory structure setup, command execution (either locally or via SLURM
scheduler), and post-processing of results.

Features:
    - Abstract interface (BaseQuantifier) for RNA-seq read quantification tools (e.g., featureCounts, HTSeq)
    - Automated input validation for BAM mapping dictionaries and annotation files (GFF/GTF)
    - Resource management with intelligent fallback CPU thread and memory allocation strategies
    - Unified execution layer supporting local multi-processing and SLURM workload manager execution
    - Generic result aggregation helper methods including output cleaning and summary statistics calculation

Configuration:
    Configured via parameters passed to the constructor (such as bam_dict, annotation_file,
    out_dir, param_dir, paired, slurm, dryrun, cpu_threads) and by parsing tool-specific INI config
    files (e.g., featureCount.ini, htseq.ini) via ConfigManager.

Dependencies:
    - pandas
    - pyseqrna.utils (FileManager, CommandExecutor, LogManager, ConfigManager, ResourceManager, DryRunManager)

Classes / Functions / Exceptions:
    - QuantificationError: Custom exception for quantification-related errors.
    - BaseQuantifier: Abstract base class for all quantification tools.

:Created: January 20, 2025
:Updated: February 25, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd

# Import utility modules
from pyseqrna.utils.file_manager import FileManager
from pyseqrna.utils.command_executor import CommandExecutor
from pyseqrna.utils.log_manager import LogManager
from pyseqrna.utils.config_manager import ConfigManager
from pyseqrna.utils.resource_manager import ResourceManager
from pyseqrna.utils.dry_run_manager import DryRunManager


class QuantificationError(Exception):
    """Custom exception for quantification-related errors."""

    pass


class BaseQuantifier(ABC):
    """
    Abstract base class for all quantification tools.

    This class provides the common interface and shared functionality
    for gene expression quantification tools like featureCounts and HTSeq.

    Attributes:
        tool_name (str): Name of the quantification tool
        logger: Logger instance for tracking progress
        file_manager: FileManager instance for file operations
        command_executor: CommandExecutor instance for running commands
        config_manager: ConfigManager instance for configuration management
    """

    def __init__(
        self,
        bam_dict: Dict[str, List[str]],
        annotation_file: str,
        out_dir: Optional[str] = None,
        param_dir: Optional[str] = None,
        paired: bool = False,
        slurm: bool = False,
        dryrun: bool = False,
        job_id: Optional[str] = None,
        cpu_threads: Optional[int] = None,
        memory: Optional[int] = None,
        logger: Optional[Any] = None,
        dry_run_manager: Optional[DryRunManager] = None,
        slurm_config: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the BaseQuantifier.

        Args:
            bam_dict: Dictionary mapping sample names to BAM file paths
            annotation_file: Path to annotation file (GFF/GTF)
            out_dir: Output directory path. Defaults to current directory.
            param_dir: Directory containing parameter files. Defaults to None.
            paired: Whether the data is paired-end
            slurm: Whether to use SLURM for job scheduling. Defaults to False.
            dryrun: Whether to perform a dry run. Defaults to False.
            job_id: SLURM job dependency ID. Defaults to None.
            cpu_threads: Number of CPU cores to use
            memory: Memory limit in GB
            logger: Logger instance
            dry_run_manager: Dry run manager instance
            slurm_config: Optional SLURM configuration overrides
            **kwargs: Additional keyword arguments
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.logger
        else:
            self.logger = logger

        # Extract class name and set tool name
        self.tool_name = self.__class__.__name__.replace("Quantifier", "").lower()
        self.logger.debug(f"Initializing {self.tool_name} quantifier")

        # Initialize utilities
        self.file_manager = FileManager(logger=self.logger)
        self.command_executor = CommandExecutor(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)
        self.config_manager = ConfigManager(logger=self.logger)

        # Store basic attributes
        self.bam_dict = bam_dict
        self.annotation_file = annotation_file
        self.paired = paired
        self.slurm = slurm
        self.dryrun = dryrun
        self.job_id = job_id
        self.param_dir = param_dir
        self.slurm_config = slurm_config or {}
        self.local_jobs = max(1, int(kwargs.get("local_jobs", 1) or 1))

        # Store dry run manager
        self.dry_run_manager = dry_run_manager

        # Set up output directory
        self.out_dir = Path(out_dir or os.getcwd())

        # Initialize config to None - will be loaded by tool-specific classes
        self.config = None

        # Initialize threads to None, will be set after config load
        self.cpu_threads = None

        # Set up CPU threads with intelligent fallback strategy:
        # 1. Use explicitly provided threads if available
        # 2. Otherwise check config file
        # 3. Finally fall back to ResourceManager
        self._setup_cpu_threads(cpu_threads)

        # Set up memory allocation
        if memory is not None:
            self.memory = memory
        else:
            self.memory = self.resource_manager.get_memory_gb(0.8)

        # Initialize additional attributes from kwargs (including param_dir)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate inputs
        self._validate_inputs()

    def _setup_cpu_threads(self, explicit_threads: Optional[int] = None) -> None:
        """
        Set up CPU threads with intelligent fallback strategy.

        This method sets the cpu_threads attribute using the following priority:
        1. Explicit threads parameter provided to the constructor
        2. Threads parameter from configuration file
        3. ResourceManager automatic allocation

        Args:
            explicit_threads: Explicitly provided thread count from constructor
        """
        if explicit_threads is not None:
            # Use explicitly provided threads
            self.cpu_threads = self.resource_manager.resolve_threads(explicit_threads)
            self.logger.debug(f"Using explicitly provided thread count: {self.cpu_threads}")
            return

        # Try to get threads from config
        config_threads = self._get_threads_from_config()
        if config_threads is not None:
            self.cpu_threads = self.resource_manager.resolve_threads(config_threads)
            self.logger.debug(f"Using thread count from config: {self.cpu_threads}")
            return

        # Fall back to ResourceManager automatic allocation
        self.cpu_threads = self.resource_manager.get_cpu_count(0.8)  # Use 80% of available CPUs
        self.logger.debug(f"Using automatically determined thread count: {self.cpu_threads}")

    def _get_threads_from_config(self) -> Optional[int]:
        """
        Try to get thread count from configuration file.

        Different quantification tools use different parameter names for threads, so this
        method checks several common patterns.

        Returns:
            Thread count if found in config, None otherwise
        """
        # Determine config files to try based on tool name
        config_files_to_try = []

        # Map tool names to actual config file names
        if self.tool_name == "featurecounts":
            config_files_to_try.append("featureCount.ini")
        elif self.tool_name == "htseq":
            config_files_to_try.append("htseq.ini")
        else:
            config_files_to_try.append(f"{self.tool_name}.ini")

        # Try each config file
        for config_name in config_files_to_try:
            try:
                config = self.load_config(config_name)
                if not config:
                    continue

                # Check for common thread parameter names in the tool's section
                # Try multiple section names
                possible_sections = []
                if self.tool_name == "featurecounts":
                    possible_sections = ["featureCount", "featurecounts"]
                elif self.tool_name == "htseq":
                    possible_sections = ["htseq-count", "htseq", "HTSeq"]
                else:
                    possible_sections = [self.tool_name]

                for section_name in possible_sections:
                    tool_section = config.get(section_name, {})
                    for thread_param in [
                        "threads",
                        "cpu",
                        "cpu_threads",
                        "cores",
                        "processors",
                        "thread",
                        "T",
                    ]:
                        if thread_param in tool_section:
                            thread_value = tool_section[thread_param]

                            # Handle different formats like "-T 8" or just "8"
                            if isinstance(thread_value, str):
                                # Extract number from string like "-T 8" or "--threads=8"
                                import re

                                match = re.search(r"\b(\d+)\b", thread_value)
                                if match:
                                    return int(match.group(1))
                            elif isinstance(thread_value, int):
                                return thread_value

            except Exception as e:
                self.logger.debug(f"Failed to get threads from config {config_name}: {str(e)}")

        # No thread setting found
        return None

    def load_config(self, default_config_name: str) -> Dict[str, Any]:
        """
        Load the tool-specific configuration file using the dedicated tool config method.

        Args:
            default_config_name: Name of the tool config file

        Returns:
            Dict containing configuration parameters
        """
        try:
            if self.param_dir:
                # Try user-provided param directory first
                self.logger.info(f"Attempting to load tool config from user path: {self.param_dir}")
                return self.config_manager.read_tool_config(default_config_name, self.param_dir)
            else:
                # Fall back to default locations
                self.logger.info(f"Loading default tool config: {default_config_name}")
                return self.config_manager.read_tool_config(default_config_name, None)

        except Exception as e:
            self.logger.warning(f"Error loading config file {default_config_name}: {str(e)}")
            return {}

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.bam_dict:
            raise QuantificationError("BAM dictionary cannot be empty")

        if not self.annotation_file:
            raise QuantificationError("Annotation file path is required")

        if not self.file_manager.verify_files_exist(self.annotation_file):
            raise QuantificationError(f"Annotation file not found: {self.annotation_file}")

        if self.dryrun:
            self.logger.info("DRYRUN: Skipping BAM existence validation for quantification inputs")
            return

        # Validate BAM files exist
        missing_files = []
        for sample_id, sample_info in self.bam_dict.items():
            # Instead of checking length, try to extract BAM path and validate format
            try:
                bam_file = self._extract_bam_path(sample_info)
                if not self.file_manager.verify_files_exist(bam_file):
                    missing_files.append(bam_file)
            except (ValueError, KeyError, IndexError) as e:
                raise QuantificationError(
                    f"Invalid sample info for {sample_id}: {sample_info}. Could not extract BAM path: {str(e)}"
                )

        if missing_files:
            raise QuantificationError(f"Missing BAM files: {missing_files}")

    def _extract_bam_path(self, sample_info: Union[str, List[str], Dict[str, str]]) -> str:
        """
        Extract the BAM file path from sample_info, which may be a string, list, or dict with 'bam' key.
        """
        if isinstance(sample_info, dict) and "bam" in sample_info:
            return sample_info["bam"]
        elif isinstance(sample_info, (list, tuple)):
            # Use the last element if it's a path, or fallback to index 2
            if len(sample_info) > 2:
                return sample_info[2]
            return sample_info[-1]
        elif isinstance(sample_info, str):
            return sample_info
        else:
            raise ValueError(f"Cannot extract BAM path from sample_info: {sample_info}")

    def _detect_annotation_format(self) -> str:
        """
        Detect if the annotation file is GTF or GFF based on extension.
        Returns 'GTF' or 'GFF'.
        """
        ext = os.path.splitext(self.annotation_file)[1].lower()
        if ext in [".gtf"]:
            return "GTF"
        return "GFF"

    @staticmethod
    def _clean_gene_column(df: pd.DataFrame, gene_col: str = "Gene") -> pd.DataFrame:
        """
        Clean up the gene column by removing 'gene:' and 'gene-'.
        """
        if gene_col in df.columns:
            df[gene_col] = df[gene_col].str.replace("gene:", "", regex=False)
            df[gene_col] = df[gene_col].str.replace("gene-", "", regex=False)
        return df

    def _build_sample_command_list(self) -> List[Tuple[str, str]]:
        """
        Build list of commands for each sample.

        Returns:
            List of tuples containing (sample_id, command)
        """
        commands = []

        for sample_id, sample_info in self.bam_dict.items():
            try:
                # Get BAM file path
                bam_file = self._extract_bam_path(sample_info)

                # Generate command for this sample
                command = self._build_command(sample_id, bam_file)
                commands.append((sample_id, command))

            except Exception as e:
                self.logger.error(f"Error building command for sample {sample_id}: {str(e)}")
                continue

        return commands

    def _process_output_files(self) -> pd.DataFrame:
        """
        Process and combine output files into a single count matrix.

        Returns:
            DataFrame containing the final count matrix
        """
        try:
            # Default implementation - subclasses should override
            results_file = self.out_dir / "Raw_Counts.xlsx"

            if results_file.exists():
                return pd.read_excel(str(results_file))
            else:
                self.logger.warning(f"Results file not found: {results_file}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error processing output files: {str(e)}")
            return pd.DataFrame()

    def execute_command(self, commands: Dict[str, str], output_dir: str, tool_name: str) -> Dict[str, str]:
        """
        Execute the given commands either in SLURM or locally.

        Args:
            commands: Dictionary of commands to execute (sample_id: command)
            output_dir: Directory for command output
            tool_name: Name of the tool being executed

        Returns:
            Dict: SLURM job IDs if using SLURM, empty dict otherwise

        Raises:
            QuantificationError: If command execution fails
        """
        try:
            # Check if we have a dry run manager available
            dry_run_manager = getattr(self, "dry_run_manager", None)
            if dry_run_manager is not None and dry_run_manager.is_enabled():
                # Use dry run manager to simulate command execution
                dry_run_manager.simulate_command_execution(
                    stage_name=tool_name,
                    commands=commands,
                    execution_type="slurm" if self.slurm else "local",
                )
                return {}

            # Normal execution
            if self.slurm:
                return self.command_executor.execute_slurm(
                    commands=commands,
                    tool_name=tool_name,
                    job_name=tool_name,
                    outdir=output_dir,
                    dependency=self.job_id,
                    slurm_config=self.slurm_config,
                )
            else:
                self.command_executor.execute_local(
                    commands=commands,
                    tool_name=tool_name,
                    outdir=output_dir,
                    max_workers=self.local_jobs,
                )
                return {}
        except Exception as e:
            raise QuantificationError(f"Command execution failed: {str(e)}")

    @abstractmethod
    def _build_command(self, sample_id: str, bam_file: str) -> str:
        """
        Build the quantification command for a single sample.

        Args:
            sample_id: Sample identifier
            bam_file: Path to BAM file

        Returns:
            str: Complete command string
        """
        pass

    @abstractmethod
    def check_tool_availability(self) -> bool:
        """
        Check if the quantification tool is available.

        Returns:
            bool: True if tool is available, False otherwise
        """
        pass

    def run(self) -> pd.DataFrame:
        """
        Run the quantification process.

        Returns:
            DataFrame containing the count matrix

        Raises:
            QuantificationError: If quantification fails
        """
        self.logger.info(f"Starting {self.tool_name} quantification")

        try:
            # Check tool availability
            if not self.check_tool_availability():
                raise QuantificationError(f"{self.tool_name} is not available in PATH")

            # Create output directory
            if not self.dryrun:
                self.file_manager.create_subdirectory(str(self.out_dir), dry_run=False, preserve_existing=True)

            # Build commands for all samples
            command_list = self._build_sample_command_list()

            if not command_list:
                raise QuantificationError("No valid commands generated")

            # Convert to dictionary format
            commands = {sample_id: cmd for sample_id, cmd in command_list}

            # Execute commands
            self.execute_command(commands, str(self.out_dir), self.tool_name)

            if not self.dryrun:
                # Process output files and create count matrix
                count_matrix = self._process_output_files()

                self.logger.info(f"{self.tool_name} quantification completed successfully")
                return count_matrix
            else:
                self.logger.info(f"DRYRUN: {self.tool_name} quantification simulation completed")
                # Return mock DataFrame for dry run
                sample_names = ["Gene"] + list(self.bam_dict.keys())
                mock_data = [
                    ["gene1"] + [100] * len(self.bam_dict),
                    ["gene2"] + [200] * len(self.bam_dict),
                ]
                return pd.DataFrame(mock_data, columns=sample_names)

        except Exception as e:
            self.logger.error(f"{self.tool_name} quantification failed: {str(e)}")
            raise QuantificationError(f"{self.tool_name} quantification failed: {str(e)}")

    def get_summary_stats(self, count_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the quantification results.

        Args:
            count_matrix: The count matrix DataFrame

        Returns:
            Dict: Summary statistics
        """
        if count_matrix.empty:
            return {}

        # Assume first column is gene names
        count_data = count_matrix.iloc[:, 1:] if count_matrix.shape[1] > 1 else count_matrix

        stats = {
            "total_genes": len(count_matrix),
            "total_samples": count_data.shape[1],
            "total_reads": count_data.sum().sum(),
            "mean_reads_per_gene": count_data.sum(axis=1).mean(),
            "mean_reads_per_sample": count_data.sum(axis=0).mean(),
            "genes_with_zero_counts": (count_data.sum(axis=1) == 0).sum(),
            "samples_processed": list(count_data.columns),
        }

        return stats
