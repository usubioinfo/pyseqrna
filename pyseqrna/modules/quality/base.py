#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Quality Control Module

This module defines the abstract base class and custom exceptions for all quality control implementations
in pySeqRNA. The QualityControl class provides a common interface, configuration loading, resource optimization,
and shell execution features (both local and SLURM cluster) that specific quality control tools inherit.

Features:
    - Abstract base class establishing standard quality control tool interfaces
    - Intelligent CPU resource resolution utilizing ResourceManager fallbacks
    - Support for local multi-threaded execution and SLURM workload manager job submissions
    - Automated sample and read file validation for raw/processed FASTQ input sequences
    - Output directory management and directory status checking
    - Configurable execution dry-runs simulating quality control tool command preparation

Configuration:
    Configured programmatically via constructor parameters, including 'sample_dict', 'out_dir',
    'param_dir', 'paired', 'slurm', 'dryrun', 'cpu_threads', and 'slurm_config'. Specific settings
    for individual tools are loaded from external configuration files (e.g. fastqc.ini) via ConfigManager.

Dependencies:
    - pySeqRNA utilities (LogManager, FileManager, ConfigManager, ResourceManager, CommandExecutor, DryRunManager)

Classes / Functions / Exceptions:
    - QualityControlError: Custom exception class raised for quality control execution errors.
    - QualityControl: Abstract base class outlining standard quality control, execution, and validation interfaces.

:Created: May 20, 2021
:Updated: January 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import shutil

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    pass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

# Import utility modules
from pyseqrna.utils import (
    LogManager,
    FileManager,
    ConfigManager,
    ResourceManager,
    CommandExecutor,
    DryRunManager,
)


class QualityControlError(Exception):
    """Custom exception for quality control-related errors."""

    pass


class QualityControl(ABC):
    """
    Base abstract class for quality control tools.

    This class defines the common interface and provides shared functionality
    for all quality control implementations. Specific quality control implementations
    should inherit from this class and implement the required abstract methods.

    Attributes:
        name (str): Name of the quality control tool
        sample_dict (Dict[str, List[str]]): Dictionary mapping sample names to input files
        out_dir (Path): Directory for output files
        paired (bool): Whether using paired-end reads
        slurm (bool): Whether to use SLURM for job scheduling
        dryrun (bool): Whether to perform a dry run without executing commands
        job_id (Optional[str]): SLURM job dependency ID
        cpu_threads (int): Number of CPU threads to use
        logger (logging.Logger): Logger for the quality control tool
    """

    # Supported file extensions for input reads
    VALID_READ_EXTENSIONS: Set[str] = {".fq", ".fastq", ".fq.gz", ".fastq.gz"}

    def __init__(
        self,
        sample_dict: Dict[str, List[str]],
        out_dir: Optional[str] = None,
        param_dir: Optional[str] = None,
        paired: bool = False,
        slurm: bool = False,
        dryrun: bool = False,
        job_id: Optional[str] = None,
        cpu_threads: Optional[int] = None,
        logger: Optional[Any] = None,
        dry_run_manager: Optional[DryRunManager] = None,
        slurm_config: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the QualityControl base class.

        Args:
            sample_dict: Dictionary mapping sample names to input files
            out_dir: Output directory path. Defaults to current directory.
            param_dir: Directory containing parameter files. Defaults to None.
            paired: Whether using paired-end reads. Defaults to False.
            slurm: Whether to use SLURM for job scheduling. Defaults to False.
            dryrun: Whether to perform a dry run. Defaults to False.
            job_id: SLURM job dependency ID. Defaults to None.
            cpu_threads: Number of CPU threads to use. If None, will be determined automatically.
            logger: Custom logger instance. Defaults to None.
            **kwargs: Additional keyword arguments for specific quality control implementations

        Raises:
            ValueError: If sample_dict is invalid or empty
            FileNotFoundError: If config_file is specified but doesn't exist
        """
        # Initialize logger
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.get_logger(__name__)
        else:
            self.logger = logger

        # Extract class name and set quality control tool name
        self.name = self.__class__.__name__.replace("QualityControl", "").lower()
        self.logger.debug(f"Initializing {self.name} quality control tool")

        # Initialize utilities
        self.file_manager = FileManager(logger=self.logger)
        self.command_executor = CommandExecutor(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)

        # Ensure critical attributes are explicitly assigned before validation
        # This prevents attribute errors in child classes that might access these
        # attributes during their initialization
        self.paired = paired
        self.slurm = slurm
        self.dryrun = dryrun
        self.job_id = job_id
        self.param_dir = param_dir
        self.slurm_config = slurm_config or {}
        self.local_jobs = max(1, int(kwargs.get("local_jobs", 1) or 1))

        # Store paired status in kwargs to ensure it's available to subclasses
        kwargs["paired"] = paired

        # Store dry run manager
        self.dry_run_manager = dry_run_manager

        # Validate sample dictionary
        self._validate_sample_dict(sample_dict)
        self.sample_dict = sample_dict

        # Set up output directory. Absolute paths are valid because users often
        # run standalone modules from a project directory but write elsewhere.
        self.out_dir = Path(out_dir or os.getcwd()).resolve()

        # Load configuration if provided
        self.config = None
        if param_dir:
            config_manager = ConfigManager(logger=self.logger)
            self.config = config_manager.read_config(param_dir)

        # Initialize threads to None, will be set after config load
        self.cpu_threads = None

        # Initialize additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Perform additional initialization specific to the quality control tool type
        self._init_specific(**kwargs)

        # Log the paired status to help with debugging
        self.logger.debug(f"Paired-end mode: {self.paired}")

        # Set up CPU threads with intelligent fallback strategy:
        # 1. Use explicitly provided threads if available
        # 2. Otherwise check config file
        # 3. Finally fall back to ResourceManager
        self._setup_cpu_threads(cpu_threads)

        self.logger.debug(f"{self.name} quality control tool initialization complete")

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

        Different quality control tools use different parameter names for threads, so this
        method checks several common patterns.

        Returns:
            Thread count if found in config, None otherwise
        """
        # Determine config files to try based on tool name
        config_files_to_try = [f"{self.name}.ini"]

        # Try each config file
        for config_name in config_files_to_try:
            try:
                config = self.load_config(config_name)
                if not config:
                    continue

                # Check for common thread parameter names in the tool's section
                tool_section = config.get(self.name, {})
                for thread_param in [
                    "threads",
                    "cpu",
                    "cpu_threads",
                    "cores",
                    "processors",
                    "thread",
                    "t",
                ]:
                    if thread_param in tool_section:
                        thread_value = tool_section[thread_param]

                        # Handle different formats like "-t 8" or just "8"
                        if isinstance(thread_value, str):
                            # Extract number from string like "-t 8" or "--threads=8"
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

    def _validate_sample_dict(self, sample_dict: Dict[str, List[str]]) -> None:
        """
        Validate the sample dictionary.

        Args:
            sample_dict: Dictionary mapping sample names to input files

        Raises:
            ValueError: If sample_dict is invalid or empty
        """
        if not sample_dict:
            raise ValueError("sample_dict cannot be empty")

        if not isinstance(sample_dict, dict):
            raise ValueError("sample_dict must be a dictionary")

        # Validate that each entry has appropriate file paths
        for sample, reads in sample_dict.items():
            if not isinstance(reads, list):
                raise ValueError(f"Sample {sample} reads must be a list")

            if not reads:
                self.logger.warning(f"Sample {sample} has no read files")

            # Detect expanded sample format [sample_id, group, fastq_path(s)]
            file_paths = []
            if len(reads) >= 3:
                # This appears to be the expanded format [sample_id, group, fastq_path, ...]
                if self.paired and len(reads) >= 4:
                    # For paired-end: extract read files at positions 2 and 3
                    file_paths = [reads[2], reads[3]]
                else:
                    # For single-end: extract read file at position 2
                    file_paths = [reads[2]]
            else:
                # Standard format: all elements are file paths
                file_paths = reads

            # Validate extracted file paths
            for read_file in file_paths:
                # Skip validation for metadata elements that are not file paths
                if not isinstance(read_file, str):
                    self.logger.warning(f"Non-string value found in sample data: {read_file}")
                    continue

                # Add debug logging to show full path
                abs_path = os.path.abspath(read_file)
                self.logger.debug(f"Validating read file for sample {sample}: {read_file}")
                self.logger.debug(f"Absolute path: {abs_path}")
                self.logger.debug(f"File exists check: {os.path.exists(abs_path)}")

                if self.dryrun:
                    self.logger.debug(f"DRYRUN: Skipping read file existence check for {read_file}")
                elif not self.file_manager.verify_files_exist(read_file):
                    raise ValueError(f"Read file does not exist: {read_file}")

                # Check file extension
                if not any(read_file.endswith(ext) for ext in self.VALID_READ_EXTENSIONS):
                    self.logger.warning(f"File {read_file} has an unusual extension for a FASTQ file")

    def _init_specific(self, **kwargs: Any) -> None:
        """
        Initialize attributes specific to the quality control implementation.

        This method can be overridden by subclasses to initialize additional
        attributes or perform specific initialization steps.

        Args:
            **kwargs: Additional keyword arguments from constructor
        """
        pass

    def load_config(self, default_config_name: str) -> Dict[str, Any]:
        """
        Load the tool-specific configuration file using the dedicated tool config method.

        Args:
            default_config_name: Name of the tool config file

        Returns:
            Dict containing configuration parameters
        """
        config_manager = ConfigManager(logger=self.logger)

        try:
            if self.param_dir:
                # Try user-provided param directory first
                self.logger.info(f"Attempting to load tool config from user path: {self.param_dir}")
                return config_manager.read_tool_config(default_config_name, self.param_dir)
            else:
                # Fall back to default locations
                self.logger.info(f"Loading default tool config: {default_config_name}")
                return config_manager.read_tool_config(default_config_name, None)

        except Exception as e:
            self.logger.warning(f"Error loading config file {default_config_name}: {str(e)}")
            return {}

    @abstractmethod
    def prepare_command(self, sample_name: str, reads: List[str]) -> str:
        """
        Prepare the command string for quality control of the specified sample.

        This method must be implemented by each quality control tool to construct the
        command string for the specific quality control tool.

        Args:
            sample_name: Name of the sample
            reads: Paths to read files for this sample

        Returns:
            Command string to be executed

        Raises:
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError("Subclass must implement prepare_command method")

    def prepare_output_file(self, sample_name: str) -> str:
        """
        Prepare output file path for the specified sample.

        Args:
            sample_name: Name of the sample

        Returns:
            Output file path for the sample
        """
        # Use the quality control tool name in the output directory for consistent organization
        out_dir = Path(self.out_dir) / f"{self.name}_results"
        if not self.dryrun:
            # Use force flag if provided
            force_flag = getattr(self, "force", False)
            self.file_manager.create_subdirectory(str(out_dir), preserve_existing=not force_flag)

        return str(out_dir)

    def validate_output_files(self, output_dir: str) -> bool:
        """
        Validate that quality control output files exist and are not empty.

        Args:
            output_dir: Path to output directory

        Returns:
            True if files exist and are valid, False otherwise
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
            self.logger.debug(f"Output directory is empty: {repr(output_dir)}")
            return False

        return True

    def execute_command(self, commands: Dict[str, str], output_dir: str, tool_name: str) -> Dict[str, str]:
        """
        Execute the given command either in SLURM or locally.

        Args:
            commands: Dictionary of commands to execute (sample_id: command)
            output_dir: Directory for command output
            tool_name: Name of the tool being executed

        Returns:
            Dict: SLURM job IDs if using SLURM, empty dict otherwise

        Raises:
            QualityControlError: If command execution fails
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
                    dependency=self.job_id or "",
                    slurm_config=self.slurm_config,
                )
            else:
                local_results = self.command_executor.execute_local(
                    commands=commands,
                    tool_name=tool_name,
                    outdir=output_dir,
                    max_workers=self.local_jobs,
                )
                failed = [sample for sample, ok in local_results.items() if not ok]
                if failed:
                    raise QualityControlError(f"{tool_name} failed for {len(failed)} sample(s): {', '.join(failed)}")
                return local_results
        except Exception as e:
            raise QualityControlError(f"Command execution failed: {str(e)}")

    def run(self) -> Dict[str, str]:
        """
        Run quality control for all samples in the sample dictionary.

        This method iterates through all samples, prepares the command for each,
        executes the command, and returns a dictionary of output file paths.

        Returns:
            Dictionary mapping sample names to output file paths

        Raises:
            QualityControlError: If command execution fails for any sample
        """
        self.logger.info(f"Running {self.name} quality control for {len(self.sample_dict)} samples")
        self.logger.debug(f"Sample dictionary keys: {list(self.sample_dict.keys())}")

        results = {}
        commands = {}

        try:
            self.logger.debug(f"Sample dict: {self.sample_dict}")
        except Exception as e:
            self.logger.debug(f"Error logging sample dict: {str(e)}")

        for sample_name, sample_info in self.sample_dict.items():
            self.logger.debug(f"Processing sample: {sample_name}")
            self.logger.debug(f"Sample info: {sample_info}")

            # Extract fastq file path from sample_info
            # Check if this is expanded sample info format
            if isinstance(sample_info, list):
                self.logger.debug(f"List format detected for {sample_name}")

                if self.paired:
                    # For paired-end data, check sample format
                    if len(sample_info) >= 4:
                        # This is the expanded format: [sample_id, group, fastq_path_R1, fastq_path_R2]
                        self.logger.debug(f"Expanded paired-end format detected for {sample_name}")
                        reads = [
                            sample_info[2],
                            sample_info[3],
                        ]  # Extract R1 and R2 paths
                    else:
                        self.logger.warning(
                            f"Paired mode requires 2 read files (at positions 2 and 3), but only {len(sample_info)} "
                            f"elements provided for {sample_name}, skipping"
                        )
                        continue
                else:
                    # For single-end data
                    if len(sample_info) >= 3:
                        # This is the expanded format: [sample_id, group, fastq_path]
                        self.logger.debug(f"Expanded single-end format detected for {sample_name}")
                        reads = [sample_info[2]]  # Extract just the FASTQ path
                    else:
                        # If fewer than 3 elements, assume all are read files
                        self.logger.debug(f"Simple list format detected for {sample_name}")
                        reads = sample_info
            else:
                # Not a list - this is unexpected
                self.logger.warning(f"Unexpected sample format for {sample_name}: {type(sample_info)}")
                continue

            self.logger.debug(f"Read files: {reads}")

            # Validate sample reads
            if not reads:
                self.logger.warning(f"No reads for sample {sample_name}, skipping")
                continue

            if self.paired and len(reads) < 2:
                self.logger.warning(
                    f"Paired mode requires 2 read files, but only {len(reads)} provided for {sample_name}, skipping"
                )
                continue

            # Prepare output file(s)
            output_dir = self.prepare_output_file(sample_name)
            results[sample_name] = output_dir
            self.logger.debug(f"Output directory for {sample_name}: {output_dir}")

            # Prepare command
            try:
                cmd = self.prepare_command(sample_name, reads)
                commands[sample_name] = cmd
                self.logger.debug(f"Prepared command for {sample_name}: {cmd}")
            except Exception as e:
                self.logger.error(f"Error preparing command for sample {sample_name}: {str(e)}")
                raise QualityControlError(f"Command preparation failed for {sample_name}: {str(e)}")

        # Execute all commands
        # Use quality and trimming directory for logs
        quality_trim_dir = self.out_dir / "1.Quality_and_trimming"
        if not self.dryrun:
            self.logger.debug(f"Executing commands for {len(commands)} samples")
            job_ids = self.execute_command(commands, str(quality_trim_dir), self.name)
        else:
            self.logger.debug(f"DRY RUN: Would execute commands for {len(commands)} samples")
            job_ids = self.execute_command(commands, str(quality_trim_dir), self.name)

        # If not using SLURM, validate output files after command execution.
        if not self.slurm:
            self.logger.debug("Validating output files")
            for sample_name, output_dir in results.items():
                self.logger.debug(f"Checking output directory for {sample_name}")
                if not self.validate_output_files(output_dir):
                    self.logger.warning(f"Output files missing for {sample_name}")

        # Store job IDs if using SLURM.
        if self.slurm and job_ids:
            self.job_id = ",".join(job_ids.values()) if isinstance(job_ids, dict) else ",".join(job_ids)
            self.logger.debug(f"Job IDs: {self.job_id}")

        self.logger.info(f"{self.name} quality control completed successfully for all samples")
        return results

    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are installed and available.

        Returns:
            True if all dependencies are available, False otherwise
        """
        tool_cmd = self._get_tool_command()

        try:
            # Check if command is in PATH
            if not shutil.which(tool_cmd):
                if self.dryrun:
                    self.logger.warning(f"{self.name} command not found in PATH: {tool_cmd}; continuing in dry-run mode")
                    return True
                self.logger.error(f"{self.name} command not found in PATH: {tool_cmd}")
                return False

            # Check version
            version = self.get_version()
            self.logger.info(f"Found {self.name} version: {version}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to check {self.name} dependencies: {str(e)}")
            return False

    def _get_tool_command(self) -> str:
        """
        Get the base command for the quality control tool.

        This method should be overridden by subclasses if the command name
        differs from the tool name.

        Returns:
            Base command for the quality control tool
        """
        return self.name

    @abstractmethod
    def get_version(self) -> str:
        """
        Get the version of the quality control tool.

        Returns:
            Version string of the quality control tool

        Raises:
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError("Subclass must implement get_version method")

    def validate_arguments(self, args: List[str], valid_args_list: List[str]) -> List[str]:
        """
        Validate the provided arguments against a list of valid arguments.

        Args:
            args: List of arguments to validate
            valid_args_list: List of valid arguments

        Returns:
            List[str]: List of valid arguments

        Raises:
            QualityControlError: If invalid arguments are found
        """
        invalid_args = []
        for arg in args:
            # Extract the argument name from options like '-p 8' or '--threads 8'
            arg_name = arg.split()[0] if " " in arg else arg
            if arg_name not in valid_args_list:
                invalid_args.append(arg_name)

        if invalid_args:
            self.logger.warning(f"Invalid arguments found: {', '.join(invalid_args)}")

        return [arg for arg in args if (arg.split()[0] if " " in arg else arg) not in invalid_args]
