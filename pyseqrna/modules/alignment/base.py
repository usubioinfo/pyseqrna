#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Alignment Module

This module defines the abstract base class and custom exceptions for RNA-seq read alignment
in pySeqRNA. It provides a common interface, configuration loading, resource management,
and shell execution features (both local and SLURM cluster) that specific aligner implementations inherit.

Features:
    - Abstract base class representing a unified API for RNA-seq aligners
    - Automatic CPU thread resolution and resource management
    - Support for local multi-threaded and SLURM cluster-based job submission
    - Validation for input reference genome (FASTA) and sequencing read (FASTQ) formats
    - Automatic handling of gzip-compressed reference genomes
    - Integrated dry-run execution mode simulation

Configuration:
    Configured via constructor/function arguments including 'genome', 'param_dir', 'out_dir',
    'slurm', 'dryrun', 'dep', 'cpu_threads', and 'slurm_config'. It also supports loading tool-specific
    options from external configuration files (e.g., bowtie2.ini, bwa.ini) via the ConfigManager.

Dependencies:
    - pySeqRNA utilities (FileManager, CommandExecutor, LogManager, ConfigManager, ResourceManager, DryRunManager)

Classes / Functions / Exceptions:
    - AlignmentError: Exception class raised for alignment-specific execution failures.
    - BaseAligner: Abstract base class outlining standard alignment, indexing, and validation interfaces.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import shutil
import re

try:
    from importlib.resources import files  # noqa: F401
except ImportError:
    # Fallback for Python < 3.9
    pass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

# Import utility modules
from pyseqrna.utils.file_manager import FileManager
from pyseqrna.utils.command_executor import CommandExecutor
from pyseqrna.utils.log_manager import LogManager
from pyseqrna.utils.config_manager import ConfigManager
from pyseqrna.utils.resource_manager import ResourceManager
from pyseqrna.utils.dry_run_manager import DryRunManager


class AlignmentError(Exception):
    """Custom exception for alignment-related errors."""

    pass


class BaseAligner(ABC):
    """
    Abstract base class for RNA-seq read aligners.

    This class defines the common interface and structure for all aligner implementations
    in pyseqrna. It provides base functionality for configuration management, file handling,
    and command execution.

    Attributes:
        genome (str): Path to the reference genome file
        param_dir (str): Directory containing parameter files
        out_dir (Path): Output directory for results
        slurm (bool): Whether to use SLURM for job execution
        dryrun (bool): Whether to perform a dry run (print commands without executing)
        dep (str): SLURM job ID on which this job depends
        logger (Any): Logger instance for logging messages
    """

    VALID_GENOME_EXTENSIONS = (
        ".fa",
        ".fasta",
        ".fna",
        ".fa.gz",
        ".fna.gz",
        ".fasta.gz",
    )

    def __init__(
        self,
        genome: Optional[str] = None,
        param_dir: Optional[str] = None,
        out_dir: str = ".",
        slurm: bool = False,
        dryrun: bool = False,
        dep: str = "",
        cpu_threads: Optional[int] = None,
        logger: Any = None,
        dry_run_manager=None,
        slurm_config: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the BaseAligner class.

        Args:
            genome: Path to the reference genome file
            param_dir: Directory containing parameter files
            out_dir: Output directory for results
            slurm: Whether to use SLURM for job execution
            dryrun: Whether to perform a dry run (print commands without executing)
            dep: SLURM job ID on which this job depends
            cpu_threads: Number of CPU threads to use. If None, will be determined automatically.
            logger: Logger instance for logging messages

        Raises:
            ValueError: If genome file is provided but doesn't exist
            ValueError: If config_file is provided but doesn't exist
        """
        # Initialize logger if not provided
        if logger is None:
            log_manager = LogManager()
            self.logger = log_manager.logger
        else:
            self.logger = logger

        # Initialize the utility managers
        self.file_manager = FileManager(logger=self.logger)
        self.resource_manager = ResourceManager(logger=self.logger)
        self.command_executor = CommandExecutor(logger=self.logger)

        # Initialize dry-run manager
        if dry_run_manager is not None:
            self.dry_run_manager = dry_run_manager
        else:
            self.dry_run_manager = DryRunManager(enabled=dryrun, logger=self.logger)

        # Validate genome file if provided
        if genome and not self.file_manager.verify_files_exist(genome):
            raise ValueError(f"Genome file not found: {genome}")

        # Validate param directory if provided
        if param_dir and not self.file_manager.verify_directories_exist(param_dir):
            raise ValueError(f"Parameter directory not found: {param_dir}")

        self.genome = genome
        self.param_dir = param_dir
        self.out_dir = Path(out_dir)
        self.slurm = slurm
        self.dryrun = dryrun
        self.dep = dep
        self.slurm_config = slurm_config or {}
        self.local_jobs = max(1, int((self.slurm_config or {}).get("local_jobs", 1) or 1))

        # Initialize threads to None, will be set after config load
        self.cpu_threads = None

        # Load configuration if provided
        self.config = None
        if param_dir:
            self.config = self.load_config(param_dir)

        # Determine aligner name from class name
        self.name = self.__class__.__name__.replace("Aligner", "").lower()
        self.logger.debug(f"Initializing {self.name} aligner")

        # Set up alignment directories
        self.alignment_dir = self.out_dir / "2.Alignment"
        self.index_dir = self.alignment_dir / f"{self.name}_index"
        self.results_dir = self.alignment_dir / f"{self.name}_results"

        # Create output directory if it doesn't exist
        if not self.dryrun and not os.path.exists(out_dir):
            self.file_manager.create_directory(out_dir)

        # Create alignment directories if they don't exist
        if not self.dryrun:
            self.alignment_dir.mkdir(parents=True, exist_ok=True)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set up CPU threads with intelligent fallback strategy
        self._setup_cpu_threads(cpu_threads)

        self.logger.debug(f"{self.name} aligner initialization complete")

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

        Different aligner tools use different parameter names for threads, so this
        method checks several common patterns.

        Returns:
            Thread count if found in config, None otherwise
        """
        # Determine config file to try based on aligner
        config_file = f"{self.name}.ini"

        try:
            config = self.load_config(config_file)
            if not config:
                return None

            # Check for common thread parameter names in the tool's section
            tool_section = config.get(self.name, {})
            for thread_param in [
                "threads",
                "cpu",
                "cpu_threads",
                "cores",
                "processors",
                "thread",
                "p",
            ]:
                if thread_param in tool_section:
                    thread_value = tool_section[thread_param]

                    # Handle different formats like "-p 8" or just "8"
                    if isinstance(thread_value, str):
                        # Extract number from string like "-p 8" or "--threads=8"
                        match = re.search(r"\b(\d+)\b", thread_value)
                        if match:
                            return int(match.group(1))
                    elif isinstance(thread_value, int):
                        return thread_value

        except Exception as e:
            self.logger.debug(f"Failed to get threads from config {config_file}: {str(e)}")

        # No thread setting found
        return None

    @abstractmethod
    def check_index(self) -> bool:
        """
        Check if the genome index exists and is valid.

        Returns:
            bool: True if index exists and is valid, False otherwise
        """
        pass

    @abstractmethod
    def build_index(self, gff: Optional[str] = None) -> Union[str, None]:
        """
        Build the genome index for read alignment.

        Args:
            gff: Optional path to GFF/GTF annotation file

        Returns:
            str: Command string if dryrun is True, None otherwise

        Raises:
            AlignmentError: If index building fails
        """
        pass

    @abstractmethod
    def run_alignment(self, target: Optional[Dict[str, List[str]]] = None, paired: bool = False) -> Dict[str, Any]:
        """
        Align reads against the indexed reference genome.

        Args:
            target: Dictionary mapping sample IDs to their file paths
            paired: Boolean indicating if reads are paired-end

        Returns:
            Dict containing output file paths and job IDs if using SLURM

        Raises:
            AlignmentError: If alignment fails
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
            AlignmentError: If command execution fails
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
                    dependency=self.dep,
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
                    raise AlignmentError(f"{tool_name} failed for {len(failed)} sample(s): {', '.join(failed)}")
                return local_results
        except Exception as e:
            raise AlignmentError(f"Command execution failed: {str(e)}")

    def validate_arguments(self, args: List[str], valid_args_list: List[str]) -> List[str]:
        """
        Validate the provided arguments against a list of valid arguments.

        Args:
            args: List of arguments to validate
            valid_args_list: List of valid arguments

        Returns:
            List[str]: List of valid arguments

        Raises:
            AlignmentError: If invalid arguments are found
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

    def _is_valid_fasta(self, file_path: str) -> bool:
        """
        Check if a file is a valid FASTA file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file is a valid FASTA file, False otherwise
        """
        try:
            # Handle gzipped files
            if file_path.endswith(".gz"):
                import gzip

                with gzip.open(file_path, "rt") as f:
                    first_line = f.readline().strip()
                    return first_line.startswith(">")
            else:
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    return first_line.startswith(">")
        except Exception as e:
            self.logger.error(f"Error validating FASTA file: {str(e)}")
            return False

    def _is_valid_fastq(self, file_path: str) -> bool:
        """
        Check if a file is a valid FASTQ file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file is a valid FASTQ file, False otherwise
        """
        try:
            # Handle gzipped files
            if file_path.endswith(".gz"):
                import gzip

                with gzip.open(file_path, "rt") as f:
                    first_line = f.readline().strip()
                    return first_line.startswith("@")
            else:
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    return first_line.startswith("@")
        except Exception as e:
            self.logger.error(f"Error validating FASTQ file: {str(e)}")
            return False

    def _copy_genome_to_index(self) -> str:
        """
        Copy genome file to the index directory.

        This method copies the reference genome file to the index directory,
        decompressing it if it's gzipped. This follows the pattern from the
        original pySeqRNA implementation.

        Returns:
            str: Path to the copied genome file in the index directory

        Raises:
            AlignmentError: If genome file is invalid or copying fails
        """
        if not self.genome:
            raise AlignmentError("No genome file provided")

        # Check if genome file has valid extension
        if not self.genome.endswith(self.VALID_GENOME_EXTENSIONS):
            raise AlignmentError(f"Invalid genome file extension. Valid extensions: {self.VALID_GENOME_EXTENSIONS}")

        # Get the basename of the genome file
        genome_basename = os.path.basename(self.genome)

        # Handle gzipped files
        if self.genome.endswith(".gz"):
            # Remove .gz extension from basename for the copied file
            genome_basename = genome_basename.replace(".gz", "")

            if not self.dryrun:
                # Decompress and copy
                import gzip

                genome_path_in_index = self.index_dir / genome_basename

                self.logger.info(f"Decompressing and copying {self.genome} to {genome_path_in_index}")

                try:
                    with gzip.open(self.genome, "rt") as gz_file:
                        with open(genome_path_in_index, "w") as out_file:
                            shutil.copyfileobj(gz_file, out_file)

                    self.logger.info(f"{genome_basename} copied successfully to {self.index_dir}")

                except Exception as e:
                    raise AlignmentError(f"Failed to decompress and copy genome file: {str(e)}")

            else:
                genome_path_in_index = self.index_dir / genome_basename
                self.logger.info(f"DRYRUN: Would decompress and copy {self.genome} to {genome_path_in_index}")
        else:
            # Copy uncompressed file
            genome_path_in_index = self.index_dir / genome_basename

            if not self.dryrun:
                self.logger.info(f"Copying {self.genome} to {genome_path_in_index}")

                try:
                    shutil.copy2(self.genome, genome_path_in_index)
                    self.logger.info(f"{genome_basename} copied successfully to {self.index_dir}")

                except Exception as e:
                    raise AlignmentError(f"Failed to copy genome file: {str(e)}")
            else:
                self.logger.info(f"DRYRUN: Would copy {self.genome} to {genome_path_in_index}")

        return str(genome_path_in_index)
