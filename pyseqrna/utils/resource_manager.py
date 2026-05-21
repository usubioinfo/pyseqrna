#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resource Manager Module

This module provides functionality to manage system resource allocation and SLURM
configurations for PySeqRNA, determining CPU and memory limits.

Features:
    - Thread-safe system CPU core and memory detection using psutil with safe fallbacks
    - Interactive thread parameter adjustment for command-line arguments
    - Auto-generation of INI-formatted SLURM configuration files
    - Resource availability validation for local and cluster execution tasks
    - Hierarchical allocation resolution prioritizing user overrides over auto-detection

Classes:
    - ResourceManager: System resource management for CPU and memory allocation

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import math
import psutil
from typing import List, Optional, Dict, Any, Union
import logging
import os
import configparser


class ResourceManager:
    """Handles system resource allocation and management."""

    DEFAULT_CPU_COUNT = 1
    DEFAULT_MEMORY_GB = 8.0

    def __init__(self, logger: logging.Logger):
        """
        Initialize ResourceManager.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for recording operations
        """
        self.logger = logger

    def get_cpu_count(self, percentage: float = 0.8) -> int:
        """
        Calculate available CPU cores based on percentage.

        Parameters
        ----------
        percentage : float
            Percentage of total CPUs to use (0.0-1.0)

        Returns
        -------
        int
            Number of CPU cores to use
        """
        total_cpus = self._safe_cpu_count()
        if total_cpus is None or total_cpus < 1:
            self.logger.warning("Could not determine CPU count, defaulting to 1")
            return 1

        cpu_count = max(1, math.floor(total_cpus * percentage))
        self.logger.info(f"Allocating {cpu_count} CPUs ({percentage * 100}% of total available {total_cpus} CPUs)")
        return cpu_count

    def get_memory_gb(self, percentage: float = 0.8) -> float:
        """
        Get available system memory in GB.

        Parameters
        ----------
        percentage : float
            Percentage of total memory to use (0.0-1.0)

        Returns
        -------
        float
            Available system memory in GB
        """
        total_memory = self._safe_memory_gb()
        available_memory = total_memory * percentage
        self.logger.info(f"Allocating {available_memory:.1f}GB memory ({percentage * 100}% of {total_memory:.1f}GB)")
        return available_memory

    def _safe_cpu_count(self) -> int:
        """Return logical CPU count with a conservative fallback."""
        try:
            total_cpus = psutil.cpu_count(logical=True)
            if total_cpus and total_cpus > 0:
                return int(total_cpus)
        except Exception as exc:
            self.logger.warning("Could not determine CPU count via psutil: %s", exc)
        return self.DEFAULT_CPU_COUNT

    def _safe_memory_gb(self) -> float:
        """Return total system memory in GB with a conservative fallback."""
        try:
            total_memory = psutil.virtual_memory().total / (1024**3)
            if total_memory > 0:
                return float(total_memory)
        except Exception as exc:
            self.logger.warning("Could not determine memory via psutil: %s", exc)
        return self.DEFAULT_MEMORY_GB

    def resolve_threads(self, threads_option: Union[str, int]) -> int:
        """
        Resolve the number of threads to use.

        Parameters
        ----------
        threads_option : Union[str, int]
            Thread option, either a number or '80% of available CPU'

        Returns
        -------
        int
            Number of threads to use
        """
        # Handle string input
        if isinstance(threads_option, str):
            if threads_option == "80% of available CPU":
                return self.get_cpu_count(0.8)
            try:
                threads_option = int(threads_option)
            except ValueError:
                self.logger.warning(f"Invalid thread option: {threads_option}, defaulting to 1")
                return 1

        # Now threads_option should be an integer. User-provided/configured
        # values are authoritative, especially on SLURM where psutil may see
        # the login/driver allocation rather than the resources requested for
        # submitted child jobs.
        return int(threads_option)

    def adjust_threads(self, command_args: List[str], thread_keywords: List[str], max_threads: int) -> List[str]:
        """
        Adjust thread counts in command arguments.

        Parameters
        ----------
        command_args : List[str]
            List of command arguments
        thread_keywords : List[str]
            Keywords that indicate thread parameters
        max_threads : int
            Maximum number of threads to use

        Returns
        -------
        List[str]
            Updated command arguments
        """
        updated_args = []
        for arg in command_args:
            if any(keyword in arg for keyword in thread_keywords):
                try:
                    opt, num_str = arg.split(" ")
                    original_num = int(num_str)
                    adjusted_num = min(original_num, max_threads)
                    if adjusted_num < original_num:
                        self.logger.warning(f"Adjusted thread count from {original_num} to {adjusted_num}")
                    updated_args.append(f"{opt} {adjusted_num}")
                except ValueError:
                    self.logger.error(f"Could not parse thread count in argument: {arg}")
                    updated_args.append(arg)
            else:
                updated_args.append(arg)
        return updated_args

    def write_slurm_ini(
        self,
        partition: str,
        cpus: int,
        memory: int,
        ntasks: int,
        account: str = "",
        time: str = "24:00:00",
        email: str = "",
        qos: str = "",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Create a SLURM configuration file with the specified parameters.
        Only includes parameters that have non-empty values.

        Args:
            partition (str): SLURM partition name or "NA"
            cpus (int): Number of CPUs per task or "NA"
            memory (int): Memory allocation in MB or "NA"
            ntasks (int): Number of tasks or "NA"
            account (str, optional): SLURM account name. Defaults to "".
            time (str, optional): Time limit for SLURM jobs (format: HH:MM:SS). Defaults to "24:00:00".
            email (str, optional): Email address for job notifications. Defaults to "".
            qos (str, optional): Quality of Service (QoS) level. Defaults to "".
            output_dir (str, optional): Directory where slurm.ini should be written.
                If omitted, writes to the current working directory with a warning.

        Returns:
            str: Path to the created configuration file

        Example:
            >>> resource_manager = ResourceManager()
            >>> config_path = resource_manager.write_slurm_ini(
            ...     partition="compute",
            ...     cpus=4,
            ...     memory=8000,
            ...     ntasks=1,
            ...     account="myproject",
            ...     time="12:00:00",
            ...     email="user@example.com",
            ...     qos="normal"
            ... )
        """
        # Initialize the config parser
        config = configparser.ConfigParser()

        # Create base SLURM configuration with required values
        slurm_config = {}

        # Add required parameters if they're not "NA"
        if partition != "NA":
            slurm_config["partition"] = partition
        if cpus != "NA":
            slurm_config["cpus"] = str(cpus)
        if memory != "NA":
            slurm_config["memory"] = str(memory)
        if ntasks != "NA":
            slurm_config["ntasks"] = str(ntasks)

        # Add optional parameters only if they have values
        if account:
            slurm_config["slurm_account"] = account
        if time:
            slurm_config["slurm_time"] = time
        if email:
            slurm_config["slurm_email"] = email
        if qos:
            slurm_config["slurm_qos"] = qos

        # Add the SLURM configuration section
        config["slurm"] = slurm_config

        if output_dir is not None:
            target_dir = os.path.abspath(str(output_dir))
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = os.getcwd()
            self.logger.warning(
                "write_slurm_ini called without output_dir; writing slurm.ini to current working directory: %s",
                target_dir,
            )

        config_path = os.path.join(target_dir, "slurm.ini")

        # Write to the .ini file
        with open(config_path, "w") as configfile:
            config.write(configfile)

        self.logger.info(f"Created SLURM configuration file: {config_path}")
        if slurm_config:
            self.logger.debug(
                "SLURM configuration parameters: %s",
                ", ".join(f"{k}={v}" for k, v in slurm_config.items()),
            )
        return config_path

    def read_slurm_config(self, config_path: str) -> Dict[str, Any]:
        """
        Read a SLURM configuration file.

        Args:
            config_path (str): Path to the SLURM configuration file

        Returns:
            Dict[str, Any]: Dictionary containing SLURM configuration parameters

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            configparser.Error: If there's an error parsing the configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SLURM configuration file not found: {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        if "slurm" not in config:
            raise ValueError("No SLURM section found in configuration file")

        return dict(config["slurm"])

    def validate_resources(self, cpus: int, memory: int) -> bool:
        """
        Validate if requested resources are available.

        Args:
            cpus (int): Number of requested CPUs
            memory (int): Requested memory in MB

        Returns:
            bool: True if resources are available, False otherwise
        """
        total_cpus = self._safe_cpu_count()
        total_memory_mb = self._safe_memory_gb() * 1024

        valid = True

        if cpus > total_cpus:
            self.logger.warning(f"Requested {cpus} CPUs exceeds available {total_cpus} CPUs")
            valid = False

        if memory > total_memory_mb:
            self.logger.warning(f"Requested {memory}MB memory exceeds available {int(total_memory_mb)}MB")
            valid = False

        return valid

    def allocate_resources(
        self,
        user_threads: Optional[int] = None,
        user_memory: Optional[int] = None,
        use_slurm: bool = False,
    ) -> Dict[str, int]:
        """
        Intelligently allocate CPU and memory resources.

        Hierarchy:
        1. User-provided values (--threads, --memory) override everything
        2. SLURM mode: Use user values for job submission
        3. Local mode: Use efficient auto-detection (80% CPU, 60% memory)

        Parameters
        ----------
        user_threads : Optional[int]
            User-specified thread count
        user_memory : Optional[int]
            User-specified memory in GB
        use_slurm : bool
            Whether running in SLURM mode

        Returns
        -------
        Dict[str, int]
            Dictionary with 'threads' and 'memory' keys
        """
        # Start with auto-detected efficient defaults
        auto_threads = self.get_cpu_count(0.8)  # 80% of available CPUs
        auto_memory = int(self.get_memory_gb(0.6))  # 60% of available memory

        # Apply user overrides if provided
        final_threads = user_threads if user_threads is not None else auto_threads
        final_memory = user_memory if user_memory is not None else auto_memory

        if use_slurm:
            self.logger.info(f"SLURM mode: Using {final_threads} threads, {final_memory}GB memory for job submission")
        else:
            if user_threads is not None:
                self.logger.info(f"Local mode: Using user-specified {final_threads} threads")
            else:
                self.logger.info(f"Local mode: Auto-detected {final_threads} threads (80% of {self._safe_cpu_count()} CPUs)")

            if user_memory is not None:
                self.logger.info("Local mode: Using user-specified %dGB memory", final_memory)
            else:
                total_mem = self._safe_memory_gb()
                # Use parameterized logging to avoid log injection (do not inline variables into the log message)
                self.logger.info(
                    "Local mode: Auto-detected %dGB memory (60%% of %.1fGB)",
                    final_memory,
                    total_mem,
                )

        return {"threads": final_threads, "memory": final_memory}
