#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Manager Module

This module provides a robust configuration management system for PySeqRNA,
handling the parsing, validation, and loading of configuration files.

Features:
    - INI configuration parsing and validation
    - Schema-based validation for parameters (types, ranges, choices)
    - Auto-detection and resolution of CPU and memory resource allocations
    - Package resource and fallback path resolution for tool-specific configurations
    - Custom validations for times, emails, and file paths

Classes:
    - ConfigManager: Main configuration management class

Functions:
    - get_cpu: Retrieves the number of available CPU cores

Example:
    config = ConfigManager(logger)
    options = config.read_runconfig("input_config.ini")
    # Access options directly
    genome_path = options.reference_genome
    threads = options.threads

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import configparser
from typing import Any, Dict, Optional, Union, List

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files
from pathlib import Path
import logging
import re
from argparse import Namespace
import multiprocessing


def get_cpu():
    """Get the number of available CPU cores."""
    return multiprocessing.cpu_count()


class ConfigManager:
    """
    A comprehensive configuration management system for PySeqRNA.

    This class handles:
    - Reading and parsing INI configuration files
    - Parameter validation and type conversion
    - Path resolution and verification
    - Resource allocation calculation
    """

    # Define configuration schema with expected types and validation rules
    CONFIG_SCHEMA = {
        "General": {
            "input_file": {"type": "path", "required": True},
            "samples_path": {"type": "path", "required": True},
            "reference_genome": {"type": "path", "required": True},
            "feature_file": {"type": "path", "required": True},
            "outdir": {"type": "path", "required": True},
        },
        "Analysis": {
            "source": {"type": "str", "choices": ["ENSEMBL", "NCBI", "UCSC"]},
            "species": {"type": "str", "required": True},
            "organism_type": {"type": "str", "required": True},
            "paired_end": {"type": "bool", "default": False},
            "fdr_threshold": {"type": "float", "range": [0, 1], "default": 0.05},
            "fold_change": {"type": "float", "default": 2.0},
            "no_replicates": {"type": "bool", "default": False},
            "normalization": {"type": "str", "choices": ["RPKM", "TPM", "CPM"]},
            "sample_comparisons": {"type": "str", "default": "all"},
        },
        "Quality_Control": {
            "fastqc_raw": {"type": "bool", "default": True},
            "fastqc_trimmed": {"type": "bool", "default": True},
            "remove_rrna": {"type": "bool", "default": False},
            "rrna_db": {"type": "path", "required_if": "remove_rrna"},
        },
        "Gene_Analysis": {
            "multimapped_genes": {"type": "bool", "default": False},
            "min_reads": {"type": "int", "min": 0, "default": 100},
            "sample_percentage": {"type": "float", "range": [0, 1], "default": 0.5},
        },
        "Visualization": {
            "pca_plot": {"type": "bool", "default": True},
            "tsne_plot": {"type": "bool", "default": True},
            "volcano_plot": {"type": "bool", "default": True},
            "ma_plot": {"type": "bool", "default": True},
            "deg_heatmap": {"type": "bool", "default": True},
            "heatmap_top_genes": {"type": "int", "min": 1, "default": 50},
            "venn": {"type": "bool", "default": True},
            "venn_comparisons": {"type": "str", "default": ""},
            "venn_label": {
                "type": "str",
                "choices": ["updown", "total"],
                "default": "updown",
            },
            "upset": {"type": "bool", "default": True},
        },
        "Functional_Annotation": {
            "go_analysis": {"type": "bool", "default": False},
            "kegg_analysis": {"type": "bool", "default": False},
        },
        "Tools": {
            "trimmer": {"type": "str", "choices": ["trim_galore", "trimmomatic"]},
            "aligner": {"type": "str", "choices": ["STAR", "hisat2", "bowtie2"]},
            "quantifier": {
                "type": "str",
                "choices": ["featureCounts", "htseq-count", "genomic_overlaps"],
            },
            "de_tool": {
                "type": "str",
                "choices": [
                    "DESeq2",
                    "edgeR",
                    "PyDiffExpress",
                    "deseq2",
                    "edger",
                    "pydiffexpress",
                ],
            },
        },
        "Alignment": {
            "alignment_stats": {"type": "bool", "default": True},
            "alignment_stats_source": {
                "type": "str",
                "choices": ["auto", "logs", "bam"],
                "default": "auto",
            },
        },
        "Computational": {
            "threads": {"type": "resource", "default": "80%"},
            "memory": {"type": "int", "min": 1},
            "param_dir": {"type": "path"},
            "resume_from": {
                "type": "str",
                "choices": [
                    "all",
                    "trimming",
                    "alignment",
                    "bam_preparation",
                    "alignment_stats",
                    "quantification",
                    "differential",
                    "functional",
                ],
            },
        },
        "SLURM": {
            "slurm": {"type": "bool", "default": False},
            "slurm_partition": {"type": "str"},
            "slurm_account": {"type": "str"},
            "slurm_time": {"type": "time"},
            "slurm_email": {"type": "email"},
            "slurm_qos": {"type": "str"},
            "slurm_array_max_parallel": {"type": "int", "min": 1, "default": 10},
            "slurm_cpus_per_task": {"type": "int", "min": 0, "default": 0},
            "slurm_memory_per_task": {"type": "int", "min": 0, "default": 0},
        },
        "Logging": {
            "verbose": {"type": "bool", "default": False},
            "quiet": {"type": "bool", "default": False},
            "logs_dir": {"type": "path", "default": "logs"},
            "log_file_prefix": {"type": "str", "default": "pyseqrna"},
            "error_file_prefix": {"type": "str", "default": "pyseqrna"},
            "no_colors": {"type": "bool", "default": False},
            "no_file_paths": {"type": "bool", "default": False},
        },
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ConfigManager with an optional logger instance.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for recording operations and errors
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = configparser.ConfigParser()
        self.config_file = None
        self.validated_config = {}

    def read_runconfig(self, config_file: str) -> Namespace:
        """
        Reads a configuration file and parses the settings into an options Namespace object.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            Namespace: Parsed configuration settings as a Namespace object.

        Raises:
            ValueError: If the configuration file cannot be parsed.
            FileNotFoundError: If the configuration file does not exist.
        """
        # Log the start of the configuration reading
        self.logger.info(f"Reading configuration file: {config_file}")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # Attempt to read the configuration file. Library code raises exceptions;
        # only CLI entrypoints should decide whether to exit the process.
        try:
            self.config.read(config_file)
            self.logger.info("Configuration file read successfully.")
        except configparser.Error as e:
            raise ValueError(f"Error reading configuration file '{config_file}': {e}") from e

        # Create Namespace object to hold the values
        options = Namespace()

        # Initialize command-line specific arguments
        options.supported_organism = False
        options.config_file = config_file
        options.version = False

        def get_config_value(section: str, option: str, default: Any = None, data_type: type = str) -> Any:
            """Helper function to retrieve configuration values with error handling."""
            try:
                if data_type is bool:
                    return self.config.getboolean(section, option, fallback=default)
                if data_type is int:
                    return self.config.getint(section, option, fallback=default)
                if data_type is float:
                    return self.config.getfloat(section, option, fallback=default)
                return self.config.get(section, option, fallback=default)
            except ValueError as e:
                self.logger.warning(f"Error parsing option '{option}' in section '{section}': {e}")
                return default

        # [General] section
        general_params = {
            "input_file": ("", str),
            "samples_path": ("", str),
            "reference_genome": ("", str),
            "feature_file": ("", str),
            "outdir": ("pySeqRNA_results", str),
        }
        for param, (default, data_type) in general_params.items():
            setattr(options, param, get_config_value("General", param, default, data_type))

        # [Analysis] section
        analysis_params = {
            "source": ("ENSEMBL", str),
            "species": ("", str),
            "organism_type": ("plants", str),
            "paired_end": (False, bool),
            "fdr_threshold": (0.05, float),
            "fold_change": (2, float),
            "no_replicates": (False, bool),
            "normalization": ("RPKM", str),
            "sample_comparisons": ("all", str),
        }
        for param, (default, data_type) in analysis_params.items():
            setattr(options, param, get_config_value("Analysis", param, default, data_type))

        # [Quality_Control] section
        qc_params = {
            "fastqc_raw": (True, bool),
            "fastqc_trimmed": (True, bool),
            "remove_rrna": (False, bool),
            "rrna_db": ("", str),
        }
        for param, (default, data_type) in qc_params.items():
            setattr(
                options,
                param,
                get_config_value("Quality_Control", param, default, data_type),
            )

        # [Gene_Analysis] section
        gene_params = {
            "multimapped_genes": (False, bool),
            "min_reads": (100, int),
            "sample_percentage": (0.5, float),
        }
        for param, (default, data_type) in gene_params.items():
            setattr(
                options,
                param,
                get_config_value("Gene_Analysis", param, default, data_type),
            )

        # [Visualization] section
        viz_params = {
            "pca_plot": (True, bool),
            "tsne_plot": (True, bool),
            "volcano_plot": (True, bool),
            "ma_plot": (True, bool),
            "deg_heatmap": (True, bool),
            "heatmap_top_genes": (50, int),
            "venn": (True, bool),
            "venn_comparisons": ("", str),
            "venn_label": ("updown", str),
            "upset": (True, bool),
        }
        for param, (default, data_type) in viz_params.items():
            setattr(
                options,
                param,
                get_config_value("Visualization", param, default, data_type),
            )

        # [Functional_Annotation] section
        func_params = {"go_analysis": (False, bool), "kegg_analysis": (False, bool)}
        for param, (default, data_type) in func_params.items():
            setattr(
                options,
                param,
                get_config_value("Functional_Annotation", param, default, data_type),
            )

        # [Tools] section
        tools_params = {
            "trimmer": ("trim_galore", str),
            "aligner": ("STAR", str),
            "quantifier": ("genomic_overlaps", str),
            "de_tool": ("DESeq2", str),
        }
        for param, (default, data_type) in tools_params.items():
            setattr(options, param, get_config_value("Tools", param, default, data_type))

        # [Computational] section
        try:
            package_param_dir = str(Path(str(files("pyseqrna") / "param")))
        except Exception:
            package_param_dir = str(Path(__file__).resolve().parents[1] / "param")
        comp_params = {
            "threads": (10, int),
            "memory": (16, int),
            "param_dir": (package_param_dir, str),
            "resume_from": ("all", str),
        }
        for param, (default, data_type) in comp_params.items():
            setattr(
                options,
                param,
                get_config_value("Computational", param, default, data_type),
            )

        # [SLURM] section
        slurm_params = {
            "slurm": (False, bool),
            "slurm_partition": ("compute", str),
            "slurm_account": ("", str),
            "slurm_time": ("24:00:00", str),
            "slurm_email": ("", str),
            "slurm_qos": ("", str),
            "slurm_array_max_parallel": (10, int),
            "slurm_cpus_per_task": (0, int),
            "slurm_memory_per_task": (0, int),
        }
        for param, (default, data_type) in slurm_params.items():
            setattr(options, param, get_config_value("SLURM", param, default, data_type))

        # [Logging] section
        logging_params = {
            "verbose": (False, bool),
            "quiet": (False, bool),
            "logs_dir": ("logs", str),
            "log_file_prefix": ("pyseqrna", str),
            "error_file_prefix": ("pyseqrna", str),
            "no_colors": (False, bool),
            "no_file_paths": (False, bool),
        }
        for param, (default, data_type) in logging_params.items():
            setattr(options, param, get_config_value("Logging", param, default, data_type))

        # Handle special cases for list conversions
        if options.sample_comparisons != "all":
            options.sample_comparisons = options.sample_comparisons.split()
        if options.venn_comparisons not in {"", None}:
            options.venn_comparisons = [item.strip() for item in options.venn_comparisons.split(",") if item.strip()]

        self.logger.info("Configuration settings parsed successfully.")
        return options

    def read_config(self, config_file: Union[str, Path]) -> Dict:
        """
        Read and validate the configuration file.

        Args:
            config_file: Path to the configuration file

        Returns:
            Dict: Validated configuration dictionary
        """
        self.config_file = str(config_file)
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        self.config.read(self.config_file)
        self._validate_config()
        return self.validated_config

    def _validate_config(self) -> None:
        """
        Validate all configuration parameters against the schema.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid
        """
        for section, params in self.CONFIG_SCHEMA.items():
            if section not in self.config:
                if any(param.get("required", False) for param in params.values()):
                    raise ValueError(f"Required section '{section}' missing from config")
                continue

            self.validated_config[section] = {}
            for param_name, rules in params.items():
                value = self.config.get(section, param_name, fallback=None)
                if value is None:
                    if rules.get("required", False):
                        raise ValueError(f"Required parameter '{param_name}' missing in section '{section}'")
                    value = rules.get("default")

                self.validated_config[section][param_name] = self._validate_parameter(value, rules, section, param_name)

    def _validate_parameter(self, value: str, rules: Dict, section: str, param_name: str) -> Any:
        """
        Validate and convert a single parameter according to its rules.

        Parameters
        ----------
        value : str
            Parameter value from config file
        rules : Dict
            Validation rules for the parameter
        section : str
            Configuration section name
        param_name : str
            Parameter name

        Returns
        -------
        Any
            Validated and converted parameter value
        """
        if value is None:
            return rules.get("default")

        param_type = rules["type"]

        try:
            if param_type == "bool":
                return value.lower() in ("true", "yes", "1", "on")
            elif param_type == "int":
                val = int(value)
                if "min" in rules and val < rules["min"]:
                    raise ValueError(f"Value must be >= {rules['min']}")
                return val
            elif param_type == "float":
                val = float(value)
                if "range" in rules:
                    min_val, max_val = rules["range"]
                    if not min_val <= val <= max_val:
                        raise ValueError(f"Value must be between {min_val} and {max_val}")
                return val
            elif param_type == "path":
                return os.path.abspath(value)
            elif param_type == "resource":
                return self._parse_resource_value(value)
            elif param_type == "time":
                return self._validate_time_format(value)
            elif param_type == "email":
                return self._validate_email(value)
            elif param_type == "str":
                if "choices" in rules and value not in rules["choices"]:
                    raise ValueError(f"Value must be one of: {', '.join(rules['choices'])}")
                return value
        except Exception as e:
            raise ValueError(f"Invalid value for '{param_name}' in section '{section}': {str(e)}")

    def _parse_resource_value(self, value: str) -> Union[int, str]:
        """
        Parse resource values that can be specified as percentages.

        Parameters
        ----------
        value : str
            Resource value (e.g., "80%", "16")

        Returns
        -------
        Union[int, str]
            Parsed resource value
        """
        if isinstance(value, (int, float)):
            return int(value)

        if "%" in value:
            return value  # Keep as string for later resolution
        return int(value)

    def _validate_time_format(self, value: str) -> str:
        """
        Validate SLURM time format (HH:MM:SS).

        Parameters
        ----------
        value : str
            Time value to validate

        Returns
        -------
        str
            Validated time string
        """
        if not re.match(r"^\d+:\d{2}:\d{2}$", value):
            raise ValueError("Time must be in format HH:MM:SS")
        return value

    def _validate_email(self, value: str) -> str:
        """
        Validate email address format.

        Parameters
        ----------
        value : str
            Email address to validate

        Returns
        -------
        str
            Validated email address
        """
        if value and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value):
            raise ValueError("Invalid email address format")
        return value

    def get_value(self, section: str, parameter: str, default: Any = None) -> Any:
        """
        Get a configuration value with type conversion.

        Parameters
        ----------
        section : str
            Configuration section name
        parameter : str
            Parameter name
        default : Any, optional
            Default value if parameter is not found

        Returns
        -------
        Any
            Configuration value
        """
        try:
            return self.validated_config[section][parameter]
        except KeyError:
            return default

    def get_path(self, section: str, parameter: str) -> Optional[Path]:
        """
        Get a path value from the configuration.

        Parameters
        ----------
        section : str
            Configuration section name
        parameter : str
            Parameter name

        Returns
        -------
        Optional[Path]
            Path object or None if not found
        """
        value = self.get_value(section, parameter)
        if value:
            # Expand common user paths like ~/data and normalize any .. segments.
            # The resolved absolute path is safer and less surprising than
            # rejecting legitimate path syntax in user configuration.
            path_obj = Path(str(value)).expanduser().resolve()
            return path_obj
        return None

    def get_compute_resource(self, section: str, parameter: str) -> int:
        """
        Get and resolve compute resource values.

        Parameters
        ----------
        section : str
            Configuration section name
        parameter : str
            Parameter name

        Returns
        -------
        int
            Resolved resource value
        """
        value = self.get_value(section, parameter)
        if value is None:
            raise ValueError(f"Compute resource '{parameter}' not set in section '{section}'")

        # Only allow known compute parameters to be resolved here
        if parameter not in ("threads", "memory"):
            raise ValueError(f"Unsupported compute parameter '{parameter}'")

        # If already numeric, return as int
        if isinstance(value, (int, float)):
            return int(value)

        # Accept strict percentage formats like "80%" or plain integers "8"
        if isinstance(value, str):
            val = value.strip()
            # Match percentage (one to three digits) with optional surrounding whitespace
            match_pct = re.match(r"^\s*(\d{1,3})\s*%\s*$", val)
            if match_pct:
                percentage = int(match_pct.group(1))
                if not (0 < percentage <= 100):
                    raise ValueError("Percentage must be between 1 and 100")

                if parameter == "threads":
                    total = get_cpu()
                    resolved = max(1, int(total * percentage / 100))
                    return resolved
                else:  # memory
                    try:
                        import psutil
                    except Exception:
                        raise RuntimeError("psutil is required to resolve memory percentages")
                    total_memory_gb = psutil.virtual_memory().total / (1024**3)
                    resolved = max(1, int(total_memory_gb * percentage / 100))
                    return resolved

            # Match plain integer values
            if re.match(r"^\s*\d+\s*$", val):
                return int(val)

        # If we reach here, the format is invalid
        raise ValueError(f"Invalid compute resource value for '{parameter}': {value}")

    def get_bool(self, section: str, parameter: str) -> bool:
        """
        Get a boolean value from the configuration.

        Parameters
        ----------
        section : str
            Configuration section name
        parameter : str
            Parameter name

        Returns
        -------
        bool
            Boolean configuration value
        """
        return bool(self.get_value(section, parameter, False))

    def get_sections(self) -> List[str]:
        """
        Get all available configuration sections.

        Returns
        -------
        List[str]
            List of section names
        """
        return list(self.validated_config.keys())

    def read_tool_config(self, config_file_name: str, param_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Read a tool-specific configuration file without applying the schema validation.
        This is used for tool configs that don't need to follow the main config schema.

        Args:
            config_file_name: Name of the tool config file (e.g., "trim_galore.ini")
            param_dir: Optional directory containing the config file. If None,
                       looks in standard locations

        Returns:
            Dict[str, Dict[str, str]]: Dictionary containing tool configuration parameters
        """
        # Validate inputs to prevent path traversal attacks
        if ".." in str(config_file_name) or "/" in str(config_file_name) or "\\" in str(config_file_name):
            raise ValueError("Invalid config file name: contains path separators or traversal sequences")

        if param_dir and (".." in str(param_dir) or not os.path.isabs(param_dir)):
            raise ValueError("Invalid param directory: must be absolute path without traversal sequences")

        def _find_case_insensitive_file(directory: Path, filename: str) -> Optional[Path]:
            """Return an exact or case-insensitive filename match within a directory."""
            candidate = directory / filename
            if candidate.is_file():
                return candidate

            target = filename.lower()
            try:
                for entry in directory.iterdir():
                    if entry.is_file() and entry.name.lower() == target:
                        return entry
            except Exception as e:
                self.logger.debug(f"Could not scan config directory {directory}: {e}")

            return None

        tool_config = configparser.ConfigParser()
        config_paths = []

        # Build list of possible config paths to try
        if param_dir:
            # User-specified param directory (already validated)
            matched = _find_case_insensitive_file(Path(param_dir), config_file_name)
            if matched is not None:
                config_paths.append(str(matched))
            else:
                config_paths.append(os.path.join(param_dir, config_file_name))

        # Add package param directory
        try:
            # Use importlib.resources for modern Python resource access
            package_param_dir = Path(str(files("pyseqrna") / "param"))
            config_path = _find_case_insensitive_file(package_param_dir, config_file_name)
            self.logger.debug(f"Checking package config directory: {package_param_dir}")
            if config_path is not None:
                config_paths.append(str(config_path))
                self.logger.debug(f"Added package config path: {str(config_path)}")
        except Exception as e:
            self.logger.debug(f"Exception in importlib.resources: {e}")
            # Fallback to relative path
            package_param_dir = Path(os.path.join(os.path.dirname(__file__), "..", "param")).resolve()
            fallback_path = _find_case_insensitive_file(package_param_dir, config_file_name)
            if fallback_path is not None:
                config_paths.append(str(fallback_path))
                safe_fallback_path = str(fallback_path).replace("\n", "").replace("\r", "")
                self.logger.debug(f"Added fallback config path: {safe_fallback_path}")
            else:
                unresolved_path = package_param_dir / config_file_name
                config_paths.append(str(unresolved_path))
                safe_unresolved_path = str(unresolved_path).replace("\n", "").replace("\r", "")
                self.logger.debug(f"Added unresolved fallback config path: {safe_unresolved_path}")

        # Log attempted paths
        self.logger.debug(f"Looking for tool config '{config_file_name}' in paths: {config_paths}")

        # Try each path
        for config_path in config_paths:
            if os.path.exists(config_path):
                # Sanitize config path for logging to prevent log injection
                safe_config_path = str(config_path).replace("\n", "").replace("\r", "")
                self.logger.info(f"Loading tool config from: {safe_config_path}")
                try:
                    tool_config.read(config_path)
                    # Convert to dictionary format
                    config_dict = {section: dict(tool_config[section]) for section in tool_config.sections()}
                    self.logger.debug(f"Loaded tool config: {config_dict}")
                    return config_dict
                except Exception as e:
                    # Sanitize error message and path for logging to prevent log injection
                    safe_error = str(e).replace("\n", "").replace("\r", "")
                    safe_config_path = str(config_path).replace("\n", "").replace("\r", "")
                    self.logger.warning(f"Failed to parse tool config {safe_config_path}: {safe_error}")

        # If we get here, no config was found
        # Sanitize config file name for logging to prevent log injection
        safe_config_file_name = str(config_file_name).replace("\n", "").replace("\r", "")
        self.logger.warning(f"No valid tool config found for {safe_config_file_name}")
        return {}
