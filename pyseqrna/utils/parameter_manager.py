#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Manager Module

This module provides functionality to validate and manage tool-specific parameters
for various tools used in the PySeqRNA pipeline.

Features:
    - Whitelist validation of arguments for FASTERQDUMP, TRIM_GALORE, STAR, HISAT2,
      HISAT2BUILD, BOWTIE2, FASTQC, CLUST, TRIMMOMATIC, FLEXBAR, and FEATURECOUNTS
    - Safe loading of parameters from INI files with path traversal protection
    - Writing/saving configuration parameters back to INI files

Classes:
    - ParameterManager: Class to manage and validate tool-specific parameters

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import configparser  # Import the configparser library for .ini files
from typing import Dict, Any, List, Optional
from .log_manager import LogManager
from .valid_args import (
    _args_FASTERQDUMP,
    _args_TRIM_GALORE,
    _args_STAR,
    _args_HISAT2,
    _args_HISAT2BUILD,
    _args_BOWTIE2,
    _args_FASTQC,
    _args_CLUST,
    _args_TRIMMOMATIC,
    _args_FLEXBAR,
    _args_FEATURECOUNTS,
)


class ParameterManager:
    """
    A class to manage and validate tool-specific parameters.

    This class provides methods to:
    - Validate user-provided parameters against predefined valid arguments
    - Load parameters from .ini files
    - Save parameters to .ini files
    - Get parameter descriptions and valid values
    """

    def __init__(self, logger: LogManager):
        """
        Initialize the ParameterManager.

        Args:
            logger: LogManager instance for logging messages
        """
        self.logger = logger
        self.tool_args = {
            "FASTERQDUMP": _args_FASTERQDUMP,
            "TRIM_GALORE": _args_TRIM_GALORE,
            "STAR": _args_STAR,
            "HISAT2": _args_HISAT2,
            "HISAT2BUILD": _args_HISAT2BUILD,
            "BOWTIE2": _args_BOWTIE2,
            "FASTQC": _args_FASTQC,
            "CLUST": _args_CLUST,
            "TRIMMOMATIC": _args_TRIMMOMATIC,
            "FLEXBAR": _args_FLEXBAR,
            "FEATURECOUNTS": _args_FEATURECOUNTS,
        }

    def validate_parameter(self, tool: str, param_name: str, param_value: Any) -> bool:
        """
        Validate a single parameter value against its predefined valid values.

        Args:
            tool: Name of the tool (e.g., 'STAR', 'HISAT2')
            param_name: Name of the parameter to validate
            param_value: Value to validate

        Returns:
            bool: True if parameter is valid, False otherwise
        """
        if tool not in self.tool_args:
            self.logger.error(f"Unknown tool: {tool}")
            return False

        # Check if the parameter is valid for the specified tool
        if param_name not in self.tool_args[tool]:
            self.logger.error(f"Unknown parameter '{param_name}' for tool '{tool}'")
            return False

        return True

    def validate_parameters(self, tool: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate a dictionary of parameters for a specific tool.

        Args:
            tool: Name of the tool
            parameters: Dictionary of parameter names and values

        Returns:
            bool: True if all parameters are valid, False otherwise
        """
        if tool not in self.tool_args:
            self.logger.error(f"Unknown tool: {tool}")
            return False

        all_valid = True
        for param_name, param_value in parameters.items():
            if not self.validate_parameter(tool, param_name, param_value):
                all_valid = False

        return all_valid

    def get_tool_parameters(self, tool: str) -> Optional[List[str]]:
        """
        Get all valid parameters for a specific tool.

        Args:
            tool: Name of the tool

        Returns:
            List: List of valid parameters, or None if tool not found
        """
        if tool not in self.tool_args:
            self.logger.error(f"Unknown tool: {tool}")
            return None
        return self.tool_args[tool]

    def load_parameters(self, param_file: str) -> Optional[Dict[str, Any]]:
        """
        Load parameters from a .ini file.

        Args:
            param_file: Path to the parameter .ini file

        Returns:
            Dict: Dictionary of parameters, or None if loading fails
        """
        # Resolve the path and prevent path traversal by ensuring the file is
        # located within this package's directory.
        real_path = os.path.realpath(param_file)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(real_path) or not (real_path == base_dir or real_path.startswith(base_dir + os.sep)):
            self.logger.error(f"Parameter file not found or outside allowed directory: {param_file}")
            return None

        config = configparser.ConfigParser()
        try:
            # Read the resolved path to avoid using an attacker-controlled relative path
            config.read(real_path)
            parameters = {section: dict(config.items(section)) for section in config.sections()}
            return parameters
        except Exception as e:
            self.logger.error(f"Error loading parameter file: {e}")
            return None

    def save_parameters(self, parameters: Dict[str, Any], param_file: str) -> bool:
        """
        Save parameters to a .ini file.

        Args:
            parameters: Dictionary of parameters to save
            param_file: Path to save the parameters to

        Returns:
            bool: True if saving was successful, False otherwise
        """
        config = configparser.ConfigParser()
        for section, params in parameters.items():
            config[section] = params
        try:
            with open(param_file, "w") as f:
                config.write(f)  # Save as .ini
            return True
        except Exception as e:
            self.logger.error(f"Error saving parameter file: {e}")
            return False
