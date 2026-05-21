#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Logger Module

This module provides a robust logging system for the PySeqRNA package with colored
console logging, file logging capabilities, and system diagnostics reporting.

Features:
    - Colored terminal logs matching severity levels
    - Log injection sanitization (control/HTML escape characters removal)
    - Dynamic regex-based file path highlighting in output logs
    - Separate general (DEBUG) and error (ERROR) log handlers with unique timestamps
    - Detailed system diagnostics banners (OS, python version, CPU cores, memory stats)
    - Thread-safe resource closing and handler cleanup

Classes:
    - LogManager: Main logger manager class
    - LogManager.ColoredFormatter: Helper formatter to color logs and highlight file paths

Example:
    logger = LogManager()
    logger.info("Processing file: /path/to/file.fastq")
    logger.error("Failed to process file")
    logger.close_logger()

:Created: July 11, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import logging
import re
from datetime import datetime
import platform
import psutil
import sys

try:
    from pyseqrna.__version__ import __version__ as PYSEQRNA_VERSION
except Exception:
    PYSEQRNA_VERSION = "unknown"


class LogManager:
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[94m",  # Blue
            "INFO": "\033[96m",  # Cyan
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "CRITICAL": "\033[41m",  # Red background
            "RESET": "\033[0m",  # Reset to default
            "PATH": "\033[92m",  # Green for paths
        }

        def format(self, record):
            # Color the log level
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            message = super().format(record)

            # Sanitize message to reduce risk of injection:
            # 1) remove non-printable/control characters (including ANSI escapes)
            # 2) escape HTML special chars so logs rendered in HTML aren't vulnerable
            message = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", message)
            message = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            # Regular expression to match file paths
            path_pattern = r"(/[^/ ]*)+/?"

            # Add color to detected file paths (only around the matched path)
            def _color_path(match):
                path_text = match.group(0)
                return f"{self.COLORS['PATH']}{path_text}{reset}"

            message = re.sub(path_pattern, _color_path, message)

            return f"{color}{message}{reset}"

    def __init__(
        self,
        log_file_prefix="pyseqrna",
        error_file_prefix="pyseqrna",
        logs_dir=None,
        print_end_banner=True,
    ):
        """
        Initialize the PySeqRNA logger with customizable file prefixes and directory.

        Parameters
        ----------
        log_file_prefix : str, optional
            Prefix for the general log file name (default: 'pyseqrna')
        error_file_prefix : str, optional
            Prefix for the error log file name (default: 'pyseqrna')
        logs_dir : str, optional
            Directory to store log files (default: None - will be set later)
        """
        self.logs_dir = logs_dir
        self.log_file_prefix = log_file_prefix
        self.error_file_prefix = error_file_prefix
        self.print_end_banner = print_end_banner
        self._closed = False
        self.logger = self.setup_logger(log_file_prefix, error_file_prefix)

    def setup_logger(self, log_file_prefix, error_file_prefix):
        """
        Configure and set up the logger with file and console handlers.

        Parameters
        ----------
        log_file_prefix : str
            Prefix for the general log file name
        error_file_prefix : str
            Prefix for the error log file name

        Returns
        -------
        logging.Logger
            Configured logger instance with both file and console handlers
        """
        # Create or get the logger
        logger = logging.getLogger("PySeqRNALogger")

        # Prevent adding multiple handlers (to avoid duplicate logging)
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)  # Set the minimum logging level

            # Send normal console logs to stdout so SLURM .out contains the
            # user-facing run log alongside printed banners. Error-only file
            # handlers still capture failures separately.
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)  # Set level for command line output

            # Create formatters
            formatter = self.ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m-%d-%Y %H:%M:%S")

            # Add formatter to console handler
            console_handler.setFormatter(formatter)

            # Add console handler to logger
            logger.addHandler(console_handler)

            # Only create file handlers if logs_dir is specified
            if self.logs_dir:
                # Normalize to an absolute path. The output directory is user
                # controlled, so logs should follow it even when it is outside
                # the current working directory.
                sanitized_dir = os.path.normpath(self.logs_dir)
                abs_logs_dir = os.path.abspath(sanitized_dir)

                # Ensure the final directory exists and store the normalized path
                os.makedirs(abs_logs_dir, exist_ok=True)
                self.logs_dir = abs_logs_dir

                # Sanitize prefixes to remove any directory components
                safe_log_prefix = os.path.basename(log_file_prefix) or "pyseqrna"
                safe_error_prefix = os.path.basename(error_file_prefix) or "pyseqrna"

                # Generate timestamp for unique file names
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Create safe file paths with timestamp in logs directory
                log_file_path = os.path.join(self.logs_dir, f"{safe_log_prefix}_{timestamp}.log")
                error_file_path = os.path.join(self.logs_dir, f"{safe_error_prefix}_{timestamp}.err")

                # Create file handlers
                log_handler = logging.FileHandler(log_file_path)
                log_handler.setLevel(logging.DEBUG)  # Log all levels to the log file

                error_handler = logging.FileHandler(error_file_path)
                error_handler.setLevel(logging.ERROR)  # Only log errors to the error file

                # Create formatters
                fileformatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%m-%d-%Y %H:%M:%S",
                )

                # Add formatters to handlers
                log_handler.setFormatter(fileformatter)
                error_handler.setFormatter(fileformatter)

                # Add file handlers to logger
                logger.addHandler(log_handler)
                logger.addHandler(error_handler)

                # Log the starting banner
                self._log_banner("start", log_file_path)

        return logger

    def get_logger(self, name=None):
        """
        Get the logger instance.

        Args:
            name: Optional logger name accepted by module callers.

        Returns:
            logging.Logger: The configured logger instance
        """
        return self.logger

    def update_log_directory(self, new_logs_dir):
        """
        Update the log directory and recreate handlers.

        Parameters
        ----------
        new_logs_dir : str
            New directory for log files
        """
        # Create new file handlers in the normalized output directory.
        sanitized_dir = os.path.normpath(new_logs_dir)
        abs_logs_dir = os.path.abspath(sanitized_dir)

        os.makedirs(abs_logs_dir, exist_ok=True)
        self.logs_dir = abs_logs_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize prefixes to remove any directory components
        safe_log_prefix = os.path.basename(self.log_file_prefix) or "pyseqrna"
        safe_error_prefix = os.path.basename(self.error_file_prefix) or "pyseqrna"

        log_file_path = os.path.join(self.logs_dir, f"{safe_log_prefix}_{timestamp}.log")
        error_file_path = os.path.join(self.logs_dir, f"{safe_error_prefix}_{timestamp}.err")

        # Create new file handlers
        log_handler = logging.FileHandler(log_file_path)
        log_handler.setLevel(logging.DEBUG)

        error_handler = logging.FileHandler(error_file_path)
        error_handler.setLevel(logging.ERROR)

        # Create formatters
        fileformatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m-%d-%Y %H:%M:%S")

        # Add formatters to handlers
        log_handler.setFormatter(fileformatter)
        error_handler.setFormatter(fileformatter)

        # Add handlers to logger
        self.logger.addHandler(log_handler)
        self.logger.addHandler(error_handler)

        # Log the starting banner to the new log file
        self._log_banner("start", log_file_path)

    def _get_system_stats(self):
        """
        Get system statistics including CPU, memory, OS details, and Python version.

        Returns
        -------
        str
            Formatted string containing system information
        """
        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)  # Convert to GB

            stats = (
                "\n           System Information"
                "\n           -----------------"
                f"\n           OS: {platform.system()} {platform.release()}"
                f"\n           Python Version: {sys.version.split()[0]}"
                f"\n           CPU Cores: {cpu_count} (Physical), {cpu_threads} (Logical)"
                f"\n           Total Memory: {memory_gb:.1f} GB"
                f"\n           Memory Available: {(memory.available / (1024**3)):.1f} GB"
                f"\n           Memory Used: {memory.percent}%\n"
            )
            return stats
        except Exception as e:
            return f"\n           Failed to get complete system stats: {str(e)}\n"

    def _log_banner(self, mode, log_file_path):
        """
        Log a formatted banner message with system statistics at the start
        or end of logging session.

        Parameters
        ----------
        mode : str
            Either 'start' or 'end' to determine which banner to display
        log_file_path : str
            Path to the log file where the banner should be written
        """
        if mode == "start":
            banner = (
                "\n"
                "           ----------------------------------------------------------                 \n"
                f"                                PySeqRNA {PYSEQRNA_VERSION}                                          \n"
                "                Written by Naveen Duhan (naveen.duhan@outlook.com),                   \n"
                "               Kaundal Bioinformatics Lab, Utah State University,                     \n"
                "            Released under the terms of GNU General Public License v3                 \n"
                "           -----------------------------------------------------------                \n"
            )
            # Add system stats to the start banner
            banner += self._get_system_stats()
            banner += "           ----------------------------------------------------------                 \n\n"

        elif mode == "end":
            banner = (
                "\n"
                "           ----------------------------------------------------------                 \n"
                f"                            End of PySeqRNA {PYSEQRNA_VERSION} Session                               \n"
                "                                                                                      \n"
                "                                 Beer Time! Enjoy!                                    \n"
                "           -----------------------------------------------------------                \n"
                "\n"
            )

        if mode == "end" and self.print_end_banner:
            print(banner, end="")

        # Log to file
        with open(log_file_path, "a") as f:
            f.write(banner)

    def close_logger(self):
        """Logs the end banner and closes the logger."""
        if self._closed:
            return
        self._closed = True

        # Find the first file handler to get the log file path
        log_file_path = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename
                break

        # Log the end banner if we found a log file. Standalone commands may
        # not have file handlers, but users should still see completion.
        if log_file_path:
            self._log_banner("end", log_file_path)
        elif self.print_end_banner:
            banner = (
                "\n"
                "           ----------------------------------------------------------                 \n"
                f"                            End of PySeqRNA {PYSEQRNA_VERSION} Session                               \n"
                "                                                                                      \n"
                "                                 Beer Time! Enjoy!                                    \n"
                "           -----------------------------------------------------------                \n"
                "\n"
            )
            print(banner, end="")

        # Remove all handlers to close the logger properly
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    # Wrapper methods for logging with variable arguments
    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)
