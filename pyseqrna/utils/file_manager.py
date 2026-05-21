#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Manager Module

This module provides a comprehensive set of file and directory operations for PySeqRNA,
including path manipulations, directory creation policies, and file search.

Features:
    - Path parsing (basename, directory name, extension, filename without extension)
    - Main output directory creation with user confirmation and directory auto-incrementing
    - Subdirectory management with modular tool output preservation
    - File and directory existence checking
    - Glob-based and regex-based file searching and retrieval

Classes:
    - Colors: ANSI color codes for terminal output
    - FileManager: Comprehensive utility class for file/directory management

Functions:
    - print_colored: Helper function to output colored and styled text to stdout

Example:
    file_mgr = FileManager(logger)
    files = file_mgr.search_files(pattern=".*\\.fastq$", directory="data/")
    output_dir = file_mgr.create_directory("results")

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
import re
import fnmatch
import logging
import sys
from typing import List, Optional, Union
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"  # Reset to default color


def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False) -> None:
    """Print colored text to terminal."""
    formatting = color
    if bold:
        formatting = Colors.BOLD + color
    print(f"{formatting}{text}{Colors.END}")


class FileManager:
    """
    A comprehensive utility class for file and directory operations.

    This class provides methods for:
    - Path manipulation and extraction
    - Directory creation and verification
    - File searching and pattern matching
    - File existence validation
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize the FileManager with a logger instance.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for recording operations and errors
        """
        self.logger = logger

    def extract_filename(self, file_path: Union[str, Path]) -> str:
        """
        Extract the filename from a full path.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file

        Returns
        -------
        str
            Base name of the file
        """
        return os.path.basename(str(file_path))

    def extract_directory(self, file_path: Union[str, Path]) -> str:
        """
        Extract the directory path from a full file path.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file

        Returns
        -------
        str
            Directory containing the file
        """
        return os.path.dirname(str(file_path))

    def extract_name_without_extension(self, file_path: Union[str, Path]) -> str:
        """
        Extract the filename without its extension.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file

        Returns
        -------
        str
            Filename without extension
        """
        return os.path.splitext(str(file_path))[0]

    def extract_extension(self, file_path: Union[str, Path]) -> str:
        """
        Extract the file extension.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file

        Returns
        -------
        str
            File extension including the dot
        """
        return os.path.splitext(str(file_path))[1]

    def create_main_output_directory(
        self,
        directory_path: Union[str, Path],
        dry_run: bool = False,
        force_overwrite: bool = False,
        dry_run_manager=None,
        allow_prompt: Optional[bool] = None,
    ) -> str:
        """
        Create main output directory with user confirmation for existing paths.

        Parameters
        ----------
        directory_path : Union[str, Path]
            Path where directory should be created
        dry_run : bool, optional
            If True, only return the path without creating (default: False)
        force_overwrite : bool, optional
            If True, overwrite existing directory without asking (default: False)
        allow_prompt : Optional[bool], optional
            If True, ask before reusing/removing existing paths. If False, fail fast
            instead of blocking for input. Defaults to terminal interactivity.

        Returns
        -------
        str
            Path to the created directory

        Notes
        -----
        If directory exists, asks user whether to overwrite or create numbered version
        """
        if dry_run:
            return str(directory_path)

        output_dir = os.path.abspath(str(directory_path))
        prompt_allowed = sys.stdin.isatty() if allow_prompt is None else bool(allow_prompt)

        if os.path.exists(output_dir):
            if force_overwrite:
                # Remove existing directory and create fresh
                import shutil

                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                self.logger.info(f"Overwrote existing main output directory: {output_dir}")
                return output_dir

            if not prompt_allowed:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Non-interactive runs cannot ask whether to overwrite it. "
                    "Use --force/force = True for a fresh overwrite, or set resume to an existing stage."
                )

            # Ask user what to do
            while True:
                print()  # Empty line for spacing
                print_colored("=" * 60, Colors.RED, bold=True)
                print_colored(
                    f"WARNING: DIRECTORY ALREADY EXISTS: '{output_dir}'",
                    Colors.RED,
                    bold=True,
                )
                print_colored("=" * 60, Colors.RED, bold=True)
                print_colored("What would you like to do?", Colors.WHITE, bold=True)
                print_colored(
                    "1. Overwrite existing directory (WARNING: This will delete all contents)",
                    Colors.RED,
                    bold=True,
                )
                print_colored(
                    "2. Create new directory with increment (e.g., {}.1)".format(os.path.basename(output_dir)),
                    Colors.YELLOW,
                )
                print_colored("3. Cancel operation", Colors.CYAN)
                print_colored("=" * 60, Colors.RED, bold=True)

                try:
                    choice = input("Enter your choice (1/2/3): ").strip()

                    if choice == "1":
                        # Overwrite existing directory
                        import shutil

                        shutil.rmtree(output_dir)
                        os.makedirs(output_dir)
                        print_colored(
                            f"Successfully overwrote directory: {output_dir}",
                            Colors.GREEN,
                            bold=True,
                        )
                        self.logger.info(f"Overwrote existing main output directory: {output_dir}")
                        return output_dir

                    elif choice == "2":
                        # Create numbered directory
                        parent, base = os.path.split(output_dir)
                        counter = self._get_next_directory_number(parent, base)
                        new_dir = os.path.join(parent, f"{base}.{counter}")
                        os.makedirs(new_dir)
                        print_colored(
                            f"Successfully created directory: {new_dir}",
                            Colors.GREEN,
                            bold=True,
                        )
                        self.logger.info(f"Created numbered main output directory: {new_dir}")
                        return new_dir

                    elif choice == "3":
                        # Cancel operation
                        print_colored("Operation cancelled by user", Colors.RED, bold=True)
                        raise KeyboardInterrupt("Operation cancelled by user")

                    else:
                        print_colored("Invalid choice. Please enter 1, 2, or 3.", Colors.RED)

                except KeyboardInterrupt:
                    print_colored("\nOperation cancelled.", Colors.RED)
                    raise

        os.makedirs(output_dir)
        safe_output_dir = str(output_dir).replace("\n", "\\n").replace("\r", "\\r")
        self.logger.info(f"Created main output directory: {safe_output_dir}")
        return output_dir

    def create_subdirectory(
        self,
        directory_path: Union[str, Path],
        dry_run: bool = False,
        preserve_existing: bool = False,
    ) -> str:
        """
        Create a subdirectory with smart handling of existing directories.

        Parameters
        ----------
        directory_path : Union[str, Path]
            Path where directory should be created
        dry_run : bool, optional
            If True, only return the path without creating (default: False)
        preserve_existing : bool, optional
            If True, preserve existing directory and contents (default: False)

        Returns
        -------
        str
            Path to the created directory

        Notes
        -----
        - If preserve_existing=True: Only creates directory if it doesn't exist
        - If preserve_existing=False: Removes and recreates directory if it exists
        - Use preserve_existing=True for modular tools where we want to keep existing results
        """
        if dry_run:
            return str(directory_path)

        output_dir = os.path.abspath(str(directory_path))

        if os.path.exists(output_dir):
            if preserve_existing:
                self.logger.info(f"Using existing subdirectory: {output_dir}")
                return output_dir
            else:
                # Remove directory if it exists and we're not preserving
                import shutil

                shutil.rmtree(output_dir)
                self.logger.info(f"Removed existing subdirectory: {output_dir}")

        # Create directory
        os.makedirs(output_dir)
        safe_output_dir = str(output_dir).replace("\n", "\\n").replace("\r", "\\r")
        self.logger.info(f"Created subdirectory: {safe_output_dir}")
        return output_dir

    def create_directory(self, directory_path: Union[str, Path], dry_run: bool = False) -> str:
        """
        Create a directory using PySeqRNA's numbered-directory policy.

        Parameters
        ----------
        directory_path : Union[str, Path]
            Path where directory should be created
        dry_run : bool, optional
            If True, only return the path without creating (default: False)

        Returns
        -------
        str
            Path to the created directory

        Notes
        -----
        If directory exists, creates a numbered version (e.g., dir.1, dir.2).
        """
        return self.create_main_output_directory(directory_path, dry_run)

    def _get_next_directory_number(self, parent: str, base: str) -> int:
        """
        Get the next available number for numbered directory creation.

        Parameters
        ----------
        parent : str
            Parent directory path
        base : str
            Base name of the directory

        Returns
        -------
        int
            Next available number for directory naming
        """
        counter = 0
        for existing_dir in os.listdir(parent):
            if existing_dir.startswith(f"{base}."):
                try:
                    num = int(existing_dir.split(".")[-1])
                    counter = max(counter, num)
                except ValueError:
                    continue
        return counter + 1

    def verify_files_exist(self, *file_paths: Union[str, Path]) -> bool:
        """
        Verify that all specified files exist.

        Parameters
        ----------
        *file_paths : Union[str, Path]
            Variable number of file paths to check

        Returns
        -------
        bool
            True if all files exist, False otherwise
        """
        missing_files = []
        for f in file_paths:
            if not os.path.isfile(str(f)):
                f_str = str(f)
                abs_path = os.path.abspath(f_str)
                self.logger.debug(f"File check failed: {f_str}")
                self.logger.debug(f"Absolute path: {abs_path}")
                self.logger.debug(f"Path exists but not a file: {os.path.exists(abs_path) and not os.path.isfile(abs_path)}")

                # Check parent directory
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    self.logger.debug(f"Parent directory ({parent_dir}) exists")
                    try:
                        # List files in parent dir to help debug
                        dir_contents = os.listdir(parent_dir)
                        if dir_contents:
                            self.logger.debug(f"First 5 files in parent directory: {dir_contents[:5]}")
                        else:
                            self.logger.debug("Parent directory is empty")
                    except Exception as e:
                        self.logger.debug(f"Could not list parent directory: {str(e)}")
                else:
                    self.logger.debug(f"Parent directory ({parent_dir}) does not exist")

                missing_files.append(f)

        if missing_files:
            self.logger.warning(f"Missing files: {', '.join(map(str, missing_files))}")
            return False
        return True

    def verify_directories_exist(self, *dir_paths: Union[str, Path]) -> bool:
        """
        Verify that all specified directories exist.

        Parameters
        ----------
        *dir_paths : Union[str, Path]
            Variable number of directory paths to check

        Returns
        -------
        bool
            True if all directories exist, False otherwise
        """
        missing_dirs = [d for d in dir_paths if not os.path.isdir(str(d))]
        if missing_dirs:
            self.logger.warning(f"Missing directories: {', '.join(map(str, missing_dirs))}")
            return False
        return True

    def search_files(
        self,
        directory: Union[str, Path] = ".",
        pattern: Optional[str] = None,
        recursive: bool = False,
        verbose: bool = False,
    ) -> List[str]:
        r"""
        Search for files matching a pattern in specified directory.

        Parameters
        ----------
        directory : Union[str, Path], optional
            Directory to search in (default: current directory)
        pattern : str, optional
            Regular expression pattern to match filenames
        recursive : bool, optional
            Whether to search in subdirectories (default: False)
        verbose : bool, optional
            Whether to log detailed search results (default: False)

        Returns
        -------
        List[str]
            List of matching file paths

        Examples
        --------
        >>> file_mgr = FileManager(logger)
        >>> fastq_files = file_mgr.search_files(
        ...     directory="data",
        ...     pattern=r".*\.fastq$",
        ...     recursive=True
        ... )
        """
        directory = str(directory)
        if not self.verify_directories_exist(directory):
            return []

        matching_files = []
        pattern_regex = re.compile(pattern) if pattern else None

        if recursive:
            for root, _, files in os.walk(directory):
                matching_files.extend(os.path.join(root, f) for f in files if not pattern_regex or pattern_regex.search(f))
        else:
            matching_files.extend(
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and (not pattern_regex or pattern_regex.search(f))
            )

        if verbose:
            safe_pattern = str(pattern).replace("\n", "\\n").replace("\r", "\\r") if pattern else ""
            safe_directory = str(directory).replace("\n", "\\n").replace("\r", "\\r")
            self.logger.info(f"Found {len(matching_files)} file(s) matching '{safe_pattern}' in '{safe_directory}'")

        return matching_files

    def find_files_by_pattern(self, pattern: str, directory: Union[str, Path]) -> List[str]:
        """
        Find files matching a glob pattern in a directory.

        Parameters
        ----------
        pattern : str
            Glob pattern to match (e.g., ``"*.fastq"``)
        directory : Union[str, Path]
            Directory to search in

        Returns
        -------
        List[str]
            List of matching file paths
        """
        directory = str(directory)
        matches = [
            os.path.join(root, name)
            for root, _, files in os.walk(directory)
            for name in files
            if fnmatch.fnmatch(name, pattern)
        ]

        self.logger.info(f"Found {len(matches)} file(s) matching '{pattern}' in '{directory}'")
        return matches
