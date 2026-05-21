"""
PySeqRNA CLI Package

This package provides command-line interface functionality for PySeqRNA,
exposing argument parsers and utility helpers.

Features:
    - CLI argument parsing management
    - CLI command implementations and helper utilities

Classes:
    - ArgumentManager: Argument parsing manager

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .argument_manager import ArgumentManager

__all__ = ["ArgumentManager"]
