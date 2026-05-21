#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dry Run Manager Module

This module provides functionality to manage dry-run operations in the PySeqRNA
pipeline, allowing simulation and validation of pipeline stages, commands, and file systems.

Features:
    - Centralized dry-run mode status management
    - Simulation of command executions (local and SLURM)
    - Simulation of file and directory operations (creation, copy, move, deletion)
    - Simulation of checkpoint operations
    - Generation of comprehensive dry-run reports
    - Generation of execution reports from actual pipeline runs

Classes:
    - DryRunManager: Manages dry-run operations for the pipeline

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime


class DryRunManager:
    """
    Manages dry-run operations for the PySeqRNA pipeline.

    This class provides a centralized way to handle dry-run mode,
    including command simulation, file operations, and checkpoint management.
    """

    def __init__(self, enabled: bool = False, logger: Optional[logging.Logger] = None):
        """
        Initialize DryRunManager.

        Parameters
        ----------
        enabled : bool, optional
            Whether dry-run mode is enabled, by default False
        logger : logging.Logger, optional
            Logger instance for recording operations, by default None
        """
        self.enabled = enabled
        self.logger = logger or logging.getLogger(__name__)
        self.simulated_operations: List[Dict[str, Any]] = []
        self.executed_operations: List[Dict[str, Any]] = []
        self.stage_results: Dict[str, bool] = {}

        if self.enabled:
            self.logger.info("Dry-run mode enabled - no actual operations will be performed")

    def is_enabled(self) -> bool:
        """
        Check if dry-run mode is enabled.

        Returns
        -------
        bool
            True if dry-run mode is enabled
        """
        return self.enabled

    def simulate_command_execution(self, stage_name: str, commands: Dict[str, str], execution_type: str = "local") -> bool:
        """
        Simulate command execution in dry-run mode.

        Parameters
        ----------
        stage_name : str
            Name of the pipeline stage
        commands : Dict[str, str]
            Dictionary of commands to simulate
        execution_type : str, optional
            Type of execution (local/slurm), by default "local"

        Returns
        -------
        bool
            True if simulation was successful
        """
        if not self.enabled:
            return True

        self.logger.info(f"DRYRUN: Would execute {execution_type.upper()} commands for {stage_name}:")

        for sample_name, command in commands.items():
            self.logger.info(f"  {sample_name}: {command}")

            # Record the operation
            self.simulated_operations.append(
                {
                    "timestamp": self._get_timestamp(),
                    "stage": stage_name,
                    "sample": sample_name,
                    "command": command,
                    "execution_type": execution_type,
                    "operation": "command_execution",
                }
            )

        self.stage_results[stage_name] = True
        return True

    def record_executed_commands(self, executed_commands: List[Dict[str, Any]]) -> None:
        """
        Record actually executed commands for reporting.

        Parameters
        ----------
        executed_commands : List[Dict[str, Any]]
            List of executed command records from command executor
        """
        self.executed_operations.extend(executed_commands)

    def simulate_file_operation(self, operation: str, source: str, destination: Optional[str] = None, **kwargs) -> bool:
        """
        Simulate file operations in dry-run mode.

        Parameters
        ----------
        operation : str
            Type of file operation (create, copy, move, delete)
        source : str
            Source file/directory path
        destination : str, optional
            Destination file/directory path, by default None
        **kwargs
            Additional operation-specific parameters

        Returns
        -------
        bool
            True if simulation was successful
        """
        if not self.enabled:
            return True

        operation_map = {
            "create": f"Would create: {source}",
            "copy": f"Would copy: {source} -> {destination}",
            "move": f"Would move: {source} -> {destination}",
            "delete": f"Would delete: {source}",
            "mkdir": f"Would create directory: {source}",
        }

        message = operation_map.get(operation, f"Would perform {operation}: {source}")
        if destination:
            message = operation_map.get(operation, f"Would perform {operation}: {source} -> {destination}")

        self.logger.info(f"DRYRUN: {message}")

        # Record the operation
        self.simulated_operations.append(
            {
                "timestamp": self._get_timestamp(),
                "operation": f"file_{operation}",
                "source": source,
                "destination": destination,
                "kwargs": kwargs,
            }
        )

        return True

    def simulate_directory_creation(self, directory_path: str, force_overwrite: bool = False) -> str:
        """
        Simulate directory creation in dry-run mode.

        Parameters
        ----------
        directory_path : str
            Path to the directory to create
        force_overwrite : bool, optional
            Whether to force overwrite existing directory, by default False

        Returns
        -------
        str
            The directory path (unchanged in dry-run mode)
        """
        if not self.enabled:
            return directory_path

        if force_overwrite:
            self.logger.info(f"DRYRUN: Would overwrite existing directory: {directory_path}")
        else:
            self.logger.info(f"DRYRUN: Would create directory: {directory_path}")

        # Record the operation
        self.simulated_operations.append(
            {
                "timestamp": self._get_timestamp(),
                "operation": "directory_creation",
                "path": directory_path,
                "force_overwrite": force_overwrite,
            }
        )

        return directory_path

    def simulate_checkpoint_operation(self, operation: str, stage: str, metadata: Optional[Dict] = None) -> bool:
        """
        Simulate checkpoint operations in dry-run mode.

        Parameters
        ----------
        operation : str
            Type of checkpoint operation (mark_complete, save, load)
        stage : str
            Stage name for the checkpoint operation
        metadata : Dict, optional
            Additional metadata for the checkpoint, by default None

        Returns
        -------
        bool
            True if simulation was successful
        """
        if not self.enabled:
            return True

        operation_map = {
            "mark_complete": f"Would mark stage '{stage}' as complete",
            "save": f"Would save checkpoint for stage '{stage}'",
            "load": f"Would load checkpoint for stage '{stage}'",
        }

        message = operation_map.get(operation, f"Would perform {operation} for stage '{stage}'")
        self.logger.info(f"DRYRUN: {message}")

        if metadata:
            self.logger.debug(f"DRYRUN: Metadata: {metadata}")

        # Record the operation
        self.simulated_operations.append(
            {
                "timestamp": self._get_timestamp(),
                "operation": f"checkpoint_{operation}",
                "stage": stage,
                "metadata": metadata,
            }
        )

        return True

    def simulate_slurm_config_creation(self, config_path: str, config_params: Dict[str, str]) -> bool:
        """
        Simulate SLURM configuration file creation in dry-run mode.

        Parameters
        ----------
        config_path : str
            Path where SLURM config would be created
        config_params : Dict[str, str]
            SLURM configuration parameters

        Returns
        -------
        bool
            True if simulation was successful
        """
        if not self.enabled:
            return True

        safe_config_path = str(config_path).replace("\n", "").replace("\r", "")
        safe_config_params = str(config_params).replace("\n", "").replace("\r", "")
        self.logger.info(f"DRYRUN: Would create SLURM config file: {safe_config_path}")
        self.logger.info(f"DRYRUN: SLURM parameters: {safe_config_params}")

        # Record the operation
        self.simulated_operations.append(
            {
                "timestamp": self._get_timestamp(),
                "operation": "slurm_config_creation",
                "config_path": config_path,
                "config_params": config_params,
            }
        )

        return True

    def get_simulated_operations(self) -> List[Dict[str, Any]]:
        """
        Get list of all simulated operations.

        Returns
        -------
        List[Dict[str, Any]]
            List of simulated operations with timestamps and details
        """
        return self.simulated_operations.copy()

    def get_stage_results(self) -> Dict[str, bool]:
        """
        Get results of simulated stage executions.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping stage names to their success status
        """
        return self.stage_results.copy()

    def generate_dry_run_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive dry-run report.

        Parameters
        ----------
        output_path : str, optional
            Path to save the report file, by default None

        Returns
        -------
        str
            The generated report content
        """
        if not self.enabled:
            return "Dry-run mode was not enabled."

        report_lines = [
            "=" * 60,
            "PYSEQRNA DRY-RUN REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Operations Simulated: {len(self.simulated_operations)}",
            f"Stages Simulated: {len(self.stage_results)}",
            "",
            "EXECUTING DRYRUN - COMMANDS THAT WOULD BE EXECUTED:",
            "=" * 60,
            "",
        ]

        # Group operations by type
        operation_types = {}
        for op in self.simulated_operations:
            op_type = op.get("operation", "unknown")
            if op_type not in operation_types:
                operation_types[op_type] = []
            operation_types[op_type].append(op)

        # Show command executions first
        if "command_execution" in operation_types:
            report_lines.append("COMMAND EXECUTIONS:")
            report_lines.append("-" * 30)

            # Group commands by stage
            stage_commands = {}
            for op in operation_types["command_execution"]:
                stage_name = op.get("stage", "Unknown Stage")
                execution_type = op.get("execution_type", "local").upper()
                sample_name = op.get("sample", "Unknown Sample")
                command = op.get("command", "Unknown Command")

                if stage_name not in stage_commands:
                    stage_commands[stage_name] = {
                        "execution_type": execution_type,
                        "commands": {},
                    }
                stage_commands[stage_name]["commands"][sample_name] = command

            # Display commands by stage
            for stage_name, stage_info in stage_commands.items():
                execution_type = stage_info["execution_type"]
                commands = stage_info["commands"]

                report_lines.append(f"\n{stage_name} ({execution_type}):")
                for sample_name, command in commands.items():
                    report_lines.append(f"  {sample_name}: {command}")
                report_lines.append("")
        else:
            report_lines.append("No command executions recorded.")
            report_lines.append("")

        # Show other operations
        for op_type, operations in operation_types.items():
            if op_type == "command_execution":
                continue  # Already handled above

            if op_type == "diffexp_internal":
                # Special handling for Differential Expression internal operations
                report_lines.append("DIFFERENTIAL EXPRESSION INTERNAL OPERATIONS:")
                report_lines.append("-" * 30)

                # Group by operation type
                internal_ops = {}
                for op in operations:
                    op_subtype = op.get("operation_type", "unknown")
                    if op_subtype not in internal_ops:
                        internal_ops[op_subtype] = []
                    internal_ops[op_subtype].append(op)

                for op_subtype, ops in internal_ops.items():
                    report_lines.append(f"\n  {op_subtype.upper()}:")
                    for op in ops:
                        timestamp = op.get("timestamp", "unknown")
                        details = op.get("details", "unknown")
                        comparison = op.get("comparison", "")
                        tool = op.get("tool", "unknown")

                        if comparison:
                            report_lines.append(f"    [{timestamp}] {tool} - {comparison}: {details}")
                        else:
                            report_lines.append(f"    [{timestamp}] {tool}: {details}")

                report_lines.append("")
                continue

            report_lines.append(f"{op_type.upper()} OPERATIONS:")
            report_lines.append("-" * 30)
            for op in operations:
                timestamp = op.get("timestamp", "unknown")
                if "stage" in op:
                    report_lines.append(f"  [{timestamp}] {op_type}: {op['stage']}")
                elif "path" in op:
                    report_lines.append(f"  [{timestamp}] {op_type}: {op['path']}")
                elif "config_path" in op:
                    report_lines.append(f"  [{timestamp}] {op_type}: {op['config_path']}")
                else:
                    report_lines.append(f"  [{timestamp}] {op_type}")
            report_lines.append("")

        # Add execution summary for dry run
        simulated_commands = len([op for op in self.simulated_operations if op.get("operation") == "command_execution"])

        report_lines.append("DRY-RUN SUMMARY:")
        report_lines.append("-" * 30)
        report_lines.append(f"  Total Commands Simulated: {simulated_commands}")
        report_lines.append(f"  Successful Stages: {sum(1 for success in self.stage_results.values() if success)}")
        report_lines.append(f"  Failed Stages: {sum(1 for success in self.stage_results.values() if not success)}")
        report_lines.append(f"  Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Show stage results
        if self.stage_results:
            report_lines.append("STAGE RESULTS:")
            report_lines.append("-" * 30)
            for stage, success in self.stage_results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                report_lines.append(f"  {stage}: {status}")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save report if output path provided
        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(report_content)
                # Sanitize file path for logging to prevent log injection
                safe_output_path = str(output_path).replace("\n", "").replace("\r", "").replace("\x00", "")
                self.logger.info("Dry-run report saved to: %s", safe_output_path)
            except Exception as e:
                # Sanitize error message for logging to prevent log injection
                safe_error = str(e).replace("\n", "").replace("\r", "")
                self.logger.error(f"Failed to save dry-run report: {safe_error}")

        return report_content

    def generate_execution_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive execution report for actual pipeline runs.

        Parameters
        ----------
        output_path : str, optional
            Path to save the report file, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing execution report data and content
        """
        # Combine simulated and executed operations
        all_operations = self.simulated_operations + self.executed_operations

        report_lines = [
            "=" * 60,
            "PYSEQRNA EXECUTION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Operations: {len(all_operations)}",
            f"Simulated Operations: {len(self.simulated_operations)}",
            f"Executed Operations: {len(self.executed_operations)}",
            f"Stages: {len(self.stage_results)}",
            "",
            "COMMAND EXECUTIONS:",
            "=" * 60,
            "",
        ]

        # Group operations by type
        operation_types = {}
        command_operations = []

        for op in all_operations:
            op_type = op.get("operation", "command_execution")  # Default to command_execution
            if op_type not in operation_types:
                operation_types[op_type] = []
            operation_types[op_type].append(op)

            # Also collect all commands (both simulated and executed)
            if "command" in op:
                command_operations.append(op)

        # Show command executions first
        if command_operations:
            report_lines.append("COMMAND EXECUTIONS:")
            report_lines.append("-" * 30)

            # Group commands by stage
            stage_commands = {}
            for op in command_operations:
                stage_name = op.get("stage", "Unknown Stage")
                execution_type = op.get("execution_type", "local").upper()
                sample_name = op.get("sample", "Unknown Sample")
                command = op.get("command", "Unknown Command")

                if stage_name not in stage_commands:
                    stage_commands[stage_name] = {
                        "execution_type": execution_type,
                        "commands": {},
                    }
                stage_commands[stage_name]["commands"][sample_name] = command

            # Display commands by stage
            for stage_name, stage_info in stage_commands.items():
                execution_type = stage_info["execution_type"]
                commands = stage_info["commands"]

                report_lines.append(f"\n{stage_name} ({execution_type}):")
                for sample_name, command in commands.items():
                    report_lines.append(f"  {sample_name}: {command}")
                report_lines.append("")
        else:
            report_lines.append("No command executions recorded.")
            report_lines.append("")

        # Show other operations
        for op_type, operations in operation_types.items():
            if op_type == "command_execution":
                continue  # Already handled above

            if op_type == "diffexp_internal":
                # Special handling for Differential Expression internal operations
                report_lines.append("DIFFERENTIAL EXPRESSION INTERNAL OPERATIONS:")
                report_lines.append("-" * 30)

                # Group by operation type
                internal_ops = {}
                for op in operations:
                    op_subtype = op.get("operation_type", "unknown")
                    if op_subtype not in internal_ops:
                        internal_ops[op_subtype] = []
                    internal_ops[op_subtype].append(op)

                for op_subtype, ops in internal_ops.items():
                    report_lines.append(f"\n  {op_subtype.upper()}:")
                    for op in ops:
                        timestamp = op.get("timestamp", "unknown")
                        details = op.get("details", "unknown")
                        comparison = op.get("comparison", "")
                        tool = op.get("tool", "unknown")

                        if comparison:
                            report_lines.append(f"    [{timestamp}] {tool} - {comparison}: {details}")
                        else:
                            report_lines.append(f"    [{timestamp}] {tool}: {details}")

                report_lines.append("")
                continue

            if op_type == "genomic_overlaps_internal":
                # Special handling for GenomicOverlaps internal operations
                report_lines.append("GENOMIC OVERLAPS INTERNAL OPERATIONS:")
                report_lines.append("-" * 30)

                # Group by operation type
                internal_ops = {}
                for op in operations:
                    op_subtype = op.get("operation_type", "unknown")
                    if op_subtype not in internal_ops:
                        internal_ops[op_subtype] = []
                    internal_ops[op_subtype].append(op)

                for op_subtype, ops in internal_ops.items():
                    report_lines.append(f"\n  {op_subtype.upper()}:")
                    for op in ops:
                        timestamp = op.get("timestamp", "unknown")
                        details = op.get("details", "unknown")
                        sample = op.get("sample", "")
                        filter_strategy = op.get("filter_strategy", "unknown")
                        overlap_mode = op.get("overlap_mode", "unknown")

                        if sample:
                            report_lines.append(f"    [{timestamp}] {sample}: {details}")
                        else:
                            report_lines.append(f"    [{timestamp}] {details}")

                        # Add mode info for sample operations
                        if sample and op_subtype == "sample_quantification":
                            report_lines.append(f"      Mode: {overlap_mode}, Filter: {filter_strategy}")

                report_lines.append("")
                continue

            if op_type == "normalization_internal":
                # Special handling for Normalization internal operations
                report_lines.append("NORMALIZATION INTERNAL OPERATIONS:")
                report_lines.append("-" * 30)

                # Group by normalization method
                method_ops = {}
                for op in operations:
                    method = op.get("normalization_method", "unknown")
                    if method not in method_ops:
                        method_ops[method] = {}

                    op_subtype = op.get("operation_type", "unknown")
                    if op_subtype not in method_ops[method]:
                        method_ops[method][op_subtype] = []
                    method_ops[method][op_subtype].append(op)

                for method, method_operations in method_ops.items():
                    report_lines.append(f"\n  {method.upper()} NORMALIZATION:")
                    for op_subtype, ops in method_operations.items():
                        report_lines.append(f"    {op_subtype.upper()}:")
                        for op in ops:
                            timestamp = op.get("timestamp", "unknown")
                            details = op.get("details", "unknown")
                            report_lines.append(f"      [{timestamp}] {details}")

                report_lines.append("")
                continue

            report_lines.append(f"{op_type.upper()} OPERATIONS:")
            report_lines.append("-" * 30)
            for op in operations:
                timestamp = op.get("timestamp", "unknown")
                if "stage" in op:
                    report_lines.append(f"  [{timestamp}] {op_type}: {op['stage']}")
                elif "path" in op:
                    report_lines.append(f"  [{timestamp}] {op_type}: {op['path']}")
                elif "config_path" in op:
                    report_lines.append(f"  [{timestamp}] {op_type}: {op['config_path']}")
                else:
                    report_lines.append(f"  [{timestamp}] {op_type}")
            report_lines.append("")

        # Show stage results
        if self.stage_results:
            report_lines.append("STAGE RESULTS:")
            report_lines.append("-" * 30)
            for stage, success in self.stage_results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                report_lines.append(f"  {stage}: {status}")
            report_lines.append("")

        # Add execution summary
        report_lines.append("EXECUTION SUMMARY:")
        report_lines.append("-" * 30)

        # Count all commands (both simulated and executed)
        total_commands = len(command_operations)
        simulated_commands = len([op for op in self.simulated_operations if op.get("operation") == "command_execution"])
        executed_commands = len(self.executed_operations)

        report_lines.append(f"  Total Commands: {total_commands}")
        report_lines.append(f"  Simulated Commands: {simulated_commands}")
        report_lines.append(f"  Executed Commands: {executed_commands}")
        report_lines.append(f"  Successful Stages: {sum(1 for success in self.stage_results.values() if success)}")
        report_lines.append(f"  Failed Stages: {sum(1 for success in self.stage_results.values() if not success)}")
        report_lines.append(f"  Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save report if output path provided
        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(report_content)
                # Sanitize file path for logging to prevent log injection
                safe_output_path = str(output_path).replace("\n", "").replace("\r", "")
                self.logger.info(f"Execution report saved to: {safe_output_path}")
            except Exception as e:
                # Sanitize error message for logging to prevent log injection
                safe_error = str(e).replace("\n", "").replace("\r", "")
                self.logger.error(f"Failed to save execution report: {safe_error}")

        # Return both the operations data and the report content
        return {
            "executed_operations": all_operations,
            "simulated_operations": self.simulated_operations,
            "total_operations": len(all_operations),
            "stage_results": self.stage_results,
            "report_content": report_content,
        }

    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.

        Returns
        -------
        str
            Current timestamp in ISO format
        """
        return datetime.now().isoformat()

    def reset(self) -> None:
        """Reset the dry-run manager state."""
        self.simulated_operations.clear()
        self.stage_results.clear()
        self.logger.info("Dry-run manager state reset")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enabled and exc_type is None:
            self.logger.info("Dry-run session completed successfully")
        elif self.enabled:
            self.logger.warning("Dry-run session completed with errors")
