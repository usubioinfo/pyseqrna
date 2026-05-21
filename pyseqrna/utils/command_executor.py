#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Command Executor Module

This module provides functionality for executing shell commands and managing jobs
both locally and on a SLURM cluster within the PySeqRNA pipeline.

Features:
    - Local command execution with ThreadPoolExecutor concurrency
    - SLURM job submission and tracking
    - SLURM array job submission for batch processing
    - Output and error log redirection
    - SLURM environment validation
    - Robust error handling and command failure logging
    - Support for shell pipelines with pipefail safety

Classes:
    - CommandExecutor: Manages execution of commands locally or on SLURM cluster

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import subprocess
import shlex
import shutil
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import logging
import time


class CommandExecutor:
    """Handles command execution both locally and on SLURM."""

    DEFAULT_SLURM_WAIT_TIMEOUT_SECONDS = 72 * 3600

    def __init__(self, logger: logging.Logger, slurm_wait_timeout_seconds: Optional[int] = None):
        """
        Initialize CommandExecutor.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for recording operations
        """
        self.logger = logger
        self.slurm_wait_timeout_seconds = int(
            self.DEFAULT_SLURM_WAIT_TIMEOUT_SECONDS if slurm_wait_timeout_seconds is None else slurm_wait_timeout_seconds
        )
        if self.slurm_wait_timeout_seconds < 1:
            raise ValueError(f"slurm_wait_timeout_seconds must be >= 1, got {self.slurm_wait_timeout_seconds}")
        self.executed_commands: List[Dict[str, Any]] = []
        self._slurm_environment_validated = False
        self._commands_lock = threading.Lock()

    def execute_local(
        self,
        commands: Dict[str, str],
        tool_name: str,
        outdir: str,
        sample_names: Optional[Dict[str, str]] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, bool]:
        """
        Execute commands locally with comprehensive logging.

        Parameters
        ----------
        commands : Dict[str, str]
            Dictionary of commands to execute
        tool_name : str
            Name of the tool being executed
        outdir : str
            Output directory for logs
        sample_names : Optional[Dict[str, str]]
            Mapping of command keys to sample names
        max_workers : Optional[int]
            Maximum local commands to run concurrently. If omitted, reads
            PYSEQRNA_LOCAL_JOBS and otherwise preserves sequential execution.

        Returns
        -------
        Dict[str, bool]
            Dictionary of sample names and their execution status
        """
        # Create logs directory based on the output directory provided
        logs_dir = Path(outdir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_logs_dir = str(logs_dir).replace("\n", "").replace("\r", "")
        self.logger.info(f"Created logs directory: {safe_logs_dir}")

        tasks = [(key, sample_names.get(key, key) if sample_names else key, command) for key, command in commands.items()]
        worker_count = self._resolve_local_workers(max_workers, len(tasks))
        if worker_count > 1:
            self.logger.info(f"Executing {len(tasks)} {tool_name} command(s) locally with {worker_count} worker(s)")

        results: Dict[str, bool] = {}
        if worker_count == 1:
            for task in tasks:
                sample_name, ok = self._execute_single_local(task, tool_name, logs_dir)
                results[sample_name] = ok
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self._execute_single_local, task, tool_name, logs_dir) for task in tasks]
            for future in as_completed(futures):
                sample_name, ok = future.result()
                results[sample_name] = ok
        return results

    def _resolve_local_workers(self, max_workers: Optional[int], task_count: int) -> int:
        """Resolve local command concurrency while preserving sequential default."""
        if task_count <= 1:
            return 1
        if max_workers is None:
            env_value = os.environ.get("PYSEQRNA_LOCAL_JOBS")
            if env_value:
                try:
                    max_workers = int(env_value)
                except ValueError:
                    self.logger.warning(
                        "Invalid PYSEQRNA_LOCAL_JOBS=%r; using sequential execution",
                        env_value,
                    )
                    max_workers = 1
            else:
                max_workers = 1
        return max(1, min(int(max_workers), task_count))

    def _execute_single_local(
        self,
        task: Tuple[str, str, str],
        tool_name: str,
        logs_dir: Path,
    ) -> Tuple[str, bool]:
        """Execute one local command and stream stdout/stderr directly to logs."""
        _key, sample_name, command = task
        stdout_log = logs_dir / f"{sample_name}_{tool_name}.out"
        stderr_log = logs_dir / f"{sample_name}_{tool_name}.err"
        self.logger.info(f"Executing {tool_name} for sample {sample_name}")
        self.logger.debug(f"Command: {command}")
        self.logger.debug(f"Log files: {stdout_log}, {stderr_log}")

        record = {
            "timestamp": datetime.now().isoformat(),
            "stage": tool_name,
            "sample": sample_name,
            "command": command,
            "execution_type": "local",
            "status": "started",
        }
        with self._commands_lock:
            self.executed_commands.append(record)

        start_time = time.time()
        try:
            with (
                open(stderr_log, "w") as stderr_file,
                open(stdout_log, "w") as stdout_file,
            ):
                stdout_file.write(f"=== {tool_name.upper()} LOCAL EXECUTION LOG ===\n")
                stdout_file.write("Generated by: PySeqRNA\n")
                stdout_file.write("Author: Naveen Duhan\n")
                stdout_file.write(f"Sample: {sample_name}\n")
                stdout_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                stdout_file.write(f"Command: {command}\n")
                stdout_file.write("=" * 50 + "\n\n")
                stdout_file.write("=== STDOUT ===\n")
                stdout_file.flush()

                cmd_args = self._command_subprocess_args(command)
                process = subprocess.run(
                    cmd_args,
                    shell=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    bufsize=1,
                    check=False,
                )

                execution_time = time.time() - start_time
                stdout_file.write("\n" + "=" * 50 + "\n")
                stdout_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                stdout_file.write(f"Execution Time: {execution_time:.2f} seconds\n")
                stdout_file.write(f"Return Code: {process.returncode}\n")
                stdout_file.write(f"Status: {'SUCCESS' if process.returncode == 0 else 'FAILED'}\n")
                stdout_file.write("=" * 50 + "\n")

            record.update(
                {
                    "status": "completed" if process.returncode == 0 else "failed",
                    "return_code": process.returncode,
                    "execution_time": execution_time,
                }
            )
            if process.returncode == 0:
                self.logger.info(f"Successfully completed {tool_name} for sample {sample_name} in {execution_time:.2f}s")
                return sample_name, True

            self.logger.error(f"{tool_name} failed for sample {sample_name} with return code {process.returncode}")
            self.logger.error(f"Check logs: {stdout_log}, {stderr_log}")
            return sample_name, False

        except Exception as e:
            record.update({"status": "error", "error": str(e)})
            self.logger.exception("Error executing %s for sample %s", tool_name, sample_name)
            try:
                with open(stderr_log, "a") as f:
                    f.write(f"Exception occurred: {str(e)}\n")
                with open(stdout_log, "a") as f:
                    f.write(f"Exception occurred: {str(e)}\n")
            except OSError:
                pass
            return sample_name, False

    def _command_subprocess_args(self, command: Any) -> List[str]:
        """
        Return subprocess arguments for a PySeqRNA command.

        Most tool commands are plain argv-style command strings and can run via
        shlex.split. Alignment commands intentionally use shell pipelines such
        as ``hisat2 | samtools view | samtools sort``; those must be interpreted
        by a shell. We still avoid ``shell=True`` and run bash explicitly with
        pipefail so upstream failures are not hidden by the final command.
        """
        if not isinstance(command, str):
            return list(command)

        if self._requires_shell_pipeline(command):
            return ["bash", "-o", "pipefail", "-c", command]

        return shlex.split(command)

    @staticmethod
    def _requires_shell_pipeline(command: str) -> bool:
        """Return True when a command uses shell syntax that argv cannot express."""
        shell_tokens = ("|", "&&", "||", ";", ">", "<", "$(", "`")
        return any(token in command for token in shell_tokens)

    def execute_slurm(
        self,
        commands: Dict[str, str],
        tool_name: str,
        job_name: str = "pyseqRNA",
        outdir: str = ".",
        dependency: str = "",
        slurm_config: Optional[Dict[str, str]] = None,
        sample_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Submit jobs to SLURM.

        Parameters
        ----------
        commands : Dict[str, str]
            Commands to execute
        job_name : str
            Base name for SLURM jobs
        outdir : str
            Output directory
        dependency : str
            SLURM job dependency
        slurm_config : Optional[Dict[str, str]]
            SLURM configuration parameters

        Returns
        -------
        Dict[str, str]
            Dictionary of job IDs
        """
        # Create logs directory based on the output directory provided
        logs_dir = Path(outdir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created SLURM logs directory: {logs_dir}")

        if not slurm_config:
            self.logger.warning(
                "No SLURM configuration was provided to CommandExecutor; "
                "using safe defaults. Check pipeline config propagation if this was unexpected."
            )
        slurm_config = self._normalize_slurm_config(slurm_config or {})
        self.validate_slurm_environment(partition=slurm_config.get("partition"))

        if len(commands) > 1 and str(slurm_config.get("use_array", "true")).lower() in {
            "1",
            "true",
            "yes",
        }:
            return self.execute_slurm_array(
                commands=commands,
                tool_name=tool_name,
                job_name=job_name,
                outdir=outdir,
                dependency=dependency,
                slurm_config=slurm_config,
                sample_names=sample_names,
            )

        job_ids = {}

        for key, command in commands.items():
            sample_name = sample_names.get(key, key) if sample_names else key

            try:
                # Create SLURM script with enhanced logging
                slurm_script = self._create_slurm_script(sample_name, tool_name, command, logs_dir, slurm_config)

                sbatch_command = self._construct_sbatch_command(
                    sample_name,
                    job_name,
                    logs_dir,
                    slurm_script,
                    dependency,
                    slurm_config,
                )

                self.logger.info(f"Submitting {tool_name} SLURM job for {sample_name}")
                self.logger.debug(f"SLURM command: {sbatch_command}")

                sbatch_response = self._submit_sbatch(sbatch_command)
                job_id = sbatch_response.split()[-1].strip()
                job_ids[sample_name] = job_id

                self.logger.info(f"Submitted {tool_name} SLURM job {job_id} for {sample_name}")

            except Exception as e:
                self.logger.error(f"Failed to submit SLURM job for {sample_name}: {str(e)}")
                raise

        wait_for_completion = str(slurm_config.get("wait_for_completion", "true")).lower() in {"1", "true", "yes"}
        if job_ids and wait_for_completion:
            check_interval = int(slurm_config.get("check_interval", 60))
            initial_delay = int(slurm_config.get("initial_delay", 10))
            self._wait_for_slurm_jobs(
                job_ids,
                logs_dir,
                tool_name,
                check_interval=check_interval,
                initial_delay=initial_delay,
                max_wait_seconds=self._slurm_wait_timeout(slurm_config),
            )
            # In blocking mode, there are no outstanding jobs for downstream
            # stages to depend on. Returning IDs here would create invalid
            # afterok dependencies on already-finished jobs/array tasks.
            return {}

        return job_ids

    def execute_slurm_array(
        self,
        commands: Dict[str, str],
        tool_name: str,
        job_name: str = "pyseqRNA",
        outdir: str = ".",
        dependency: str = "",
        slurm_config: Optional[Dict[str, str]] = None,
        sample_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Submit multiple sample commands as one SLURM array job."""
        logs_dir = Path(outdir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        slurm_config = self._normalize_slurm_config(slurm_config or {})
        self.validate_slurm_environment(partition=slurm_config.get("partition"))

        tasks = [(sample_names.get(key, key) if sample_names else key, command) for key, command in commands.items()]

        slurm_script = self._create_slurm_array_script(
            tasks=tasks,
            tool_name=tool_name,
            job_name=job_name,
            logs_dir=logs_dir,
            slurm_config=slurm_config,
        )
        sbatch_command = self._construct_sbatch_command(
            sample_name=job_name,
            job_name=job_name,
            logs_dir=logs_dir,
            script_path=slurm_script,
            dependency=dependency,
            slurm_config=slurm_config,
        )

        self.logger.info(f"Submitting {tool_name} SLURM array with {len(tasks)} task(s)")
        self.logger.debug(f"SLURM array command: {sbatch_command}")

        sbatch_response = self._submit_sbatch(sbatch_command, array=True)

        array_job_id = sbatch_response.split()[-1].strip()
        job_ids = {sample_name: f"{array_job_id}_{idx}" for idx, (sample_name, _) in enumerate(tasks)}
        self.logger.info(f"Submitted {tool_name} SLURM array job {array_job_id}")

        wait_for_completion = str(slurm_config.get("wait_for_completion", "true")).lower() in {"1", "true", "yes"}
        if wait_for_completion:
            check_interval = int(slurm_config.get("check_interval", 60))
            initial_delay = int(slurm_config.get("initial_delay", 10))
            self._wait_for_slurm_array_job(
                array_job_id=array_job_id,
                job_ids=job_ids,
                logs_dir=logs_dir,
                tool_name=tool_name,
                check_interval=check_interval,
                initial_delay=initial_delay,
                max_wait_seconds=self._slurm_wait_timeout(slurm_config),
            )
            return {}

        return job_ids

    def _create_slurm_script(
        self,
        sample_name: str,
        tool_name: str,
        command: str,
        logs_dir: Path,
        slurm_config: Dict[str, str],
    ) -> str:
        """Create SLURM script with enhanced logging."""
        logs_dir.mkdir(parents=True, exist_ok=True)
        slurm_config = self._normalize_slurm_config(slurm_config)
        command_runner = self._slurm_command_runner_block()
        script_content = f"""#!/bin/bash
# Generated by: PySeqRNA
# Author: Naveen Duhan
#SBATCH --job-name={sample_name}_{tool_name}
#SBATCH --output={logs_dir}/{sample_name}_{tool_name}.slurm.out
#SBATCH --error={logs_dir}/{sample_name}_{tool_name}.slurm.err
#SBATCH --time={slurm_config["time"]}
#SBATCH --partition={slurm_config["partition"]}
#SBATCH --mem={slurm_config["memory"]}
#SBATCH --cpus-per-task={slurm_config["cpus"]}
#SBATCH --ntasks={slurm_config["ntasks"]}
{f"#SBATCH --account={slurm_config.get('account')}" if slurm_config.get("account") else ""}
{f"#SBATCH --mail-user={slurm_config.get('email')}" if slurm_config.get("email") else ""}
{"#SBATCH --mail-type=END,FAIL" if slurm_config.get("email") else ""}
{f"#SBATCH --qos={slurm_config.get('qos')}" if slurm_config.get("qos") else ""}

SAMPLE="{sample_name}"
TOOL="{tool_name}"
COMMAND={shlex.quote(command)}
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== {tool_name.upper()} SLURM EXECUTION LOG ==="
echo "Generated by: PySeqRNA"
echo "Author: Naveen Duhan"
echo "Sample: $SAMPLE"
echo "Start Time: $START_TIME"
echo "Command: $COMMAND"
echo "=================================================="
echo ""

echo "=== STDOUT ==="
{command_runner}

EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "=================================================="
echo "End Time: $END_TIME"
echo "Return Code: $EXIT_CODE"
echo "Status: $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "=================================================="

exit $EXIT_CODE
"""

        script_path = logs_dir / f"{sample_name}_{tool_name}.slurm"
        with open(script_path, "w") as f:
            f.write(script_content)

        return str(script_path)

    def _slurm_command_runner_block(self) -> str:
        """Return a shell block that runs COMMAND without shell=True or eval."""
        return r"""PYSEQRNA_COMMAND="$COMMAND" "${PYSEQRNA_PYTHON:-python}" - <<'PYSEQRNA_RUNNER'
import os
import shlex
import subprocess
import sys

command = os.environ["PYSEQRNA_COMMAND"]
shell_tokens = ("|", "&&", "||", ";", ">", "<", "$(", "`")
if any(token in command for token in shell_tokens):
    command_args = ["bash", "-o", "pipefail", "-c", command]
else:
    try:
        command_args = shlex.split(command)
    except ValueError as exc:
        print(f"Failed to parse PySeqRNA command: {exc}", file=sys.stderr)
        sys.exit(2)

if not command_args:
    print("Empty PySeqRNA command", file=sys.stderr)
    sys.exit(2)

completed = subprocess.run(command_args)
sys.exit(completed.returncode)
PYSEQRNA_RUNNER"""

    def _create_slurm_array_script(
        self,
        tasks: List[tuple],
        tool_name: str,
        job_name: str,
        logs_dir: Path,
        slurm_config: Dict[str, str],
    ) -> str:
        """Create one SLURM array script for multiple sample commands."""
        logs_dir.mkdir(parents=True, exist_ok=True)
        slurm_config = self._normalize_slurm_config(slurm_config)
        command_runner = self._slurm_command_runner_block()
        max_parallel = slurm_config.get("array_max_parallel")
        array_range = f"0-{len(tasks) - 1}"
        if max_parallel:
            array_range = f"{array_range}%{max_parallel}"

        sample_lines = "\n".join(f"SAMPLES[{idx}]={shlex.quote(sample_name)}" for idx, (sample_name, _) in enumerate(tasks))
        command_lines = "\n".join(f"COMMANDS[{idx}]={shlex.quote(command)}" for idx, (_, command) in enumerate(tasks))

        script_content = f"""#!/bin/bash
# Generated by: PySeqRNA
# Author: Naveen Duhan
#SBATCH --job-name={job_name}_{tool_name}
#SBATCH --output={logs_dir}/.{job_name}_{tool_name}.array.%A_%a.out
#SBATCH --error={logs_dir}/.{job_name}_{tool_name}.array.%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={slurm_config["time"]}
#SBATCH --partition={slurm_config["partition"]}
#SBATCH --mem={slurm_config["memory"]}
#SBATCH --cpus-per-task={slurm_config["cpus"]}
#SBATCH --ntasks={slurm_config["ntasks"]}
{f"#SBATCH --account={slurm_config.get('account')}" if slurm_config.get("account") else ""}
{f"#SBATCH --mail-user={slurm_config.get('email')}" if slurm_config.get("email") else ""}
{"#SBATCH --mail-type=END,FAIL" if slurm_config.get("email") else ""}
{f"#SBATCH --qos={slurm_config.get('qos')}" if slurm_config.get("qos") else ""}

declare -a SAMPLES
declare -a COMMANDS
{sample_lines}
{command_lines}

SAMPLE="${{SAMPLES[$SLURM_ARRAY_TASK_ID]}}"
COMMAND="${{COMMANDS[$SLURM_ARRAY_TASK_ID]}}"
TOOL="{tool_name}"
LOG_DIR="{logs_dir}"

exec > "$LOG_DIR/${{SAMPLE}}_${{TOOL}}.slurm.out" 2> "$LOG_DIR/${{SAMPLE}}_${{TOOL}}.slurm.err"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "=== {tool_name.upper()} SLURM ARRAY EXECUTION LOG ==="
echo "Generated by: PySeqRNA"
echo "Author: Naveen Duhan"
echo "Sample: $SAMPLE"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $START_TIME"
echo "Command: $COMMAND"
echo "=================================================="
echo ""

echo "=== STDOUT ==="
{command_runner}

EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "=================================================="
echo "End Time: $END_TIME"
echo "Return Code: $EXIT_CODE"
echo "Status: $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "=================================================="

exit $EXIT_CODE
"""

        script_path = logs_dir / f"{job_name}_{tool_name}.array.slurm"
        with open(script_path, "w") as f:
            f.write(script_content)

        return str(script_path)

    def _normalize_slurm_config(self, slurm_config: Dict[str, str]) -> Dict[str, str]:
        """Normalize SLURM config values for script generation."""
        config = dict(slurm_config or {})
        config["partition"] = str(config.get("partition") or "compute")
        config["time"] = str(config.get("time") or "24:00:00")
        config["memory"] = self._format_slurm_memory(config.get("memory") or "16")
        config["cpus"] = str(config.get("cpus") or "1")
        config["ntasks"] = str(config.get("ntasks") or "1")
        return config

    def _format_slurm_memory(self, memory: Any) -> str:
        """Format memory for SBATCH --mem. Numeric values are interpreted as GB."""
        value = str(memory).strip()
        if not value:
            return "16G"

        upper_value = value.upper()
        if upper_value.endswith(("K", "M", "G", "T", "KB", "MB", "GB", "TB")):
            return value

        try:
            numeric = float(value)
        except ValueError:
            return value

        if numeric.is_integer():
            return f"{int(numeric)}G"
        return f"{numeric}G"

    def validate_slurm_environment(self, partition: Optional[str] = None) -> None:
        """
        Validate that SLURM client commands can reach the cluster controller.

        This catches scheduler/DNS/configuration problems before PySeqRNA starts
        writing stage outputs or submitting a large batch of internal jobs.
        """
        if self._slurm_environment_validated:
            return

        missing = [cmd for cmd in ("sbatch", "squeue") if shutil.which(cmd) is None]
        if missing:
            raise RuntimeError(
                "SLURM mode requested, but required command(s) are not available "
                f"on PATH: {', '.join(missing)}. Activate the HPC SLURM environment "
                "or run PySeqRNA with slurm = False."
            )

        checks = [
            ["scontrol", "ping"],
            ["sinfo", "-h", "-o", "%P"],
        ]
        errors = []
        for command in checks:
            if shutil.which(command[0]) is None:
                errors.append(f"{command[0]}: command not found")
                continue
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                self._slurm_environment_validated = True
                if partition:
                    self.logger.debug(f"SLURM controller reachable; using partition {partition}")
                else:
                    self.logger.debug("SLURM controller reachable")
                return
            errors.append(self._format_command_failure(command, result))

        raise RuntimeError(
            "SLURM mode requested, but the SLURM controller/configuration is not "
            "reachable from this shell.\n"
            "Checks attempted:\n"
            + "\n".join(f"  - {error}" for error in errors)
            + "\nRun `scontrol ping` or `sinfo` in the same job/session to confirm "
            "the scheduler is available, or run PySeqRNA with `slurm = False` if "
            "you are already inside an allocated SLURM job and want local execution."
        )

    def _submit_sbatch(self, sbatch_command: str, array: bool = False) -> str:
        """Submit an sbatch command and return its stdout with clean errors."""
        result = subprocess.run(
            shlex.split(sbatch_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        response = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        if result.returncode != 0 or not result.stdout.strip().startswith("Submitted batch job "):
            label = "sbatch array submission" if array else "sbatch submission"
            raise RuntimeError(f"{label} failed:\n{response or '(no sbatch output)'}")
        return result.stdout.strip()

    def _format_command_failure(self, command: List[str], result: subprocess.CompletedProcess) -> str:
        """Format a failed scheduler probe with stdout/stderr preserved."""
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        if not output:
            output = f"exit code {result.returncode}"
        return f"{' '.join(command)}: {output}"

    def _construct_sbatch_command(
        self,
        sample_name: str,
        job_name: str,
        logs_dir: Path,
        script_path: str,
        dependency: str,
        slurm_config: Dict[str, str],
    ) -> str:
        """Construct SLURM sbatch command."""
        dep_str = ""
        if dependency:
            dependency_ids = [dep.strip() for dep in str(dependency).replace(":", ",").split(",") if dep.strip()]
            if dependency_ids:
                dep_str = f"--dependency=afterok:{':'.join(dependency_ids)} --kill-on-invalid-dep=yes "

        return f"sbatch {dep_str}{shlex.quote(script_path)}"

    def _wait_for_slurm_jobs(
        self,
        job_ids: Dict[str, str],
        logs_dir: Path,
        tool_name: str,
        check_interval: int = 60,
        initial_delay: int = 10,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        """Wait for submitted SLURM jobs and validate their per-sample logs."""
        if len(job_ids) == 1:
            only_sample, only_job = next(iter(job_ids.items()))
            self.logger.info(f"Waiting for {tool_name} SLURM job {only_job} ({only_sample}) to finish")
        else:
            self.logger.info(f"Waiting for {len(job_ids)} {tool_name} SLURM job(s) to finish (poll every {check_interval}s)")
        if initial_delay > 0:
            self.logger.debug(f"Giving SLURM {initial_delay}s to register submitted {tool_name} job(s)")
            time.sleep(initial_delay)

        for sample_name, job_id in job_ids.items():
            stdout_log = logs_dir / f"{sample_name}_{tool_name}.slurm.out"
            stderr_log = logs_dir / f"{sample_name}_{tool_name}.slurm.err"
            self._wait_for_single_slurm_job(
                sample_name=sample_name,
                job_id=job_id,
                tool_name=tool_name,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                check_interval=check_interval,
                max_wait_seconds=max_wait_seconds,
            )
            self.logger.info(f"{tool_name} completed for {sample_name} (job {job_id})")

        self.logger.info(f"All SLURM {tool_name} job(s) completed successfully")

    def _wait_for_slurm_array_job(
        self,
        array_job_id: str,
        job_ids: Dict[str, str],
        logs_dir: Path,
        tool_name: str,
        check_interval: int = 60,
        initial_delay: int = 10,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        """Wait for one SLURM array job, then validate each task wrapper log."""
        self.logger.info(f"Waiting for {tool_name} SLURM array job {array_job_id} ({len(job_ids)} task(s)) to finish")
        if initial_delay > 0:
            time.sleep(initial_delay)

        max_wait_seconds = int(self.slurm_wait_timeout_seconds if max_wait_seconds is None else max_wait_seconds)
        elapsed = 0
        while elapsed < max_wait_seconds:
            state = self._get_slurm_job_state(array_job_id)
            if state == "COMPLETED":
                break
            if state in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
            }:
                raise RuntimeError(
                    f"{tool_name} SLURM array job {array_job_id} ended with state {state}. Check logs in {logs_dir}"
                )

            # Some clusters do not expose sacct immediately. In that case, use
            # the sample wrapper logs as the source of truth.
            statuses = [self._slurm_job_log_status(logs_dir / f"{sample}_{tool_name}.slurm.out") for sample in job_ids]
            if statuses and all(status == "SUCCESS" for status in statuses):
                break
            if any(status == "FAILED" for status in statuses):
                failed = [
                    sample
                    for sample in job_ids
                    if self._slurm_job_log_status(logs_dir / f"{sample}_{tool_name}.slurm.out") == "FAILED"
                ]
                raise RuntimeError(f"{tool_name} SLURM array task(s) failed: {', '.join(failed)}. Check logs in {logs_dir}")

            time.sleep(check_interval)
            elapsed += check_interval
        else:
            raise TimeoutError(
                f"{tool_name} SLURM array job {array_job_id} did not complete within "
                f"{max_wait_seconds // 3600}h. Check logs in {logs_dir}"
            )

        failed = []
        for sample_name in job_ids:
            stdout_log = logs_dir / f"{sample_name}_{tool_name}.slurm.out"
            stderr_log = logs_dir / f"{sample_name}_{tool_name}.slurm.err"
            if not self._slurm_job_log_success(stdout_log):
                failed.append(f"{sample_name} ({stdout_log}, {stderr_log})")

        if failed:
            raise RuntimeError(f"{tool_name} SLURM array task(s) did not finish successfully: {', '.join(failed)}")

        self._cleanup_slurm_array_scheduler_logs(logs_dir, tool_name)
        self.logger.info(f"All {len(job_ids)} {tool_name} SLURM array task(s) completed successfully")

    def _cleanup_slurm_array_scheduler_logs(self, logs_dir: Path, tool_name: str) -> None:
        """Remove internal SLURM array scheduler logs after sample logs are validated."""
        for log_file in logs_dir.glob(f".*_{tool_name}.array.*"):
            try:
                log_file.unlink()
            except OSError as e:
                self.logger.debug(f"Could not remove internal array log {log_file}: {e}")

    def _wait_for_single_slurm_job(
        self,
        sample_name: str,
        job_id: str,
        tool_name: str,
        stdout_log: Path,
        stderr_log: Path,
        check_interval: int,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        """Wait until one SLURM job writes a final wrapper status."""
        max_wait_seconds = int(self.slurm_wait_timeout_seconds if max_wait_seconds is None else max_wait_seconds)
        elapsed = 0
        while elapsed < max_wait_seconds:
            log_status = self._slurm_job_log_status(stdout_log)

            if log_status == "SUCCESS":
                return
            if log_status == "FAILED":
                raise RuntimeError(f"{tool_name} SLURM job failed for {sample_name}: {stdout_log}, {stderr_log}")

            slurm_state = self._get_slurm_job_state(job_id)
            if slurm_state in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
            }:
                raise RuntimeError(
                    f"{tool_name} SLURM job {job_id} for {sample_name} ended with state "
                    f"{slurm_state}. Check logs: {stdout_log}, {stderr_log}"
                )

            time.sleep(check_interval)
            elapsed += check_interval
        raise TimeoutError(
            f"{tool_name} SLURM job {job_id} for {sample_name} did not complete within "
            f"{max_wait_seconds // 3600}h. Check logs: {stdout_log}, {stderr_log}"
        )

    def _slurm_job_log_status(self, stdout_log: Path) -> str:
        """Return SUCCESS, FAILED, or UNKNOWN from the generated SLURM wrapper log."""
        try:
            if not stdout_log.exists():
                return "UNKNOWN"
            content = stdout_log.read_text(errors="replace")
            if "Status: SUCCESS" in content:
                return "SUCCESS"
            if "Status: FAILED" in content:
                return "FAILED"
            return "UNKNOWN"
        except Exception as e:
            self.logger.warning(f"Could not inspect SLURM stdout log {stdout_log}: {e}")
            return "UNKNOWN"

    def _slurm_job_log_success(self, stdout_log: Path) -> bool:
        """Return True when the generated SLURM wrapper recorded success."""
        return self._slurm_job_log_status(stdout_log) == "SUCCESS"

    def _slurm_wait_timeout(self, slurm_config: Optional[Dict[str, str]] = None) -> int:
        """Resolve the maximum wait time for blocking SLURM submissions."""
        if slurm_config and slurm_config.get("wait_timeout_seconds") is not None:
            timeout = int(slurm_config["wait_timeout_seconds"])
            if timeout < 1:
                raise ValueError(f"SLURM wait_timeout_seconds must be >= 1, got {timeout}")
            return timeout
        return self.slurm_wait_timeout_seconds

    def _get_slurm_job_state(self, job_id: str) -> str:
        """Get SLURM job state from squeue first, then sacct when available."""
        try:
            output = subprocess.check_output(
                ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                universal_newlines=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if output:
                return output.splitlines()[0].strip().upper()
        except subprocess.CalledProcessError:
            pass

        try:
            output = subprocess.check_output(
                ["sacct", "-n", "-j", str(job_id), "--format=State", "--parsable2"],
                universal_newlines=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            states = [line.split("|")[0].split()[0].strip().upper() for line in output.splitlines() if line.strip()]
            if states:
                failed_states = {
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "PREEMPTED",
                    "BOOT_FAIL",
                    "DEADLINE",
                }
                active_states = {
                    "PENDING",
                    "RUNNING",
                    "CONFIGURING",
                    "COMPLETING",
                    "SUSPENDED",
                }
                failed = [state for state in states if state in failed_states]
                if failed:
                    return failed[0]
                if any(state in active_states for state in states):
                    return "RUNNING"
                if all(state == "COMPLETED" for state in states):
                    return "COMPLETED"
                return states[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return "UNKNOWN"
