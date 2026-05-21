#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Preprocessing Runner

This module coordinates raw sequencing read preprocessing steps. It handles initial quality control (QC),
executes adapter and quality trimming using supported tools, calculates trimming statistics,
and performs post-trimming quality checks.

Features:
    - Raw read quality control using FastQC
    - Read trimming using Trim Galore, Flexbar, or Trimmomatic
    - Trimming statistics collection and summary reporting
    - Post-trimming quality checks for trimming validation
    - Interactive user validation when re-running or resuming preprocessing steps

Configuration:
    - Configured via Pipeline context properties (skip_quality, quality_tool, quality_trim, skip_trim, trimming_tool, threads, local_jobs, slurm, force).

Dependencies:
    - Python packages: pathlib
    - External tools: FastQC, Trim Galore, Flexbar, Trimmomatic

Classes / Functions / Exceptions:
    - PreProcessingRunner: Coordinates preprocessing steps including raw quality control, trimming, and post-trimming quality control.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

from ...modules.quality import create_quality_control, get_available_quality_tools
from ...modules.trimming import create_trimmer, get_available_trimmers


class PreProcessingRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_quality_control(self) -> bool:
        """
        Run quality control using modular implementation.

        Returns:
            True if successful, False otherwise
        """
        if self.ctx.skip_quality:
            self.ctx.logger.info("Quality control skipped by user request")
            return True

        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("quality"):
            quality_metadata = self.ctx.checkpoint_manager.get_stage_metadata("quality")
            completed_tool = quality_metadata.get("tool", "unknown") if quality_metadata else "unknown"

            if completed_tool == self.ctx.quality_tool:
                # Ask user if they want to re-run or skip
                user_choice = self.ctx._ask_user_rerun_stage("quality", completed_tool)

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    # Load quality results from checkpoint
                    if quality_metadata and "output_directories" in quality_metadata:
                        self.ctx.quality_results = quality_metadata["output_directories"]
                        # Validate results exist
                        if self.ctx.validate_stage_results("quality", self.ctx.quality_results):
                            self.ctx.logger.info(
                                "Quality control already completed, loaded and validated results from checkpoint"
                            )
                        else:
                            self.ctx.logger.warning("Quality control results missing, will re-run stage")
                            self.ctx.checkpoint_manager.mark_stage_incomplete("quality")
                            self.ctx.quality_results = None
                    else:
                        self.ctx.logger.info("Quality control already completed, skipping")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run quality control stage")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("quality")
                    self.ctx.quality_results = None
            else:
                # Tool mismatch - inform user and ask
                self.ctx.logger.warning(
                    f"Quality control stage completed with {completed_tool}, but {self.ctx.quality_tool} requested."
                )
                user_choice = self.ctx._ask_user_rerun_stage("quality", f"{completed_tool} (different tool)")

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    self.ctx.logger.warning("User chose to skip despite tool mismatch. This may cause issues.")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run quality control stage with new tool")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("quality")
                    self.ctx.quality_results = None

        self.ctx.logger.info(f"Starting quality control with {self.ctx.quality_tool.upper()}")

        try:
            # Validate quality tool
            available_quality_tools = get_available_quality_tools()
            if self.ctx.quality_tool not in available_quality_tools:
                self.ctx.logger.error(
                    f"Unknown quality tool: {self.ctx.quality_tool}. Available: {', '.join(available_quality_tools)}"
                )
                return False

            quality_threads, quality_slurm_config = self.ctx._get_sample_parallel_resources(
                self.ctx.quality_tool, len(self.ctx.sample_dict)
            )

            # Create quality control module
            quality_module = create_quality_control(
                tool_name=self.ctx.quality_tool,
                sample_dict=self.ctx.sample_dict,
                out_dir=str(self.ctx.output_dir),
                param_dir=self.ctx.param_dir,
                paired=self.ctx._detect_paired_end_data(),  # Auto-detect paired-end
                slurm=self.ctx.slurm,
                dryrun=self.ctx.dryrun,
                cpu_threads=quality_threads,
                logger=self.ctx.logger.logger,
                dry_run_manager=self.ctx.dry_run_manager,
                slurm_config=quality_slurm_config,
                local_jobs=int(quality_slurm_config.get("local_jobs", 1)),
                force=self.ctx.force,
            )

            # Run quality control
            try:
                results = quality_module.run()
                self.ctx.logger.info(f"{self.ctx.quality_tool.upper()} quality control completed successfully")
                self.ctx.logger.info(f"{self.ctx.quality_tool.upper()} results: {len(results)} samples processed")

                # Store quality results for potential resume
                self.ctx.quality_results = results

                # Record executed commands from the module
                if hasattr(quality_module, "command_executor") and hasattr(
                    quality_module.command_executor, "executed_commands"
                ):
                    self.ctx.dry_run_manager.record_executed_commands(quality_module.command_executor.executed_commands)

                # Mark quality control as complete and save results in checkpoint
                self.ctx.checkpoint_manager.mark_stage_complete(
                    "quality",
                    metadata={
                        "tool": self.ctx.quality_tool,
                        "output_directories": results,
                    },
                    dry_run=self.ctx.dryrun,
                )
                return True
            except Exception as e:
                self.ctx.logger.error(f"{self.ctx.quality_tool.upper()} failed: {e}")
                return False

        except Exception as e:
            self.ctx.logger.error(f"{self.ctx.quality_tool.upper()} failed: {e}")
            return False

    def run_quality_control_trim(self) -> bool:
        """
        Run quality control on trimmed reads using modular implementation.

        Returns:
            True if successful, False otherwise
        """
        if not self.ctx.quality_trim:
            self.ctx.logger.info("Post-trimming quality control not requested, skipping")
            return True

        # Check if trimming was done
        if self.ctx.skip_trim:
            self.ctx.logger.warning("Post-trimming quality control requested but trimming was skipped")
            return True

        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("quality_trim"):
            # Load post-trimming quality control results from checkpoint
            quality_trim_metadata = self.ctx.checkpoint_manager.get_stage_metadata("quality_trim")
            if quality_trim_metadata and "output_directories" in quality_trim_metadata:
                self.ctx.quality_trim_results = quality_trim_metadata["output_directories"]
                # Validate results exist
                if self.ctx.validate_stage_results("quality_trim", self.ctx.quality_trim_results):
                    self.ctx.logger.info(
                        "Post-trimming quality control already completed, loaded and validated results from checkpoint"
                    )
                else:
                    self.ctx.logger.warning("Post-trimming quality control results missing, will re-run stage")
                    self.ctx.checkpoint_manager.mark_stage_incomplete("quality_trim")
                    self.ctx.quality_trim_results = None
                    # Don't recursively call - let the pipeline continue normally
            else:
                self.ctx.logger.info("Post-trimming quality control already completed, skipping")
            return True

        self.ctx.logger.info(f"Starting {self.ctx.quality_tool.upper()} quality control on trimmed reads")

        try:
            if not self.ctx.trimming_results:
                trim_meta = self.ctx.checkpoint_manager.get_stage_metadata("trimming")
                if trim_meta and "output_files" in trim_meta:
                    self.ctx.trimming_results = trim_meta["output_files"]

            if not self.ctx.trimming_results:
                self.ctx.logger.error("Trimmed read files not found. Cannot run post-trimming quality control.")
                return False

            quality_threads, quality_slurm_config = self.ctx._get_sample_parallel_resources(
                f"{self.ctx.quality_tool}_trim", len(self.ctx.trimming_results)
            )

            # Create quality control module for trimmed reads
            quality_module_trim = create_quality_control(
                tool_name=self.ctx.quality_tool,
                sample_dict=self.ctx.trimming_results,
                out_dir=str(self.ctx.output_dir),
                param_dir=self.ctx.param_dir,
                paired=self.ctx._detect_paired_end_data(),  # Auto-detect paired-end
                slurm=self.ctx.slurm,
                dryrun=self.ctx.dryrun,
                cpu_threads=quality_threads,
                logger=self.ctx.logger.logger,
                dry_run_manager=self.ctx.dry_run_manager,
                slurm_config=quality_slurm_config,
                local_jobs=int(quality_slurm_config.get("local_jobs", 1)),
                force=self.ctx.force,
            )

            # Set a custom name for post-trimming quality control to distinguish it in dry run
            quality_module_trim.name = f"{self.ctx.quality_tool}_trim"

            # Run FastQC on trimmed reads
            try:
                results = quality_module_trim.run()
                self.ctx.logger.info(f"Post-trimming {self.ctx.quality_tool.upper()} quality control completed successfully")
                self.ctx.logger.info(
                    f"Post-trimming {self.ctx.quality_tool.upper()} results: {len(results)} samples processed"
                )

                # Store post-trimming quality control results
                self.ctx.quality_trim_results = results

                # Record executed commands from the module
                if hasattr(quality_module_trim, "command_executor") and hasattr(
                    quality_module_trim.command_executor, "executed_commands"
                ):
                    self.ctx.dry_run_manager.record_executed_commands(quality_module_trim.command_executor.executed_commands)

                # Mark post-trimming quality control as complete and save results in checkpoint
                self.ctx.checkpoint_manager.mark_stage_complete(
                    "quality_trim",
                    metadata={
                        "tool": self.ctx.quality_tool,
                        "output_directories": results,
                    },
                    dry_run=self.ctx.dryrun,
                )
                return True
            except Exception as e:
                self.ctx.logger.error(f"Post-trimming FastQC failed: {e}")
                return False

        except Exception as e:
            self.ctx.logger.error(f"Post-trimming FastQC failed: {e}")
            return False

    def run_trimming(self) -> bool:
        """
        Run read trimming using the selected trimming tool.

        Returns:
            True if successful, False otherwise
        """
        if self.ctx.skip_trim:
            self.ctx.logger.info("Trimming skipped by user request")
            return True

        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("trimming"):
            stage_metadata = self.ctx.checkpoint_manager.get_stage_metadata("trimming")
            completed_tool = stage_metadata.get("tool", "unknown") if stage_metadata else "unknown"

            if completed_tool == self.ctx.trimming_tool:
                # Load trimming results from checkpoint
                if stage_metadata and "output_files" in stage_metadata:
                    self.ctx.trimming_results = stage_metadata["output_files"]
                    # Validate results exist
                    if self.ctx.validate_stage_results("trimming", self.ctx.trimming_results):
                        self.ctx.logger.info(
                            f"Trimming already completed with {self.ctx.trimming_tool}, loaded and validated results from checkpoint"
                        )
                    else:
                        self.ctx.logger.warning("Trimming results missing, will re-run stage")
                        self.ctx.checkpoint_manager.mark_stage_incomplete("trimming")
                        self.ctx.trimming_results = None
                        # Don't recursively call - let the pipeline continue normally
                else:
                    self.ctx.logger.info(f"Trimming already completed with {self.ctx.trimming_tool}, skipping")
                return True
            else:
                self.ctx.logger.warning(
                    f"Trimming stage completed with {completed_tool}, but {self.ctx.trimming_tool} requested. Will re-run with {self.ctx.trimming_tool}"
                )
                # Mark the stage as incomplete so we can re-run with the new tool
                self.ctx.checkpoint_manager.mark_stage_incomplete("trimming")

        self.ctx.logger.info(f"Starting read trimming with {self.ctx.trimming_tool}")

        try:
            # Validate trimming tool
            available_trimmers = get_available_trimmers()
            if self.ctx.trimming_tool not in available_trimmers:
                self.ctx.logger.error(
                    f"Unknown trimming tool: {self.ctx.trimming_tool}. Available: {', '.join(available_trimmers)}"
                )
                return False

            trimming_threads, trimming_slurm_config = self.ctx._get_sample_parallel_resources(
                self.ctx.trimming_tool, len(self.ctx.sample_dict)
            )

            # Create the trimming module
            trimming_module = create_trimmer(
                trimmer_name=self.ctx.trimming_tool,
                sample_dict=self.ctx.sample_dict,
                out_dir=str(self.ctx.output_dir),
                param_dir=self.ctx.param_dir,
                paired=self.ctx._detect_paired_end_data(),  # Auto-detect paired-end
                slurm=self.ctx.slurm,
                dryrun=self.ctx.dryrun,
                cpu_threads=trimming_threads,
                logger=self.ctx.logger.logger,
                dry_run_manager=self.ctx.dry_run_manager,
                slurm_config=trimming_slurm_config,
                local_jobs=int(trimming_slurm_config.get("local_jobs", 1)),
                force=self.ctx.force,  # Pass force for supported trimmers
            )

            # Run the trimming
            try:
                results = trimming_module.run()
                self.ctx.logger.info(f"{self.ctx.trimming_tool} trimming completed successfully")
                self.ctx.logger.info(f"{self.ctx.trimming_tool} results: {len(results)} samples processed")

                # Store trimming results for use in alignment
                self.ctx.trimming_results = results

                # Capture SLURM job ID for dependency chaining
                if self.ctx.slurm and hasattr(trimming_module, "job_id") and trimming_module.job_id:
                    self.ctx._last_slurm_job_id = trimming_module.job_id

                # Record executed commands from the module
                if hasattr(trimming_module, "command_executor") and hasattr(
                    trimming_module.command_executor, "executed_commands"
                ):
                    self.ctx.dry_run_manager.record_executed_commands(trimming_module.command_executor.executed_commands)

                if self.ctx.slurm and not self.ctx.dryrun:
                    self.ctx.logger.info(
                        "Skipping trimming statistics during SLURM submission; "
                        "trimmed FASTQ files will exist after submitted trimming jobs finish."
                    )
                else:
                    # Calculate trimming statistics only when files are available
                    self.ctx.logger.info("Calculating trimming statistics")
                    try:
                        from pyseqrna.modules.trimming import TrimmingStats

                        # Create trimming statistics directory
                        trimming_stats_dir = Path(self.ctx.output_dir) / "1.Quality_and_trimming" / "trimming_stats"

                        # Initialize trimming statistics module
                        trim_stats_module = TrimmingStats(
                            samples_dict=self.ctx.sample_dict,
                            trimmed_dict=self.ctx.trimming_results,
                            out_dir=str(trimming_stats_dir),
                            paired=self.ctx._detect_paired_end_data(),  # Auto-detect paired-end
                            cpu_threads=self.ctx.threads,
                            logger=self.ctx.logger.logger,
                            dryrun=self.ctx.dryrun,
                            dry_run_manager=self.ctx.dry_run_manager,
                        )

                        # Run trimming statistics
                        trimming_stats_result = trim_stats_module.run()

                        # Record executed commands from statistics module
                        if hasattr(trim_stats_module, "command_executor") and hasattr(
                            trim_stats_module.command_executor, "executed_commands"
                        ):
                            self.ctx.dry_run_manager.record_executed_commands(
                                trim_stats_module.command_executor.executed_commands
                            )

                        # Store trimming statistics results
                        self.ctx.trimming_stats_results = trimming_stats_result

                        # Generate summary
                        summary = trim_stats_module.summarize_results(trimming_stats_result)
                        self.ctx.logger.info("Trimming statistics summary:")
                        for line in summary.split("\n"):
                            if line.strip():
                                self.ctx.logger.info(line)

                        self.ctx.logger.info("Trimming statistics completed successfully")

                    except Exception as e:
                        self.ctx.logger.error(f"Trimming statistics failed: {e}")
                        # Don't fail the whole pipeline if stats fail

                # Mark trimming as complete and save results in checkpoint
                self.ctx.checkpoint_manager.mark_stage_complete(
                    "trimming",
                    metadata={"tool": self.ctx.trimming_tool, "output_files": results},
                    dry_run=self.ctx.dryrun,
                )
                return True
            except Exception as e:
                self.ctx.logger.error(f"{self.ctx.trimming_tool} trimming failed: {e}")
                return False

        except Exception as e:
            self.ctx.logger.error(f"{self.ctx.trimming_tool} trimming failed: {e}")
            return False
