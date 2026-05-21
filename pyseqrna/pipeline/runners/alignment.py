#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Alignment Runner

This module coordinates sequence alignment steps as part of the pipeline execution.
It handles indexing the reference genome, executing read alignment, sorting and indexing
resulting BAM files using samtools, and calculating alignment statistics.

Features:
    - Reference genome index building for supported aligners (STAR, HISAT2, Bowtie2)
    - Local or SLURM cluster-based read alignment execution
    - Dynamic resource allocation for parallel sample processing
    - BAM post-processing including coordinate sorting and indexing using samtools
    - Alignment statistics computation and reporting

Configuration:
    - Configured via Pipeline context properties (alignment_tool, reference_genome, skip_alignment, alignment_stats, alignment_stats_source, threads, memory, slurm).

Dependencies:
    - Python packages: shlex, pathlib
    - External tools: samtools, STAR, HISAT2, Bowtie2

Classes / Functions / Exceptions:
    - AlignmentRunner: Coordinates the alignment, BAM preparation, and alignment statistics stages of the pipeline.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

import shlex

from typing import Optional, Any

from ...modules.alignment import create_aligner, get_available_aligners


class AlignmentRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_alignment(self) -> bool:
        """
        Run read alignment using the selected alignment tool.

        Returns:
            True if successful, False otherwise
        """
        if self.ctx.skip_alignment:
            self.ctx.logger.info("Alignment skipped by user request")
            return True

        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("alignment"):
            stage_metadata = self.ctx.checkpoint_manager.get_stage_metadata("alignment")
            completed_tool = stage_metadata.get("tool", "unknown") if stage_metadata else "unknown"

            if completed_tool == self.ctx.alignment_tool:
                # Load alignment results from checkpoint
                if stage_metadata and "output_files" in stage_metadata:
                    self.ctx.alignment_results = stage_metadata["output_files"]
                    # Validate results exist
                    if self.ctx.validate_stage_results("alignment", self.ctx.alignment_results):
                        self.ctx.logger.info(
                            f"Alignment already completed with {self.ctx.alignment_tool}, loaded and validated results from checkpoint"
                        )
                    else:
                        self.ctx.logger.warning("Alignment results missing, will re-run stage")
                        self.ctx.checkpoint_manager.mark_stage_incomplete("alignment")
                        self.ctx.alignment_results = None
                        # Don't recursively call - let the pipeline continue normally
                else:
                    self.ctx.logger.info(f"Alignment already completed with {self.ctx.alignment_tool}, skipping")
                return True
            else:
                self.ctx.logger.warning(
                    f"Alignment stage completed with {completed_tool}, but {self.ctx.alignment_tool} requested. Will re-run with {self.ctx.alignment_tool}"
                )
                # Mark the stage as incomplete so we can re-run with the new tool
                self.ctx.checkpoint_manager.mark_stage_incomplete("alignment")

        self.ctx.logger.info(f"Starting read alignment with {self.ctx.alignment_tool}")

        # Load trimming results from checkpoint if not already loaded
        if self.ctx.trimming_results is None:
            trimming_metadata = self.ctx.checkpoint_manager.get_stage_metadata("trimming")
            if trimming_metadata and "output_files" in trimming_metadata:
                self.ctx.trimming_results = trimming_metadata["output_files"]
                self.ctx.logger.info("Loaded trimming results from checkpoint")

        try:
            # Create alignment directory structure
            alignment_dir = Path(self.ctx.output_dir) / "2.Alignment"
            if not self.ctx.dryrun:
                alignment_dir.mkdir(parents=True, exist_ok=True)
                self.ctx.logger.info(f"Created alignment directory: {alignment_dir}")
            else:
                self.ctx.logger.info(f"Would create alignment directory: {alignment_dir}")

            # Validate alignment tool
            available_aligners = get_available_aligners()
            if self.ctx.alignment_tool not in available_aligners:
                self.ctx.logger.error(
                    f"Unknown alignment tool: {self.ctx.alignment_tool}. Available: {', '.join(available_aligners)}"
                )
                return False

            # Create the alignment module
            aligner = create_aligner(
                aligner_name=self.ctx.alignment_tool,
                genome=self.ctx.reference_genome,
                out_dir=str(self.ctx.output_dir),
                param_dir=self.ctx.param_dir,
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                cpu_threads=self.ctx.threads,
                slurm=self.ctx.slurm,
                dep=self.ctx._last_slurm_job_id,
                dry_run_manager=self.ctx.dry_run_manager,
                slurm_config=self.ctx._get_slurm_config(),
            )

            # Keep index checks quiet inside aligners and log the user-facing
            # message here based on whether this is a resume/reuse path.
            index_exists = aligner.check_index()
            if index_exists:
                if self.ctx.resume != "all":
                    self.ctx.logger.info(f"Using existing {self.ctx.alignment_tool} index in {aligner.index_dir}")
                else:
                    self.ctx.logger.info(f"Found existing {self.ctx.alignment_tool} index in {aligner.index_dir}")
            else:
                self.ctx.logger.info(f"Building {self.ctx.alignment_tool} index")
                try:
                    aligner.build_index(gff=self.ctx.feature_file)
                except Exception as e:
                    self.ctx.logger.error(f"Failed to build {self.ctx.alignment_tool} index: {e}")
                    return False

            # In dry-run mode, skip the second index check since files weren't actually created
            if not self.ctx.dryrun:
                # Double-check index exists after building (only in non-dry-run mode)
                if not aligner.check_index():
                    self.ctx.logger.error(f"{self.ctx.alignment_tool} index still not found after building")
                    return False

            # Run alignment
            try:
                # Auto-detect paired-end from sample data
                paired = self.ctx._detect_paired_end_data()

                # Determine which files to use for alignment
                alignment_target = {}

                if self.ctx.skip_trim or self.ctx.trimming_results is None:
                    # Use original input files when trimming was skipped or not performed
                    self.ctx.logger.info("Using original input files for alignment")
                    for sample_name, sample_info in self.ctx.sample_dict.items():
                        if paired and len(sample_info) >= 4:
                            alignment_target[sample_name] = [
                                sample_info[2],
                                sample_info[3],
                            ]
                        else:
                            alignment_target[sample_name] = [sample_info[2]]
                else:
                    # Use trimmed files when trimming was performed
                    self.ctx.logger.info("Using trimmed files for alignment")
                    for sample_name, sample_info in self.ctx.sample_dict.items():
                        if sample_name in self.ctx.trimming_results:
                            # Get the trimmed file path from trimming results
                            trimmed_file = self.ctx.trimming_results[sample_name]
                            if paired and isinstance(trimmed_file, (list, tuple)) and len(trimmed_file) >= 2:
                                file_path = [trimmed_file[0], trimmed_file[1]]
                            elif isinstance(trimmed_file, str):
                                file_path = [trimmed_file]
                            else:
                                # Fallback to original file if trimmed file not found
                                self.ctx.logger.warning(f"Trimmed file not found for {sample_name}, using original file")
                                file_path = (
                                    [sample_info[2], sample_info[3]] if paired and len(sample_info) >= 4 else [sample_info[2]]
                                )
                        else:
                            # Fallback to original file if sample not in trimming results
                            self.ctx.logger.warning(f"Sample {sample_name} not found in trimming results, using original file")
                            file_path = (
                                [sample_info[2], sample_info[3]] if paired and len(sample_info) >= 4 else [sample_info[2]]
                            )
                        alignment_target[sample_name] = file_path

                alignment_threads, alignment_slurm_config = self.ctx._get_sample_parallel_resources(
                    f"{self.ctx.alignment_tool}_alignment", len(alignment_target)
                )
                aligner.cpu_threads = alignment_threads
                aligner.slurm_config = alignment_slurm_config
                aligner.local_jobs = int(alignment_slurm_config.get("local_jobs", 1))

                results = aligner.run_alignment(target=alignment_target, paired=paired)
                self.ctx.logger.info(f"{self.ctx.alignment_tool} alignment completed successfully")
                self.ctx.logger.info(f"{self.ctx.alignment_tool} results: {len(results)} samples processed")

                # Store alignment results for potential resume
                self.ctx.alignment_results = results

                # Capture SLURM job IDs for dependency chaining
                if self.ctx.slurm and results:
                    job_ids = [
                        info.get("job_id", "") for info in results.values() if isinstance(info, dict) and info.get("job_id")
                    ]
                    if job_ids:
                        self.ctx._last_slurm_job_id = ",".join(job_ids)

                # Record executed commands from the module
                if hasattr(aligner, "command_executor") and hasattr(aligner.command_executor, "executed_commands"):
                    self.ctx.dry_run_manager.record_executed_commands(aligner.command_executor.executed_commands)

                # Mark alignment as complete and save results in checkpoint
                self.ctx.checkpoint_manager.mark_stage_complete(
                    "alignment",
                    metadata={
                        "tool": self.ctx.alignment_tool,
                        "output_files": results,
                        "input_files": alignment_target,
                    },
                    dry_run=self.ctx.dryrun,
                )
                return True
            except Exception as e:
                self.ctx.logger.error(f"{self.ctx.alignment_tool} alignment failed: {e}")
                return False

        except Exception as e:
            self.ctx.logger.error(f"{self.ctx.alignment_tool} alignment failed: {e}")
            return False

    def _extract_bam_path(self, sample_info: Any) -> Optional[str]:
        """Extract a BAM path from a string, list, or result dictionary."""
        if isinstance(sample_info, dict):
            return sample_info.get("bam")
        if isinstance(sample_info, str):
            return sample_info
        if isinstance(sample_info, (list, tuple)):
            for value in reversed(sample_info):
                if isinstance(value, str) and value.endswith(".bam"):
                    return value
        return None

    def _load_alignment_results_from_checkpoint(self) -> bool:
        """Load alignment outputs from checkpoint into pipeline state."""
        if self.ctx.alignment_results:
            return True
        if self.ctx.checkpoint_manager and self.ctx.checkpoint_manager.is_stage_complete("alignment"):
            alignment_metadata = self.ctx.checkpoint_manager.get_stage_metadata("alignment")
            if alignment_metadata and "output_files" in alignment_metadata:
                self.ctx.alignment_results = alignment_metadata["output_files"]
                self.ctx.logger.info("Loaded alignment results from checkpoint")
                return True
        return False

    def _load_prepared_bams_from_checkpoint(self) -> bool:
        """Load prepared BAM outputs from checkpoint into pipeline state."""
        if self.ctx.prepared_bam_results:
            return True
        if self.ctx.checkpoint_manager and self.ctx.checkpoint_manager.is_stage_complete("bam_preparation"):
            bam_metadata = self.ctx.checkpoint_manager.get_stage_metadata("bam_preparation")
            if bam_metadata and "output_files" in bam_metadata:
                prepared = bam_metadata["output_files"]
                if not self.ctx.dryrun:
                    for sample_name, sample_info in prepared.items():
                        bam_path = self._extract_bam_path(sample_info)
                        if not bam_path or not self._prepared_bam_is_current(Path(bam_path), Path(str(bam_path) + ".bai")):
                            self.ctx.logger.warning(
                                f"Prepared BAM checkpoint is stale or incomplete for {sample_name}; BAM preparation will be rerun"
                            )
                            self.ctx.checkpoint_manager.mark_stage_incomplete("bam_preparation")
                            return False
                self.ctx.prepared_bam_results = prepared
                self.ctx.alignment_results = self.ctx.prepared_bam_results
                self.ctx.logger.info("Loaded prepared BAMs from checkpoint")
                return True
        return False

    def _prepared_bam_is_current(self, bam_file: Path, bai_file: Path) -> bool:
        """Return True when a BAM and its index exist and the index is current."""
        return bam_file.exists() and bai_file.exists() and bai_file.stat().st_mtime >= bam_file.stat().st_mtime

    def _build_bam_preparation_command(
        self,
        sample_id: str,
        input_bam: Path,
        prepared_bam: Path,
        marker_file: Path,
        threads: int,
        memory_gb: int,
    ) -> str:
        """Build a portable shell command that coordinate-sorts and indexes one BAM."""
        sort_memory_gb = max(1, int(memory_gb or 4) // max(1, int(threads or 1) + 1))
        quoted_input = shlex.quote(str(input_bam))
        quoted_output = shlex.quote(str(prepared_bam))
        quoted_marker = shlex.quote(str(marker_file))
        quoted_outdir = shlex.quote(str(prepared_bam.parent))
        return (
            "set -eu\n"
            f"mkdir -p {quoted_outdir}\n"
            f"INPUT_BAM={quoted_input}\n"
            f"OUTPUT_BAM={quoted_output}\n"
            f"MARKER_FILE={quoted_marker}\n"
            "if samtools view -H \"$INPUT_BAM\" | grep -q 'SO:coordinate'; then\n"
            '  PREPARED_BAM="$INPUT_BAM"\n'
            "else\n"
            f'  samtools sort -@ {int(threads)} -m {sort_memory_gb}G -o "$OUTPUT_BAM" "$INPUT_BAM"\n'
            '  PREPARED_BAM="$OUTPUT_BAM"\n'
            "fi\n"
            'if [ ! -s "${PREPARED_BAM}.bai" ] || [ "${PREPARED_BAM}.bai" -ot "$PREPARED_BAM" ]; then\n'
            f'  samtools index -@ {int(threads)} "$PREPARED_BAM"\n'
            "fi\n"
            'printf \'%s\\n\' "$PREPARED_BAM" > "$MARKER_FILE"\n'
        )

    def run_bam_preparation(self) -> bool:
        """
        Prepare alignment BAMs once for downstream reuse.

        Coordinate-sorted and indexed BAMs are reused by alignment statistics,
        quantification, and multimapped-group analysis.
        """
        if self.ctx.skip_alignment:
            self.ctx.logger.info("Skipping BAM preparation because alignment is skipped")
            return True

        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("bam_preparation"):
            if self._load_prepared_bams_from_checkpoint():
                return True
            self.ctx.checkpoint_manager.mark_stage_incomplete("bam_preparation")

        if not self._load_alignment_results_from_checkpoint():
            self.ctx.logger.error("Alignment must be completed before BAM preparation")
            return False

        bam_inputs = {}
        for sample_name, sample_info in self.ctx.alignment_results.items():
            bam_path = self._extract_bam_path(sample_info)
            if bam_path:
                bam_inputs[sample_name] = Path(bam_path)

        if not bam_inputs:
            self.ctx.logger.error("No BAM files found for BAM preparation")
            return False

        prep_dir = Path(self.ctx.output_dir) / "2.Alignment" / "bam_preparation"
        if not self.ctx.dryrun:
            prep_dir.mkdir(parents=True, exist_ok=True)

        task_threads, slurm_config = self.ctx._get_sample_parallel_resources("bam_preparation", len(bam_inputs))
        memory_gb = int(slurm_config.get("memory", self.ctx.memory or 16)) if self.ctx.slurm else int(self.ctx.memory or 16)
        commands = {}
        marker_files = {}
        prepared_results = {}

        for sample_name, input_bam in bam_inputs.items():
            prepared_bam = prep_dir / f"{sample_name}_sorted.bam"
            marker_file = prep_dir / f"{sample_name}.prepared_bam.path"
            marker_files[sample_name] = marker_file

            if not self.ctx.dryrun and not input_bam.exists():
                self.ctx.logger.error(f"BAM file not found for {sample_name}: {input_bam}")
                return False

            # Check if marker file exists and the BAM it references is current
            reused = False
            if not self.ctx.dryrun and marker_file.exists():
                try:
                    reused_path = Path(marker_file.read_text().strip())
                    if self._prepared_bam_is_current(reused_path, Path(str(reused_path) + ".bai")):
                        self.ctx.logger.info(f"Using existing prepared BAM for {sample_name}: {reused_path}")
                        reused = True
                except Exception:
                    pass

            if reused:
                continue

            if not self.ctx.dryrun and self._prepared_bam_is_current(prepared_bam, Path(str(prepared_bam) + ".bai")):
                marker_file.write_text(str(prepared_bam) + "\n")
                self.ctx.logger.info(f"Using existing prepared BAM for {sample_name}: {prepared_bam}")
            else:
                commands[sample_name] = self._build_bam_preparation_command(
                    sample_id=sample_name,
                    input_bam=input_bam,
                    prepared_bam=prepared_bam,
                    marker_file=marker_file,
                    threads=max(1, int(task_threads or 1)),
                    memory_gb=max(1, int(memory_gb or 16)),
                )

        try:
            if self.ctx.dryrun:
                self.ctx.dry_run_manager.simulate_command_execution(
                    stage_name="bam_preparation",
                    commands=commands or {sample: "reuse existing prepared BAM" for sample in bam_inputs},
                    execution_type="slurm" if self.ctx.slurm else "local",
                )
                for sample_name, input_bam in bam_inputs.items():
                    prepared_results[sample_name] = {
                        "bam": str(prep_dir / f"{sample_name}_sorted.bam"),
                        "original_bam": str(input_bam),
                        "bai": str(prep_dir / f"{sample_name}_sorted.bam.bai"),
                    }
            elif commands:
                self.ctx.logger.info(f"Preparing {len(commands)} BAM file(s) for downstream reuse")
                if self.ctx.slurm:
                    self.ctx.command_executor.execute_slurm(
                        commands=commands,
                        tool_name="bam_preparation",
                        job_name="bam_preparation",
                        outdir=str(prep_dir),
                        dependency=self.ctx._last_slurm_job_id,
                        slurm_config=slurm_config,
                    )
                else:
                    local_results = self.ctx.command_executor.execute_local(
                        commands=commands,
                        tool_name="bam_preparation",
                        outdir=str(prep_dir),
                        max_workers=int(slurm_config.get("local_jobs", 1)),
                    )
                    failed = [sample for sample, ok in local_results.items() if not ok]
                    if failed:
                        self.ctx.logger.error(f"BAM preparation failed for sample(s): {', '.join(failed)}")
                        return False

            if not self.ctx.dryrun:
                for sample_name, input_bam in bam_inputs.items():
                    marker_file = marker_files[sample_name]
                    prepared_path = Path(marker_file.read_text().strip()) if marker_file.exists() else input_bam
                    bai_path = Path(str(prepared_path) + ".bai")
                    if not self._prepared_bam_is_current(prepared_path, bai_path):
                        self.ctx.logger.error(f"Prepared BAM or index is missing for {sample_name}: {prepared_path}")
                        return False
                    prepared_results[sample_name] = {
                        "bam": str(prepared_path),
                        "original_bam": str(input_bam),
                        "bai": str(bai_path),
                    }

            self.ctx.prepared_bam_results = prepared_results
            self.ctx.alignment_results = prepared_results
            self.ctx.checkpoint_manager.mark_stage_complete(
                "bam_preparation",
                metadata={
                    "tool": "samtools",
                    "output_files": prepared_results,
                    "samples_processed": len(prepared_results),
                },
                dry_run=self.ctx.dryrun,
            )
            self.ctx.logger.info("BAM preparation completed successfully")
            return True

        except Exception as e:
            self.ctx.logger.error(f"BAM preparation failed: {e}")
            return False

    def run_alignment_statistics(self) -> bool:
        """Run alignment statistics using logs first when configured."""
        if not self.ctx.alignment_stats:
            self.ctx.logger.info("Skipping alignment statistics as requested")
            return True

        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("alignment_stats"):
            self.ctx.logger.info("Alignment statistics already completed, skipping")
            return True

        if not self._load_prepared_bams_from_checkpoint() and not self._load_alignment_results_from_checkpoint():
            self.ctx.logger.error("Alignment or BAM preparation must be completed before alignment statistics")
            return False

        self.ctx.logger.info(f"Calculating alignment statistics (source={self.ctx.alignment_stats_source})")
        try:
            from pyseqrna.modules.alignment import AlignmentStats

            alignment_stats_dir = Path(self.ctx.output_dir) / "2.Alignment" / "alignment_stats"
            stats_module = AlignmentStats(
                sample_dict=self.ctx.sample_dict,
                trimmed_dict=self.ctx.trimming_results,
                bam_dict=self.ctx.alignment_results,
                trimming_stats=self.ctx.trimming_stats_results,
                out_dir=str(alignment_stats_dir),
                cpu_threads=self.ctx.threads,
                paired=self.ctx._detect_paired_end_data(),
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
                source=self.ctx.alignment_stats_source,
                alignment_tool=self.ctx.alignment_tool,
            )

            self.ctx.alignment_stats_results = stats_module.run()
            summary = stats_module.summarize_results(self.ctx.alignment_stats_results)
            self.ctx.logger.info("Alignment statistics summary:")
            for line in summary.split("\n"):
                if line.strip():
                    self.ctx.logger.info(line)

            self.ctx.checkpoint_manager.mark_stage_complete(
                "alignment_stats",
                metadata={
                    "tool": "alignment_stats",
                    "source": stats_module.resolved_source,
                    "output_dir": str(alignment_stats_dir),
                },
                dry_run=self.ctx.dryrun,
            )
            self.ctx.logger.info("Alignment statistics completed successfully")
            return True

        except Exception as e:
            self.ctx.logger.error(f"Alignment statistics failed: {e}")
            return False
