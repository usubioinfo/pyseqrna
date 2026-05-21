#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Expression Quantification Runner

This module coordinates transcript/gene expression quantification, count normalization,
and multimapped groups (MMG) analyses. It interfaces with quantification engines,
executes normalization algorithms, and manages checkpoint logging.

Features:
    - Gene expression quantification using Genomic Overlaps, FeatureCounts, or HTSeq
    - Expression count normalization using RPKM, TPM, CPM, or TMM
    - Multimapped group (MMG) analysis using aligned BAM files
    - Dynamic verification of sorted BAM files before quantification runs
    - Checkpoint metadata generation including summary statistics and output file logs
    - Diagnostic plot generation during normalization stages

Configuration:
    - Configured via Pipeline context properties (quant_method, skip_quantification, threads, memory, slurm, local_jobs, param_dir, feature_file, normalize_counts, normalization_method, skip_normalization_plots, enable_multimapped_groups, mmg_feature, mmg_min_count, mmg_percent_sample, mmg_min_overlap, mmg_fraction_overlap, mmg_include_ambiguous_unique, mmg_collapse_contained_groups).

Dependencies:
    - Python packages: pathlib, pandas, datetime
    - External tools: featureCounts, HTSeq (optional, depends on chosen method)

Classes / Functions / Exceptions:
    - QuantificationRunner: Coordinates expression quantification, count normalization, and multimapped groups analyses.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

import pandas as pd

from datetime import datetime

from ...modules.quantification import create_quantifier, get_available_quantifiers
from ...modules.normalization import create_normalizer, get_available_normalizers
from ...modules.multimapped_groups import create_multimapped_groups_analyzer


class QuantificationRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_multimapped_groups(self) -> bool:
        """
        Run multimapped groups analysis using aligned BAM files.

        Returns:
            True if successful, False otherwise
        """
        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("multimapped_groups"):
            stage_metadata = self.ctx.checkpoint_manager.get_stage_metadata("multimapped_groups")
            completed_tool = stage_metadata.get("tool", "unknown") if stage_metadata else "unknown"

            # Ask user if they want to re-run or skip
            user_choice = self.ctx._ask_user_rerun_stage("multimapped_groups", completed_tool)

            if user_choice is None:  # User cancelled
                return False
            elif user_choice is False:  # User chose to skip
                self.ctx.logger.info("Multimapped groups analysis already completed, skipping")
                return True
            else:  # User chose to re-run
                self.ctx.logger.info("User chose to re-run multimapped groups analysis")
                # Mark as incomplete to force re-run
                self.ctx.checkpoint_manager.mark_stage_incomplete("multimapped_groups")

        self.ctx.logger.info("Starting multimapped groups analysis")

        # Check dependencies - need alignment results
        if (
            not self.ctx.alignment._load_prepared_bams_from_checkpoint()
            and not self.ctx.alignment._load_alignment_results_from_checkpoint()
        ):
            self.ctx.logger.error("Alignment must be completed before multimapped groups analysis")
            return False

        # Validate annotation file exists
        if not Path(self.ctx.feature_file).exists():
            self.ctx.logger.error(f"Annotation file not found: {self.ctx.feature_file}")
            return False

        try:
            # Create multimapped groups output directory within quantification directory
            quantification_dir = Path(self.ctx.output_dir) / "3.Quantification"
            mmg_dir = quantification_dir / "multimapped_groups"
            self.ctx._clean_and_create_directory(mmg_dir, "multimapped_groups")

            # Extract BAM files from alignment results
            bam_files = {}
            for sample_name, file_info in self.ctx.alignment_results.items():
                if isinstance(file_info, dict):
                    bam_path = file_info.get("bam")
                else:
                    bam_path = file_info

                if isinstance(bam_path, str) and bam_path.endswith(".bam"):
                    bam_files[sample_name] = bam_path

            if not bam_files:
                self.ctx.logger.error("No BAM files found in alignment results")
                return False

            self.ctx.logger.info(f"Found {len(bam_files)} BAM files for multimapped groups analysis")

            # Create multimapped groups analyzer
            mmg_analyzer = create_multimapped_groups_analyzer(
                bam_files=bam_files,
                gff_file=self.ctx.feature_file,
                out_dir=str(mmg_dir),
                feature=self.ctx.mmg_feature,
                min_count=self.ctx.mmg_min_count,
                percent_sample=self.ctx.mmg_percent_sample,
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
                cpu_threads=self.ctx.threads,
                min_overlap=self.ctx.mmg_min_overlap,
                fraction_overlap=self.ctx.mmg_fraction_overlap,
                include_ambiguous_unique=self.ctx.mmg_include_ambiguous_unique,
                collapse_contained_groups=self.ctx.mmg_collapse_contained_groups,
            )

            # Run multimapped groups analysis
            results = mmg_analyzer.run()

            if results:
                results_df = results.get("results")
                raw_mmg_file = None
                if isinstance(results_df, pd.DataFrame) and not results_df.empty and not self.ctx.dryrun:
                    raw_mmg_file = quantification_dir / "Raw_MMGcounts.xlsx"
                    results_df.to_excel(raw_mmg_file, index=False)
                    results.setdefault("output_files", []).append(str(raw_mmg_file))
                    self.ctx.logger.info(f"Saved raw multimapped group counts to: {raw_mmg_file}")

                # Save results to checkpoint
                summary_stats = mmg_analyzer.get_summary_stats(results)

                # Convert NumPy types for JSON serialization
                summary_stats = self.ctx._convert_numpy_for_json(summary_stats)

                self.ctx.checkpoint_manager.mark_stage_complete(
                    "multimapped_groups",
                    metadata={
                        "tool": "multimapped_groups_analyzer",
                        "output_files": results.get("output_files", []),
                        "output_file": str(raw_mmg_file) if raw_mmg_file else None,
                        "summary_stats": summary_stats,
                        "bam_files_processed": len(bam_files),
                    },
                    dry_run=self.ctx.dryrun,
                )

                self.ctx.logger.info("Multimapped groups analysis completed successfully")
                return True
            else:
                self.ctx.logger.error("Multimapped groups analysis failed")
                return False

        except Exception as e:
            self.ctx.logger.error(f"Multimapped groups analysis failed: {e}")
            return False

    def run_quantification(self) -> bool:
        """
        Run gene expression quantification using the selected quantification tool.

        Returns:
            True if successful, False otherwise
        """
        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("quantification"):
            stage_metadata = self.ctx.checkpoint_manager.get_stage_metadata("quantification")
            completed_tool = stage_metadata.get("tool", "unknown") if stage_metadata else "unknown"

            if completed_tool == self.ctx.quant_method:
                # Ask user if they want to re-run or skip
                user_choice = self.ctx._ask_user_rerun_stage("quantification", completed_tool)

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    # Load quantification results from checkpoint
                    if stage_metadata and "output_file" in stage_metadata:
                        self.ctx.quantification_results = stage_metadata["output_file"]
                        self.ctx.logger.info(
                            f"Quantification already completed with {self.ctx.quant_method}, loaded results from checkpoint"
                        )
                    else:
                        self.ctx.logger.info(f"Quantification already completed with {self.ctx.quant_method}, skipping")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run quantification stage")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("quantification")
            else:
                # Tool mismatch - inform user and ask
                self.ctx.logger.warning(
                    f"Quantification stage completed with {completed_tool}, but {self.ctx.quant_method} requested."
                )
                user_choice = self.ctx._ask_user_rerun_stage("quantification", f"{completed_tool} (different tool)")

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    self.ctx.logger.warning("User chose to skip despite tool mismatch. This may cause issues.")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run quantification stage with new tool")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("quantification")

        self.ctx.logger.info(f"Starting gene expression quantification with {self.ctx.quant_method}")

        # Check dependencies - need alignment results
        if (
            not self.ctx.alignment._load_prepared_bams_from_checkpoint()
            and not self.ctx.alignment._load_alignment_results_from_checkpoint()
        ):
            self.ctx.logger.error("Alignment must be completed before quantification")
            return False

        # Validate annotation file exists
        if not Path(self.ctx.feature_file).exists():
            self.ctx.logger.error(f"Annotation file not found: {self.ctx.feature_file}")
            return False

        try:
            # Create quantification directory structure
            quantification_dir = Path(self.ctx.output_dir) / "3.Quantification"
            self.ctx._clean_and_create_directory(quantification_dir, "quantification")

            # Validate quantification method
            available_quantifiers = get_available_quantifiers()
            if self.ctx.quant_method.lower() not in available_quantifiers:
                self.ctx.logger.error(
                    f"Unsupported quantification tool: {self.ctx.quant_method}. Available: {', '.join(available_quantifiers)}"
                )
                return False

            # Create quantification module
            quantification_module = create_quantifier(
                quantifier_name=self.ctx.quant_method,
                bam_dict=self.ctx.alignment_results,
                annotation_file=self.ctx.feature_file,
                out_dir=str(quantification_dir),
                param_dir=self.ctx.param_dir,
                paired=self.ctx._detect_paired_end_data(),  # Auto-detect paired-end
                slurm=self.ctx.slurm,
                dryrun=self.ctx.dryrun,
                job_id=self.ctx._last_slurm_job_id,
                cpu_threads=self.ctx.threads,
                memory=self.ctx.memory,
                logger=self.ctx.logger.logger,
                dry_run_manager=self.ctx.dry_run_manager,
                slurm_config=self.ctx._get_slurm_config(),
                local_jobs=self.ctx.local_jobs,
            )

            # Run quantification
            count_matrix = quantification_module.run()

            # Store quantification results
            self.ctx.quantification_results = count_matrix

            # Record executed commands from the quantification module
            if hasattr(quantification_module, "command_executor") and hasattr(
                quantification_module.command_executor, "executed_commands"
            ):
                self.ctx.dry_run_manager.record_executed_commands(quantification_module.command_executor.executed_commands)

            # Generate summary statistics
            if not count_matrix.empty:
                stats = quantification_module.get_summary_stats(count_matrix)
                self.ctx.logger.info("Quantification completed successfully:")
                self.ctx.logger.info(f"  - Tool: {self.ctx.quant_method}")
                self.ctx.logger.info(f"  - Genes: {stats.get('total_genes', 0)}")
                self.ctx.logger.info(f"  - Samples: {stats.get('total_samples', 0)}")
                self.ctx.logger.info(f"  - Total reads: {stats.get('total_reads', 0):,}")
                self.ctx.logger.info(f"  - Genes with zero counts: {stats.get('genes_with_zero_counts', 0)}")

            # Enhanced metadata collection for quantification
            output_excel = str(quantification_dir / "Raw_Counts.xlsx")

            # Collect detailed file information
            detailed_output_files = {}
            total_size = 0

            if Path(output_excel).exists():
                file_size = Path(output_excel).stat().st_size
                detailed_output_files[output_excel] = {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "exists": True,
                    "modified": datetime.fromtimestamp(Path(output_excel).stat().st_mtime).isoformat(),
                    "type": "count_matrix",
                }
                total_size += file_size

            # Enhanced metadata with comprehensive information
            metadata = {
                "tool": self.ctx.quant_method,
                "output_file": output_excel,
                "detailed_output_files": detailed_output_files,
                "total_output_size_mb": round(total_size / (1024 * 1024), 2),
                "feature_file": self.ctx.feature_file,
                "execution_time": datetime.now().isoformat(),
                "paired_end": self.ctx._detect_paired_end_data(),
                "summary_stats": self.ctx._convert_numpy_for_json(stats if not count_matrix.empty else {}),
            }

            self.ctx.checkpoint_manager.mark_stage_complete("quantification", metadata=metadata, dry_run=self.ctx.dryrun)

            return True

        except Exception as e:
            self.ctx.logger.error(f"Quantification with {self.ctx.quant_method} failed: {e}")
            return False

    def run_normalization(self) -> bool:
        """
        Run count normalization step.

        Returns:
            bool: True if normalization successful, False otherwise
        """
        if not self.ctx.normalize_counts:
            self.ctx.logger.info("Skipping count normalization (not requested)")
            return True

        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("normalization"):
            stage_metadata = self.ctx.checkpoint_manager.get_stage_metadata("normalization")
            completed_method = stage_metadata.get("method", "unknown") if stage_metadata else "unknown"

            if completed_method == self.ctx.normalization_method:
                # Ask user if they want to re-run or skip
                user_choice = self.ctx._ask_user_rerun_stage("normalization", completed_method)

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    self.ctx.logger.info(f"Normalization already completed with {self.ctx.normalization_method}, skipping")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run normalization stage")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("normalization")
            else:
                # Method mismatch - inform user and ask
                self.ctx.logger.warning(
                    f"Normalization stage completed with {completed_method}, but {self.ctx.normalization_method} requested."
                )
                user_choice = self.ctx._ask_user_rerun_stage("normalization", f"{completed_method} (different method)")

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    self.ctx.logger.warning("User chose to skip despite method mismatch. This may cause issues.")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run normalization stage with new method")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("normalization")

        self.ctx.logger.info(f"Starting count normalization with {self.ctx.normalization_method}")

        # Check if quantification results are available
        if self.ctx.quantification_results is None:
            quantification_metadata = self.ctx.checkpoint_manager.get_stage_metadata("quantification")
            if quantification_metadata:
                count_matrix_file = (
                    Path(quantification_metadata["output_file"]) if "output_file" in quantification_metadata else None
                )

                if count_matrix_file and count_matrix_file.exists():
                    self.ctx.logger.info(f"Found count matrix file from checkpoint: {count_matrix_file}")
                else:
                    self.ctx.logger.error("Quantification results not found in checkpoint or file missing")
                    return False
            else:
                self.ctx.logger.error("Quantification results not available. Run quantification first.")
                return False
        else:
            # Use current quantification results
            quantification_dir = Path(self.ctx.output_dir) / "3.Quantification"
            count_matrix_file = quantification_dir / "Raw_Counts.xlsx"
            count_matrix_input = (
                self.ctx.quantification_results
                if isinstance(self.ctx.quantification_results, pd.DataFrame)
                else str(count_matrix_file)
            )

            # If we have a DataFrame but no saved file, save the DataFrame first
            if isinstance(self.ctx.quantification_results, pd.DataFrame) and not count_matrix_file.exists():
                self.ctx.logger.info("Saving quantification DataFrame to Excel file for normalization")
                if not self.ctx.dryrun:
                    count_matrix_file.parent.mkdir(parents=True, exist_ok=True)
                    self.ctx.quantification_results.to_excel(str(count_matrix_file), index=False)
                    self.ctx.logger.info(f"Saved count matrix to: {count_matrix_file}")
                else:
                    self.ctx.logger.info(f"DRYRUN: Would save count matrix to: {count_matrix_file}")

            if not self.ctx.dryrun and not count_matrix_file.exists():
                self.ctx.logger.error(f"Count matrix file not found: {count_matrix_file}")
                return False

        try:
            # Create normalization directory structure
            normalization_dir = Path(self.ctx.output_dir) / "4.Normalization"
            self.ctx._clean_and_create_directory(normalization_dir, "normalization")

            # Validate normalization method
            available_normalizers = get_available_normalizers()
            if self.ctx.normalization_method not in available_normalizers:
                self.ctx.logger.error(
                    f"Unknown normalization method: {self.ctx.normalization_method}. Available: {', '.join(available_normalizers)}"
                )
                return False

            # Create the normalizer
            normalizer = create_normalizer(
                normalizer_name=self.ctx.normalization_method,
                count_matrix_file=count_matrix_input
                if self.ctx.quantification_results is not None
                else str(count_matrix_file),
                annotation_file=self.ctx.feature_file,
                out_dir=str(normalization_dir),
                param_dir=self.ctx.param_dir,
                logger=self.ctx.logger.logger,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
            )

            # Run normalization
            normalized_data = normalizer.run(plot=not self.ctx.skip_normalization_plots, save_results=True)

            # Store normalization results
            self.ctx.normalization_results = normalized_data

            # Enhanced metadata collection for normalization
            output_filename = f"{self.ctx.normalization_method.upper()}_normalized_counts.xlsx"
            output_file_path = normalization_dir / output_filename
            self.ctx.normalization_results = {
                "normalized_counts_file": str(output_file_path),
                "data": normalized_data,
            }

            # Collect detailed file information
            detailed_output_files = {}
            total_size = 0

            if Path(output_file_path).exists():
                file_size = Path(output_file_path).stat().st_size
                detailed_output_files[str(output_file_path)] = {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "exists": True,
                    "modified": datetime.fromtimestamp(Path(output_file_path).stat().st_mtime).isoformat(),
                    "type": "normalized_counts",
                }
                total_size += file_size

            # Check for plot files if plots were created
            if not self.ctx.skip_normalization_plots:
                plot_files = list(normalization_dir.glob("*.png")) + list(normalization_dir.glob("*.pdf"))
                for plot_file in plot_files:
                    if plot_file.exists():
                        plot_size = plot_file.stat().st_size
                        detailed_output_files[str(plot_file)] = {
                            "size_bytes": plot_size,
                            "size_mb": round(plot_size / (1024 * 1024), 2),
                            "exists": True,
                            "modified": datetime.fromtimestamp(plot_file.stat().st_mtime).isoformat(),
                            "type": "plot",
                        }
                        total_size += plot_size

            # Enhanced metadata with comprehensive information
            normalization_metadata = {
                "method": self.ctx.normalization_method,
                "input_file": str(count_matrix_file),
                "output_file": str(output_file_path),
                "detailed_output_files": detailed_output_files,
                "total_output_size_mb": round(total_size / (1024 * 1024), 2),
                "plots_created": not self.ctx.skip_normalization_plots,
                "total_genes": len(normalized_data) if normalized_data is not None else 0,
                "total_samples": len(normalized_data.columns) - 1 if normalized_data is not None else 0,  # Exclude gene column
                "execution_time": datetime.now().isoformat(),
                "summary_stats": {
                    "total_genes": len(normalized_data) if normalized_data is not None else 0,
                    "total_samples": len(normalized_data.columns) - 1 if normalized_data is not None else 0,
                    "plot_files": len([f for f in detailed_output_files.values() if f.get("type") == "plot"]),
                },
            }

            # Mark normalization as complete
            if not self.ctx.dryrun:
                self.ctx.checkpoint_manager.mark_stage_complete("normalization", normalization_metadata)
                self.ctx.logger.info("Normalization stage marked as complete")

            self.ctx.logger.info(f"Count normalization completed successfully using {self.ctx.normalization_method}")
            return True

        except Exception as e:
            self.ctx.logger.error(f"Count normalization failed: {str(e)}")
            return False
