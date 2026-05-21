#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Differential Expression Runner

This module coordinates differential gene expression analysis steps. It interfaces with
statistical packages (DESeq2, edgeR, or PyDiffExpress) to discover differentially expressed genes (DEGs),
manages target comparisons, filters results by significance thresholds, and archives output logs.

Features:
    - Statistical testing for DEGs using DESeq2, edgeR, or PyDiffExpress
    - Dynamic comparison setup from experimental factor designs
    - Checkpoint evaluation and smart resume verification on tool/normalization changes
    - File tracking including size, modification times, and existence audits
    - Optional multi-mapped groups (MMG) differential expression analysis

Configuration:
    - Configured via Pipeline context properties (skip_diffexp, diffexp_tool, diffexp_normalization, diffexp_abundance, diffexp_dispersion, diffexp_test, fdr_threshold, log2fc_threshold, pvalue_threshold, subset, enable_multimapped_groups).

Dependencies:
    - Python packages: pathlib, datetime
    - External packages/tools: R environment with DESeq2/edgeR installed (optional, depends on tool choice)

Classes / Functions / Exceptions:
    - DifferentialExpressionRunner: Coordinates differential expression analysis steps using DESeq2, edgeR, or PyDiffExpress.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

from datetime import datetime


class DifferentialExpressionRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_differential_expression(self) -> bool:
        """
        Run differential expression analysis using the selected tool.

        Returns:
            True if successful, False otherwise
        """
        if self.ctx.skip_diffexp:
            self.ctx.logger.info("Differential expression analysis skipped by user request")
            return True

        # Only check for completed stages if we're in resume mode (not 'all')
        if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("differential"):
            stage_metadata = self.ctx.checkpoint_manager.get_stage_metadata("differential")
            completed_tool = stage_metadata.get("tool", "unknown") if stage_metadata else "unknown"

            # Check if tool and native component settings match.
            component_matches = True
            if self.ctx.diffexp_tool == "pydiffexpress":
                component_matches = (
                    stage_metadata.get("diffexp_normalization") == self.ctx.diffexp_normalization
                    and stage_metadata.get("diffexp_abundance") == self.ctx.diffexp_abundance
                    and stage_metadata.get("diffexp_dispersion") == self.ctx.diffexp_dispersion
                    and stage_metadata.get("diffexp_test") == self.ctx.diffexp_test
                )
            tool_matches = completed_tool == self.ctx.diffexp_tool and component_matches

            if tool_matches:
                # Ask user if they want to re-run or skip
                user_choice = self.ctx._ask_user_rerun_stage("differential", completed_tool)

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    # Load differential expression results from checkpoint
                    if stage_metadata and "output_file" in stage_metadata:
                        self.ctx.differential_expression_results = stage_metadata["output_file"]
                        self.ctx.logger.info(
                            f"Differential expression already completed with {self.ctx.diffexp_tool}, loaded results from checkpoint"
                        )
                    else:
                        self.ctx.logger.info(
                            f"Differential expression already completed with {self.ctx.diffexp_tool}, skipping"
                        )
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run differential expression stage")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("differential")
            else:
                # Tool mismatch - inform user and ask
                mismatch_info = f"Tool/settings mismatch: {completed_tool} vs {self.ctx.diffexp_tool}"

                self.ctx.logger.warning(f"Differential expression stage completed with different settings: {mismatch_info}")
                user_choice = self.ctx._ask_user_rerun_stage("differential", f"{completed_tool} (different settings)")

                if user_choice is None:  # User cancelled
                    return False
                elif user_choice is False:  # User chose to skip
                    self.ctx.logger.warning("User chose to skip despite tool mismatch. This may cause issues.")
                    return True
                else:  # User chose to re-run
                    self.ctx.logger.info("User chose to re-run differential expression stage with new settings")
                    # Mark as incomplete to force re-run
                    self.ctx.checkpoint_manager.mark_stage_incomplete("differential")

        self.ctx.logger.info(f"Starting differential expression analysis with {self.ctx.diffexp_tool}")

        # Check dependencies - need quantification results
        if self.ctx.quantification_results is None:
            if self.ctx.checkpoint_manager.is_stage_complete("quantification"):
                quantification_metadata = self.ctx.checkpoint_manager.get_stage_metadata("quantification")
                if quantification_metadata and "output_file" in quantification_metadata:
                    self.ctx.quantification_results = quantification_metadata["output_file"]
                    self.ctx.logger.info("Loaded quantification results from checkpoint")
                else:
                    self.ctx.logger.error("Quantification results not found in checkpoint")
                    return False
            else:
                self.ctx.logger.error("Quantification must be completed before differential expression")
                return False

        # Validate annotation file exists
        if not Path(self.ctx.feature_file).exists():
            self.ctx.logger.error(f"Annotation file not found: {self.ctx.feature_file}")
            return False

        try:
            # Create differential expression directory structure
            diffexp_dir = Path(self.ctx.output_dir) / "4.Differential_Expression"
            self.ctx._clean_and_create_directory(diffexp_dir, "differential expression")

            # Available differential expression tools
            from ...modules.diffexp import create_diffexp_analyzer, get_available_tools

            available_tools = get_available_tools()
            if self.ctx.diffexp_tool.lower() not in available_tools:
                self.ctx.logger.error(
                    f"Unsupported differential expression tool: {self.ctx.diffexp_tool}. Available: {', '.join(available_tools)}"
                )
                return False

            # Create differential expression analyzer using factory function
            # Get actual combinations from input processor instead of hardcoded ones
            sample_data = self.ctx.input_processor.process_sample_file(
                self.ctx.input_file,
                self.ctx.samples_path,
                paired=self.ctx.force_paired,
            )
            self.ctx._paired_end_cache = sample_data.get("paired", self.ctx._paired_end_cache)
            actual_combinations = sample_data["combinations"]
            processed_targets = sample_data["targets"]  # Get the processed targets DataFrame

            self.ctx.logger.info(f"Using sample combinations: {actual_combinations}")

            diffexp_module = create_diffexp_analyzer(
                tool_name=self.ctx.diffexp_tool,
                count_matrix_file=self.ctx.quantification_results,  # Pass DataFrame directly
                sample_info_file=processed_targets,  # Pass DataFrame directly
                comparisons=actual_combinations,  # Use actual combinations from input processor
                out_dir=str(diffexp_dir),
                param_dir=self.ctx.param_dir,
                species=self.ctx.species,
                organism_type=self.ctx.organism_type,
                add_gene_names=self.ctx.add_gene_names,
                logger=self.ctx.logger.logger,
                fdr_threshold=self.ctx.fdr_threshold,
                log2fc_threshold=self.ctx.log2fc_threshold,
                dryrun=self.ctx.dryrun,
                dry_run_manager=self.ctx.dry_run_manager,
                subset=self.ctx.subset,
                diffexp_normalization=self.ctx.diffexp_normalization,
                diffexp_abundance=self.ctx.diffexp_abundance,
                diffexp_dispersion=self.ctx.diffexp_dispersion,
                diffexp_test=self.ctx.diffexp_test,
            )

            # Run differential expression
            results = diffexp_module.run()

            # Store differential expression results
            self.ctx.differential_expression_results = results

            # Record executed commands from the differential expression module
            if hasattr(diffexp_module, "command_executor") and hasattr(diffexp_module.command_executor, "executed_commands"):
                self.ctx.dry_run_manager.record_executed_commands(diffexp_module.command_executor.executed_commands)

            mmg_results = None
            if self.ctx.enable_multimapped_groups:
                mmg_results = self.ctx._run_mmg_differential_expression(
                    diffexp_dir=diffexp_dir,
                    processed_targets=processed_targets,
                    actual_combinations=actual_combinations,
                )

            # Generate summary statistics
            if results is not None and isinstance(results, dict):
                stats = diffexp_module.get_summary_stats(results)
                self.ctx.logger.info("Differential expression completed successfully:")
                self.ctx.logger.info(f"  - Tool: {self.ctx.diffexp_tool}")
                if self.ctx.diffexp_tool == "pydiffexpress":
                    self.ctx.logger.info(
                        "  - Components: normalization=%s, abundance=%s, dispersion=%s, test=%s",
                        self.ctx.diffexp_normalization,
                        self.ctx.diffexp_abundance,
                        self.ctx.diffexp_dispersion,
                        self.ctx.diffexp_test,
                    )
                self.ctx.logger.info(f"  - Genes: {stats.get('total_genes', 0)}")
                self.ctx.logger.info(f"  - Samples: {stats.get('total_samples', 0)}")
                self.ctx.logger.info(f"  - Total genes tested: {stats.get('total_genes_tested', 0)}")
                self.ctx.logger.info(f"  - Genes with significant results: {stats.get('genes_with_significant_results', 0)}")

            # Enhanced metadata collection with detailed file tracking
            output_files = results.get("output_files", []) if results else []
            if mmg_results:
                output_files.extend(mmg_results.get("output_files", []))
            main_output_file = (
                str(diffexp_dir / "All_gene_expression.xlsx")
                if output_files
                else str(diffexp_dir / "Differential_Expression_Results.xlsx")
            )

            # Collect detailed file information
            detailed_output_files = {}
            total_size = 0
            for file_path in output_files:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    detailed_output_files[file_path] = {
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "exists": True,
                        "modified": datetime.fromtimestamp(Path(file_path).stat().st_mtime).isoformat(),
                    }
                    total_size += file_size
                else:
                    detailed_output_files[file_path] = {
                        "exists": False,
                        "error": "File not found",
                    }

            # Enhanced metadata with comprehensive information
            metadata = {
                "tool": self.ctx.diffexp_tool,
                "diffexp_normalization": self.ctx.diffexp_normalization,
                "diffexp_abundance": self.ctx.diffexp_abundance,
                "diffexp_dispersion": self.ctx.diffexp_dispersion,
                "diffexp_test": self.ctx.diffexp_test,
                "output_file": main_output_file,
                "output_files": output_files,
                "detailed_output_files": detailed_output_files,
                "total_output_size_mb": round(total_size / (1024 * 1024), 2),
                "feature_file": self.ctx.feature_file,
                "execution_time": datetime.now().isoformat(),
                "comparisons": len(actual_combinations),
                "fdr_threshold": self.ctx.fdr_threshold,
                "log2fc_threshold": self.ctx.log2fc_threshold,
                "summary_stats": self.ctx._convert_numpy_for_json(results.get("summary", {})) if results else {},
                "mmg_summary_stats": self.ctx._convert_numpy_for_json(mmg_results.get("summary", {})) if mmg_results else {},
            }

            self.ctx.checkpoint_manager.mark_stage_complete("differential", metadata=metadata, dry_run=self.ctx.dryrun)

            return True

        except Exception as e:
            self.ctx.logger.error(f"Differential expression with {self.ctx.diffexp_tool} failed: {e}")
            return False
