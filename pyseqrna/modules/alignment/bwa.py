#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BWA Aligner Module

This module provides the BWA (Burrows-Wheeler Aligner) implementation for RNA-seq read alignment.
BWA is a software package for mapping low-divergent sequencing reads against a large reference genome,
supporting multi-threading, configurable execution parameters, and pipeline-based SAM-to-sorted-BAM conversion.

Features:
    - Reference genome indexing using `bwa index`
    - Read mapping using `bwa mem` for high-throughput single-end and paired-end sequencing datasets
    - Direct pipelining to `samtools` for on-the-fly SAM-to-BAM conversion and sorting
    - Integration with SLURM scheduler for cluster job execution or local execution
    - Support for custom command options through `bwa.ini` config file

Configuration:
    The aligner loads custom command-line options from a `bwa.ini` configuration file.
    It can also be customized programmatically via constructor parameters, including CPU thread count,
    output directory, and SLURM workload settings.

Dependencies:
    - BWA: Must be installed and available in system PATH
    - Samtools: Required for BAM conversion and sorting pipeline

Classes / Functions / Exceptions:
    - BwaAligner: Class implementing BWA indexing and alignment methods.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
from typing import Dict, List, Optional, Union, Any
from .base import BaseAligner, AlignmentError


class BwaAligner(BaseAligner):
    """
    BWA aligner implementation for RNA-seq read alignment.

    This class provides methods for building BWA genome indices and aligning
    RNA-seq reads against the reference genome.
    """

    def __init__(
        self,
        genome: str,
        out_dir: str,
        param_dir: Optional[str] = None,
        logger=None,
        dryrun: bool = False,
        cpu_threads: Optional[int] = None,
        slurm: bool = False,
        dep: str = "",
        dry_run_manager=None,
        slurm_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize BWA aligner.

        Args:
            genome: Path to reference genome file
            out_dir: Output directory for results
            param_dir: Directory containing parameter files
            logger: Logger instance
            dryrun: Whether to perform dry run
            cpu_threads: Number of CPU threads
            slurm: Whether to use SLURM
            dep: SLURM job dependency
        """
        super().__init__(
            genome=genome,
            param_dir=param_dir,
            out_dir=out_dir,
            slurm=slurm,
            dryrun=dryrun,
            dep=dep,
            cpu_threads=cpu_threads,
            logger=logger,
            dry_run_manager=dry_run_manager,
            slurm_config=slurm_config,
        )

        # Load BWA-specific configuration
        self.bwa_config = self.load_config("bwa.ini")
        self.logger.info(f"BWA aligner initialized with {self.cpu_threads} threads")

    def check_index(self) -> bool:
        """
        Check if BWA genome index exists and is valid.

        Returns:
            bool: True if index exists and is valid, False otherwise
        """
        try:
            # Check for BWA index files
            genome_base = os.path.basename(self.genome)
            index_files = [
                self.index_dir / f"{genome_base}.amb",
                self.index_dir / f"{genome_base}.ann",
                self.index_dir / f"{genome_base}.bwt",
                self.index_dir / f"{genome_base}.pac",
                self.index_dir / f"{genome_base}.sa",
            ]

            # Check if all required index files exist (quietly)
            all_exist = all(f.exists() for f in index_files)

            return all_exist

        except Exception as e:
            self.logger.error(f"Error checking BWA index: {str(e)}")
            return False

    def build_index(self, gff: Optional[str] = None) -> Union[str, None]:
        """
        Build BWA genome index.

        Args:
            gff: Optional path to GFF/GTF annotation file (not used by BWA)

        Returns:
            str: Command string if dryrun is True, None otherwise

        Raises:
            AlignmentError: If index building fails
        """
        try:
            # Copy genome file to index directory first
            genome_in_index = self._copy_genome_to_index()

            # Get BWA parameters from config
            bwa_params = self.bwa_config.get("bwa", {})

            # Build BWA index command
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(genome_in_index))
            str(self.index_dir / genome_base)

            cmd_parts = ["bwa", "index"]

            # Add additional parameters from config
            for param, value in bwa_params.items():
                cmd_parts.extend([f"-{param}", str(value)])

            cmd_parts.append(genome_in_index)  # Use copied genome
            command = " ".join(cmd_parts)

            if self.dryrun:
                self.logger.info(f"Would run BWA index build command: {command}")
                return command

            self.logger.info(f"Building BWA index in {self.index_dir}")
            self.logger.debug(f"BWA command: {command}")

            # Execute the command
            commands = {"bwa_index": command}
            job_ids = self.execute_command(commands, str(self.alignment_dir), "bwa_index")

            self.logger.info("BWA index built successfully")
            if self.slurm and job_ids:
                return ",".join(job_ids.values())
            return None

        except Exception as e:
            raise AlignmentError(f"BWA index building failed: {str(e)}")

    def run_alignment(self, target: Optional[Dict[str, List[str]]] = None, paired: bool = False) -> Dict[str, Any]:
        """
        Run BWA alignment on RNA-seq reads.

        Args:
            target: Dictionary mapping sample IDs to their file paths
            paired: Boolean indicating if reads are paired-end

        Returns:
            Dict containing output file paths and job IDs if using SLURM

        Raises:
            AlignmentError: If alignment fails
        """
        try:
            if not target:
                raise AlignmentError("No target files provided for alignment")

            # Check if index exists (skip in dry-run mode since files are not created)
            if not self.dryrun and not self.check_index():
                raise AlignmentError("BWA index not found. Please build index first.")

            # Get BWA parameters from config
            bwa_params = self.bwa_config.get("bwa", {})

            commands = {}
            results = {}

            for sample_id, file_paths in target.items():
                # Build BWA alignment command
                genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
                str(self.index_dir / genome_base)
                # Put output files directly in tool results directory
                output_bam = str(self.results_dir / f"{sample_id}_aligned.bam")

                # BWA mem command
                cmd_parts = ["bwa", "mem"]
                cmd_parts.extend(["-t", str(self.cpu_threads)])

                # Add additional parameters from config
                for param, value in bwa_params.items():
                    if param not in ["t", "threads"]:
                        cmd_parts.extend([f"-{param}", str(value)])

                # Add reference genome
                cmd_parts.append(self.genome)

                # Handle input files
                if paired and len(file_paths) >= 2:
                    cmd_parts.extend([file_paths[0], file_paths[1]])
                else:
                    cmd_parts.append(file_paths[0])

                # Add output processing
                cmd_parts.extend(["|", "samtools", "view", "-bS", "-"])
                cmd_parts.extend(["|", "samtools", "sort", "-o", output_bam, "-"])

                command = " ".join(cmd_parts)
                commands[sample_id] = command

                # Store expected output files
                results[sample_id] = {
                    "bam": output_bam,
                    "log": str(self.results_dir / f"{sample_id}_alignment.log"),
                }

            if self.dryrun:
                self.logger.info(f"Would run BWA alignment for {len(commands)} samples")
                for sample_id, cmd in commands.items():
                    self.logger.info(f"Sample {sample_id}: {cmd}")
                return results

            self.logger.info(f"Running BWA alignment for {len(commands)} samples")

            # Execute alignment commands
            job_ids = self.execute_command(commands, str(self.alignment_dir), "bwa_alignment")

            # Add job IDs to results if using SLURM
            if job_ids:
                for sample_id, job_id in job_ids.items():
                    results[sample_id]["job_id"] = job_id

            self.logger.info("BWA alignment completed successfully")
            return results

        except Exception as e:
            raise AlignmentError(f"BWA alignment failed: {str(e)}")
