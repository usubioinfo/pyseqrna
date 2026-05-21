#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HISAT2 Aligner Module

This module provides the HISAT2 aligner implementation for RNA-seq read alignment.
HISAT2 (Hierarchical Indexing for Spliced Alignment of Transcripts 2) is a fast
and sensitive alignment system for mapping next-generation sequencing reads (both DNA and RNA)
to reference genomes, specifically optimized for spliced RNA alignment.

Features:
    - Hierarchical reference index building via `hisat2-build`
    - Spliced RNA-seq read mapping supporting single-end and paired-end modes
    - Inline streaming to `samtools` for outputting sorted BAM format directly
    - Concurrent sample processing using SLURM job submission or local execution
    - Extensible parameter configuration via `hisat2.ini` configuration file

Configuration:
    The aligner loads custom command-line options from a `hisat2.ini` configuration file.
    Standard pipeline settings (CPUs, reference genome, directories, slurm settings)
    are managed via constructor parameters.

Dependencies:
    - HISAT2 (hisat2 and hisat2-build): Must be installed and available in system PATH
    - Samtools: Required for BAM conversion and sorting pipeline

Classes / Functions / Exceptions:
    - Hisat2Aligner: Class implementing HISAT2 indexing and alignment methods.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
from typing import Dict, List, Optional, Union, Any
from .base import BaseAligner, AlignmentError


class Hisat2Aligner(BaseAligner):
    """
    HISAT2 aligner implementation for RNA-seq read alignment.

    This class provides methods for building HISAT2 genome indices and aligning
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
        Initialize HISAT2 aligner.

        Args:
            genome: Path to reference genome file
            out_dir: Output directory for results
            param_dir: Directory containing parameter files
            logger: Logger instance
            dryrun: Whether to perform dry run
            cpu_threads: Number of CPU threads
            slurm: Whether to use SLURM
            dep: SLURM job dependency
            dry_run_manager: Dry run manager for command simulation
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

        # Load HISAT2-specific configuration
        self.hisat2_config = self.load_config("hisat2.ini")
        self.logger.info(f"HISAT2 aligner initialized with {self.cpu_threads} threads")

    def check_index(self) -> bool:
        """
        Check if HISAT2 genome index exists and is valid.

        Returns:
            bool: True if index exists and is valid, False otherwise
        """
        try:
            # Check for HISAT2 index files (base name without extension)
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
            index_files = [
                self.index_dir / f"{genome_base}.1.ht2",
                self.index_dir / f"{genome_base}.2.ht2",
                self.index_dir / f"{genome_base}.3.ht2",
                self.index_dir / f"{genome_base}.4.ht2",
                self.index_dir / f"{genome_base}.5.ht2",
                self.index_dir / f"{genome_base}.6.ht2",
                self.index_dir / f"{genome_base}.7.ht2",
                self.index_dir / f"{genome_base}.8.ht2",
            ]

            # Check if all required index files exist (quietly)
            all_exist = all(f.exists() for f in index_files)

            return all_exist

        except Exception as e:
            self.logger.error(f"Error checking HISAT2 index: {str(e)}")
            return False

    def build_index(self, gff: Optional[str] = None) -> Union[str, None]:
        """
        Build HISAT2 genome index.

        Args:
            gff: Optional path to GFF/GTF annotation file (not used by HISAT2)

        Returns:
            str: Command string if dryrun is True, None otherwise

        Raises:
            AlignmentError: If index building fails
        """
        try:
            # Copy genome file to index directory first
            genome_in_index = self._copy_genome_to_index()

            # Get HISAT2 parameters from config
            hisat2_params = self.hisat2_config.get("hisat2", {})

            # Build HISAT2 index command
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(genome_in_index))
            index_prefix = str(self.index_dir / genome_base)

            cmd_parts = ["hisat2-build"]
            cmd_parts.extend(["-p", str(self.cpu_threads)])

            # Add additional parameters from config
            for param, value in hisat2_params.items():
                if param not in ["p", "threads"]:
                    cmd_parts.extend([f"-{param}", str(value)])

            cmd_parts.extend([genome_in_index, index_prefix])  # Use copied genome
            command = " ".join(cmd_parts)

            if self.dryrun:
                self.logger.info(f"Would run HISAT2 index build command: {command}")
                return command

            self.logger.info(f"Building HISAT2 index in {self.index_dir}")
            self.logger.debug(f"HISAT2 command: {command}")

            # Execute the command
            commands = {"hisat2_index": command}
            # Use alignment directory for logs
            job_ids = self.execute_command(commands, str(self.alignment_dir), "hisat2_index")

            self.logger.info("HISAT2 index built successfully")
            if self.slurm and job_ids:
                return ",".join(job_ids.values())
            return None

        except Exception as e:
            raise AlignmentError(f"HISAT2 index building failed: {str(e)}")

    def run_alignment(self, target: Optional[Dict[str, List[str]]] = None, paired: bool = False) -> Dict[str, Any]:
        """
        Run HISAT2 alignment on RNA-seq reads.

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
                raise AlignmentError("HISAT2 index not found. Please build index first.")

            # Get HISAT2 parameters from config
            hisat2_params = self.hisat2_config.get("hisat2", {})

            commands = {}
            results = {}

            for sample_id, file_paths in target.items():
                # Build HISAT2 alignment command
                genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
                index_prefix = str(self.index_dir / genome_base)
                # Put output files directly in tool results directory
                output_bam = str(self.results_dir / f"{sample_id}_aligned.bam")

                cmd_parts = ["hisat2"]
                cmd_parts.extend(["-x", index_prefix])
                cmd_parts.extend(["-p", str(self.cpu_threads)])

                # Handle input files
                if paired and len(file_paths) >= 2:
                    cmd_parts.extend(["-1", file_paths[0], "-2", file_paths[1]])
                else:
                    cmd_parts.extend(["-U", file_paths[0]])

                # Add additional parameters from config
                for param, value in hisat2_params.items():
                    if param not in ["x", "p", "1", "2", "U"]:
                        cmd_parts.extend([f"-{param}", str(value)])

                # HISAT2 writes SAM to stdout when -S is omitted. Some builds
                # treat "-S -" as a literal file named "-", so omit -S for pipes.
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
                self.logger.info(f"Would run HISAT2 alignment for {len(commands)} samples")
                for sample_id, cmd in commands.items():
                    self.logger.info(f"Sample {sample_id}: {cmd}")
                return results

            self.logger.info(f"Running HISAT2 alignment for {len(commands)} samples")

            # Execute alignment commands
            # Use alignment directory for logs
            job_ids = self.execute_command(commands, str(self.alignment_dir), "hisat2_alignment")

            # Add job IDs to results if using SLURM
            if job_ids:
                for sample_id, job_id in job_ids.items():
                    results[sample_id]["job_id"] = job_id

            self.logger.info("HISAT2 alignment completed successfully")
            return results

        except Exception as e:
            raise AlignmentError(f"HISAT2 alignment failed: {str(e)}")
