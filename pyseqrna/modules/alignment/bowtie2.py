#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bowtie2 Aligner Module

This module provides the Bowtie2 aligner implementation for RNA-seq read alignment.
Bowtie2 is a fast and memory-efficient aligner designed for aligning short sequencing reads
to relatively long reference genomes. It supports multi-threading, custom config execution,
and pipelined SAM-to-sorted-BAM conversion.

Features:
    - Reference genome index building via `bowtie2-build`
    - High-throughput read alignment supporting both single-end and paired-end modes
    - Pipeling to `samtools` for on-the-fly SAM-to-BAM conversion and sorting
    - Local execution and SLURM workload manager integration
    - Advanced parameter configuration using `bowtie2.ini`

Configuration:
    The aligner loads custom command-line options from `bowtie2.ini` config file.
    Additionally, standard settings such as thread counts, reference genomes, and output
    directories are configured through constructor arguments.

Dependencies:
    - Bowtie2 (bowtie2 and bowtie2-build): Must be installed and available in system PATH
    - Samtools: Required for BAM conversion and sorting pipeline

Classes / Functions / Exceptions:
    - Bowtie2Aligner: Class implementing Bowtie2 indexing and alignment methods.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
from typing import Dict, List, Optional, Union, Any
from .base import BaseAligner, AlignmentError


class Bowtie2Aligner(BaseAligner):
    """
    Bowtie2 aligner implementation for RNA-seq read alignment.

    This class provides methods for building Bowtie2 genome indices and aligning
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
        Initialize Bowtie2 aligner.

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

        # Load Bowtie2-specific configuration
        self.bowtie2_config = self.load_config("bowtie2.ini")
        self.logger.info(f"Bowtie2 aligner initialized with {self.cpu_threads} threads")

    def check_index(self) -> bool:
        """
        Check if Bowtie2 genome index exists and is valid.

        Returns:
            bool: True if index exists and is valid, False otherwise
        """
        try:
            # Check for Bowtie2 index files
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
            index_files = [
                self.index_dir / f"{genome_base}.1.bt2",
                self.index_dir / f"{genome_base}.2.bt2",
                self.index_dir / f"{genome_base}.3.bt2",
                self.index_dir / f"{genome_base}.4.bt2",
                self.index_dir / f"{genome_base}.rev.1.bt2",
                self.index_dir / f"{genome_base}.rev.2.bt2",
            ]

            # Check if all required index files exist (quietly)
            all_exist = all(f.exists() for f in index_files)

            return all_exist

        except Exception as e:
            self.logger.error(f"Error checking Bowtie2 index: {str(e)}")
            return False

    def build_index(self, gff: Optional[str] = None) -> Union[str, None]:
        """
        Build Bowtie2 genome index.

        Args:
            gff: Optional path to GFF/GTF annotation file (not used by Bowtie2)

        Returns:
            str: Command string if dryrun is True, None otherwise

        Raises:
            AlignmentError: If index building fails
        """
        try:
            # Copy genome file to index directory first
            genome_in_index = self._copy_genome_to_index()

            # Get Bowtie2 parameters from config
            bowtie2_params = self.bowtie2_config.get("bowtie2", {})

            # Build Bowtie2 index command
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(genome_in_index))
            index_prefix = str(self.index_dir / genome_base)

            cmd_parts = ["bowtie2-build"]
            cmd_parts.extend(["--threads", str(self.cpu_threads)])

            # Add additional parameters from config
            for param, value in bowtie2_params.items():
                if param not in ["threads"]:
                    cmd_parts.extend([f"--{param}", str(value)])

            cmd_parts.extend([genome_in_index, index_prefix])  # Use copied genome
            command = " ".join(cmd_parts)

            if self.dryrun:
                self.logger.info(f"Would run Bowtie2 index build command: {command}")
                return command

            self.logger.info(f"Building Bowtie2 index in {self.index_dir}")
            self.logger.debug(f"Bowtie2 command: {command}")

            # Execute the command
            commands = {"bowtie2_index": command}
            job_ids = self.execute_command(commands, str(self.alignment_dir), "bowtie2_index")

            self.logger.info("Bowtie2 index built successfully")
            if self.slurm and job_ids:
                return ",".join(job_ids.values())
            return None

        except Exception as e:
            raise AlignmentError(f"Bowtie2 index building failed: {str(e)}")

    def run_alignment(self, target: Optional[Dict[str, List[str]]] = None, paired: bool = False) -> Dict[str, Any]:
        """
        Run Bowtie2 alignment on RNA-seq reads.

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
                raise AlignmentError("Bowtie2 index not found. Please build index first.")

            # Get Bowtie2 parameters from config
            bowtie2_params = self.bowtie2_config.get("bowtie2", {})

            commands = {}
            results = {}

            for sample_id, file_paths in target.items():
                # Build Bowtie2 alignment command
                genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
                index_prefix = str(self.index_dir / genome_base)
                # Put output files directly in tool results directory
                output_bam = str(self.results_dir / f"{sample_id}_aligned.bam")

                cmd_parts = ["bowtie2"]
                cmd_parts.extend(["-x", index_prefix])
                cmd_parts.extend(["-p", str(self.cpu_threads)])

                # Handle input files
                if paired and len(file_paths) >= 2:
                    cmd_parts.extend(["-1", file_paths[0], "-2", file_paths[1]])
                else:
                    cmd_parts.extend(["-U", file_paths[0]])

                # Add additional parameters from config
                for param, value in bowtie2_params.items():
                    if param not in ["x", "p", "1", "2", "U"]:
                        cmd_parts.extend([f"-{param}", str(value)])

                # Add output
                cmd_parts.extend(["-S", "/dev/stdout"])
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
                self.logger.info(f"Would run Bowtie2 alignment for {len(commands)} samples")
                for sample_id, cmd in commands.items():
                    self.logger.info(f"Sample {sample_id}: {cmd}")
                return results

            self.logger.info(f"Running Bowtie2 alignment for {len(commands)} samples")

            # Execute alignment commands
            job_ids = self.execute_command(commands, str(self.alignment_dir), "bowtie2_alignment")

            # Add job IDs to results if using SLURM
            if job_ids:
                for sample_id, job_id in job_ids.items():
                    results[sample_id]["job_id"] = job_id

            self.logger.info("Bowtie2 alignment completed successfully")
            return results

        except Exception as e:
            raise AlignmentError(f"Bowtie2 alignment failed: {str(e)}")
