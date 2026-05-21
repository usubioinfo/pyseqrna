#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimap2 Aligner Module

This module provides the Minimap2 aligner implementation for RNA-seq read alignment.
Minimap2 is a versatile, high-performance sequence alignment program that maps DNA or long mRNA/RNA-seq reads
against a large reference genome, supporting multi-threading, custom config execution, and direct SAM-to-sorted-BAM pipeline conversion.

Features:
    - Reference genome indexing using `minimap2 -d` (producing `.mmi` files)
    - High-throughput sequence alignment supporting single-end and paired-end modes
    - Flexible support for diverse read types (short reads, splice-aware mRNA, and long reads)
    - Streaming stdout SAM output directly to `samtools` for sorting and BAM compression
    - Dynamic cluster job submission via SLURM or multi-threaded local runs
    - Index and alignment parameter customization via `minimap2.ini` config file

Configuration:
    The aligner loads custom command-line options from a `minimap2.ini` configuration file
    containing `[index]` and `[alignment]` sections. Standard pipeline parameters (threads,
    reference genome path, directories) are configured through the constructor.

Dependencies:
    - Minimap2: Must be installed and available in system PATH
    - Samtools: Required for BAM conversion and sorting pipeline

Classes / Functions / Exceptions:
    - Minimap2Aligner: Class implementing Minimap2 indexing and alignment methods.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import os
from typing import Dict, List, Optional, Union, Any
from .base import BaseAligner, AlignmentError


class Minimap2Aligner(BaseAligner):
    """
    Minimap2 aligner implementation for RNA-seq read alignment.

    This class provides methods for building Minimap2 genome indices and aligning
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
        Initialize Minimap2 aligner.

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

        # Load Minimap2-specific configuration
        self.minimap2_config = self.load_config("minimap2.ini")
        self.logger.info(f"Minimap2 aligner initialized with {self.cpu_threads} threads")

    def check_index(self) -> bool:
        """
        Check if Minimap2 genome index exists and is valid.

        Returns:
            bool: True if index exists and is valid, False otherwise
        """
        try:
            # Check for Minimap2 index file
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
            index_file = self.index_dir / f"{genome_base}.mmi"

            # Check if index file exists (quietly)
            return index_file.exists()

        except Exception as e:
            self.logger.error(f"Error checking Minimap2 index: {str(e)}")
            return False

    def build_index(self, gff: Optional[str] = None) -> Union[str, None]:
        """
        Build Minimap2 genome index.

        Args:
            gff: Optional path to GFF/GTF annotation file (not used by Minimap2)

        Returns:
            str: Command string if dryrun is True, None otherwise

        Raises:
            AlignmentError: If index building fails
        """
        try:
            # Copy genome file to index directory first
            genome_in_index = self._copy_genome_to_index()

            # Get Minimap2 indexing parameters from config
            index_params = self.minimap2_config.get("index", {})

            # Build Minimap2 index command
            genome_base = self.file_manager.extract_name_without_extension(os.path.basename(genome_in_index))
            index_file = str(self.index_dir / f"{genome_base}.mmi")

            cmd_parts = ["minimap2", "-d", index_file]
            cmd_parts.extend(["-t", str(self.cpu_threads)])

            # Add additional parameters from config (skip preset for indexing)
            for param, value in index_params.items():
                if param not in ["index_dir", "basename"] and value and value != "NA":
                    # Map config parameter names to minimap2 flags
                    if param == "kmer_size":
                        cmd_parts.extend(["-k", str(value)])
                    elif param == "window_size":
                        cmd_parts.extend(["-w", str(value)])
                    elif param == "split_index":
                        cmd_parts.extend(["-I", str(value)])
                    elif value.startswith("-"):
                        # Handle parameters that already include the flag
                        cmd_parts.extend(value.split())
                    else:
                        cmd_parts.append(value)

            cmd_parts.append(genome_in_index)  # Use copied genome
            command = " ".join(cmd_parts)

            if self.dryrun:
                self.logger.info(f"Would run Minimap2 index build command: {command}")
                return command

            self.logger.info(f"Building Minimap2 index in {self.index_dir}")
            self.logger.debug(f"Minimap2 command: {command}")

            # Execute the command
            commands = {"minimap2_index": command}
            job_ids = self.execute_command(commands, str(self.alignment_dir), "minimap2_index")

            self.logger.info("Minimap2 index built successfully")
            if self.slurm and job_ids:
                return ",".join(job_ids.values())
            return None

        except Exception as e:
            raise AlignmentError(f"Minimap2 index building failed: {str(e)}")

    def run_alignment(self, target: Optional[Dict[str, List[str]]] = None, paired: bool = False) -> Dict[str, Any]:
        """
        Run Minimap2 alignment on RNA-seq reads.

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
                raise AlignmentError("Minimap2 index not found. Please build index first.")

            # Get Minimap2 alignment parameters from config
            alignment_config = self.minimap2_config.get("alignment", {})

            commands = {}
            results = {}

            for sample_id, file_paths in target.items():
                # Build Minimap2 alignment command
                genome_base = self.file_manager.extract_name_without_extension(os.path.basename(self.genome))
                index_file = str(self.index_dir / f"{genome_base}.mmi")
                # Put output files directly in tool results directory
                output_bam = str(self.results_dir / f"{sample_id}_aligned.bam")

                # Minimap2 alignment command - start with base command
                cmd_parts = ["minimap2"]
                cmd_parts.extend(["-t", str(self.cpu_threads)])

                # Add all parameters from config including preset and SAM output
                alignment_config = self.minimap2_config.get("alignment", {})
                for param, value in alignment_config.items():
                    if param not in ["threads", "index_dir", "reference"] and value and value != "NA":
                        # Handle parameters that already include the flag
                        if value.startswith("-"):
                            cmd_parts.extend(value.split())
                        else:
                            cmd_parts.append(value)

                # Add index file
                cmd_parts.append(index_file)

                # Handle input files
                if paired and len(file_paths) >= 2:
                    cmd_parts.extend([file_paths[0], file_paths[1]])
                else:
                    cmd_parts.append(file_paths[0])

                # Add output processing pipeline
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
                self.logger.info(f"Would run Minimap2 alignment for {len(commands)} samples")
                for sample_id, cmd in commands.items():
                    self.logger.info(f"Sample {sample_id}: {cmd}")
                return results

            self.logger.info(f"Running Minimap2 alignment for {len(commands)} samples")

            # Execute alignment commands
            job_ids = self.execute_command(commands, str(self.alignment_dir), "minimap2_alignment")

            # Add job IDs to results if using SLURM
            if job_ids:
                for sample_id, job_id in job_ids.items():
                    results[sample_id]["job_id"] = job_id

            self.logger.info("Minimap2 alignment completed successfully")
            return results

        except Exception as e:
            raise AlignmentError(f"Minimap2 alignment failed: {str(e)}")
