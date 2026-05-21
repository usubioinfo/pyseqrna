#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAR Aligner Module

This module provides the STAR (Spliced Transcripts Alignment to a Reference) aligner implementation
for RNA-seq read alignment. STAR is a fast and highly accurate spliced aligner designed specifically
to handle eukaryotic RNA-seq data, supporting splice junction discovery, platform-aware compression/decompression,
and customizable mapping parameters.

Features:
    - Spliced reference genome index generation using `STAR --runMode genomeGenerate`
    - Splice junction database integration (GFF/GTF annotations and custom overhangs)
    - Ultra-fast RNA-seq read mapping supporting single-end and paired-end read inputs
    - Platform-aware resolution of compressed inputs (handles `gzcat`, `zcat`, and `gunzip` dynamically)
    - SLURM job scheduler and local multi-threaded job execution management
    - Detailed output extraction, including BAM alignments, run logs, and splice junction tables

Configuration:
    The aligner loads custom command-line options from a `star.ini` configuration file
    under the `[star]` and `[alignment]` sections. Additionally, configurations like reference genomes,
    output folders, and threads are managed via constructor arguments.

Dependencies:
    - STAR: Must be installed and available in system PATH
    - System decompression utilities (e.g., `zcat`, `gzcat`, `gunzip`) for processing compressed fastq files

Classes / Functions / Exceptions:
    - StarAligner: Class implementing STAR index construction and alignment methods.

:Created: May 20, 2021
:Updated: February 4, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import shutil
from typing import Dict, List, Optional, Union, Any
from .base import BaseAligner, AlignmentError


class StarAligner(BaseAligner):
    """
    STAR aligner implementation for RNA-seq read alignment.

    This class provides methods for building STAR genome indices and aligning
    RNA-seq reads against the reference genome.
    """

    def __init__(
        self,
        genome: str,
        param_dir: Optional[str] = None,
        out_dir: str = ".",
        slurm: bool = False,
        dryrun: bool = False,
        dep: str = "",
        cpu_threads: Optional[int] = None,
        logger=None,
        dry_run_manager=None,
        slurm_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the STAR aligner.

        Args:
            genome: Path to reference genome FASTA file
            param_dir: Directory containing parameter files
            out_dir: Output directory for results
            slurm: Whether to use SLURM for job execution
            dryrun: Whether to perform a dry run
            dep: SLURM job ID on which this job depends
            cpu_threads: Number of CPU threads to use
            logger: Logger instance for logging messages
            dry_run_manager: Dry run manager instance
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

        # Load STAR-specific configuration with platform adjustments
        self.star_config = self._load_platform_aware_config()
        self.logger.info(f"STAR aligner initialized with {self.cpu_threads} threads")

    def _load_platform_aware_config(self) -> Dict[str, Any]:
        """
        Load STAR configuration with platform-specific adjustments.

        Returns:
            Dict containing STAR configuration parameters
        """
        # Load base configuration. The shared config loader resolves filenames
        # case-insensitively, so users are not penalized for STAR.ini vs star.ini.
        config = self.load_config("star.ini")
        if not config:
            self.logger.warning("Could not load STAR config file")
            return {}

        import platform

        if "alignment" in config and "zipped_file" in config["alignment"]:
            config["alignment"]["zipped_file"] = self._resolve_read_files_command(
                config["alignment"]["zipped_file"],
                prefer_gzcat=platform.system() == "Darwin",
            )

        return config

    def _resolve_read_files_command(self, zipped_config: str, prefer_gzcat: bool = False) -> str:
        """Resolve STAR --readFilesCommand to an executable path when possible."""
        parts = zipped_config.split()
        if "--readFilesCommand" not in parts:
            return zipped_config

        command_index = parts.index("--readFilesCommand") + 1
        if command_index >= len(parts):
            return zipped_config

        command = parts[command_index]
        command_args = parts[command_index + 1 :]
        candidates = []
        if prefer_gzcat and command in {"zcat", "gzcat"}:
            candidates.extend(["gzcat", "gunzip", "zcat"])
        else:
            candidates.append(command)
            if command == "zcat":
                candidates.extend(["gzcat", "gunzip"])

        for candidate in candidates:
            resolved = shutil.which(candidate)
            if not resolved:
                continue
            if candidate == "gunzip" and not command_args:
                command_args = ["-c"]
            parts[command_index:] = [resolved] + command_args
            self.logger.info(f"Using {resolved} for STAR compressed read input")
            return " ".join(parts)

        self.logger.warning(f"Could not resolve STAR readFilesCommand executable '{command}'; keeping configured value")
        return zipped_config

    def check_index(self) -> bool:
        """
        Check if STAR genome index exists and is valid.

        Returns:
            bool: True if index exists and is valid, False otherwise
        """
        try:
            # Check for STAR index files
            index_files = [
                self.index_dir / "Genome",
                self.index_dir / "SA",
                self.index_dir / "SAindex",
            ]

            # Check if all required index files exist (quietly)
            all_exist = all(f.exists() for f in index_files)

            return all_exist

        except Exception as e:
            self.logger.error(f"Error checking STAR index: {str(e)}")
            return False

    def build_index(self, gff: Optional[str] = None) -> Union[str, None]:
        """
        Build STAR genome index.

        Args:
            gff: Optional path to GFF/GTF annotation file

        Returns:
            str: Command string if dryrun is True, None otherwise

        Raises:
            AlignmentError: If index building fails
        """
        try:
            # Copy genome file to index directory first
            genome_in_index = self._copy_genome_to_index()

            # Get STAR parameters from config
            star_params = self.star_config.get("star", {})

            # Build STAR index command
            cmd_parts = ["STAR", "--runMode", "genomeGenerate"]
            cmd_parts.extend(["--genomeDir", str(self.index_dir)])
            cmd_parts.extend(["--genomeFastaFiles", genome_in_index])  # Use copied genome
            cmd_parts.extend(["--runThreadN", str(self.cpu_threads)])

            # Add GFF/GTF annotation if provided
            if gff:
                cmd_parts.extend(["--sjdbGTFfile", gff])
                cmd_parts.extend(["--sjdbOverhang", "100"])

            # Add additional parameters from config
            for param, value in star_params.items():
                if param not in [
                    "genomeDir",
                    "genomeFastaFiles",
                    "runThreadN",
                    "sjdbGTFfile",
                    "sjdbOverhang",
                ]:
                    cmd_parts.extend([f"--{param}", str(value)])

            command = " ".join(cmd_parts)

            self.logger.info(f"Building STAR index in {self.index_dir}")
            self.logger.debug(f"STAR command: {command}")

            # Execute the command (handles both dry-run and normal execution)
            commands = {"star_index": command}
            job_ids = self.execute_command(commands, str(self.alignment_dir), "star_index")

            self.logger.info("STAR index built successfully")
            if self.slurm and job_ids:
                return ",".join(job_ids.values())
            return None

        except Exception as e:
            raise AlignmentError(f"STAR index building failed: {str(e)}")

    def run_alignment(self, target: Optional[Dict[str, List[str]]] = None, paired: bool = False) -> Dict[str, Any]:
        """
        Run STAR alignment on RNA-seq reads.

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

            # Check if index exists (skip in dry-run mode since files weren't actually created)
            if not self.dryrun and not self.check_index():
                raise AlignmentError("STAR index not found. Please build index first.")

            # Get STAR parameters from config
            star_params = self.star_config.get("alignment", {})

            commands = {}
            results = {}

            for sample_id, file_paths in target.items():
                # Build STAR alignment command
                cmd_parts = ["STAR"]
                cmd_parts.extend(["--genomeDir", str(self.index_dir)])
                cmd_parts.extend(["--runThreadN", str(self.cpu_threads)])
                cmd_parts.extend(["--outFileNamePrefix", str(self.results_dir / f"{sample_id}_")])

                # Handle input files
                if paired and len(file_paths) >= 2:
                    cmd_parts.extend(["--readFilesIn", file_paths[0], file_paths[1]])
                else:
                    cmd_parts.extend(["--readFilesIn", file_paths[0]])

                # Add additional parameters from config
                for param, value in star_params.items():
                    # Skip parameters that are already handled or are directory paths
                    if param in ["index_dir", "threads"] or not value or value == "NA":
                        continue

                    if param == "zipped_file":
                        # Handle special zipped_file parameter (contains --readFilesCommand)
                        cmd_parts.extend(value.split())
                    else:
                        # Handle regular parameters that already include -- prefix
                        cmd_parts.extend(value.split())

                command = " ".join(cmd_parts)
                commands[sample_id] = command

                # Store expected output files
                results[sample_id] = {
                    "bam": str(self.results_dir / f"{sample_id}_Aligned.out.bam"),
                    "log": str(self.results_dir / f"{sample_id}_Log.final.out"),
                    "sj": str(self.results_dir / f"{sample_id}_SJ.out.tab"),
                }

            self.logger.info(f"Running STAR alignment for {len(commands)} samples")

            # Execute alignment commands (handles both dry-run and normal execution)
            job_ids = self.execute_command(commands, str(self.alignment_dir), "star_alignment")

            # Add job IDs to results if using SLURM
            if job_ids:
                for sample_id, job_id in job_ids.items():
                    results[sample_id]["job_id"] = job_id

            self.logger.info("STAR alignment completed successfully")
            return results

        except Exception as e:
            raise AlignmentError(f"STAR alignment failed: {str(e)}")
