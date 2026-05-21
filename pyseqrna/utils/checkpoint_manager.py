#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint Manager Module

This module provides functionality for managing checkpoints in the RNA-seq analysis pipeline.
It allows tracking progress across different stages and enables resuming from specific points
in the analysis workflow.

Features:
    - Track completion status of analysis stages
    - Save and load checkpoint information
    - Support for multiple analysis stages:
        * Quality Control
        * Read Trimming
        * Alignment
        * Alignment Statistics
        * Quantification
        * Normalization
        * Sample Clustering
        * Differential Expression
        * Visualization
        * Functional Annotation
    - JSON-based checkpoint storage
    - Detailed logging of checkpoint operations
    - Error handling and validation
    - Dry-run support for completed stages
    - Pipeline resumption capabilities
    - Smart directory management for fresh vs resumed analyses

Usage::

    from pyseqrna.utils import CheckpointManager

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir="analysis_output",
        pipeline_name="pyseqrna",
        logger=custom_logger  # Optional
    )

    # Mark a stage as complete
    checkpoint_manager.mark_stage_complete("quality_control")

    # Check if a stage is complete
    if checkpoint_manager.is_stage_complete("read_trimming"):
        print("Read trimming already completed")

    # Get incomplete stages
    incomplete_stages = checkpoint_manager.get_incomplete_stages()

    # Save checkpoint state
    checkpoint_manager.save_checkpoint()

    # Resume pipeline with dry-run for completed stages
    checkpoint_manager.resume_pipeline(dry_run_completed=True)

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime

from .stage_registry import AVAILABLE_STAGE_LABELS, CHECKPOINT_STAGE_ORDER


class CheckpointManager:
    """
    Manages checkpoints for different stages of RNA-seq analysis pipeline.

    This class provides functionality to track the completion status of various
    stages in the RNA-seq analysis workflow, allowing for resuming from specific
    points and maintaining analysis state.

    Attributes:
        output_dir (str): Directory for storing checkpoint information
        pipeline_name (str): Name of the analysis pipeline
        checkpoint_file (str): Path to the checkpoint JSON file
        stages (Set[str]): Set of all available analysis stages
        completed_stages (Set[str]): Set of completed stages
        stage_metadata (Dict[str, Dict]): Metadata for each stage
        dry_run_completed (bool): Whether to run completed stages in dry-run mode
        stage_directories (Dict[str, Path]): Mapping of stages to their output directories
        is_resumed (bool): Whether this is a resumed analysis
        logger (logging.Logger): Logger instance for recording operations
    """

    STAGE_ORDER = list(CHECKPOINT_STAGE_ORDER)
    AVAILABLE_STAGES = dict(AVAILABLE_STAGE_LABELS)

    def __init__(
        self,
        output_dir: str,
        pipeline_name: str = "pyseqrna",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the CheckpointManager.

        Args:
            output_dir (str): Directory for storing checkpoint information, provided by main.
            pipeline_name (str, optional): Name of the analysis pipeline.
                Defaults to "pyseqrna".
            logger (logging.Logger, optional): Logger instance for recording operations.
                If None, creates a new logger. Defaults to None.

        Raises:
            ValueError: If output_dir is empty or invalid
        """
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir must be a non-empty string")

        # Set up logger
        self.logger = logger or logging.getLogger(__name__)

        self.output_dir = Path(output_dir)
        self.pipeline_name = pipeline_name
        self.checkpoint_file = self.output_dir / f"{pipeline_name}_checkpoint.json"
        self.stages = set(self.AVAILABLE_STAGES.keys())
        self.stage_order = list(self.STAGE_ORDER)
        self.completed_stages: Set[str] = set()
        self.stage_metadata: Dict[str, Dict] = {}
        self.dry_run_completed: bool = False
        self.stage_directories: Dict[str, Path] = {}
        self.is_resumed: bool = False

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint if available
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load checkpoint information from JSON file if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)
                    self.completed_stages = set(checkpoint_data.get("completed_stages", []))
                    self.stage_metadata = checkpoint_data.get("stage_metadata", {})
                    self.dry_run_completed = checkpoint_data.get("dry_run_completed", False)
                    self.stage_directories = {
                        stage: Path(dir_path) for stage, dir_path in checkpoint_data.get("stage_directories", {}).items()
                    }
                    self.is_resumed = True
                self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {str(e)}")
                self.completed_stages = set()
                self.stage_metadata = {}
                self.dry_run_completed = False
                self.stage_directories = {}
                self.is_resumed = False

    def save_checkpoint(self) -> None:
        """Save current checkpoint state to JSON file."""
        try:
            checkpoint_data = {
                "pipeline_name": self.pipeline_name,
                "last_updated": datetime.now().isoformat(),
                "completed_stages": [stage for stage in self.stage_order if stage in self.completed_stages],
                "stage_metadata": self.stage_metadata,
                "dry_run_completed": self.dry_run_completed,
                "stage_directories": {stage: str(dir_path) for stage, dir_path in self.stage_directories.items()},
            }
            tmp_checkpoint = self.checkpoint_file.with_suffix(self.checkpoint_file.suffix + ".tmp")
            with open(tmp_checkpoint, "w") as f:
                json.dump(checkpoint_data, f, indent=4)
            tmp_checkpoint.replace(self.checkpoint_file)
            self.logger.info(f"Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def get_stage_directory(self, stage: str, base_name: Optional[str] = None, create: bool = True) -> Path:
        """
        Get or create the output directory for a stage.

        Args:
            stage (str): Name of the stage
            base_name (str, optional): Base name for the directory. If None,
                uses the stage name. Defaults to None.
            create (bool, optional): Whether to create the directory if it
                doesn't exist. Defaults to True.

        Returns:
            Path: Path to the stage's output directory

        Raises:
            ValueError: If stage is not a valid stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.stages)}")

        # If directory already exists for this stage, return it
        if stage in self.stage_directories:
            return self.stage_directories[stage]

        # For new stages or fresh analysis, create directory
        base_name = base_name or stage
        if self.is_resumed:
            # For resumed analysis, use the existing output directory
            stage_dir = self.output_dir / base_name
        else:
            # For fresh analysis, create incremental directory
            counter = 1
            while (self.output_dir / f"{base_name}_{counter}").exists():
                counter += 1
            stage_dir = self.output_dir / f"{base_name}_{counter}"

        if create:
            stage_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory for stage '{stage}': {stage_dir}")

        self.stage_directories[stage] = stage_dir
        self.save_checkpoint()
        return stage_dir

    def mark_stage_complete(self, stage: str, metadata: Optional[Dict] = None, dry_run: bool = False) -> None:
        """
        Mark a stage as complete and optionally store metadata.

        Args:
            stage (str): Name of the stage to mark as complete
            metadata (Dict, optional): Additional metadata for the stage
            dry_run (bool, optional): Whether the stage was run in dry-run mode

        Raises:
            ValueError: If stage is not a valid stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.stages)}")

        self.completed_stages.add(stage)
        if metadata:
            self.stage_metadata[stage] = {
                "completion_time": datetime.now().isoformat(),
                "dry_run": dry_run,
                **metadata,
            }
        self.save_checkpoint()
        safe_stage = str(stage).replace("\n", "").replace("\r", "")
        self.logger.info(f"Marked stage '{safe_stage}' as complete (dry_run={dry_run})")

    def mark_stage_incomplete(self, stage: str) -> None:
        """
        Mark a stage as incomplete.

        Args:
            stage (str): Name of the stage to mark as incomplete

        Raises:
            ValueError: If stage is not a valid stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.stages)}")

        self.completed_stages.discard(stage)
        self.stage_metadata.pop(stage, None)
        self.stage_directories.pop(stage, None)  # Remove directory mapping
        self.save_checkpoint()
        self.logger.info(f"Marked stage '{stage}' as incomplete")

    def is_stage_complete(self, stage: str) -> bool:
        """
        Check if a stage is marked as complete.

        Args:
            stage (str): Name of the stage to check

        Returns:
            bool: True if stage is complete, False otherwise

        Raises:
            ValueError: If stage is not a valid stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.stages)}")

        return stage in self.completed_stages

    def get_completed_stages(self) -> List[str]:
        """
        Get list of completed stages.

        Returns:
            List[str]: List of completed stage names
        """
        return [stage for stage in self.stage_order if stage in self.completed_stages]

    def get_incomplete_stages(self) -> List[str]:
        """
        Get list of incomplete stages.

        Returns:
            List[str]: List of incomplete stage names
        """
        return [stage for stage in self.stage_order if stage not in self.completed_stages]

    def get_stage_metadata(self, stage: str) -> Optional[Dict]:
        """
        Get metadata for a specific stage.

        Args:
            stage (str): Name of the stage

        Returns:
            Optional[Dict]: Stage metadata if available, None otherwise

        Raises:
            ValueError: If stage is not a valid stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.stages)}")

        return self.stage_metadata.get(stage)

    def reset_all_stages(self) -> None:
        """Reset all stages to incomplete state."""
        self.completed_stages.clear()
        self.stage_metadata.clear()
        self.stage_directories.clear()
        self.dry_run_completed = False
        self.is_resumed = False
        self.save_checkpoint()
        self.logger.info("Reset all stages to incomplete state")

    def get_pipeline_status(self) -> Dict:
        """
        Get overall pipeline status.

        Returns:
            Dict: Dictionary containing pipeline status information
        """
        return {
            "pipeline_name": self.pipeline_name,
            "total_stages": len(self.stages),
            "completed_stages": len(self.completed_stages),
            "incomplete_stages": len(self.stages - self.completed_stages),
            "completion_percentage": (len(self.completed_stages) / len(self.stages)) * 100,
            "last_updated": datetime.now().isoformat(),
            "dry_run_completed": self.dry_run_completed,
            "is_resumed": self.is_resumed,
            "stage_details": {
                stage: {
                    "status": "complete" if stage in self.completed_stages else "incomplete",
                    "metadata": self.stage_metadata.get(stage),
                    "directory": str(self.stage_directories.get(stage)),
                }
                for stage in self.stages
            },
        }

    def resume_pipeline(
        self, dry_run_completed: bool = True, start_from: Optional[str] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Resume pipeline from last completed stage or specified stage.

        Args:
            dry_run_completed (bool, optional): Whether to run completed stages
                in dry-run mode. Defaults to True.
            start_from (str, optional): Stage to start from. If None, starts
                from first incomplete stage. Defaults to None.

        Returns:
            Tuple[List[str], List[str], List[str]]: Lists of stages to run in
                normal mode, dry-run mode, and stages to skip, respectively.

        Raises:
            ValueError: If start_from is not a valid stage name
        """
        if start_from and start_from not in self.stages:
            raise ValueError(f"Invalid start_from stage: {start_from}. Must be one of {list(self.stages)}")

        # Determine stages to run
        stages_to_run = []
        stages_to_dry_run = []
        stages_to_skip = []

        # If starting from a specific stage, mark all previous stages as completed
        if start_from:
            start_index = self.stage_order.index(start_from)
            for stage in self.stage_order[:start_index]:
                if stage not in self.completed_stages:
                    self.mark_stage_complete(stage, dry_run=True)
                stages_to_skip.append(stage)

        # Process remaining stages deterministically in pipeline order.
        for stage in self.stage_order:
            if stage in stages_to_skip:
                continue

            if stage in self.completed_stages:
                if dry_run_completed:
                    stages_to_dry_run.append(stage)
                else:
                    stages_to_skip.append(stage)
            else:
                stages_to_run.append(stage)

        self.dry_run_completed = dry_run_completed
        self.save_checkpoint()

        self.logger.info("Pipeline resume plan:")
        self.logger.info(f"Stages to run normally: {stages_to_run}")
        self.logger.info(f"Stages to run in dry-run mode: {stages_to_dry_run}")
        self.logger.info(f"Stages to skip: {stages_to_skip}")

        return stages_to_run, stages_to_dry_run, stages_to_skip

    def get_stage_dependencies(self, stage: str) -> List[str]:
        """
        Get list of stages that must be completed before the given stage.

        Args:
            stage (str): Name of the stage to check dependencies for

        Returns:
            List[str]: List of dependent stage names

        Raises:
            ValueError: If stage is not a valid stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.stages)}")

        dependencies = {
            "quality": [],
            "trimming": ["quality"],
            "quality_trim": ["trimming"],
            "alignment": ["trimming"],
            "bam_preparation": ["alignment"],
            "alignment_stats": ["bam_preparation"],
            "quantification": ["bam_preparation"],
            "multimapped_groups": ["quantification"],
            "normalization": ["quantification"],
            "sample_clustering": ["normalization"],
            "coexpression": ["normalization"],
            "differential": ["quantification"],
            "visualization": ["differential"],
            "gene_ontology": ["differential"],
            "pathway_enrichment": ["differential"],
        }

        return dependencies.get(stage, [])

    def validate_stage_sequence(self, stages: List[str]) -> bool:
        """
        Validate that the sequence of stages respects dependencies.

        Args:
            stages (List[str]): List of stages in execution order

        Returns:
            bool: True if sequence is valid, False otherwise
        """
        completed = set()
        for stage in stages:
            dependencies = self.get_stage_dependencies(stage)
            if not all(dep in completed for dep in dependencies):
                self.logger.error(f"Invalid stage sequence: {stage} requires {dependencies}")
                return False
            completed.add(stage)
        return True
