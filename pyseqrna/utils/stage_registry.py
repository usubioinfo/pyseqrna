#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Stage Registry Module

This module defines central pipeline stage metadata for PySeqRNA, allowing utilities
such as checkpointing to share stage ordering without importing the Pipeline class directly.

Features:
    - Centralized stage-to-method name mapping
    - Checkpoint ordering for normal and annotation-specific stages
    - Mapping of stage keys to user-facing labels

Functions:
    - get_pipeline_stages: Binds registered stage method names to a pipeline instance

:Created: May 20, 2021
:Updated: January 22, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

PIPELINE_STAGE_METHODS = (
    ("quality", "run_quality_control"),
    ("trimming", "run_trimming"),
    ("quality_trim", "run_quality_control_trim"),
    ("alignment", "run_alignment"),
    ("bam_preparation", "run_bam_preparation"),
    ("alignment_stats", "run_alignment_statistics"),
    ("quantification", "run_quantification"),
    ("multimapped_groups", "run_multimapped_groups"),
    ("normalization", "run_normalization"),
    ("sample_clustering", "run_sample_clustering"),
    ("coexpression", "run_coexpression_analysis"),
    ("differential", "run_differential_expression"),
    ("visualization", "run_visualization"),
    ("annotation", "run_annotation"),
)

PIPELINE_STAGE_NAMES = tuple(stage_name for stage_name, _method_name in PIPELINE_STAGE_METHODS)

ANNOTATION_CHECKPOINT_STAGES = (
    "gene_ontology",
    "pathway_enrichment",
)

CHECKPOINT_STAGE_ORDER = tuple(
    checkpoint_stage
    for stage_name, _method_name in PIPELINE_STAGE_METHODS
    for checkpoint_stage in (ANNOTATION_CHECKPOINT_STAGES if stage_name == "annotation" else (stage_name,))
)

AVAILABLE_STAGE_LABELS = {
    "quality": "Quality Control",
    "quality_trim": "Post-trimming Quality Control",
    "trimming": "Read Trimming",
    "alignment": "Read Alignment",
    "bam_preparation": "BAM Preparation",
    "alignment_stats": "Alignment Statistics",
    "multimapped_groups": "Multimapped Groups",
    "quantification": "Gene Quantification",
    "normalization": "Count Normalization",
    "sample_clustering": "Sample Clustering",
    "coexpression": "Gene Co-expression",
    "differential": "Differential Expression",
    "visualization": "Results Visualization",
    "gene_ontology": "Gene Ontology Enrichment",
    "pathway_enrichment": "KEGG Pathway Enrichment",
}


def get_pipeline_stages(pipeline):
    """Bind registered stage method names to a pipeline instance."""
    return [(stage_name, getattr(pipeline, method_name)) for stage_name, method_name in PIPELINE_STAGE_METHODS]
