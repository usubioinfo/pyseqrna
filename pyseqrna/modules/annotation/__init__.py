#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gene Annotation Module for PySeqRNA
===================================

This module provides comprehensive gene annotation functionality including:
- Gene descriptions from BioMart
- Gene Ontology (GO) enrichment analysis
- KEGG pathway enrichment analysis

Exceptions:
    AnnotationError - Base exception for annotation module errors.

Functions:
    get_available_annotation_services - Get list of available annotation services.
    get_default_annotation_service - Get the default annotation service name.

:Created: May 20, 2021
:Updated: March 10, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from typing import List

from .gene_description import GeneDescriptionService, GeneAnnotator
from .gene_ontology import GeneOntology, GeneOntologyError, GeneOntologyPlotter
from .pathway import Pathway, PathwayError, PathwayPlotter
from .factory import (
    create_gene_description_service,
    create_gene_annotator,
    create_gene_ontology,
    create_pathway,
)
from pyseqrna.__version__ import __version__


class AnnotationError(Exception):
    """Base exception for annotation module errors."""

    pass


# Available annotation services registry
AVAILABLE_SERVICES = {
    "gene_description": GeneDescriptionService,
    "gene_annotator": GeneAnnotator,
    "gene_ontology": GeneOntology,
    "pathway": Pathway,
}

# Default service
DEFAULT_SERVICE = "gene_annotator"


def get_available_annotation_services() -> List[str]:
    """
    Get list of available annotation services.

    Returns:
        List of available service names
    """
    return list(AVAILABLE_SERVICES.keys())


def get_default_annotation_service() -> str:
    """
    Get the default annotation service name.

    Returns:
        Default service name
    """
    return DEFAULT_SERVICE


__author__ = "Naveen Duhan"

__all__ = [
    "AnnotationError",
    "GeneDescriptionService",
    "GeneAnnotator",
    "GeneOntology",
    "GeneOntologyError",
    "GeneOntologyPlotter",
    "Pathway",
    "PathwayError",
    "PathwayPlotter",
    "create_gene_description_service",
    "create_gene_annotator",
    "create_gene_ontology",
    "create_pathway",
    "get_available_annotation_services",
    "get_default_annotation_service",
    "AVAILABLE_SERVICES",
    "DEFAULT_SERVICE",
]
