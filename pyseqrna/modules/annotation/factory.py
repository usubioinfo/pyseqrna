#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Factory Functions for Gene Annotation Services

This module provides convenient factory functions for creating and initializing gene annotation
services, including gene descriptions, Gene Ontology (GO) enrichment, and KEGG pathway analysis.
It abstracts the setup details and handles logging and dry-run state propagation.

Features:
    - Factory function for GeneDescriptionService and GeneAnnotator instances
    - Simple instantiation of GeneOntology enrichment analyzers supporting Ensembl and NCBI keys
    - Configuration of KEGG Pathway enrichment analyzers with customizable organism types
    - Standardized options for setting species, organism type, taxonomy ID, and GFF paths
    - Unified passing of logging contexts and dry-run execution switches

Configuration:
    - Configured via parameters passed to individual factory functions (species, organism_type,
      key_type, taxid, gff, dryrun, logger, and dry_run_manager).

Dependencies:
    - Internal modules: pyseqrna.modules.annotation.gene_description, pyseqrna.modules.annotation.gene_ontology, pyseqrna.modules.annotation.pathway

Classes / Functions / Exceptions:
    - create_gene_description_service: Create a gene description service.
    - create_gene_annotator: Create a gene annotator with an underlying description service.
    - create_gene_annotator_from_service: Create a gene annotator from an existing service.
    - create_gene_ontology: Create a Gene Ontology enrichment analyzer.
    - create_pathway: Create a KEGG Pathway enrichment analyzer.

:Created: May 20, 2021
:Updated: March 10, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import logging
from typing import Optional, Any

from .gene_description import GeneDescriptionService, GeneAnnotator
from .gene_ontology import GeneOntology
from .pathway import Pathway


def create_gene_description_service(
    species: str, organism_type: str = "plants", logger: Optional[logging.Logger] = None
) -> GeneDescriptionService:
    """
    Create a gene description service.

    Args:
        species: Species identifier (e.g., 'athaliana' for Arabidopsis thaliana)
        organism_type: Type of organism - 'plants' or 'animals'
        logger: Optional logger instance

    Returns:
        Configured GeneDescriptionService instance

    Example:
        >>> service = create_gene_description_service('athaliana', 'plants')
        >>> descriptions = service.fetch_gene_descriptions()
    """
    if logger:
        logger.info(f"Creating gene description service for {species} ({organism_type})")

    return GeneDescriptionService(species=species, organism_type=organism_type)


def create_gene_annotator(
    species: str, organism_type: str = "plants", logger: Optional[logging.Logger] = None
) -> GeneAnnotator:
    """
    Create a gene annotator with an underlying description service.

    Args:
        species: Species identifier (e.g., 'athaliana' for Arabidopsis thaliana)
        organism_type: Type of organism - 'plants' or 'animals'
        logger: Optional logger instance

    Returns:
        Configured GeneAnnotator instance

    Example:
        >>> annotator = create_gene_annotator('athaliana', 'plants')
        >>> annotated_df = annotator.add_descriptions_to_dataframe(df)
    """
    if logger:
        logger.info(f"Creating gene annotator for {species} ({organism_type})")

    # Create the underlying service
    service = create_gene_description_service(species, organism_type, logger)

    return GeneAnnotator(service)


def create_gene_annotator_from_service(
    service: GeneDescriptionService, logger: Optional[logging.Logger] = None
) -> GeneAnnotator:
    """
    Create a gene annotator from an existing service.

    Args:
        service: Existing GeneDescriptionService instance
        logger: Optional logger instance

    Returns:
        GeneAnnotator instance using the provided service

    Example:
        >>> service = create_gene_description_service('athaliana', 'plants')
        >>> annotator = create_gene_annotator_from_service(service)
    """
    if logger:
        logger.info("Creating gene annotator from existing service")

    return GeneAnnotator(service)


def create_gene_ontology(
    species: str,
    organism_type: str = "plants",
    key_type: str = "ensembl",
    taxid: Optional[str] = None,
    gff: Optional[str] = None,
    dryrun: bool = False,
    logger: Optional[Any] = None,
    dry_run_manager: Optional[Any] = None,
) -> GeneOntology:
    """
    Create a Gene Ontology enrichment analyzer.

    Args:
        species: Species identifier (e.g., 'athaliana' for Arabidopsis thaliana)
        organism_type: Type of organism - 'plants' or 'animals'
        key_type: Gene ID type - 'ensembl' or 'ncbi'
        taxid: Taxonomy ID (required if key_type is 'ncbi')
        gff: Path to GFF/GTF annotation file (required if key_type is 'ncbi')
        dryrun: Whether to perform a dry run (no actual file operations)
        logger: Optional logger instance
        dry_run_manager: Dry run manager instance

    Returns:
        Configured GeneOntology instance

    Example:
        >>> go = create_gene_ontology('athaliana', 'plants', 'ensembl')
        >>> results = go.enrich_go('deg_genes.txt')

    Example with NCBI IDs:
        >>> go = create_gene_ontology('athaliana', 'plants', 'ncbi',
        ...                          taxid='3702', gff='genes.gff')
        >>> results = go.enrich_go('deg_genes.csv')
    """
    if logger:
        logger.info(f"Creating Gene Ontology analyzer for {species} ({organism_type}, {key_type})")

    return GeneOntology(
        species=species,
        organism_type=organism_type,
        key_type=key_type,
        taxid=taxid,
        gff=gff,
        dryrun=dryrun,
        logger=logger,
        dry_run_manager=dry_run_manager,
    )


def create_pathway(
    species: str,
    organism_type: str = "plants",
    key_type: str = "ensembl",
    gff: Optional[str] = None,
    dryrun: bool = False,
    logger: Optional[Any] = None,
    dry_run_manager: Optional[Any] = None,
) -> Pathway:
    """
    Create a KEGG Pathway enrichment analyzer.

    Args:
        species: Species identifier (e.g., 'ath' for Arabidopsis thaliana)
        organism_type: Type of organism - 'plants' or 'animals'
        key_type: Gene ID type - 'ensembl' or 'ncbi'
        gff: Path to GFF/GTF annotation file (required if key_type is 'ncbi')
        dryrun: Whether to perform a dry run (no actual file operations)
        logger: Optional logger instance
        dry_run_manager: Dry run manager instance

    Returns:
        Configured Pathway instance

    Example:
        >>> pathway = create_pathway('ath', 'plants', 'ensembl')
        >>> results = pathway.enrich_kegg('deg_genes.txt')

    Example with NCBI IDs:
        >>> pathway = create_pathway('ath', 'plants', 'ncbi', gff='genes.gff')
        >>> results = pathway.enrich_kegg('deg_genes.csv')
    """
    if logger:
        logger.info(f"Creating KEGG Pathway analyzer for {species} ({organism_type}, {key_type})")

    return Pathway(
        species=species,
        type=organism_type,
        keyType=key_type,
        gff=gff,
        dryrun=dryrun,
        logger=logger,
        dry_run_manager=dry_run_manager,
    )


__all__ = [
    "create_gene_description_service",
    "create_gene_annotator",
    "create_gene_annotator_from_service",
    "create_gene_ontology",
    "create_pathway",
]
