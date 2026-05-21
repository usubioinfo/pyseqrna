#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Pipeline Annotation Runner

This module coordinates functional annotation steps, specifically Gene Ontology (GO)
enrichment and KEGG pathway enrichment, on differential expression outputs.
It manages execution flow, input checks, result directories, and checkpoint recording.

Features:
    - Gene Ontology (GO) enrichment analysis on identified DEGs
    - KEGG pathway enrichment analysis on identified DEGs
    - Automatic search and identification of comparison-specific DEG lists
    - Annotation output directory organization and management
    - Detailed checkpointing to support step-by-step pipeline recovery

Configuration:
    - Configured via Pipeline context properties (species, organism_type, source, gene_ontology, kegg_pathway, go_pvalue_threshold, kegg_pvalue_threshold, skip_functional_annotation).

Dependencies:
    - Python packages: pathlib
    - External R packages: clusterProfiler, pathview (loaded via underlying modules)

Classes / Functions / Exceptions:
    - AnnotationRunner: Runner class for functional annotation steps including GO and KEGG pathway enrichment.

:Created: May 20, 2021
:Updated: May 12, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path

from ...modules.annotation import create_gene_ontology, create_pathway


class AnnotationRunner:
    def __init__(self, pipeline_context):
        self.ctx = pipeline_context

    def run_annotation(self) -> bool:
        """
        Run functional annotation stages enabled by the user.

        Returns:
            True if all requested annotation stages complete successfully, False otherwise
        """
        if self.ctx.skip_functional_annotation:
            self.ctx.logger.info("Skipping functional annotation as requested")
            return True
        if not self.ctx.gene_ontology and not self.ctx.kegg_pathway:
            self.ctx.logger.info("No functional annotation analyses requested")
            return True
        if not self.ctx.species:
            self.ctx.logger.warning("Skipping functional annotation: species not specified")
            return True
        if not self.ctx.checkpoint_manager.is_stage_complete("differential"):
            recovered = self.ctx._discover_differential_results()
            if recovered:
                self.ctx.differential_expression_results = recovered["output_file"]
                self.ctx.checkpoint_manager.mark_stage_complete("differential", metadata=recovered, dry_run=self.ctx.dryrun)
                self.ctx.logger.info("Recovered differential expression results from existing files")
            else:
                self.ctx.logger.warning("Skipping functional annotation: differential expression outputs not found")
                return True

        success = True
        if self.ctx.gene_ontology:
            success = self.run_gene_ontology() and success
        else:
            self.ctx.logger.info("Skipping Gene Ontology enrichment as requested")

        if self.ctx.kegg_pathway:
            success = self.run_pathway_enrichment() and success
        else:
            self.ctx.logger.info("Skipping KEGG pathway enrichment as requested")

        return success

    def run_gene_ontology(self) -> bool:
        """
        Run Gene Ontology enrichment analysis on DEG files.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if gene ontology should be skipped
            if self.ctx.skip_functional_annotation or not self.ctx.gene_ontology:
                self.ctx.logger.info("Skipping Gene Ontology enrichment analysis")
                return True

            # Check if stage is already complete (resume logic)
            if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("gene_ontology"):
                go_metadata = self.ctx.checkpoint_manager.get_stage_metadata("gene_ontology")
                if go_metadata:
                    self.ctx.logger.info("Gene Ontology enrichment already completed")
                    self.ctx.logger.info(
                        f"  - Processed {go_metadata.get('deg_files_processed', 'N/A')}/{go_metadata.get('total_deg_files', 'N/A')} DEG files"
                    )
                    self.ctx.logger.info(f"  - Output directory: {go_metadata.get('output_directory', 'N/A')}")
                    return True

            # Check for required parameters
            if not self.ctx.species:
                self.ctx.logger.warning("Species not specified. Skipping Gene Ontology analysis.")
                return True

            # Record operation
            self.ctx._record_internal_operation(
                "gene_ontology_start",
                f"Starting Gene Ontology enrichment for species: {self.ctx.species}",
            )

            self.ctx.logger.info(f"Running Gene Ontology enrichment analysis for {self.ctx.species}")

            # Find DEG files from differential expression analysis
            diffexp_dir = Path(self.ctx.output_dir) / "4.Differential_Expression"
            if not diffexp_dir.exists():
                self.ctx.logger.warning("Differential expression directory not found. Skipping Gene Ontology analysis.")
                return True

            deg_files = self.ctx._find_annotation_deg_files(diffexp_dir)
            if not deg_files and self.ctx.dryrun and self.ctx.comparisons:
                deg_files = [diffexp_dir / "diff_genes" / f"{comparison}.txt" for comparison in self.ctx.comparisons]
                self.ctx.logger.info(
                    "DRYRUN: Would process %d simulated DEG file(s) for Gene Ontology analysis.",
                    len(deg_files),
                )
            elif not deg_files:
                self.ctx.logger.warning(
                    "No annotation-ready DEG gene-list files found in %s. Skipping Gene Ontology analysis.",
                    diffexp_dir / "diff_genes",
                )
                return True

            # Create Gene Ontology output directory
            go_dir = Path(self.ctx.output_dir) / "6.Functional_Annotation" / "Gene_Ontology"
            self.ctx._clean_and_create_directory(go_dir, "gene ontology")

            if self.ctx.dryrun:
                for deg_file in deg_files:
                    self.ctx.logger.info(
                        "DRYRUN: Would run Gene Ontology enrichment for %s and save results to %s",
                        deg_file.name,
                        go_dir,
                    )
                    self.ctx._record_internal_operation(
                        "gene_ontology_dryrun",
                        f"Would enrich Gene Ontology for {deg_file.name}",
                    )

                self.ctx.checkpoint_manager.mark_stage_complete(
                    "gene_ontology",
                    metadata={
                        "deg_files_processed": len(deg_files),
                        "total_deg_files": len(deg_files),
                        "output_directory": str(go_dir),
                        "species": self.ctx.species,
                        "organism_type": self.ctx.organism_type,
                    },
                    dry_run=self.ctx.dryrun,
                )
                self.ctx.logger.info(
                    "DRYRUN: Gene Ontology enrichment would process %d/%d DEG files",
                    len(deg_files),
                    len(deg_files),
                )
                return True

            # Create Gene Ontology analyzer
            try:
                go_analyzer = create_gene_ontology(
                    species=self.ctx.species,
                    organism_type=self.ctx.organism_type,
                    key_type=self.ctx.source.lower(),
                    gff=self.ctx.feature_file if self.ctx.source.lower() == "ncbi" else None,
                    dryrun=self.ctx.dryrun,
                    logger=self.ctx.logger,
                    dry_run_manager=self.ctx.dry_run_manager,
                )
            except Exception as e:
                self.ctx.logger.error(f"Failed to create Gene Ontology analyzer: {e}")
                return False

            # Process each DEG file
            success_count = 0
            for deg_file in deg_files:
                try:
                    self.ctx.logger.info(f"Processing DEG file: {deg_file.name}")

                    # Run enrichment analysis
                    result = go_analyzer.enrichGO(
                        file=str(deg_file),
                        pvalueCutoff=self.ctx.go_pvalue_threshold,
                        plot=True,
                        plotType="all",
                        nrows=20,
                        outdir=str(go_dir),
                        colorBy="logPvalues",
                    )

                    if result != "No Gene Ontology results.":
                        self.ctx.logger.info(f"Gene Ontology enrichment completed for {deg_file.name}")
                        success_count += 1
                    else:
                        self.ctx.logger.warning(f"No Gene Ontology results for {deg_file.name}")

                except Exception as e:
                    self.ctx.logger.error(f"Error processing {deg_file.name}: {e}")
                    continue

            # Record completion
            self.ctx._record_internal_operation(
                "gene_ontology_complete",
                f"Processed {success_count}/{len(deg_files)} DEG files successfully",
            )

            # Mark stage as complete
            if not self.ctx.dryrun:
                self.ctx.checkpoint_manager.mark_stage_complete(
                    "gene_ontology",
                    metadata={
                        "deg_files_processed": success_count,
                        "total_deg_files": len(deg_files),
                        "output_directory": str(go_dir),
                        "species": self.ctx.species,
                        "organism_type": self.ctx.organism_type,
                    },
                )

            self.ctx.logger.info(f"Gene Ontology enrichment completed for {success_count}/{len(deg_files)} DEG files")
            return True

        except Exception as e:
            self.ctx.logger.error(f"Gene Ontology enrichment failed: {e}")
            return False

    def run_pathway_enrichment(self) -> bool:
        """
        Run KEGG pathway enrichment analysis on DEG files.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if pathway enrichment should be skipped
            if self.ctx.skip_functional_annotation or not self.ctx.kegg_pathway:
                self.ctx.logger.info("Skipping KEGG pathway enrichment analysis")
                return True

            # Check if stage is already complete (resume logic)
            if self.ctx.resume != "all" and self.ctx.checkpoint_manager.is_stage_complete("pathway_enrichment"):
                pathway_metadata = self.ctx.checkpoint_manager.get_stage_metadata("pathway_enrichment")
                if pathway_metadata:
                    self.ctx.logger.info("KEGG pathway enrichment already completed")
                    self.ctx.logger.info(
                        f"  - Processed {pathway_metadata.get('deg_files_processed', 'N/A')}/{pathway_metadata.get('total_deg_files', 'N/A')} DEG files"
                    )
                    self.ctx.logger.info(f"  - Output directory: {pathway_metadata.get('output_directory', 'N/A')}")
                    return True

            # Check for required parameters
            if not self.ctx.species:
                self.ctx.logger.warning("Species not specified. Skipping KEGG pathway analysis.")
                return True

            # Record operation
            self.ctx._record_internal_operation(
                "pathway_enrichment_start",
                f"Starting KEGG pathway enrichment for species: {self.ctx.species}",
            )

            self.ctx.logger.info(f"Running KEGG pathway enrichment analysis for {self.ctx.species}")

            # Find DEG files from differential expression analysis
            diffexp_dir = Path(self.ctx.output_dir) / "4.Differential_Expression"
            if not diffexp_dir.exists():
                self.ctx.logger.warning("Differential expression directory not found. Skipping KEGG pathway analysis.")
                return True

            deg_files = self.ctx._find_annotation_deg_files(diffexp_dir)
            if not deg_files and self.ctx.dryrun and self.ctx.comparisons:
                deg_files = [diffexp_dir / "diff_genes" / f"{comparison}.txt" for comparison in self.ctx.comparisons]
                self.ctx.logger.info(
                    "DRYRUN: Would process %d simulated DEG file(s) for KEGG pathway analysis.",
                    len(deg_files),
                )
            elif not deg_files:
                self.ctx.logger.warning(
                    "No annotation-ready DEG gene-list files found in %s. Skipping KEGG pathway analysis.",
                    diffexp_dir / "diff_genes",
                )
                return True

            # Create pathway output directory
            pathway_dir = Path(self.ctx.output_dir) / "6.Functional_Annotation" / "KEGG_Pathway"
            self.ctx._clean_and_create_directory(pathway_dir, "kegg pathway")

            if self.ctx.dryrun:
                for deg_file in deg_files:
                    self.ctx.logger.info(
                        "DRYRUN: Would run KEGG pathway enrichment for %s and save results to %s",
                        deg_file.name,
                        pathway_dir,
                    )
                    self.ctx._record_internal_operation(
                        "pathway_enrichment_dryrun",
                        f"Would enrich KEGG pathways for {deg_file.name}",
                    )

                self.ctx.checkpoint_manager.mark_stage_complete(
                    "pathway_enrichment",
                    metadata={
                        "deg_files_processed": len(deg_files),
                        "total_deg_files": len(deg_files),
                        "output_directory": str(pathway_dir),
                        "species": self.ctx.species,
                        "organism_type": self.ctx.organism_type,
                    },
                    dry_run=self.ctx.dryrun,
                )
                self.ctx.logger.info(
                    "DRYRUN: KEGG pathway enrichment would process %d/%d DEG files",
                    len(deg_files),
                    len(deg_files),
                )
                return True

            # Create Pathway analyzer
            try:
                pathway_analyzer = create_pathway(
                    species=self.ctx.species,
                    organism_type=self.ctx.organism_type,
                    key_type=self.ctx.source.lower(),
                    gff=self.ctx.feature_file if self.ctx.source.lower() == "ncbi" else None,
                    dryrun=self.ctx.dryrun,
                    logger=self.ctx.logger,
                    dry_run_manager=self.ctx.dry_run_manager,
                )
            except Exception as e:
                self.ctx.logger.error(f"Failed to create Pathway analyzer: {e}")
                return False

            # Process each DEG file
            success_count = 0
            for deg_file in deg_files:
                try:
                    self.ctx.logger.info(f"Processing DEG file: {deg_file.name}")

                    # Run enrichment analysis
                    result = pathway_analyzer.enrichKEGG(
                        file=str(deg_file),
                        pvalueCutoff=self.ctx.kegg_pvalue_threshold,
                        plot=True,
                        plotType="all",
                        nrows=20,
                        outdir=str(pathway_dir),
                        colorBy="logPvalues",
                    )

                    if result != "No Pathways results.":
                        self.ctx.logger.info(f"KEGG pathway enrichment completed for {deg_file.name}")
                        success_count += 1
                    else:
                        self.ctx.logger.warning(f"No KEGG pathway results for {deg_file.name}")

                except Exception as e:
                    self.ctx.logger.error(f"Error processing {deg_file.name}: {e}")
                    continue

            # Record completion
            self.ctx._record_internal_operation(
                "pathway_enrichment_complete",
                f"Processed {success_count}/{len(deg_files)} DEG files successfully",
            )

            # Mark stage as complete
            if not self.ctx.dryrun:
                self.ctx.checkpoint_manager.mark_stage_complete(
                    "pathway_enrichment",
                    metadata={
                        "deg_files_processed": success_count,
                        "total_deg_files": len(deg_files),
                        "output_directory": str(pathway_dir),
                        "species": self.ctx.species,
                        "organism_type": self.ctx.organism_type,
                    },
                )

            self.ctx.logger.info(f"KEGG pathway enrichment completed for {success_count}/{len(deg_files)} DEG files")
            return True

        except Exception as e:
            self.ctx.logger.error(f"KEGG pathway enrichment failed: {e}")
            return False
