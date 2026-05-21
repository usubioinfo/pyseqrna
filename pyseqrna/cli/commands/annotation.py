"""
PySeqRNA CLI Functional Annotation Command

This module implements the standalone functional annotation subcommand for the PySeqRNA CLI.
It reads lists of target genes and executes Gene Ontology (GO) and KEGG pathway enrichment analyses.

Features:
    - Standalone command-line runner for GO and KEGG pathway enrichment
    - Extracts gene identifiers from text, CSV, TSV, or Excel format files
    - Directs results to separate structured directories for GO and KEGG
    - Configures enrichment significance cutoffs, visualization types, and plots
    - Supports dry-run simulation mode to verify parsed options and files before execution

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib
    - Internal modules: pyseqrna.cli.utils, pyseqrna.modules.annotation.factory, pyseqrna.utils.dry_run_manager

Classes / Functions / Exceptions:
    - run_annotation_subcommand: Standard entry point function for running standalone GO and/or KEGG functional annotation.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
from ..utils import (
    _collect_annotation_gene_files,
    _prepare_annotation_inputs,
)


def run_annotation_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone GO and/or KEGG functional annotation."""
    from ...modules.annotation.factory import create_gene_ontology, create_pathway
    from ...utils.dry_run_manager import DryRunManager

    gene_files = _collect_annotation_gene_files(args)
    if not gene_files:
        raise ValueError("No annotation input files found.")

    missing = [str(path) for path in gene_files if not path.exists()]
    if missing:
        raise ValueError("Annotation input file(s) not found: " + ", ".join(missing))

    outdir = Path(args.outdir)
    go_dir = outdir / "6.Functional_Annotation" / "Gene_Ontology"
    kegg_dir = outdir / "6.Functional_Annotation" / "KEGG_Pathway"
    if not args.dryrun:
        outdir.mkdir(parents=True, exist_ok=True)
        if args.go:
            go_dir.mkdir(parents=True, exist_ok=True)
        if args.kegg:
            kegg_dir.mkdir(parents=True, exist_ok=True)
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create annotation output directory: {outdir}")

    pyseqrna_logger.info("Running standalone functional annotation")
    pyseqrna_logger.info(f"Species: {args.species}")
    pyseqrna_logger.info(f"Organism type: {args.organism_type}")
    pyseqrna_logger.info("Gene ID type: ensembl")
    pyseqrna_logger.warning("GO/KEGG standalone annotation currently expects Ensembl-style gene IDs.")

    prepared_inputs = _prepare_annotation_inputs(gene_files, outdir, args.dryrun)
    dry_run_manager = DryRunManager(enabled=args.dryrun, logger=pyseqrna_logger.logger)

    if args.dryrun:
        for item in prepared_inputs:
            if args.go:
                pyseqrna_logger.info(
                    "DRYRUN: Would run GO enrichment for %s (%d genes) -> %s",
                    item["source"].name,
                    item["gene_count"],
                    go_dir,
                )
            if args.kegg:
                pyseqrna_logger.info(
                    "DRYRUN: Would run KEGG enrichment for %s (%d genes) -> %s",
                    item["source"].name,
                    item["gene_count"],
                    kegg_dir,
                )
        pyseqrna_logger.info("Standalone functional annotation dry-run completed successfully")
        return 0

    go_analyzer = None
    pathway_analyzer = None
    if args.go:
        go_analyzer = create_gene_ontology(
            species=args.species,
            organism_type=args.organism_type,
            key_type=args.key_type,
            dryrun=args.dryrun,
            logger=pyseqrna_logger.logger,
            dry_run_manager=dry_run_manager,
        )
    if args.kegg:
        pathway_analyzer = create_pathway(
            species=args.species,
            organism_type=args.organism_type,
            key_type=args.key_type,
            dryrun=args.dryrun,
            logger=pyseqrna_logger.logger,
            dry_run_manager=dry_run_manager,
        )

    go_success = 0
    kegg_success = 0
    for item in prepared_inputs:
        pyseqrna_logger.info(f"Processing annotation input: {item['source'].name} ({item['gene_count']} genes)")
        if go_analyzer:
            result = go_analyzer.enrichGO(
                file=str(item["go_file"]),
                pvalueCutoff=args.go_pvalue_threshold,
                plot=not args.no_plots,
                plotType=args.plot_type,
                nrows=args.nrows,
                outdir=str(go_dir),
                colorBy=args.color_by,
            )
            if result != "No Gene Ontology results.":
                go_success += 1
        if pathway_analyzer:
            result = pathway_analyzer.enrichKEGG(
                file=str(item["kegg_file"]),
                pvalueCutoff=args.kegg_pvalue_threshold,
                plot=not args.no_plots,
                plotType=args.plot_type,
                nrows=args.nrows,
                outdir=str(kegg_dir),
                colorBy=args.color_by,
            )
            if result != "No Pathways results.":
                kegg_success += 1

    if args.go:
        pyseqrna_logger.info(f"GO enrichment completed for {go_success}/{len(prepared_inputs)} files")
    if args.kegg:
        pyseqrna_logger.info(f"KEGG enrichment completed for {kegg_success}/{len(prepared_inputs)} files")
    pyseqrna_logger.info("Standalone functional annotation completed successfully")
    return 0
