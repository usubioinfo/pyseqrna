"""
PySeqRNA CLI Differential Expression Command

This module implements the standalone differential expression subcommand for the PySeqRNA CLI.
It conducts statistical tests to identify differentially expressed genes (DEGs) using DESeq2, edgeR, or PyDiffExpress.

Features:
    - Standalone command-line runner for differential expression analysis
    - Interfaces with DESeq2, edgeR, and PyDiffExpress statistical backends
    - Automatically parses or infers sample combinations and target comparisons
    - Supports user-defined FDR, fold-change, and P-value threshold filtering
    - Outputs summary metrics regarding total assessed genes and significant up/down DEGs

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib, pandas, numpy
    - Internal modules: pyseqrna.utils, pyseqrna.cli.argument_manager, pyseqrna.modules.diffexp

Classes / Functions / Exceptions:
    - _load_tabular_dataframe: Helper function to load tabular data from Excel, CSV, TSV, or TXT.
    - run_diffexp_subcommand: Standard entry point function for running standalone differential expression.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path
import pandas as pd
import numpy as np
from ...utils import SupportedSpecies
from ..argument_manager import ArgumentManager


def _load_tabular_dataframe(file_path: str) -> pd.DataFrame:
    """Load a tabular file for standalone diffexp mode."""
    suffix = Path(file_path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(file_path, sep="\t")
    raise ValueError(f"Unsupported tabular file format: {file_path}")


def run_diffexp_subcommand(args, pyseqrna_logger) -> int:
    """Run standalone differential expression analysis."""
    from ...modules.diffexp import create_diffexp_analyzer
    from ...utils import InputProcessor

    if args.organism:
        SupportedSpecies().display_organisms()
        return 0

    if args.species:
        species_validator = SupportedSpecies(logger=pyseqrna_logger.logger)
        species_validator.validate_options(args)

    if args.input_file:
        input_processor = InputProcessor(pyseqrna_logger.logger)
        sample_data = input_processor.process_sample_file(
            args.input_file,
            args.samples_path,
            paired=args.paired,
        )
        sample_info = sample_data["targets"]
        comparisons = args.comparisons or sample_data.get("combinations", [])
    else:
        sample_info = _load_tabular_dataframe(args.sample_info_file)
        if "condition" not in sample_info.columns:
            raise ValueError("Sample info file must contain a 'condition' column")
        comparisons = args.comparisons or ArgumentManager.infer_comparisons_from_conditions(sample_info["condition"])

    if not comparisons:
        raise ValueError("No comparisons were provided or inferred. Use --comparisons or provide at least two conditions.")

    pyseqrna_logger.info("Running standalone differential expression analysis")
    pyseqrna_logger.info(f"Differential expression tool: {args.diffexp_tool}")
    if args.diffexp_tool == "pydiffexpress":
        pyseqrna_logger.info(
            "PyDiffExpress components: normalization=%s, abundance=%s, dispersion=%s, test=%s",
            args.diffexp_normalization,
            args.diffexp_abundance,
            args.diffexp_dispersion,
            args.diffexp_test,
        )
    pyseqrna_logger.info(f"Comparisons: {comparisons}")
    pyseqrna_logger.info(f"Count matrix: {args.counts}")
    pyseqrna_logger.info(f"Sample metadata source: {'input file' if args.input_file else 'sample info file'}")

    diffexp_dir = Path(args.outdir)
    if not args.dryrun:
        diffexp_dir.mkdir(parents=True, exist_ok=True)
        pyseqrna_logger.info(f"Created differential expression output directory: {diffexp_dir}")
    else:
        pyseqrna_logger.info(f"DRYRUN: Would create differential expression output directory: {diffexp_dir}")

    analyzer = create_diffexp_analyzer(
        tool_name=args.diffexp_tool,
        count_matrix_file=args.counts,
        sample_info_file=sample_info,
        comparisons=comparisons,
        out_dir=str(diffexp_dir),
        species=args.species,
        organism_type=args.organism_type,
        add_gene_names=args.add_gene_names,
        gene_column=args.gene_column,
        log2fc_threshold=np.log2(args.fold_threshold),
        fdr_threshold=args.fdr_threshold,
        subset=args.subset,
        dryrun=args.dryrun,
        logger=pyseqrna_logger.logger,
        param_dir=args.param_dir,
        diffexp_normalization=args.diffexp_normalization,
        diffexp_abundance=args.diffexp_abundance,
        diffexp_dispersion=args.diffexp_dispersion,
        diffexp_test=args.diffexp_test,
    )

    results = analyzer.run()
    stats = analyzer.get_summary_stats(results)

    pyseqrna_logger.info("Standalone differential expression completed successfully")
    pyseqrna_logger.info(f"  - Tool: {args.diffexp_tool}")
    pyseqrna_logger.info(f"  - Genes: {stats.get('total_genes', 0)}")
    pyseqrna_logger.info(f"  - Samples: {stats.get('total_samples', 0)}")
    pyseqrna_logger.info(f"  - Genes with significant results: {stats.get('genes_with_significant_results', 0)}")

    return 0
