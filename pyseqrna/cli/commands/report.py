"""
PySeqRNA CLI Report Generation Command

This module implements the standalone report generation subcommand for the PySeqRNA CLI.
It reads analysis outcomes from a run directory and packages them into HTML, Markdown, or JSON reports.

Features:
    - Standalone command-line runner for multi-format report generation
    - Supports generating HTML, Markdown, and JSON report files
    - Automatically collects statistics and output paths from previous workflow steps
    - Configures custom report titles and links to reference genome and sample configuration files
    - Simulates report generation in dry-run mode to report expected output file targets

Configuration:
    - Configured via parsed command-line options passed in through argument objects.

Dependencies:
    - Python packages: pathlib
    - Internal modules: pyseqrna.modules.reporting

Classes / Functions / Exceptions:
    - run_report_subcommand: Standard entry point function for running standalone report generation.

:Created: May 20, 2021
:Updated: May 18, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from pathlib import Path


def run_report_subcommand(args, pyseqrna_logger) -> int:
    """Generate a standalone comprehensive report from an existing run directory."""
    from ...modules.reporting import ReportGenerator

    pipeline_dir = Path(args.pipeline_dir).resolve()
    if not pipeline_dir.exists():
        raise ValueError(f"Pipeline output directory not found: {pipeline_dir}")

    report_dir = Path(args.outdir).resolve() if args.outdir else pipeline_dir / "7.Report"
    formats = ReportGenerator.parse_formats(args.formats)

    pyseqrna_logger.info("Generating standalone PySeqRNA report")
    pyseqrna_logger.info(f"Pipeline directory: {pipeline_dir}")
    pyseqrna_logger.info(f"Report directory: {report_dir}")
    pyseqrna_logger.info(f"Report formats: {', '.join(formats)}")

    if args.dryrun:
        for fmt in formats:
            suffix = "md" if fmt == "md" else fmt
            pyseqrna_logger.info(f"DRYRUN: Would write {report_dir / f'pyseqrna_report.{suffix}'}")
        return 0

    generator = ReportGenerator(
        output_dir=pipeline_dir,
        report_dir=report_dir,
        title=args.title,
        input_file=args.input_file,
        samples_path=args.samples_path,
        reference_genome=args.reference_genome,
        feature_file=args.feature_file,
        logger=pyseqrna_logger.logger,
    )
    written = generator.generate(formats=formats)
    for fmt, output in written.items():
        pyseqrna_logger.info(f"Report saved ({fmt}): {output}")
    pyseqrna_logger.info("Standalone report completed successfully")
    return 0
