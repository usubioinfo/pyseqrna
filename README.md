# PySeqRNA

[![PySeqRNA](https://img.shields.io/badge/PySeqRNA-1.0.0-2f6f73.svg)](https://github.com/navduhan/pyseqrna)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-green.svg)](#license)
[![Tests](https://github.com/navduhan/pyseqrna/actions/workflows/tests.yml/badge.svg)](https://github.com/navduhan/pyseqrna/actions/workflows/tests.yml)
[![Lint](https://github.com/navduhan/pyseqrna/actions/workflows/lint.yml/badge.svg)](https://github.com/navduhan/pyseqrna/actions/workflows/lint.yml)

PySeqRNA is a production-oriented RNA-seq analysis pipeline for moving from raw reads to interpretable biological results. Version 1.0.0 restructures the project around modular pipeline stages, INI-driven configuration, checkpoint-aware resume, local and SLURM execution, multimapped gene-group analysis, built-in differential expression, visualization, functional annotation, and report generation.

## Highlights

- End-to-end RNA-seq workflow: quality control, trimming, alignment, BAM preparation, alignment statistics, quantification, normalization, clustering, co-expression, differential expression, visualization, GO/KEGG annotation, and reporting.
- Modular CLI: run the full pipeline or use standalone subcommands for alignment, quantification, normalization, differential expression, visualization, annotation, clustering, and report generation.
- Production resume behavior: checkpointed stages, deterministic stage ordering, and configurable `resume_policy = skip|rerun|fail|prompt`.
- Local and HPC execution: local multiprocessing with `--local-jobs` plus SLURM array support for sample-level jobs.
- Modern analysis defaults: STAR alignment, genomic-overlap quantification, PyDiffExpress differential expression, sample clustering, PyCoexpression, Venn/UpSet plots, and HTML/Markdown/JSON reports.
- Multimapped gene groups: optional MMG count generation, filtering, differential expression, and downstream annotation support.

## Installation

PySeqRNA requires Python 3.10 or newer.

```bash
git clone git@github.com:navduhan/pyseqrna.git
cd pyseqrna
pip install -e .
```

For a fuller bioinformatics environment, use the included Conda environment as a starting point:

```bash
conda env create -f environment.yml
conda activate pyseqrna
pip install -e .
```

PySeqRNA orchestrates external tools where requested. Install the tools used by your selected workflow and ensure they are available in `PATH`.

Common external tools include FastQC, Trim Galore, Trimmomatic, Flexbar, STAR, HISAT2, Bowtie2, BWA, Minimap2, SAMtools, featureCounts/Subread, HTSeq, and optional R/Bioconductor packages for DESeq2 or edgeR.

## Quick Start

The recommended production entry point is an INI run configuration:

```bash
pyseqrna -c input_file.ini
```

CLI arguments can override values from the INI file:

```bash
pyseqrna -c input_file.ini --resume quantification --resume-policy skip --threads 32
```

You can also run the full pipeline using positional inputs:

```bash
pyseqrna input_samples.txt data/ reference.fa annotation.gff3 --paired --species athaliana
```

Use dry-run mode to inspect commands and planned stages before execution:

```bash
pyseqrna -c input_file.ini --dryrun
```

## Input Sample Sheet

PySeqRNA expects a tab-delimited sample sheet. Lines beginning with `#` are ignored.

```text
SampleName	Replication	Identifier	File1	File2
Mock.1h.A	M1A	M1	SRR446027_1.fastq.gz	SRR446027_2.fastq.gz
Mock.1h.B	M1B	M1	SRR446028_1.fastq.gz	SRR446028_2.fastq.gz
Avr.1h.A	A1A	A1	SRR446029_1.fastq.gz	SRR446029_2.fastq.gz
Avr.1h.B	A1B	A1	SRR446030_1.fastq.gz	SRR446030_2.fastq.gz
```

- `SampleName`: Descriptive sample name.
- `Replication`: Unique replicate/sample identifier used in output files.
- `Identifier`: Biological condition/group used for comparisons.
- `File1`: Read 1 FASTQ file.
- `File2`: Read 2 FASTQ file for paired-end data; omit or leave empty for single-end runs.

## Configuration

The repository includes a documented [input_file.ini](input_file.ini) template. Important sections include:

- `[General]`: input paths, output directory, paired-end mode, force/dry-run behavior, and resume policy.
- `[Quality]`: FastQC and trimming switches.
- `[Alignment]`: aligner selection and alignment statistics source.
- `[Quantification]`: quantification method and multimapped gene-group settings.
- `[Normalization]`: normalization method and plot settings.
- `[Clustering]`: sample similarity clustering settings.
- `[Coexpression]`: PyCoexpression settings.
- `[DifferentialExpression]`: PyDiffExpress, DESeq2, or edgeR settings.
- `[Visualization]`: PCA, t-SNE, volcano, MA, heatmap, Venn, and UpSet switches.
- `[FunctionalAnnotation]`: GO and KEGG enrichment settings.
- `[Report]`: HTML, Markdown, JSON, DOCX, or PDF report output.
- `[SLURM]`: partition, account, array parallelism, task CPU/memory, and scheduler timeout settings.

## Main CLI Options

```bash
pyseqrna --help
```

Core options:

- `--outdir`: output directory.
- `--paired`: enable paired-end mode.
- `--resume`: resume from a stage such as `quality`, `alignment`, `quantification`, `differential`, or `annotation`.
- `--resume-policy`: choose `skip`, `rerun`, `fail`, or `prompt` for completed stages.
- `--threads`, `--memory`, `--local-jobs`: local resource controls.
- `--slurm`: enable SLURM job submission.
- `--slurm-array-max-parallel`: maximum concurrent SLURM array tasks.
- `--alignment-tool`: `star`, `hisat2`, `bowtie2`, `bwa`, or `minimap2`.
- `--alignment-stats-source`: `auto`, `logs`, or `bam`.
- `--quant-method`: `genomic_overlaps`, `featureCounts`, or `htseq`.
- `--run-multimapped-groups`: enable MMG analysis.
- `--normalization-method`: `rpkm`, `cpm`, `tpm`, `fpkm`, `median_ratio`, or `tmm`.
- `--diffexp-tool`: `pydiffexpress`, `deseq2`, or `edger`.
- `--gene-ontology`, `--kegg-pathway`: enable functional annotation.
- `--report-formats`: comma-separated formats such as `html,md,json`.

## Standalone Subcommands

PySeqRNA also exposes focused subcommands:

```bash
pyseqrna alignment --help
pyseqrna quantification --help
pyseqrna normalization --help
pyseqrna diffexp --help
pyseqrna visualization --help
pyseqrna annotation --help
pyseqrna clustering --help
pyseqrna report --help
```

Example differential expression-only run:

```bash
pyseqrna diffexp \
  --counts Raw_Counts.xlsx \
  --sample-info sample_info.tsv \
  --comparisons M1-A1,V1-A1 \
  --outdir diffexp_out
```

Or infer conditions and comparisons from the standard PySeqRNA sample sheet:

```bash
pyseqrna diffexp \
  --counts Raw_Counts.xlsx \
  --input-file input_samples.txt \
  --samples-path data \
  --outdir diffexp_out
```

## Output Layout

A full run creates a structured output directory:

```text
pyseqrna_project_run/
├── 1.Quality_and_trimming/
├── 2.Alignment/
│   ├── bam_preparation/
│   └── alignment_stats/
├── 3.Quantification/
│   └── multimapped_groups/
├── 4.Normalization/
├── 4.Differential_Expression/
├── 5.Clustering/
├── 5.Coexpression/
├── 5.Visualization/
├── 6.Functional_Annotation/
│   ├── Gene_Ontology/
│   └── KEGG_Pathway/
├── 7.Report/
├── logs/
└── pyseqrna_checkpoint.json
```

Reports summarize run metadata, completed stages, major output files, column descriptions, plots, and downstream interpretation helpers.

## SLURM Notes

For HPC runs, enable SLURM in `input_file.ini` or pass `--slurm`. Sample-level tools use SLURM arrays where supported. The main controls are:

```ini
[SLURM]
slurm = True
slurm_partition = compute
slurm_array_max_parallel = 10
slurm_cpus_per_task = 8
slurm_memory_per_task = 64
slurm_wait_timeout_hours = 72
```

Set `slurm_cpus_per_task = 0` and `slurm_memory_per_task = 0` to use PySeqRNA's stage-aware defaults.

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Run formatting checks:

```bash
black --check .
```

Build docs with warnings treated as errors:

```bash
python -m sphinx -b html -W docs /tmp/pyseqrna-docs
```

## Citation

Duhan N and Kaundal R. pySeqRNA: an automated Python package for RNA sequencing data analysis [version 1; not peer reviewed]. F1000Research 2020, 9(ISCB Comm J):1128 (poster). https://doi.org/10.7490/f1000research.1118314.1

## Authors

Naveen Duhan, Kaundal Bioinformatics Lab, Utah State University.

## License

Released under the terms of the GNU General Public License v3.
