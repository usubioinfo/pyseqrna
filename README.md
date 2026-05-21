pySeqRNA
========

pySeqRNA: A Comprehensive Python-based RNA-Seq Data Analysis Package
------------------------------------------------------------------

**Version:** 1.0.0

pySeqRNA is an automated, modular, and flexible pipeline for RNA-Seq data analysis. It streamlines the process from raw reads to functional enrichment, integrating state-of-the-art tools with custom Python scripts to ensure reproducibility and efficiency.

Features
--------
*   **Comprehensive Workflow**: Covers Quality Control, Trimming, Alignment, Quantification, Differential Expression, and Functional Annotation.
*   **Modular Design**: Run the entire pipeline or specific stages.
*   **Resume Capability**: Smart checkpointing allows you to resume analysis from any stage without re-running completed steps.
*   **Dry Run Mode**: Preview the commands and workflow execution without modifying any files.
*   **HPC Support**: Built-in support for SLURM job scheduling.
*   **Flexible Configuration**: Customize tool parameters via configuration files or CLI arguments.
*   **Functional Annotation**: Integrated Gene Ontology (GO) and KEGG Pathway enrichment analysis.
*   **Multimapped Reads Analysis**: Specialized handling for multimapped reads groups.

Prerequisites
-------------
pySeqRNA requires **Python >= 3.8**.

**External Dependencies:**
The pipeline orchestrates the following tools. Ensure they are installed and in your system PATH:
*   **Quality Control**: FastQC
*   **Trimming**: Trim Galore, Trimmomatic, Flexbar
*   **Alignment**: STAR, HISAT2, Bowtie2, BWA, Minimap2
*   **Quantification**: featureCounts (Subread), HTSeq
*   **Differential Expression**: R (with DESeq2, edgeR packages)

**Python Dependencies:**
*   pandas >= 1.3.0
*   numpy >= 1.20.0
*   psutil >= 5.8.0
*   matplotlib
*   scipy

Installation
------------
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kaundal-lab/pyseqrna.git
    cd pyseqrna
    ```

2.  **Install the package:**
    ```bash
    pip install .
    ```
    *Or using setup.py:*
    ```bash
    python3 setup.py install
    ```

Input Format
------------
pySeqRNA requires a tab-delimited input file containing sample information. Lines starting with `#` are treated as comments.

**Template:**
```text
# Project Description
SampleName	Replication	Identifier	File1	File2
Sample1_Rep1	Rep1	ConditionA	Sample1_R1.fastq.gz	Sample1_R2.fastq.gz
Sample1_Rep2	Rep2	ConditionA	Sample2_R1.fastq.gz	Sample2_R2.fastq.gz
Sample2_Rep1	Rep1	ConditionB	Sample3_R1.fastq.gz	Sample3_R2.fastq.gz
```

*   **SampleName**: Unique name for the sample.
*   **Replication**: Replicate identifier (e.g., Rep1, Rep2).
*   **Identifier**: Group/Condition identifier for differential expression (e.g., Control, Treated).
*   **File1**: Path to Read 1 file.
*   **File2**: Path to Read 2 file (leave empty for single-end data).

Usage
-----
Basic command structure:
```bash
pyseqrna input_file samples_path reference_genome feature_file [options]
```

Differential expression only:
```bash
pyseqrna diffexp --counts Raw_Counts.xlsx --sample-info sample_info.tsv --outdir diffexp_out
```

Or derive conditions and comparisons from the standard PySeqRNA sample sheet:
```bash
pyseqrna diffexp --counts Raw_Counts.xlsx --input-file input_samples.txt --samples-path data_dir --outdir diffexp_out
```

### Mandatory Arguments
*   `input_file`: Tab-delimited file containing sample information.
*   `samples_path`: Directory containing raw read files.
*   `reference_genome`: Path to the reference genome FASTA file.
*   `feature_file`: Path to the annotation GTF/GFF file.

### Options

#### General
*   `--outdir`: Output directory name (default: `pySeqRNA_results`).
*   `--dryrun`: Run in dry-run mode (show commands without executing).
*   `--force`: Force overwrite existing output directory.
*   `--paired`: Enable paired-end mode.
*   `--version`: Show version information.
*   `--organism`: Display supported organisms for functional annotation.

#### Species & Annotation
*   `--species`: Species name for functional annotation (e.g., `athaliana`).
*   `--organism-type`: Type of organism: `plants` (default) or `animals`.
*   `--source`: Annotation source: `ENSEMBL` (default) or `NCBI`.

#### Quality Control & Trimming
*   `--skip-quality`: Skip FastQC on raw reads.
*   `--quality-tool`: Tool to use (default: `fastqc`).
*   `--skip-trim`: Skip read trimming.
*   `--trimming-tool`: `trim_galore` (default), `trimmomatic`, or `flexbar`.
*   `--quality-trim`: Run FastQC on trimmed reads.

#### Alignment
*   `--skip-alignment`: Skip alignment stage.
*   `--alignment-tool`: `star` (default), `hisat2`, `bowtie2`, `bwa`, `minimap2`.

#### Quantification
*   `--skip-quantification`: Skip quantification.
*   `--quant-method`: `genomic_overlaps` (default), `featureCounts`, `htseq`.
*   `--run-multimapped-groups`: Run multimapped groups analysis.
*   `--mmg-min-count`: Min read count for multimapped groups (default: 100).
*   `--skip-normalization`: Skip count normalization.
*   `--normalization-method`: `rpkm` (default), `cpm`, `tpm`, `fpkm`, `median_ratio`.

#### Differential Expression
*   `--skip-diffexp`: Skip differential expression analysis.
*   `--diffexp-tool`: `deseq2` (default), `edger`, `pydiffexpress`.
*   `--fdr-threshold`: FDR threshold (default: 0.05).
*   `--fold-threshold`: Fold change threshold (default: 2.0).
*   `--add-gene-names`: Add gene names/descriptions to results.

#### Functional Annotation
*   `--skip-functional-annotation`: Skip GO and KEGG analysis.
*   `--gene-ontology`: Enable Gene Ontology enrichment.
*   `--kegg-pathway`: Enable KEGG Pathway enrichment.
*   `--go-pvalue-threshold`: P-value cutoff for GO (default: 0.05).
*   `--kegg-pvalue-threshold`: P-value cutoff for KEGG (default: 0.05).

#### Computational & SLURM
*   `--threads`: Number of threads (default: auto-detect).
*   `--memory`: Memory in GB (default: auto-detect).
*   `--resume`: Resume from stage: `all`, `quality`, `trimming`, `alignment`, `quantification`, `normalization`, `differential`, `gene_ontology`, `pathway_enrichment`.
*   `--slurm`: Enable SLURM job scheduling.
*   `--slurm_partition`: SLURM partition name.
*   `--slurm_account`: SLURM account.
*   `--slurm_time`: Job time limit (default: 24:00:00).
*   `--slurm_email`: Email for notifications.

Output Structure
----------------
The pipeline creates a structured output directory (default: `pySeqRNA_results`):

*   `1.Quality_and_trimming/`: FastQC reports and trimmed reads.
*   `2.Alignment/`: Aligned BAM files and indices.
*   `3.Quantification/`: Count matrices and quantification reports.
*   `differential_expression/`: DEG lists, plots (Volcano, MA, PCA, Heatmap), and summary stats.
*   `gene_ontology/`: GO enrichment results (CSV) and plots (Bar, Dot).
*   `kegg_pathway/`: KEGG enrichment results (CSV) and plots.
*   `logs/`: Execution logs and error reports.
*   `.pyseqrna_checkpoint.json`: Checkpoint file for resume capability.

Citation
--------
Duhan N and Kaundal R. pySeqRNA: an automated Python package for RNA sequencing data analysis [version 1; not peer reviewed]. F1000Research 2020, 9(ISCB Comm J):1128 (poster) (https://doi.org/10.7490/f1000research.1118314.1)

Contact
-------
**Naveen Duhan** (naveen.duhan@usu.edu)
Kaundal Bioinformatics Lab, Utah State University

**Rakesh Kaundal** (rkaundal@usu.edu)

Released under the terms of GNU General Public License v3.
