# Arabidopsis GSE97186 PySeqRNA Test Data

This folder documents a small, reproducible Arabidopsis thaliana RNA-seq test set for PySeqRNA local and SLURM testing.

## Recommended Dataset

Use GEO series GSE97186 / SRA study SRP102698 / BioProject PRJNA380922.

Why this dataset is useful:

- Arabidopsis thaliana, so TAIR10 references are small enough for local testing.
- Paired-end Illumina RNA-seq, matching the common PySeqRNA path.
- Clear biological design: control and acyl-CoA treatments in triplicate.
- Individual runs are moderate size, and some are small enough to download quickly.
- The same study can test quick local runs, SLURM runs, differential expression, normalization, visualization, and reporting.

Primary recommendation:

- Local quick test: `local4`, two control replicates vs two C18:0 CoA replicates.
- SLURM integration test: `slurm6`, three control replicates vs three C18:0 CoA replicates.

## Files

- `sra_runs.tsv`: curated run metadata for all 12 samples.
- `samples_local4_template.txt`: PySeqRNA sample sheet for 4 paired-end samples.
- `samples_slurm6_template.txt`: PySeqRNA sample sheet for 6 paired-end samples.
- `prepare_arabidopsis_test_data.sh`: download and subsample FASTQ files.

## Prepare FASTQs

The script requires `fasterq-dump` from SRA Toolkit. It uses Python from the active environment to keep the first N read pairs and gzip the tiny test FASTQs.

Local smoke-test data, 100k read pairs per sample:

```bash
bash test_data/arabidopsis_gse97186/prepare_arabidopsis_test_data.sh \
  --mode local4 \
  --reads 100000 \
  --outdir /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_local4
```

Full-data SLURM test set, no read-pair subsetting:

```bash
bash test_data/arabidopsis_gse97186/prepare_arabidopsis_test_data.sh \
  --mode slurm6 \
  --reads all \
  --outdir /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_slurm6
```

The script writes an `input_samples.txt` file into the output directory.

## Reference

Use TAIR10. Ensembl Plants currently lists Arabidopsis thaliana assembly TAIR10 and provides FASTA/GFF3 downloads. For strict reproducibility, release 52 links are stable and widely used:

```bash
curl -L -o Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz \
  http://ftp.ensemblgenomes.org/pub/plants/release-52/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz

curl -L -o Arabidopsis_thaliana.TAIR10.52.gff3.gz \
  http://ftp.ensemblgenomes.org/pub/plants/release-52/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.52.gff3.gz
```

Unzip them before running PySeqRNA:

```bash
gunzip -k Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
gunzip -k Arabidopsis_thaliana.TAIR10.52.gff3.gz
```

## Local PySeqRNA Run

```bash
pyseqrna \
  /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_local4/input_samples.txt \
  /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_local4 \
  /path/to/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
  /path/to/Arabidopsis_thaliana.TAIR10.52.gff3 \
  --paired \
  --outdir /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_local4_run \
  --threads 4 \
  --memory 16 \
  --skip-functional-annotation \
  --report-formats html,md,json
```

## SLURM PySeqRNA Run

```bash
pyseqrna \
  /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_slurm6/input_samples.txt \
  /Users/naveen/Documents/phd/pyseqrna_test/arabidopsis_gse97186_slurm6 \
  /path/to/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
  /path/to/Arabidopsis_thaliana.TAIR10.52.gff3 \
  --paired \
  --outdir /scratch/$USER/pyseqrna_arabidopsis_slurm6_run \
  --threads 8 \
  --memory 32 \
  --slurm \
  --slurm_partition compute \
  --slurm_time 08:00:00 \
  --skip-functional-annotation \
  --report-formats html,md,json
```

## Sources

- GEO GSE97186: Response of Arabidopsis thaliana seedlings to acyl-CoAs.
- SRA SRP102698 / BioProject PRJNA380922.
- Ensembl Plants: Arabidopsis thaliana TAIR10 genome and GFF3 annotation.
