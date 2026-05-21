# Arabidopsis SRP010938 PySeqRNA Test Data

This folder contains a reproducible setup for the Arabidopsis/Pseudomonas paired-end RNA-seq dataset already represented in `/Users/naveen/Documents/phd/pyseqrna_test/data`.

The existing FASTQs in that folder are small subsets of roughly 90k-100k read pairs per sample. For differential-expression and functional-annotation testing, use the full SRA downloads instead.

## Design

- Study: SRP010938
- Samples: `SRR446027` through `SRR446044`
- Conditions/time points: Mock, Avr, and Vir at 1h, 6h, and 12h
- Replicates: two per condition/time point
- Input sheet: `samples_pe_template.txt`

## Download Full FASTQs

```bash
bash test_data/arabidopsis_srp010938/prepare_srp010938_data.sh \
  --reads all \
  --threads 4 \
  --outdir /Users/naveen/Documents/phd/pyseqrna_test/srp010938_full
```

## Local PySeqRNA Run

Use the complete NCBI TAIR10 reference already present in the test data folder:

```bash
pyseqrna \
  /Users/naveen/Documents/phd/pyseqrna_test/srp010938_full/input_samples.txt \
  /Users/naveen/Documents/phd/pyseqrna_test/srp010938_full \
  /Users/naveen/Documents/phd/pyseqrna_test/data/tair_genomic.fna \
  /Users/naveen/Documents/phd/pyseqrna_test/data/tair_genomic.gff \
  --paired \
  --outdir /Users/naveen/Documents/phd/pyseqrna_test/srp010938_full_run \
  --threads 4 \
  --memory 16 \
  --alignment-tool hisat2 \
  --quant-method genomic_overlaps \
  --normalization-method rpkm \
  --species athaliana \
  --organism-type plants \
  --gene-ontology \
  --kegg-pathway \
  --report-formats html,md,json \
  --force
```
