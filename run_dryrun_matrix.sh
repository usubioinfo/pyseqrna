#!/usr/bin/env bash

set -euo pipefail

INPUT_FILE="${INPUT_FILE:-/Users/naveen/Documents/phd/pyseqrna_test/data/input_Sample_PE_test4.txt}"
SAMPLES_PATH="${SAMPLES_PATH:-/Users/naveen/Documents/phd/pyseqrna_test/data}"
REFERENCE_GENOME="${REFERENCE_GENOME:-/Users/naveen/Documents/phd/pyseqrna_test/data/tair10.fasta}"
FEATURE_FILE="${FEATURE_FILE:-/Users/naveen/Documents/phd/pyseqrna_test/data/tair10.gff}"
OUTPUT_BASE="${OUTPUT_BASE:-/Users/naveen/Documents/phd/pyseqrna_test/dryrun_matrix}"
PYSEQRNA_CMD="${PYSEQRNA_CMD:-pyseqrna}"
THREADS="${THREADS:-4}"
MEMORY="${MEMORY:-16}"

TRIMMERS=(trim_galore trimmomatic flexbar)
ALIGNERS=(star hisat2 bowtie2 bwa minimap2)
QUANTIFIERS=(genomic_overlaps featureCounts htseq)
DIFFEXP_TOOLS=(deseq2 edger pydiffexpress)

mkdir -p "$OUTPUT_BASE"

SUMMARY_FILE="$OUTPUT_BASE/summary.tsv"
{
  printf "trimmer\taligner\tquantifier\tdiffexp\tstatus\toutdir\n"
} > "$SUMMARY_FILE"

run_index=0
total_runs=$(( ${#TRIMMERS[@]} * ${#ALIGNERS[@]} * ${#QUANTIFIERS[@]} * ${#DIFFEXP_TOOLS[@]} ))

for trimmer in "${TRIMMERS[@]}"; do
  for aligner in "${ALIGNERS[@]}"; do
    for quantifier in "${QUANTIFIERS[@]}"; do
      for diffexp_tool in "${DIFFEXP_TOOLS[@]}"; do
        run_index=$((run_index + 1))
        run_name="trim_${trimmer}__align_${aligner}__quant_${quantifier}__de_${diffexp_tool}"
        outdir="$OUTPUT_BASE/$run_name"
        logfile="$outdir/run.log"

        mkdir -p "$outdir"

        echo "[$run_index/$total_runs] Running $run_name"

        if "$PYSEQRNA_CMD" \
          "$INPUT_FILE" \
          "$SAMPLES_PATH" \
          "$REFERENCE_GENOME" \
          "$FEATURE_FILE" \
          --paired \
          --trimming-tool "$trimmer" \
          --alignment-tool "$aligner" \
          --quant-method "$quantifier" \
          --diffexp-tool "$diffexp_tool" \
          --outdir "$outdir" \
          --threads "$THREADS" \
          --memory "$MEMORY" \
          --skip-functional-annotation \
          --dryrun \
          > "$logfile" 2>&1; then
          status="PASS"
        else
          status="FAIL"
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$trimmer" "$aligner" "$quantifier" "$diffexp_tool" "$status" "$outdir" \
          >> "$SUMMARY_FILE"
      done
    done
  done
done

echo
echo "Dry-run matrix complete."
echo "Summary: $SUMMARY_FILE"
