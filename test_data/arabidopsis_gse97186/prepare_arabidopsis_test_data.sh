#!/usr/bin/env bash
set -euo pipefail

MODE="local4"
READS="100000"
OUTDIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Prepare small Arabidopsis GSE97186 FASTQ subsets for PySeqRNA tests.

Required:
  --outdir DIR          Output directory for FASTQ files and input_samples.txt

Options:
  --mode local4|slurm6  Sample set to prepare [default: local4]
  --reads N|all         Read pairs to keep per sample, or all for full FASTQs [default: 100000]
  -h, --help            Show this help

Dependencies:
  fasterq-dump from SRA Toolkit
  python3 or python from the active environment
  gzip
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --reads)
      READS="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  echo "ERROR: --outdir is required" >&2
  usage >&2
  exit 1
fi

if [[ "$MODE" != "local4" && "$MODE" != "slurm6" ]]; then
  echo "ERROR: --mode must be local4 or slurm6" >&2
  exit 1
fi

if ! command -v fasterq-dump >/dev/null 2>&1; then
  echo "ERROR: fasterq-dump was not found. Install/load SRA Toolkit first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: python3 or python was not found." >&2
    exit 1
  fi
fi

mkdir -p "$OUTDIR"

if [[ "$MODE" == "local4" ]]; then
  RUNS=(SRR5388646 SRR5388647 SRR5388649 SRR5388650)
  cp "$SCRIPT_DIR/samples_local4_template.txt" "$OUTDIR/input_samples.txt"
else
  RUNS=(SRR5388646 SRR5388647 SRR5388648 SRR5388649 SRR5388650 SRR5388651)
  cp "$SCRIPT_DIR/samples_slurm6_template.txt" "$OUTDIR/input_samples.txt"
fi

echo "Preparing $MODE data in $OUTDIR"
if [[ "$READS" == "all" ]]; then
  echo "Keeping all read pairs per sample"
else
  echo "Keeping $READS read pairs per sample"
fi

for run in "${RUNS[@]}"; do
  r1_gz="$OUTDIR/${run}_1.fastq.gz"
  r2_gz="$OUTDIR/${run}_2.fastq.gz"
  if [[ -s "$r1_gz" && -s "$r2_gz" ]]; then
    echo "Already exists: $run"
    continue
  fi

  tmpdir="$OUTDIR/.tmp_${run}"
  mkdir -p "$tmpdir"
  echo "Downloading $run with fasterq-dump"
  fasterq-dump "$run" --split-files --threads 4 --outdir "$tmpdir"

  if [[ "$READS" == "all" ]]; then
    echo "Compressing full FASTQs for $run"
    gzip -c "$tmpdir/${run}_1.fastq" > "$r1_gz"
    gzip -c "$tmpdir/${run}_2.fastq" > "$r2_gz"
  else
    echo "Subsetting $run to $READS read pairs"
    "$PYTHON_BIN" - "$tmpdir/${run}_1.fastq" "$tmpdir/${run}_2.fastq" "$r1_gz" "$r2_gz" "$READS" <<'PY'
import gzip
import sys

r1_in, r2_in, r1_out, r2_out, reads = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

def copy_fastq(src, dst, n_reads):
    with open(src, "rt", encoding="utf-8", errors="replace") as inp, gzip.open(dst, "wt", encoding="utf-8") as out:
        for _ in range(n_reads):
            block = [inp.readline() for _ in range(4)]
            if not block[0]:
                break
            out.writelines(block)

copy_fastq(r1_in, r1_out, reads)
copy_fastq(r2_in, r2_out, reads)
PY
  fi

  rm -rf "$tmpdir"
done

echo "Done."
echo "Sample sheet: $OUTDIR/input_samples.txt"
