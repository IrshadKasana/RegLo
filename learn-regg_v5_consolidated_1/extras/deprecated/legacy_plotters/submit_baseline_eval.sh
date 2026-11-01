#!/bin/bash
# Submits the AdamW+SAM baseline eval + compute benchmark.
set -e
cd "$(dirname "$0")"
qsub pbs_scripts/eval_adamw_sam.pbs
qsub pbs_scripts/bench_compute.pbs
echo
echo 'Submitted. Check with: qstat -u 2025pcs0034'
