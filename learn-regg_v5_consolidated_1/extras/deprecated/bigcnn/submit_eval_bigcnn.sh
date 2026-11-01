#!/bin/bash
set -e
cd "$(dirname "$0")"
qsub pbs_scripts/eval_no_reg_bigcnn.pbs
qsub pbs_scripts/eval_sam_bigcnn.pbs
echo
echo 'Submitted. Check: qstat -u 2025pcs0034'
