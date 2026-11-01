#!/bin/bash
set -e
cd "$(dirname "$0")"
qsub pbs_scripts/train_no_reg_bigcnn.pbs
qsub pbs_scripts/train_sam_bigcnn.pbs
qsub pbs_scripts/eval_adamw_sam_bigcnn.pbs
echo
echo 'Submitted 2 trainings + baseline eval. Check: qstat -u 2025pcs0034'
