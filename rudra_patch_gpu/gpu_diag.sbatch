#!/bin/bash
#SBATCH -J v3gpudiag
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:08:00
#SBATCH -o logs/gpudiag.%j.out
#SBATCH -e logs/gpudiag.%j.err
# Read-only diagnosis of what else is on the GPU node.  sbatch [--exclude=ragpu004] this file from the project root.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1; mkdir -p logs
. "$ROOT/experiments/convnet4_sweep_v3_4datasets/slurm/env.sh"
echo "### host $(hostname) job $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
env | grep -i -E "^(CUDA|GPU|SLURM_JOB_GPUS|SLURM_GPUS|SLURM_STEP_GPUS|NVIDIA)" | sort
echo "### scontrol job"; scontrol show job -d "$SLURM_JOB_ID" | grep -i -E "gres|tres|nodes=|Features|Partition"
echo "### nvidia-smi"; nvidia-smi
echo "### compute apps (all processes the driver sees)"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total --format=csv
echo "### torch view"
python - <<'PY'
import torch
n = torch.cuda.device_count(); print("device_count", n)
for i in range(n):
    f, t = torch.cuda.mem_get_info(i); print(f"cuda:{i} {torch.cuda.get_device_name(i)} free {f/2**30:.1f} GB of {t/2**30:.1f} GB")
PY
echo "### processes on this node (visible to us)"
ps -eo pid,user,etime,rss,cmd --sort=-rss | grep -v -E "ps -eo|grep" | head -15
echo "### slurm jobs on this node"
squeue -w "$(hostname)" -o "%i %u %P %T %M %b %C" 2>/dev/null | head -20
echo "### pid 4145627?"; ps -p 4145627 -o pid,user,etime,cmd 2>&1 | head -3
