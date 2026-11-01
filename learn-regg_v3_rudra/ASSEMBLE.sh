#!/bin/bash
# Build the Rudra project root from the three uploaded zips (run in the directory that holds them):
#   learn-regg_v3_rudra_code.zip   learn-regg_v3_rudra_newdata.zip   (cifar-10 + SVHN are copied from the OLD Rudra project)
#   bash ASSEMBLE.sh [OLD_ROOT]     OLD_ROOT defaults to ~/Rego/convnet4_rudra/learn-regg_convnet4_sweep_rudra
# Nothing existing is modified: the new root is a separate directory  learn-regg_v3_rudra/.
set -eu
OLD="${1:-$HOME/Rego/convnet4_rudra/learn-regg_convnet4_sweep_rudra}"
unzip -q -n learn-regg_v3_rudra_code.zip
unzip -q -n learn-regg_v3_rudra_newdata.zip
cd learn-regg_v3_rudra
mkdir -p datasets
for f in cifar-10-batches-py train_32x32.mat test_32x32.mat; do
    if [ -e "$OLD/datasets/$f" ]; then [ -e "datasets/$f" ] || cp -r "$OLD/datasets/$f" datasets/; echo "have $f (from the old Rudra project)"
    else echo "!! $OLD/datasets/$f not found - upload learn-regg_v3_rudra_with_data.zip instead (or give OLD_ROOT)"; fi
done
echo "== checking checksums"; md5sum -c --quiet MD5SUMS_datasets.txt && echo "ALL DATASET FILES OK"
echo "== next:  cd $(pwd) && bash experiments/convnet4_sweep_v3_4datasets/slurm/setup_env.sh"
