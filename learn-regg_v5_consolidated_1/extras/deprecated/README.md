# Deprecated (design v2, 2026-07-09)

- bigcnn/: the BigConvNN CIFAR-10 proxy experiment set (configs, PBS jobs,
  submit scripts, setup patcher, logs). SUPERSEDED: the CIFAR column now
  uses resnet8 as the meta-training proxy and ResNet-18 (param_chunk_size)
  at meta-test. Note: the AdSAM_sgd_bigcnn checkpoint produced by these
  jobs is an UNTRAINED LSTM (routing no-op, Change.MD item 20) — do not use.
  The BigConvNN class itself remains in model.py (harmless, unused).
- resnet_task/: legacy exploratory LO-on-ResNet code. Broken as shipped:
  ResNet-34 layout with a 10-class head on CIFAR-100, shifted
  fit_val_loss_acc() arguments (immediate crash), missing checkpoint
  (Change.MD item 64). The working large-model path is the dmo pipeline
  (model.py resnet18 + --set param_chunk_size=2000000).
