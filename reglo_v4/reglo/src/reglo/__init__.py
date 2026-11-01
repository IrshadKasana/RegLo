"""RegLO — learned optimizers that internalise regularization.

The four methods compared in every experiment:

    M1  optimizee + AdamW
    M2  optimizee + AdamW + regularizer r          r in {L1, L2, SAM, GAM, GSAM}, computed every step
    M3  optimizee + LSTM optimizer                 meta-trained on L_meta
    M4  optimizee + LSTM optimizer + regularizer r meta-trained on L_meta + L_smooth + L_reg(r);
                                                   no regularizer computed when training the optimizee

Package layout
    reglo.study           the study design: stages, evaluation settings, metrics
    reglo.profiles        smoke vs production: every number that differs between them
    reglo.config          YAML composition and command-line overrides
    reglo.manifest        experiment files → validated run lists
    reglo.runner          a whole study on one machine (laptop, cloud GPUs)
    reglo.checks          stage-to-stage connectivity and progress
    reglo.data · models · regularizers · meta_training · training · diagnostics · reporting
    reglo.registry        name -> class registries used by the config files
    reglo.utils           paths, seeding/devices/exit codes, logging, checkpoints, run locks
"""

__version__ = "0.4.0"
