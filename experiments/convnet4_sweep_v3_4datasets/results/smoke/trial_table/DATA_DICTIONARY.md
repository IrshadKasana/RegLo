# Data dictionary (all accuracies / AUROCs are fractions 0-1; losses are mean NLL; times are seconds)

| column | meaning |
|---|---|
| trial | unique id: `m1_<ds>_lr<..>`, `m2_<ds>_<reg>_lr<..>_lam<..>[_rho<..>]`, `lo_<ds>_<reg>_ls<..>_lr<..>_mlr<..>[_rho<..>]_seed<..>` (= result file name with a method prefix) |
| phase | sweep phase that planned it: m1, m3, m4a, m5, m5blk, m5mlr, m5rho, m2, m2rho, final, (old plan) |
| method | M1 AdamW, M2 AdamW+lambda*R, M3 learned optimizer, M4a LO+smooth, M5 LO+smooth+R |
| source | dataset the model was trained / selected on (the ID dataset) |
| reg | regulariser R (l1, l2, sam, gsam, gam); `-` when none |
| adamw_lr | AdamW learning rate (M1/M2) |
| lam | M2 regularisation weight lambda |
| rho | perturbation radius of SAM / GSAM / GAM (M2, M5) |
| ls | lambda_smooth (M4a / M5); 0 for M3 |
| lam_reg | lambda_reg of M5 (the `lr` field of the lo files); 0 for M3 / M4a |
| meta_lr | Adam learning rate of the learned optimizer's own weights |
| meta_seed | meta-training seed of a learned-optimizer trial (0 = sweep, 1 / 2 = final seeds of the selected cells) |
| status | PASSED; PASSED_AFTER_RETRY (a failed / OOM attempt, then OK); FAILED; MISSING (never ran); EXTRA_NOT_IN_CURRENT_GRID |
| best_pick | method(:regulariser) this trial was selected for (validation accuracy of the source, seed 0) |
| attempts_ok / attempts_failed / oom_retries | counts parsed from the worker logs |
| n_inits, init_seeds | optimizee initialisations run (3) and their seeds |
| optimizee_steps | training steps of every optimizee (1000) |
| train_loss, train_acc | on the first 10 000 training images, final weights, mean over inits |
| val_loss, val_acc | validation split (20 %) of the source; the ONLY quantity selection uses |
| test_loss, test_acc | ID test split (10 %) of the source |
| id_auroc | macro one-vs-rest multiclass AUROC of the softmax output on the ID test split |
| ood_auroc_<t>, ood_acc_<t> | OOD AUROC (ID test vs target t's 20 % partition by max-softmax confidence) and accuracy under t's label ids |
| run_seconds | summed optimizee-run time of all inits |
| start_utc, end_utc, host, pbs_jobid, gpu, code_sha1, wall_seconds | provenance of the trial |
| weights_file / weights_exists / weights_bytes | final optimizee weights of all inits (`torch.load` -> init_seeds, states) |
| checkpoint / checkpoint_exists / checkpoint_bytes | learned-optimizer (LSTM) checkpoint of the best meta-training stage |
| meta_train_stages, meta_ckpt_saves, meta_ckpt_stage_used | curriculum stages run, stages that improved the checkpoint score, last stage saved |
| best_ckpt_score, meta_train_seconds, meta_unroll_steps_total, meta_loss_last_stage | meta-training summary |

Other tables: trials_per_init.csv (per init), trials_ood.csv (per OOD target), trials_curves.csv (per 100-step point incl. step 0),
trials_meta.csv (per meta-training stage), coverage.csv (planned vs the full Cartesian grid), selection_trace.csv (how each pick was chosen).
