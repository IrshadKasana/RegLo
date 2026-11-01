"""
experiment_config.py
====================
YAML config loading + merging for the dmo pipeline.

MERGE PRECEDENCE (do not "fix" without reading this):
  train  : base/meta_train.yaml  <  regularizers/smooth.yaml
           <  regularizers/<name>.yaml  <  --set overrides
  eval   : base/optimizee_eval.yaml
           <  regularizers/<name>.yaml   (ONLY when --if_normal_optim)
           <  CLI argument values (argparse defaults included!)
           <  --set overrides
  i.e. at eval time a YAML key that also exists as a CLI argument is
  always clobbered by the CLI value; use --set to override reliably.
"""
from argparse import Namespace
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT
CONFIG_DIR = REPO_ROOT / "configs" / "dmo"
BASE_CONFIG_DIR = CONFIG_DIR / "base"
REGULARIZER_CONFIG_DIR = CONFIG_DIR / "regularizers"

# Aliases only. Any other <name> resolves to regularizers/<name>.yaml
# automatically (auto-discovery), so new experiment configs need no code edit.
REGULARIZER_CONFIGS = {
    "no_reg": "no_reg.yaml",
    "sam": "sam.yaml",
    "sam_opzee": "sam_opzee.yaml",
    "gam": "gam.yaml",
    "gsam": "gsam.yaml",
    "adamw_sam_baseline": "adamw_sam_baseline.yaml",
}

# Keys that may legitimately be introduced by --set even if a YAML forgot
# to declare them (kept deliberately small).
_EXTRA_ALLOWED_KEYS = {
    "rho",   # penalty-mode (Alg. 1) shared ascent radius; gam/gsam YAMLs
             # do not declare it but run_sweep now forwards it (2026-07-31)
    "lambda_reg", "reg_mode", "xi", "split_seed", "param_chunk_size",
    "use_meta_val", "allow_cpu", "eval_every", "completion_at",
    "lo_train_dataset", "eval_tag", "lo_train_arch",
    "eps_list", "val_fraction", "eval_cifar10c", "cifar10c_root",
    "cifar10c_severities",
    "experiment_name", "experiment_group", "benchmark_name",
    "train_dataset", "test_dataset", "train_architecture", "test_architecture",
    "resume", "max_hours", "overwrite", "pacs_num_workers", "curriculum_steps", "cl_patience", "calc_grad_norm", "pacs_holdout", "pacs_root", "pacs_img_size", "sharpness_every", "sharpness_track_eps",
}


def load_yaml_arguments(filename):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return config["train_arguments"]


def merge_arguments(*argument_groups):
    merged = {}
    for arguments in argument_groups:
        merged.update(arguments)
    return Namespace(**merged)


def regularizer_yaml(regularizer):
    """Resolve a regularizer/experiment name to its YAML path.

    Known aliases first, then auto-discovery of regularizers/<name>.yaml.
    """
    if regularizer in REGULARIZER_CONFIGS:
        return REGULARIZER_CONFIG_DIR / REGULARIZER_CONFIGS[regularizer]
    candidate = REGULARIZER_CONFIG_DIR / f"{regularizer}.yaml"
    if candidate.exists():
        return candidate
    valid = sorted(set(list(REGULARIZER_CONFIGS) +
                       [p.stem for p in REGULARIZER_CONFIG_DIR.glob("*.yaml")]))
    raise ValueError(
        f"Unknown regularizer '{regularizer}'. Expected one of: {', '.join(valid)}")


def _coerce(value):
    """yaml.safe_load leaves '1e-05' as a STRING (PyYAML needs a dot for
    exponent floats). Coerce numeric-looking strings to float/int."""
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            as_float = float(parsed)
        except (TypeError, ValueError):
            return parsed
        as_int = int(as_float)
        return as_int if as_int == as_float and ("e" not in parsed.lower()
                                                 and "." not in parsed) else as_float
    return parsed


def parse_overrides(pairs):
    """Turn ['smooth_lamb=0.5', 'rho=1e-05', 'if_smooth=True'] into a typed
    dict. Numeric strings (incl. scientific notation) become numbers."""
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, _, value = pair.partition("=")
        overrides[key.strip()] = _coerce(value.strip())
    return overrides


def validate_overrides(merged_namespace, overrides):
    """Raise on --set keys that neither exist in the merged config nor in
    the small allowlist of new keys — kills silent typo overrides."""
    known = set(vars(merged_namespace)) | _EXTRA_ALLOWED_KEYS
    unknown = [k for k in (overrides or {}) if k not in known]
    if unknown:
        raise ValueError(
            f"--set received unknown key(s): {unknown}. "
            f"Known keys come from the merged YAML/CLI config; "
            f"check for typos (e.g. 'optimzee_wd' vs 'optzee_wd').")


PROFILE_CONFIG_DIR = CONFIG_DIR / "profiles"


def profile_yaml(name, phase):
    """configs/dmo/profiles/<name>_<phase>.yaml — the SCALE of a run
    (epochs, steps, periods, seeds, hessian...). Keeping these in YAML is
    what lets the PBS scripts carry nothing but scheduler settings plus
    model_name / resume / max_hours."""
    if not name:
        return None
    f = PROFILE_CONFIG_DIR / f"{name}_{phase}.yaml"
    if not f.exists():
        avail = sorted({q.stem.rsplit("_", 1)[0]
                        for q in PROFILE_CONFIG_DIR.glob(f"*_{phase}.yaml")})
        raise ValueError(f"unknown --profile '{name}' for phase '{phase}'. "
                         f"Available: {avail or 'none'} (looked for {f})")
    return f


def load_train_args(regularizer, overrides=None, profile=None):
    groups = [
        load_yaml_arguments(BASE_CONFIG_DIR / "meta_train.yaml"),
        load_yaml_arguments(REGULARIZER_CONFIG_DIR / "smooth.yaml"),
        load_yaml_arguments(regularizer_yaml(regularizer)),
    ]
    pf = profile_yaml(profile, "train")          # scale profile beats defaults
    if pf:
        groups.append(load_yaml_arguments(pf))
    base = merge_arguments(*groups)
    validate_overrides(base, overrides)          # validate BEFORE merging
    return merge_arguments(vars(base), overrides or {})


def load_optimizee_args(cli_args, overrides=None):
    argument_groups = [load_yaml_arguments(BASE_CONFIG_DIR / "optimizee_eval.yaml")]

    if cli_args.if_normal_optim:
        argument_groups.append(load_yaml_arguments(regularizer_yaml(cli_args.regularizer)))

    argument_groups.append(vars(cli_args))
    # profile AFTER cli_args: argparse always supplies defaults, so a
    # profile placed earlier would be silently clobbered by them. An
    # explicit --set still wins (it is merged last, below).
    pf = profile_yaml(getattr(cli_args, "profile", None), "eval")
    if pf:
        argument_groups.append(load_yaml_arguments(pf))
    base = merge_arguments(*argument_groups)
    validate_overrides(base, overrides)          # validate BEFORE merging
    return merge_arguments(vars(base), overrides or {})


def print_arguments(args):
    for key, value in vars(args).items():
        print(f"{key}: {value}")