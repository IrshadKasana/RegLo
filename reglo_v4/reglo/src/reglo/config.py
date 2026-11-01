"""Composable YAML configuration for every script in the project.

A run is described by one YAML file per *group*, merged on top of
``configs/default.yaml``:

    group           folder                    lands in
    --------------  ------------------------  --------------------
    dataset         configs/datasets/         cfg.dataset
    optimizee       configs/optimizees/       cfg.optimizee
    optimizer       configs/optimizers/       cfg.optimizer
    regularizer     configs/regularizers/     cfg.regularizer
    smooth          configs/regularizers/     cfg.smooth          (L_smooth, Stage 1 only)
    method          configs/methods/          cfg.method          (M1-M4, Stages 2–3)
    meta_training   configs/meta_training/    cfg.meta_training   (Stage 1)

Every run has one *entry* file: a ``method`` (training an optimizee) or a
``meta_training`` file (training the LSTM optimizer). The entry file's
``defaults:`` block names the other files it needs; the command line wins.
A file may start with ``extends: <name>`` to build on another file in the same
folder (``meta_training/m4_lstm_reg.yaml`` extends ``base.yaml``).

The *profile* (``configs/profiles/<name>.yaml``: production, smoke, smoke_offline — see
:mod:`reglo.profiles`) is merged next: it sets every number that differs between a smoke test
and the real study, which the group files leave as ``???``. Command-line overrides come last,
in three forms::

    group=name        choose a file            method=m2_adamw_reg regularizer=sam
    key.path=value    change an existing key   training.epochs=100 optimizer.lr=5e-4
    +key.path=value   add a new key            +notes=first_try

(``profile=<name>`` picks the profile; without it, ``$REGLO_PROFILE`` or production.)
Changing a key that does not exist raises ``ConfigError``: a typo never runs
silently. The composed config records what was chosen under ``cfg.selected``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf import __version__ as omegaconf_version
from omegaconf.errors import OmegaConfBaseException

from reglo.profiles import apply_to_run, load_profile

__all__ = [
    "ConfigError",
    "ENTRY_GROUPS",
    "GROUP_DIRS",
    "NO_REGULARIZER",
    "available",
    "compose",
    "compose_meta",
    "compose_run",
    "config_dir",
    "default_meta_tag",
    "experiment_names",
    "load_experiment",
    "load_group_file",
    "load_saved_config",
    "parse_overrides",
    "project_root",
    "save_config",
    "to_yaml",
]

PathLike = Union[str, Path]

#: config group -> folder under configs/ (dict order is the merge order)
GROUP_DIRS: Dict[str, str] = {
    "dataset": "datasets",
    "optimizee": "optimizees",
    "optimizer": "optimizers",
    "regularizer": "regularizers",
    "smooth": "regularizers",
    "method": "methods",
    "meta_training": "meta_training",
}

#: groups that can start a run: training an optimizee, or meta-training the LSTM
ENTRY_GROUPS: Tuple[str, ...] = ("method", "meta_training")

#: name of the "no regularizer" file used by M1, M3 and M3's meta-training
NO_REGULARIZER = "none"

_MISSING = "???"
_TOKEN = re.compile(r"^[A-Za-z0-9_.]+$")
_BASE_FILES = {"base"}  # shared bases: reachable through `extends:`, never chosen directly
_EXPERIMENTS_DIR = "experiments"
_SMOOTH = "smooth"  # regularizers/smooth.yaml is L_smooth, a meta-objective term — not a choice of r

#: groups sharing a folder see only their own files
_GROUP_FILTERS = {
    "regularizer": lambda name: name != _SMOOTH,
    "smooth": lambda name: name == _SMOOTH,
}


class ConfigError(ValueError):
    """The requested combination of config files and overrides is not a valid run."""


# --------------------------------------------------------------------------- locations


def project_root() -> Path:
    """Repository root: ``$REGLO_ROOT`` if set, else the folder that contains ``src/``."""
    env = os.environ.get("REGLO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Folder holding the YAML files: ``$REGLO_CONFIGS`` if set, else ``<root>/configs``."""
    env = os.environ.get("REGLO_CONFIGS")
    return Path(env).expanduser().resolve() if env else project_root() / "configs"


def _register_resolver(name: str, func) -> None:
    """OmegaConf 2.4 renamed ``register_new_resolver`` to ``register_resolver``; 2.3 only has the former."""
    major, minor = (int(x) for x in omegaconf_version.split(".")[:2])
    if (major, minor) >= (2, 4) and hasattr(OmegaConf, "register_resolver"):
        OmegaConf.register_resolver(name, func, replace=True)
    else:
        OmegaConf.register_new_resolver(name, func, replace=True)


# `${reglo_root:}` in default.yaml resolves to the repository root, wherever the job starts.
_register_resolver("reglo_root", lambda: project_root().as_posix())


def _folder(group: str, root: Optional[PathLike]) -> Path:
    if group not in GROUP_DIRS:
        raise ConfigError(f"Unknown config group '{group}'. Groups: {', '.join(GROUP_DIRS)}")
    return Path(root or config_dir()) / GROUP_DIRS[group]


def available(group: str, root: Optional[PathLike] = None) -> List[str]:
    """Names that can be chosen for ``group`` (the YAML file names in its folder)."""
    keep = _GROUP_FILTERS.get(group, lambda name: True)
    stems = (p.stem for p in _folder(group, root).glob("*.yaml"))
    return sorted(s for s in stems if s not in _BASE_FILES and keep(s))


def experiment_names(root: Optional[PathLike] = None) -> List[str]:
    """Experiment grids in ``configs/experiments/``."""
    folder = Path(root or config_dir()) / _EXPERIMENTS_DIR
    return sorted(p.stem for p in folder.glob("*.yaml"))


# --------------------------------------------------------------------------- loading


def _load_yaml(path: Path, _chain: Tuple[Path, ...] = ()) -> DictConfig:
    """Load one YAML file, merged over its ``extends:`` parent (same folder) if it names one."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        node = OmegaConf.load(path)
    except Exception as err:  # YAML syntax errors come from several libraries
        raise ConfigError(f"Could not parse {path}: {err}") from err
    if not isinstance(node, DictConfig):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    parent = node.pop("extends", None)
    if parent is None:
        return node
    parent_path = path.parent / f"{parent}.yaml"
    if parent_path == path or parent_path in _chain:
        raise ConfigError(f"Circular 'extends' chain through {path}")
    return OmegaConf.merge(_load_yaml(parent_path, _chain + (path,)), node)


def _load_choice(group: str, name: str, root: Optional[PathLike]) -> DictConfig:
    choices = available(group, root)
    if name in _BASE_FILES:
        raise ConfigError(f"'{name}' is a shared base file, not a choice. {group}= takes one of: {', '.join(choices)}")
    if name not in choices:
        raise ConfigError(f"Unknown {group} '{name}'. Choose one of: {', '.join(choices)}")
    return _load_yaml(_folder(group, root) / f"{name}.yaml")


def default_meta_tag(root: Optional[PathLike] = None) -> str:
    """The round a run uses when no ``meta_tag=`` is given (``meta_tag`` in configs/default.yaml)."""
    return str(_load_yaml(Path(root or config_dir()) / "default.yaml").get("meta_tag", ""))


def load_group_file(group: str, name: str, root: Optional[PathLike] = None) -> DictConfig:
    """One file of a group on its own, e.g. ``load_group_file("dataset", "svhn")`` (data preparation)."""
    return _load_choice(group, name, root)


def load_experiment(name: str, root: Optional[PathLike] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """Load an experiment grid from ``configs/experiments/<name>.yaml`` as plain Python data,
    at the scale of ``profile`` (default: the active profile): its seeds, Stage-4 settings and
    per-profile overrides (see :func:`reglo.profiles.apply_to_experiment`). ``spec["profile"]``
    records which.

    A grid is a list of runs, not a config: its override strings (which may contain
    ``${...}``) stay raw here and are resolved later, inside each run's own config.
    """
    path = Path(root or config_dir()) / _EXPERIMENTS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"Unknown experiment '{name}'. Choose one of: {', '.join(experiment_names(root))}")
    with open(path, encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as err:  # e.g. [a=[1,2]] inside a flow list: write such overrides as a block list
            raise ConfigError(f"{path} is not valid YAML: {err}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    if data.get("name") != name:  # manifests and audits refer to an experiment by this name
        raise ConfigError(f"{path}: 'name: {data.get('name')}' must equal the file name '{name}'")
    if "profile" in data:
        raise ConfigError(f"{path}: an experiment file runs under any profile; choose it with --profile, not in the file")
    from reglo.profiles import active_profile, apply_to_experiment

    return apply_to_experiment(data, profile or active_profile(), root)


# --------------------------------------------------------------------------- overrides


def parse_overrides(args: Iterable[str]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Split command-line arguments into (file selections, value overrides, new keys)."""
    selections: Dict[str, str] = {}
    values: List[str] = []
    additions: List[str] = []
    for raw in args:
        arg = raw.strip()
        if not arg:
            continue
        key, sep, value = arg.partition("=")
        if not sep or key in ("", "+"):
            raise ConfigError(f"Override '{raw}' must look like key=value")
        if key.startswith("+"):
            additions.append(f"{key[1:]}={value}")
        elif key in GROUP_DIRS:
            if not value:
                raise ConfigError(f"'{key}=' needs a file name, e.g. {key}=<name>")
            selections[key] = value
        else:
            values.append(arg)
    return selections, values, additions


def _apply_dotlist(cfg: DictConfig, dotlist: Sequence[str], allow_new: bool) -> None:
    try:
        patch = OmegaConf.from_dotlist(list(dotlist))
    except OmegaConfBaseException as err:
        raise ConfigError(f"Could not parse overrides {list(dotlist)}: {err}") from err
    try:
        if allow_new:
            with open_dict(cfg):
                cfg.merge_with(patch)
        else:
            cfg.merge_with(patch)
    except OmegaConfBaseException as err:
        key = getattr(err, "full_key", None) or str(err).splitlines()[0]
        if "selected" in str(key):
            raise ConfigError(f"'{key}' is read-only: choose files with group=name, e.g. regularizer=sam") from err
        raise ConfigError(
            f"Unknown or invalid key '{key}'. Check the spelling in the config files, "
            f"or use +{key}=... to add a new key on purpose."
        ) from err


# --------------------------------------------------------------------------- composition


def compose(entry_group: str, overrides: Sequence[str] = (), root: Optional[PathLike] = None) -> DictConfig:
    """Build the full configuration of one run.

    Args:
        entry_group: ``"method"`` (Stages 2–3) or ``"meta_training"`` (Stage 1).
        overrides: command-line style arguments, e.g. ``["method=m2_adamw_reg", "regularizer=sam"]``.
        root: alternative ``configs/`` folder (tests).

    Returns:
        A resolved, struct-mode ``DictConfig``.

    Raises:
        ConfigError: a missing choice, an unknown file or key, or an invalid combination
            (e.g. M1 with a regularizer, M3 with AdamW).
    """
    if entry_group not in ENTRY_GROUPS:
        raise ConfigError(f"entry_group must be one of {ENTRY_GROUPS}, got '{entry_group}'")
    selections, values, additions = parse_overrides(overrides)

    for other in ENTRY_GROUPS:
        if other != entry_group and other in selections:
            raise ConfigError(f"A {entry_group} run cannot also choose {other}=... (one entry file per run)")
    entry_name = selections.get(entry_group)
    if not entry_name:
        raise ConfigError(f"Choose a {entry_group}: {entry_group}=<{'|'.join(available(entry_group, root))}>")

    base = _load_yaml(Path(root or config_dir()) / "default.yaml")
    entry = _load_choice(entry_group, entry_name, root)
    if entry_group == "meta_training":  # a round meta-trains on its own source pair
        _select_round_source(selections, values, str(base.get("meta_tag", "")), root)

    chosen: Dict[str, Optional[str]] = {}
    for source in (base.pop("defaults", None), entry.pop("defaults", None)):
        if source is not None:
            chosen.update(OmegaConf.to_container(source))
    chosen.update(selections)
    chosen[entry_group] = entry_name

    unknown = [g for g in chosen if g not in GROUP_DIRS]
    if unknown:
        raise ConfigError(f"Unknown group(s) in a defaults block: {unknown}. Groups: {', '.join(GROUP_DIRS)}")
    missing = [g for g, name in chosen.items() if name in (None, "", _MISSING)]
    if missing:
        hints = []
        for group in missing:
            options = list(entry.get("allowed_regularizers", [])) if group == "regularizer" else []
            hints.append(f"{group}=<{'|'.join(options or available(group, root))}>")
        raise ConfigError(f"{entry_group} '{entry_name}' needs: " + "  ".join(hints))

    ordered = {g: chosen[g] for g in GROUP_DIRS if g in chosen}
    groups = {g: (entry if g == entry_group else _load_choice(g, name, root)) for g, name in ordered.items()}
    cfg = OmegaConf.merge(base, groups, {"selected": ordered})

    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg.selected, True)

    # the profile: after the group files, before every explicit override
    profile_name, values = _take_profile(values)
    profile = load_profile(profile_name, root)
    apply_to_run(cfg, profile, ordered, sorted(set(base.keys()) | set(GROUP_DIRS)), profile_name)
    cfg.profile = profile_name

    if values:
        _apply_dotlist(cfg, values, allow_new=False)
    if additions:
        _apply_dotlist(cfg, additions, allow_new=True)

    unset = sorted(OmegaConf.missing_keys(cfg))
    if unset:
        raise ConfigError("These values must be set (key=value on the command line): " + ", ".join(unset))
    try:
        OmegaConf.resolve(cfg)
    except OmegaConfBaseException as err:
        raise ConfigError(f"Could not resolve an interpolation: {err}") from err

    _validate(cfg, entry_group, profile)
    return cfg


def _select_round_source(selections: Dict[str, str], values: Sequence[str], default_tag: str,
                         root: Optional[PathLike]) -> None:
    """``meta_tag=<round>`` (or the default meta_tag) names a round of configs/rounds.yaml: choose its
    source optimizee and dataset; an explicit choice that contradicts it is an error."""
    from reglo.rounds import get_round

    tag = default_tag
    for item in values:
        key, _, value = item.partition("=")
        if key.strip() == "meta_tag":
            tag = value.strip()
    rnd = get_round(tag, root)
    if rnd is None:
        return
    for group, name in (("optimizee", rnd.optimizee), ("dataset", rnd.dataset)):
        chosen = selections.get(group)
        if chosen is not None and chosen != name:
            raise ConfigError(
                f"meta_tag={tag} is round {tag} of configs/rounds.yaml, meta-trained on {rnd.optimizee} / "
                f"{rnd.dataset}; {group}={chosen} contradicts it. Another source pair is another round (add it to "
                f"configs/rounds.yaml), or an ad-hoc family under its own name: meta_tag=<name>"
            )
        selections[group] = name


def _take_profile(values: List[str]) -> Tuple[str, List[str]]:
    """The profile named by ``profile=<name>`` (the last one wins), else the active profile."""
    from reglo.profiles import active_profile

    chosen, rest = None, []
    for item in values:
        key, _, value = item.partition("=")
        if key.strip() == "profile":
            if not value.strip():
                raise ConfigError("'profile=' needs a name, e.g. profile=smoke")
            chosen = value.strip()
        else:
            rest.append(item)
    return chosen or active_profile(), rest


def compose_run(overrides: Sequence[str] = (), root: Optional[PathLike] = None) -> DictConfig:
    """Config for training one optimizee with M1/M2/M3/M4 (Stages 2–3)."""
    return compose("method", overrides, root)


def compose_meta(overrides: Sequence[str] = (), root: Optional[PathLike] = None) -> DictConfig:
    """Config for meta-training one LSTM optimizer (Stage 1)."""
    return compose("meta_training", overrides, root)


def _validate(cfg: DictConfig, entry_group: str, profile: DictConfig) -> None:
    entry = cfg[entry_group]
    label = f"{entry.label} ({entry.name})" if entry.get("label") else f"{entry_group} '{entry.name}'"
    reg = cfg.selected.get("regularizer", NO_REGULARIZER)
    if not _TOKEN.match(str(cfg.meta_tag)):  # a folder name and the @<round> of M3/M4 run folders
        raise ConfigError(f"meta_tag '{cfg.meta_tag}' may contain letters, digits, '_' and '.' only")

    source = str(cfg.dataset.get("source", "real"))
    if source != "real" and not bool(profile.get("allow_synthetic_data", False)):
        raise ConfigError(
            f"dataset '{cfg.dataset.name}' has source: {source}, but profile '{cfg.profile}' needs the real "
            f"datasets (allow_synthetic_data: false). For a run without downloads use profile=smoke_offline."
        )

    allowed = list(entry.get("allowed_regularizers", [NO_REGULARIZER]))
    if reg not in allowed:
        raise ConfigError(f"{label} takes regularizer=<{'|'.join(allowed)}>, got '{reg}'")
    if "regularizer" in cfg and cfg.regularizer.name != reg:
        raise ConfigError(f"configs/regularizers/{reg}.yaml declares name '{cfg.regularizer.name}'")

    if entry_group == "method":
        fraction = float(cfg.training.get("validation_fraction", 0.0) or 0.0)
        if not 0.0 <= fraction < 1.0:
            raise ConfigError(f"training.validation_fraction must be in [0, 1), got {fraction}")
        if cfg.optimizer.kind != entry.optimizer_kind:
            raise ConfigError(
                f"{entry.label} trains with a {entry.optimizer_kind} optimizer, but optimizer="
                f"{cfg.selected.optimizer} is {cfg.optimizer.kind}"
            )
        return

    # meta_training
    if cfg.optimizer.kind != "learned":
        raise ConfigError(f"Meta-training needs a learned optimizer, got optimizer={cfg.selected.optimizer}")
    if entry.objective.smooth and "smooth" not in cfg:
        raise ConfigError("objective.smooth is on but no L_smooth file is chosen (smooth=smooth)")
    if not cfg.dataset.meta_split.enabled:
        raise ConfigError(
            f"Meta-training on '{cfg.dataset.name}' needs its meta split, so that Stage-2/3 optimizees train "
            f"on data the LSTM never saw: set meta_split.enabled: true in "
            f"configs/datasets/{cfg.selected.dataset}.yaml"
        )
    unroll = int(entry.inner.unroll_steps)
    sched = entry.schedule
    lengths = list(sched.curriculum.trajectory_steps) if sched.curriculum.enabled else [sched.trajectory_steps]
    ragged = [int(n) for n in lengths if int(n) < 1 or int(n) % unroll]
    if unroll < 1 or ragged:
        raise ConfigError(
            f"meta_training: every trajectory length must be a positive multiple of inner.unroll_steps={unroll} "
            f"(whole truncations only), got {ragged or lengths}"
        )
    if list(lengths) != sorted(lengths):
        raise ConfigError(f"meta_training.schedule.curriculum.trajectory_steps must grow, got {list(lengths)}")
    if int(entry.validation.optimizees) < 1 or int(entry.validation.steps) < 1 or int(entry.validation.every) < 1:
        raise ConfigError("meta_training.validation: every, optimizees and steps must be ≥ 1")


# --------------------------------------------------------------------------- saving


def save_config(cfg: DictConfig, path: PathLike) -> Path:
    """Write the fully resolved config next to a run's results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path


def load_saved_config(path: PathLike) -> DictConfig:
    """Read back a config written by :func:`save_config`."""
    return OmegaConf.load(Path(path))


def to_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)
