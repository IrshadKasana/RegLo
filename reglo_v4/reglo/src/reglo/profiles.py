"""Profiles — the scale a study runs at.

    configs/profiles/production.yaml     the study's real numbers (GPU cluster, cloud GPUs)
    configs/profiles/smoke.yaml          the same keys with tiny values: every stage, every
                                         regularizer, the real datasets — in minutes on a laptop
    configs/profiles/smoke_offline.yaml  smoke on random stand-in images: nothing to download

Every number that differs between a smoke run and the real study lives in these files and
nowhere else (the group files under ``configs/`` hold ``???`` for them), so the smoke test runs
exactly the experiment files, code paths and datasets of the real study. A profile file has:

    runs            merged into every run's configuration (Stages 1–3), before any command-line
                    override; a section the run does not have (meta_training in a training run)
                    is skipped
    by_optimizee    {<optimizee>: {...}} merged into the runs of that optimizee
    by_dataset      {<dataset>: {...}} merged into the runs on that dataset
    grids           applied to the experiment files: max_seeds, and the Stage-4 diagnostics /
                    compute settings

Precedence, lowest first: the group files → the profile → the experiment file's overrides →
the command line. Each profile writes to its own ``artifacts/<profile>/`` folder.

The active profile is ``profile=<name>`` on a run's command line, else ``$REGLO_PROFILE``, else
``production``; the scripts take ``--profile``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

__all__ = [
    "DEFAULT_PROFILE",
    "ENV_VAR",
    "PROFILE_SECTIONS",
    "active_profile",
    "apply_to_experiment",
    "apply_to_run",
    "available_profiles",
    "controlled_keys",
    "leaf_keys",
    "load_profile",
    "mirror_gaps",
    "set_active_profile",
]

PathLike = Union[str, Path]

DEFAULT_PROFILE = "production"
ENV_VAR = "REGLO_PROFILE"
PROFILE_SECTIONS = ("description", "allow_synthetic_data", "runs", "by_optimizee", "by_dataset", "grids")
_GRID_BLOCKS = ("diagnostics", "compute")
_MISSING = "???"


def _config_error(message: str) -> Exception:
    from reglo.config import ConfigError  # config imports this module: import lazily

    return ConfigError(message)


def _profiles_dir(root: Optional[PathLike] = None) -> Path:
    from reglo.config import config_dir

    return Path(root or config_dir()) / "profiles"


def available_profiles(root: Optional[PathLike] = None) -> List[str]:
    """Names of the files in ``configs/profiles/``."""
    folder = _profiles_dir(root)
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.is_dir() else []


def active_profile() -> str:
    """``$REGLO_PROFILE`` or ``production``."""
    return os.environ.get(ENV_VAR) or DEFAULT_PROFILE


def set_active_profile(name: Optional[str], root: Optional[PathLike] = None) -> str:
    """Make ``name`` the profile of this process and of every process it starts."""
    name = name or active_profile()
    load_profile(name, root)  # an unknown name stops here, with the list of valid ones
    os.environ[ENV_VAR] = name
    return name


def load_profile(name: str, root: Optional[PathLike] = None) -> DictConfig:
    """``configs/profiles/<name>.yaml`` (merged over its ``extends:`` parent, if any)."""
    from reglo.config import _load_yaml

    names = available_profiles(root)
    if name not in names:
        raise _config_error(f"Unknown profile '{name}'. Choose one of: {', '.join(names) or '(none found)'}")
    node = _load_yaml(_profiles_dir(root) / f"{name}.yaml")
    unknown = sorted(set(node.keys()) - set(PROFILE_SECTIONS))
    if unknown:
        raise _config_error(f"configs/profiles/{name}.yaml: unknown section(s) {unknown}; "
                            f"a profile has {', '.join(PROFILE_SECTIONS)}")
    return node


# ───────────────────────────────────────────────────────────────── run configurations


def _merge_strict(target: DictConfig, patch: Mapping[str, Any], where: str) -> None:
    """Write ``patch`` into ``target``; every key must already exist there (typos never pass)."""
    for key, value in patch.items():
        if key not in target.keys():
            raise _config_error(f"{where}.{key} is not a key of this configuration (check the spelling)")
        if isinstance(value, Mapping) and isinstance(target[key], DictConfig):
            _merge_strict(target[key], value, f"{where}.{key}")
            continue
        try:
            target[key] = value
        except OmegaConfBaseException as err:
            raise _config_error(f"{where}.{key}: {err}") from err


def _plain(node: Any) -> Any:
    return OmegaConf.to_container(node, resolve=False) if isinstance(node, (DictConfig, ListConfig)) else node


def apply_to_run(cfg: DictConfig, profile: DictConfig, chosen: Mapping[str, str], known_sections: Sequence[str],
                 name: str = "") -> None:
    """Merge the profile's ``runs``, ``by_optimizee`` and ``by_dataset`` sections into ``cfg``.

    A top-level section the run lacks is skipped (``meta_training`` in a training run, ``smooth``
    in M3's meta-training) unless it is not a section of any run at all — then it is a typo.
    """
    where = f"configs/profiles/{name}.yaml"
    layers: List[Tuple[str, Mapping[str, Any]]] = [("runs", _plain(profile.get("runs")) or {})]
    for section, group in (("by_optimizee", "optimizee"), ("by_dataset", "dataset")):
        table = _plain(profile.get(section)) or {}
        if chosen.get(group) in table:
            layers.append((f"{section}.{chosen[group]}", table[chosen[group]] or {}))
    for label, layer in layers:
        for key, value in layer.items():
            if key not in cfg.keys():
                if key in known_sections:
                    continue
                raise _config_error(f"{where}: {label}.{key} is not a section of any run "
                                    f"(sections: {', '.join(sorted(known_sections))})")
            if isinstance(value, Mapping) and isinstance(cfg[key], DictConfig):
                _merge_strict(cfg[key], value, f"{where}: {label}.{key}")
            else:
                cfg[key] = value


def leaf_keys(node: Any, prefix: str = "") -> Iterator[str]:
    """Dotted paths of every leaf of a nested mapping (lists count as leaves)."""
    node = _plain(node)
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping) and value:
                yield from leaf_keys(value, path)
            else:
                yield path
    elif prefix:
        yield prefix


def controlled_keys(profile: DictConfig) -> Set[str]:
    """Run-config keys this profile sets for every run or per dataset — the scale knobs that the
    experiment files must leave to the profiles (by_optimizee only narrows networks for smoke runs)."""
    keys = set(leaf_keys(profile.get("runs") or {}))
    for table in (_plain(profile.get("by_dataset")) or {}).values():
        keys |= set(leaf_keys(table or {}))
    return keys


def mirror_gaps(reference: DictConfig, other: DictConfig) -> List[str]:
    """Keys ``reference`` sets that ``other`` does not set anywhere (its runs, by_* or grids)."""

    def covered(profile: DictConfig) -> Set[str]:
        keys = set(leaf_keys(profile.get("runs") or {})) | {f"grids.{k}" for k in leaf_keys(profile.get("grids") or {})}
        for section in ("by_optimizee", "by_dataset"):
            for table in (_plain(profile.get(section)) or {}).values():
                keys |= set(leaf_keys(table or {}))
        return keys

    return sorted(covered(reference) - covered(other))


# ───────────────────────────────────────────────────────────────── experiment files


def _contains_missing(value: Any, path: str) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _contains_missing(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _contains_missing(item, f"{path}[{index}]")
    elif value == _MISSING:
        yield path


def _merge_plain(target: Dict[str, Any], patch: Mapping[str, Any], where: str) -> None:
    for key, value in patch.items():
        if key not in target:
            raise _config_error(f"{where}.{key}: the experiment file has no such key")
        if isinstance(value, Mapping) and isinstance(target[key], dict):
            _merge_plain(target[key], value, f"{where}.{key}")
        else:
            target[key] = copy.deepcopy(value)


def _override_key(override: str) -> str:
    return override.partition("=")[0].lstrip("+")


def _experiment_overrides(spec: Mapping[str, Any]) -> Iterator[Tuple[str, str]]:
    """(where, override) for every override string an experiment file applies to all profiles."""
    for block in ("meta_train", "train", "compute"):
        body = spec.get(block) or {}
        for item in body.get("overrides") or []:
            yield f"{block}.overrides", str(item)
        for key, items in (body.get("pair_overrides") or {}).items():
            for item in items or []:
                yield f"{block}.pair_overrides.{key}", str(item)
        for optimizer, table in (body.get("optimizer_overrides") or {}).items():
            for optimizee, items in (table or {}).items():
                for item in items or []:
                    yield f"{block}.optimizer_overrides.{optimizer}.{optimizee}", str(item)


def apply_to_experiment(spec: Mapping[str, Any], name: str, root: Optional[PathLike] = None) -> Dict[str, Any]:
    """An experiment file at the scale of profile ``name`` (a new dict; ``spec`` is not changed).

    * ``max_seeds`` keeps the first N seeds of every block;
    * ``grids.diagnostics`` / ``grids.compute`` fill the Stage-4 settings;
    * a block's ``profile_overrides: {<profile>: [...]}`` adds overrides for that profile only;
    * an override that pins a scale knob for every profile (``training.epochs=50``) is refused —
      that number belongs in the profiles, or else the smoke test would run it at full size.
    """
    profile = load_profile(name, root)
    out = copy.deepcopy(dict(spec))
    title = f"experiment '{out.get('name')}'"

    controlled = set()
    for other in available_profiles(root):
        controlled |= controlled_keys(load_profile(other, root))
    pinned = [f"{where}: {item}" for where, item in _experiment_overrides(out) if _override_key(item) in controlled]
    if pinned:
        raise _config_error(f"{title} pins numbers that the profiles set (configs/profiles/*.yaml), so a smoke "
                            f"run would use them too: {'; '.join(pinned)}. Move them to the profiles, or to a "
                            f"'profile_overrides: {{<profile>: [...]}}' entry of the block.")

    grids = _plain(profile.get("grids")) or {}
    max_seeds = grids.get("max_seeds")
    for block in ("meta_train", "train", "compute"):
        body = out.get(block)
        if not isinstance(body, dict):
            continue
        per_profile = body.pop("profile_overrides", None) or {}
        unknown = sorted(set(per_profile) - set(available_profiles(root)))
        if unknown:
            raise _config_error(f"{title}: {block}.profile_overrides names unknown profile(s) {unknown}")
        extra = list(per_profile.get(name) or [])
        if extra:
            body["overrides"] = [*(body.get("overrides") or []), *extra]
        if max_seeds is not None and "seeds" in body:
            body["seeds"] = list(body["seeds"])[: max(int(max_seeds), 1)]
    for block in _GRID_BLOCKS:
        if isinstance(out.get(block), dict) and grids.get(block):
            _merge_plain(out[block], grids[block], f"configs/profiles/{name}.yaml: grids.{block} → {title}: {block}")

    unset = list(_contains_missing({k: v for k, v in out.items()}, out.get("name", "experiment")))
    if unset:
        raise _config_error(f"{title}: {', '.join(unset)} must be set by configs/profiles/{name}.yaml (grids section)")
    out["profile"] = name
    return out
