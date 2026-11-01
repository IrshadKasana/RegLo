"""The experiment plan as one Excel workbook — every table of the study, built from the configs,
with blank cells for the results; ``fill_workbook`` writes finished results into them.

    Read me                      the study in brief, the legend, how to fill it in
    T1 Optimizees & Optimizers   Table 1: optimizees, optimizers, datasets and the training protocol
    T2 Regularizers              Table 2: L1, L2, SAM, GAM, GSAM and L_smooth — objective, settings, use
    T3 Sweeps                    Table 3: the hyperparameters worth sweeping, with cells for the best values
    T4 Experiments               Table 4: every experiment, stage by stage and round by round, with metrics
    Cubes                        Stage 4: method × regularizer × evaluation setting, per round and metric
    Cross-evaluation             every pair with every round's optimizers; advantage over M1; transfer gap
    Stage 1 meta-training        one row per LSTM optimizer: meta-validation, objective terms, cost
    Runs                         one row per training run — the cells the results go in
    Metrics                      what every number means and where the code computes it

Everything derived (seed means and spreads, differences to M1, the cubes, the cross-evaluation)
is an Excel formula over the Runs and Stage-1 sheets, so the workbook updates itself as results
are entered. Built with openpyxl (``pip install openpyxl``); used by scripts/make_workbook.py.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from reglo.config import compose_meta, compose_run, default_meta_tag, load_group_file
from reglo.rounds import ROLE_LABELS, Round, load_rounds
from reglo.study import (
    DATASET_NAMES,
    LABELS,
    LEARNED,
    OPTIMIZEE_NAMES,
    REG_NAMES,
    REGULARIZERS,
    SETTINGS,
    Pair,
    is_evaluation,
    spec_rounds,
    study_experiments,
    study_rounds,
    study_settings,
)

__all__ = ["SHEETS", "build_workbook", "fill_workbook", "plan_study_workbook"]

PathLike = Union[str, Path]

SHEETS = ("Read me", "T1 Optimizees & Optimizers", "T2 Regularizers", "T3 Sweeps", "T4 Experiments", "Cubes",
          "Cross-evaluation", "Stage 1 meta-training", "Runs", "Metrics")
T4, RUNS, META = "T4 Experiments", "Runs", "Stage 1 meta-training"

#: training-set sizes of the real datasets (torchvision); the test sets: SVHN 26,032, the others 10,000
TRAIN_SIZES = {"svhn": 73_257, "fmnist": 60_000, "cifar10": 50_000, "cifar100": 50_000}
TEST_SIZES = {"svhn": 26_032, "fmnist": 10_000, "cifar10": 10_000, "cifar100": 10_000}
METHOD_ROWS = ([("m1_adamw", "none")] + [("m2_adamw_reg", r) for r in REGULARIZERS] + [("m3_lstm", "none")]
               + [("m4_lstm_reg", r) for r in REGULARIZERS])
ROLE_ORDER = ("seen", "unseen_optimizee", "unseen_dataset", "unseen_both")
SETTING_TEXT = {s.key: s.label.lower() for s in SETTINGS}

Key = Tuple[str, str, str, str, str, str, int]  # study, optimizee, dataset, method, regularizer, round, seed


# ═════════════════════════════════════════════════════════════════ the plan

@dataclass
class Run:
    """One training run of an optimizee (one seed)."""

    key: Key
    experiment: str
    images: int = 0
    epochs: int = 0
    steps: int = 0
    run_id: str = ""
    row: int = 0
    serves: List[str] = field(default_factory=list)

    @property
    def study(self) -> str:
        return self.key[0]

    @property
    def pair(self) -> Pair:
        return Pair(self.key[1], self.key[2])

    @property
    def method(self) -> str:
        return self.key[3]

    @property
    def reg(self) -> str:
        return self.key[4]

    @property
    def round(self) -> str:
        return self.key[5]

    @property
    def seed(self) -> int:
        return self.key[6]

    @property
    def folder(self) -> str:
        tag = (self.method if self.reg == "none" else f"{self.method}-{self.reg}") + (f"@{self.round}" if self.round else "")
        return f"runs/{self.study}/{self.pair.tag}/{tag}/seed{self.seed}"


@dataclass
class Arm:
    """One Stage-1 meta-training run: an LSTM optimizer of one round."""

    round: str
    entry: str      # m3_lstm | m4_lstm_reg
    reg: str
    source: Pair
    exp_id: str = ""
    row: int = 0

    @property
    def folder(self) -> str:
        return f"meta_training/{self.round}/{self.entry if self.reg == 'none' else f'{self.entry}-{self.reg}'}"


@dataclass
class Exp:
    """One row of Table 4: an experiment = a (stage, round, setting, pair, method, regularizer) and its runs."""

    exp_id: str
    stage: str               # "1", "2", "3", "2–3 ext.", "pilot"
    round: str
    setting: str             # human-readable
    setting_key: str
    role: str
    pair: Pair
    method: str
    reg: str
    runs: List[Run] = field(default_factory=list)
    arm: Optional[Arm] = None
    section: str = ""
    extended: bool = False
    same_as: str = ""
    m1: Optional["Exp"] = None   # the M1 row of the same block (for the Δ columns)
    row: int = 0


@dataclass
class Plan:
    profile: str
    rounds: List[Round]
    runs: List[Run]
    arms: List[Arm]
    exps: List[Exp]
    settings: Dict[str, Dict[str, list]]
    seeds: List[int]
    stage1: Dict[str, Any]


def _args(line: Sequence[str]) -> Dict[str, str]:
    return {a.split("=", 1)[0]: a.split("=", 1)[1] for a in line[1:] if "=" in a and not a.startswith("+")}


def _training_images(cfg) -> int:
    ds = cfg.dataset
    n = TRAIN_SIZES.get(str(ds.name), 0)
    if ds.meta_split.enabled and n:
        n -= min(max(int(round(n * float(ds.meta_split.fraction))), 1), n - 1)
    if ds.get("train_subset") is not None:
        n = min(n, int(ds.train_subset))
    return n


def plan_study_workbook(profile: str = "production", study: str = "main", extended: bool = True,
                        pilot: bool = True) -> Plan:
    """Every run, Stage-1 arm and experiment row of the study, from its experiment files."""
    from reglo.manifest import expand

    default_round = default_meta_tag()
    specs = list(study_experiments(study, profile=profile))
    if pilot:
        specs += list(study_experiments("pilot", profile=profile))
    if not extended:
        specs = [s for s in specs if not s.get("extended")]
    runs: Dict[Key, Run] = {}
    sizes: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for spec in specs:
        if "train" not in spec:
            continue
        for line in expand(spec, ["train"]):
            arg = _args(line)
            method, reg = arg["method"], arg.get("regularizer", "none")
            rnd = arg.get("meta_tag", default_round) if method in LEARNED else ""
            key = (str(spec["study"]), arg["optimizee"], arg["dataset"], method, reg, rnd, int(arg["seed"]))
            if key in runs:
                continue
            size_key = (str(spec["name"]), arg["optimizee"], arg["dataset"])
            if size_key not in sizes:
                cfg = compose_run(line[1:])
                sizes[size_key] = (_training_images(cfg), int(cfg.training.epochs), int(cfg.training.batch_size))
            images, epochs, batch = sizes[size_key][0], sizes[size_key][1], sizes[size_key][2]
            runs[key] = Run(key, str(spec["name"]), images, epochs, math.ceil(images / batch) * epochs if images else 0)

    stage1 = next(s for s in specs if s["name"] == "stage1_meta_training")
    rounds = [r for r in study_rounds(study, profile=profile)]
    arms: List[Arm] = []
    for line in expand(stage1, ["meta_train"]):
        arg = _args(line)
        rnd = arg.get("meta_tag", default_round)
        arms.append(Arm(rnd, arg["meta_training"], arg.get("regularizer", "none"),
                        Pair(arg.get("optimizee", "convnet4"), arg.get("dataset", "svhn"))))

    def runs_of(pair: Pair, method: str, reg: str, rnd: str, of_study: str = study) -> List[Run]:
        found = [r for r in runs.values() if (r.study, r.pair, r.method, r.reg, r.round) == (of_study, pair, method, reg, rnd)]
        return sorted(found, key=lambda r: r.seed)

    exps: List[Exp] = []
    settings = study_settings(study, profile=profile)

    # Stage 1 — per round: the meta-training arms, M1 (reference) and M2(r) on the source pair
    for rnd in spec_rounds(stage1):
        source = Pair(rnd.optimizee, rnd.dataset)
        section = f"STAGE 1 · Training — round {rnd.name}: source {source.label}"
        number = 0
        for arm in (a for a in arms if a.round == rnd.name):
            number += 1
            arm.exp_id = f"E1.{rnd.name}.{number:02d}"
            exps.append(Exp(arm.exp_id, "1", rnd.name, "source pair (meta-training half)", "meta", "seen", source,
                            arm.entry, arm.reg, arm=arm, section=section))
        number += 1
        m1 = Exp(f"E1.{rnd.name}.{number:02d}", "1", rnd.name, "source pair (reference)", "benchmark", "seen", source,
                 "m1_adamw", "none", runs_of(source, "m1_adamw", "none", ""), section=section)
        exps.append(m1)
        for reg in REGULARIZERS:
            got = runs_of(source, "m2_adamw_reg", reg, "")
            if got:
                number += 1
                exps.append(Exp(f"E1.{rnd.name}.{number:02d}", "1", rnd.name, "source pair (benchmark)", "benchmark",
                                "seen", source, "m2_adamw_reg", reg, got, section=section, m1=m1))

    # Stages 2 and 3 (core), then the extended grid — per round, setting, pair: M1 · M2(r) · M3 · M4(r)
    def block(stage_label: str, exp_prefix: str, rnd: Round, sp, section: str, counter: List[int]) -> None:
        m1_row: Optional[Exp] = None
        for method, reg in METHOD_ROWS:
            # M1 and M2 (with r; on a source pair: its Stage-1 runs) use no φ; M3/M4 the round's
            got = runs_of(sp.pair, method, reg, rnd.name if method in LEARNED else "")
            if not got:
                continue
            counter[0] += 1
            exp = Exp(f"{exp_prefix}.{rnd.name}.{counter[0]:03d}", stage_label, rnd.name, SETTING_TEXT[sp.setting.key],
                      sp.setting.key, sp.role, sp.pair, method, reg, got, section=section, extended=sp.extended, m1=m1_row)
            if method == "m1_adamw":
                m1_row = exp
            exps.append(exp)

    for extended in (False, True):
        for stage in (2, 3):
            for rnd in rounds:
                per = settings.get(rnd.name, {})
                entries = [sp for s in SETTINGS if s.stage == stage for sp in per.get(s.key, []) if sp.extended == extended]
                if not entries:
                    continue
                source = Pair(rnd.optimizee, rnd.dataset)
                name = "Optimizer evaluation" if stage == 2 else "Optimizee evaluation"
                if extended:
                    section = f"EXTENDED (optional) · Stage {stage} {name.lower()} — round {rnd.name}: source {source.label}"
                else:
                    section = f"STAGE {stage} · {name} — round {rnd.name}: source {source.label}"
                counter = [0]  # E2.A.001 …, EX3.A.001 … — numbered per stage and round
                for sp in entries:
                    block(f"{stage} (ext.)" if extended else str(stage), f"EX{stage}" if extended else f"E{stage}", rnd, sp,
                          section, counter)

    if pilot:
        pilot_spec = next((s for s in specs if s["name"] == "pilot"), None)
        if pilot_spec is not None:
            pilot_rounds = spec_rounds(pilot_spec) or [load_rounds()[default_round]]
            rnd = pilot_rounds[0]
            for pair in {Pair(r.key[1], r.key[2]) for r in runs.values() if r.study == "pilot"}:
                section = f"PILOT (optional) — M1 against M3 (round {rnd.name}), short budget"
                number = 0
                m1_row = None
                for method in ("m1_adamw", "m3_lstm"):
                    got = runs_of(pair, method, "none", rnd.name if method in LEARNED else "", of_study="pilot")
                    if got:
                        number += 1
                        exp = Exp(f"EP.{rnd.name}.{number}", "pilot", rnd.name, "seen optimizee", "seen_optimizee",
                                  rnd.role(pair.optimizee, pair.dataset), pair, method, "none", got,
                                  section=section, m1=m1_row)
                        m1_row = m1_row or exp
                        exps.append(exp)

    # identifiers, "same runs as", and what each run serves
    ordered = [r for r in runs.values() if r.study == study and r.experiment == stage1["name"]]
    ordered += [r for r in runs.values() if r not in ordered]
    for number, run in enumerate(ordered, start=1):
        run.run_id = f"R{number:04d}"
    first: Dict[Tuple[str, ...], str] = {}
    for exp in exps:
        if not exp.runs:
            continue
        ids = tuple(r.run_id for r in exp.runs)
        if ids in first:
            exp.same_as = first[ids]
        else:
            first[ids] = exp.exp_id
        label = f"{exp.round}·S{exp.stage.split()[0]} {exp.setting}" if exp.stage != "pilot" else f"{exp.round}·pilot"
        if exp.stage == "1":
            label = f"{exp.round}·S1 {'benchmark' if exp.method == 'm2_adamw_reg' else 'reference'}"
        for run in exp.runs:
            if label not in run.serves:
                run.serves.append(label)
    seeds = sorted({r.seed for r in runs.values() if r.study == study})
    return Plan(profile, rounds, ordered, arms, exps, settings, seeds, stage1)


# ═════════════════════════════════════════════════════════════════ text of the design

def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-2 or abs(value) >= 1e4:
            mantissa, exponent = f"{value:.0e}".split("e")
            return f"{mantissa}e{int(exponent)}"
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _params(optimizee: str, dataset: str) -> Optional[int]:
    from reglo.models.common import count_parameters
    from reglo.models.optimizees import build_optimizee

    cfg = compose_run(["method=m1_adamw", f"optimizee={optimizee}", f"dataset={dataset}", "profile=production"])
    return count_parameters(build_optimizee(cfg.optimizee))


def _role_text(name: str, kind: str, rounds: Sequence[Round]) -> str:
    """What an optimizee (kind='optimizee') or dataset (kind='dataset') is to each round."""
    parts = []
    for r in rounds:
        mine = r.optimizee if kind == "optimizee" else r.dataset
        parts.append(f"round {r.name}: {'source (seen)' if name == mine else 'unseen'}")
    return "; ".join(parts)


ACTIVATIONS = {"relu": "ReLU", "gelu": "GELU", "tanh": "tanh", "silu": "SiLU"}
OPTIMIZEE_TEXT = {
    "convnet4": ("3 × [Conv 3×3 → BatchNorm → ReLU → MaxPool 2] → Linear", "the paper's protocol (Sahoo & Krishnan)"),
    "mlp": ("Flatten → 2 × [Linear → ReLU] → Linear", "Metz et al. 2022 (FashionMnistRelu128x128)"),
    "resnet18": ("ResNet-18, CIFAR stem (3×3 conv, no max-pool), 4 stages × 2 basic blocks", "He et al. 2016"),
    "wrn28": ("Wide ResNet-28-k: 3 groups × 4 pre-activation blocks", "Zagoruyko & Komodakis 2016"),
    "vit_lite7": ("ViT-Lite-7/4: patch 4, transformer encoder, class token", "Hassani et al. 2021"),
}


def _optimizee_block(rounds: Sequence[Round]) -> Tuple[List[str], List[str], List[List[str]]]:
    smoke = _profile_sizes("smoke")
    names = ["convnet4", "mlp", "resnet18", "wrn28", "vit_lite7"]
    heads = ["ConvNet-4", "MLP", "ResNet-18", "WRN-28-4 (extended)", "ViT-Lite-7/4 (extended)"]
    attributes = ["Architecture", "Size settings", "Activation", "Normalisation", "Dropout",
                  "Parameters · 28×28×1, 10 classes (FMNIST)", "Parameters · 32×32×3, 10 classes (SVHN, CIFAR-10)",
                  "Parameters · 32×32×3, 100 classes (CIFAR-100)", "Role in the rounds", "Trained on (study main)",
                  "Smoke profile", "Config file", "Reference"]
    columns = []
    trained_on = _trained_on()
    for name in names:
        cfg = load_group_file("optimizee", name)
        if name == "convnet4":
            size, norm, drop = f"channels {_fmt(list(cfg.channels))}", "BatchNorm", "—"
        elif name == "mlp":
            size, norm, drop = f"hidden units {_fmt(list(cfg.hidden))}", "none", "—"
        elif name == "resnet18":
            size, norm, drop = f"width {cfg.width} (stages {cfg.width}, {2 * cfg.width}, {4 * cfg.width}, {8 * cfg.width})", \
                "BatchNorm", "—"
        elif name == "wrn28":
            size, norm, drop = f"depth {cfg.depth}, widen factor {cfg.widen_factor}", "BatchNorm", _fmt(float(cfg.dropout))
        else:
            size = f"embedding {cfg.embed_dim}, depth {cfg.depth}, heads {cfg.num_heads}, MLP ratio {_fmt(float(cfg.mlp_ratio))}"
            norm, drop = "LayerNorm", f"{_fmt(float(cfg.dropout))} (attention {_fmt(float(cfg.attention_dropout))})"
        counts = [_params(name, d) for d in ("fmnist", "svhn", "cifar100")]
        columns.append([
            OPTIMIZEE_TEXT[name][0], size, ACTIVATIONS.get(str(cfg.activation), str(cfg.activation)), norm, drop,
            f"{counts[0]:,}", f"{counts[1]:,}", f"{counts[2]:,}", _role_text(name, "optimizee", rounds),
            ", ".join(DATASET_NAMES[d] for d in trained_on.get(name, [])) or "—",
            smoke.get(name, "same as production"), f"configs/optimizees/{name}.yaml", OPTIMIZEE_TEXT[name][1],
        ])
    return heads, attributes, columns


def _trained_on() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for spec in study_experiments("main", profile="production"):
        if is_evaluation(spec):
            from reglo.study import training_pairs

            for pair in training_pairs(spec):
                out.setdefault(pair.optimizee, [])
                if pair.dataset not in out[pair.optimizee]:
                    out[pair.optimizee].append(pair.dataset)
    order = list(DATASET_NAMES)
    return {k: sorted(v, key=order.index) for k, v in out.items()}


def _profile_sizes(profile: str) -> Dict[str, str]:
    from reglo.profiles import load_profile
    from omegaconf import OmegaConf

    node = OmegaConf.to_container(load_profile(profile).get("by_optimizee") or {}, resolve=True)
    out = {}
    for name, body in node.items():
        values = body.get("optimizee", {})
        out[name] = ", ".join(f"{k.replace('_', ' ')} {_fmt(v)}" for k, v in values.items())
    return out


def _optimizer_block() -> Tuple[List[str], List[str], List[List[str]]]:
    from reglo.diagnostics.flops import lstm_flops_per_coordinate
    from reglo.models.common import count_parameters
    from reglo.models.optimizers import build_lstm_optimizer

    adamw, lstm = load_group_file("optimizer", "adamw"), load_group_file("optimizer", "lstm")
    meta = compose_meta(["meta_training=m4_lstm_reg", "regularizer=sam", "profile=production"]).meta_training
    smoke_meta = compose_meta(["meta_training=m4_lstm_reg", "regularizer=sam", "profile=smoke"]).meta_training
    net, _ = build_lstm_optimizer(lstm)
    state_bytes = 2 * int(lstm.num_layers) * int(lstm.hidden_size) * 4
    attributes = ["Used by", "Kind", "Update", "Step size / learning rate", "β₁, β₂ · ε", "Weight decay",
                  "Learning-rate schedule", "Gradient clipping", "Architecture", "Input features · normalisation",
                  "Output scale · head initialisation", "Weights of the optimizer itself", "State kept per optimizee weight",
                  "FLOPs per optimizee weight per step", "How it is trained", "Truncated BPTT (unroll T)",
                  "Outer optimizer (meta-learning rate)", "Meta-training budget (production)",
                  "Curriculum of trajectory lengths", "Meta-validation (selects φ)", "Meta-training budget (smoke)",
                  "Which φ a run uses", "Config file", "Reference"]
    sched = adamw.scheduler
    adamw_col = [
        "M1 (Stages 2–3), M2 (Stages 1–3, with r)", "conventional (hand-designed)", "Adam step with decoupled weight decay",
        _fmt(float(adamw.lr)), f"{_fmt(list(adamw.betas))} · {_fmt(float(adamw.eps))}",
        f"{_fmt(float(adamw.weight_decay))} — M1 unregularised; M2-L2 decoupled uses its own coefficient",
        f"{sched.name}, warm-up {sched.warmup_epochs} epochs, min lr {_fmt(float(sched.min_lr))} (ViT-Lite: lr 5e-4, warm-up 5)",
        "off" if adamw.grad_clip is None else _fmt(float(adamw.grad_clip)), "—", "—", "—", "—",
        "2 moments (8 bytes)", "≈ 10 (elementwise)", "—", "—", "—", "—", "—", "—", "—", "—",
        "configs/optimizers/adamw.yaml", "Loshchilov & Hutter 2019",
    ]
    curriculum = meta.schedule.curriculum
    lstm_col = [
        "M3 (φ, meta-trained on L_meta), M4 (φ*_r, meta-trained on L_meta + λ_smooth·L_smooth + λ_reg·L_reg)",
        "learned, coordinate-wise (shared across weights)", "θ_{t+1} = θ_t + output_scale · head(LSTM(g_t, h_t))",
        "learned (no learning rate)", "—", "— (no explicit term)", "—", "—",
        f"{lstm.num_layers}-layer LSTM, {lstm.hidden_size} units; head {lstm.head_layers} layers "
        f"({lstm.hidden_size} → {lstm.head_hidden} → 1)",
        f"{', '.join(lstm.features)} · {lstm.normalization} (per-tensor RMS)",
        f"{_fmt(float(lstm.output_scale))} · last layer ×{_fmt(float(lstm.head_init_scale))}",
        f"{count_parameters(net):,}", f"h and c of {lstm.num_layers} layers: {state_bytes} bytes",
        f"{lstm_flops_per_coordinate(net):,}",
        "Stage 1, Algorithm 1: unroll the optimizee, backpropagate the meta-loss into φ (first-order)",
        f"T = {meta.inner.unroll_steps} steps, inner batch {meta.inner.batch_size}",
        f"Adam, lr {_fmt(float(meta.outer.lr))}, ‖∇φ‖ clipped at {_fmt(float(meta.outer.grad_clip))}, "
        f"loss weights {meta.outer.loss_weights}",
        f"{meta.schedule.num_trajectories:,} trajectories (fresh optimizees), each up to "
        f"{max(curriculum.trajectory_steps):,} steps",
        f"{' → '.join(str(n) for n in curriculum.trajectory_steps)} steps; next length after {curriculum.patience} "
        "validations without improvement",
        f"every {meta.validation.every} trajectories: {meta.validation.optimizees} fresh optimizees × "
        f"{meta.validation.steps} steps, scored on {_fmt(float(meta.validation.fraction) * 100)}% of the meta half",
        f"{smoke_meta.schedule.num_trajectories} trajectories, lengths {_fmt(list(smoke_meta.schedule.curriculum.trajectory_steps))}",
        "meta_tag = round: artifacts/<profile>/meta_training/<round>/<arm>/phi_best.pt",
        "configs/optimizers/lstm.yaml · configs/meta_training/*.yaml",
        "Andrychowicz et al. 2016; Chen et al. 2020; Algorithm 1 (Sahoo & Krishnan)",
    ]
    return ["AdamW", "LSTM learned optimizer F_φ"], attributes, [adamw_col, lstm_col]


def _dataset_block(rounds: Sequence[Round]) -> Tuple[List[str], List[str], List[List[str]]]:
    names = ["svhn", "fmnist", "cifar10", "cifar100"]
    attributes = ["Images (train / test)", "Shape · classes", "Normalisation mean · std", "Augmentation (training)",
                  "Meta split (meta-training / optimizee half)", "Training images of an optimizee run",
                  "Meta-training images (minus the 10% validation hold-out)", "Role in the rounds",
                  "Epochs · steps per epoch (production, batch 128)", "Smoke profile", "Config file"]
    columns = []
    for name in names:
        ds = load_group_file("dataset", name)
        cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", f"dataset={name}", "profile=production"])
        n = TRAIN_SIZES[name]
        if ds.meta_split.enabled:
            meta = min(max(int(round(n * float(ds.meta_split.fraction))), 1), n - 1)
            split = f"{meta:,} / {n - meta:,} (seed {ds.meta_split.seed})"
            held = meta - min(max(int(round(meta * 0.1)), 1), meta - 1)
            meta_text = f"{held:,}"
        else:
            split, meta_text = "none — no round meta-trains on it", "—"
        images = _training_images(cfg)
        aug = list(ds.augment)
        aug_text = ", ".join(a.replace("_", " ") for a in aug) + (f" (padding {ds.crop_padding})" if "random_crop" in aug else "")
        columns.append([
            f"{n:,} / {TEST_SIZES[name]:,}", f"{ds.image_size}×{ds.image_size}×{ds.channels} · {ds.num_classes}",
            f"{_fmt(list(ds.mean))} · {_fmt(list(ds.std))}", aug_text, split, f"{images:,}", meta_text,
            _role_text(name, "dataset", rounds),
            f"{cfg.training.epochs} · {math.ceil(images / int(cfg.training.batch_size)):,}",
            "first 512 training / 256 test images, 2 epochs", f"configs/datasets/{name}.yaml",
        ])
    return ["SVHN", "FMNIST", "CIFAR-10", "CIFAR-100 (extended)"], attributes, columns


def _protocol_rows(plan: Plan) -> List[Tuple[str, str]]:
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=production"])
    tr = cfg.training
    return [
        ("Seeds per experiment", ", ".join(str(s) for s in plan.seeds) + " (smoke: the first only)"),
        ("Same seed → same", "initial weights and data order for M1–M4 (paired comparison)"),
        ("Batch size · evaluation batch", f"{tr.batch_size} · {tr.eval_batch_size}"),
        ("Epochs (production)", "50; CIFAR-10 and CIFAR-100: 100 (configs/profiles/production.yaml)"),
        ("Test-set evaluation", f"every {tr.eval_every} epoch(s) and at the end"),
        ("Clean training-set evaluation (for the gap)", f"every {tr.train_eval_every} epochs and at the end, no augmentation"),
        ("'At convergence' snapshot", f"when the training loss has not improved for {tr.convergence_patience} steps"),
        ("FLOPs", "one training step, measured at the start of every run (torch FlopCounterMode)"),
        ("Checkpoints", f"every {tr.checkpoint_every} epochs; a run resumes where it stopped"),
        ("Device", f"{cfg.device} (production) · GPU if present, else CPU (smoke)"),
        ("Data halves", "SVHN and FMNIST (the source datasets): every optimizee run trains on the half no LSTM "
                        "was meta-trained on; CIFAR: all training images"),
    ]


def _regularizer_block() -> Tuple[List[str], List[str], List[List[str]]]:
    regs = {name: load_group_file("regularizer", name) for name in REGULARIZERS}
    smooth = load_group_file("smooth", "smooth")
    l1, l2, sam, gam, gsam = (regs[n] for n in REGULARIZERS)
    attributes = ["Full name", "Targets (the property)", "M2: objective / update (every step, Stages 1–3)",
                  "M2: hyperparameters", "M2: forward–backward passes per step", "Weights it applies to",
                  "M4: term in the meta-objective, on φ (Stage 1)", "M4: weight λ_reg (meta.coef)",
                  "M4: where / normalisation", "In Stages 2–3 (evaluation)", "Property measured in Stage 4",
                  "Config file · override keys", "Reference"]
    off = "M2 computes r at every step; M4 runs its frozen φ*_r (no r); M1, M3 never"
    cols = [
        ["L1 (lasso) penalty", "sparsity — many weights at or near zero",
         f"min_θ L(θ) + λ‖θ‖₁ ({l1.explicit.mode}; alternative: proximal soft-thresholding)",
         f"λ = {_fmt(float(l1.explicit.coef))}", "1", "all but biases and normalisation layers",
         "λ‖φ‖₁ on the LSTM's weights, biases excluded (M2's λ: normalize = explicit)", _fmt(float(l1.meta.coef)),
         f"{l1.meta.space} · {l1.meta.normalize}", off,
         "sparsity (|w| < 1e-3), ‖θ‖₁", "configs/regularizers/l1.yaml · regularizer.explicit.coef, .mode, .meta.coef",
         "Tibshirani 1996"],
        ["L2 penalty (weight decay)", "small weights",
         f"min_θ L(θ) + ½λ‖θ‖₂² ({l2.explicit.mode}; alternative: decoupled decay {_fmt(float(l2.explicit.decoupled_coef))})",
         f"λ = {_fmt(float(l2.explicit.coef))}", "1", "all but biases and normalisation layers",
         "½λ‖φ‖₂² on the LSTM's weights, biases excluded (M2's λ: normalize = explicit)", _fmt(float(l2.meta.coef)),
         f"{l2.meta.space} · {l2.meta.normalize}", off,
         "‖θ‖₂", "configs/regularizers/l2.yaml · regularizer.explicit.coef, .mode, .decoupled_coef",
         "Krogh & Hertz 1992; Loshchilov & Hutter 2019"],
        ["Sharpness-Aware Minimization", "flat minima — low loss in a ρ-ball",
         "min_θ max_{‖ε‖≤ρ} L(θ+ε): ε̂ = ρ∇L/‖∇L‖, then step with ∇L(θ+ε̂)",
         f"ρ = {_fmt(float(sam.rho))}, adaptive {str(sam.explicit.adaptive).lower()}", "2", "all weights",
         "L_meta(φ + ε̂), ε̂ = ρ∇_φL_meta/‖∇_φL_meta‖ (Algorithm 1, lines 21–22)", _fmt(float(sam.meta.coef)),
         f"{sam.meta.space} (alternative: θ_T)", off,
         "λ_max of the Hessian; max loss in an ε-ball", "configs/regularizers/sam.yaml · regularizer.rho, .meta.coef",
         "Foret et al. 2021"],
        ["Gradient-norm Aware Minimization", "first-order flatness — small ‖∇L‖ around θ",
         f"min_θ L(θ) + ρ‖∇L(θ)‖ (Hessian–vector product by a finite difference, radius {_fmt(float(gam.explicit.fd_radius))})",
         f"ρ = {_fmt(float(gam.rho))}, form {gam.form}", "2", "all weights",
         "ρ‖∇_φL_meta‖ (its gradient by a finite difference in φ)", _fmt(float(gam.meta.coef)), str(gam.meta.space), off,
         "Tr(H); max ‖∇L‖ in an ε-ball", "configs/regularizers/gam.yaml · regularizer.rho, .meta.coef",
         "Zhang et al. 2023"],
        ["Surrogate-Gap guided SAM", "flat and low — small surrogate gap h = L(θ+ε̂) − L(θ)",
         "SAM step, plus α × the component of ∇L(θ) orthogonal to ∇L(θ+ε̂) (descent on h)",
         f"ρ = {_fmt(float(gsam.rho))}, α = {_fmt(float(gsam.alpha))}", "2", "all weights",
         "GSAM's direction on φ: ∇_φL_meta(φ+ε̂) − α∇⊥_φL_meta(φ)", _fmt(float(gsam.meta.coef)),
         str(gsam.meta.space), off, "λ_max; surrogate gap in an ε-ball",
         "configs/regularizers/gsam.yaml · regularizer.rho, .alpha, .meta.coef", "Zhuang et al. 2022"],
        ["Smoothness of the learned optimizer (L_smooth)", "a smooth update rule — similar inputs, similar steps",
         "— (not a regularizer of θ; never used by M2)", "—", "—", "the LSTM's input s_t (normalised gradient)",
         f"max over ‖s'−s_t‖∞ ≤ ε of ‖u(s_t) − u(s')‖²₂, hidden state fixed: ε = {_fmt(float(smooth.epsilon))}, "
         f"{smooth.pga_steps} PGA steps of ε·{_fmt(float(smooth.step_ratio))}, start std ε·{_fmt(float(smooth.init_ratio))}, "
         f"every {smooth.every} step(s), {smooth.reduction} over weights",
         f"λ_smooth = {_fmt(float(smooth.coef))} (M4 arms only)", "the LSTM's input space", "not computed",
         "—", "configs/regularizers/smooth.yaml · smooth.coef, .epsilon, .pga_steps", "Xiong & Hsieh 2022"],
    ]
    return ["L1", "L2", "SAM", "GAM", "GSAM", "L_smooth (meta-objective term)"], attributes, cols


def _sweep_rows() -> List[List[str]]:
    regs = {name: load_group_file("regularizer", name) for name in REGULARIZERS}
    adamw, lstm, smooth = load_group_file("optimizer", "adamw"), load_group_file("optimizer", "lstm"), \
        load_group_file("smooth", "smooth")
    meta = compose_meta(["meta_training=m3_lstm", "profile=production"]).meta_training
    sources = "each round's source pair (ConvNet-4 / SVHN; MLP / FMNIST)"
    stage1 = "the round's Stage-1 meta-training (short budget)"
    val = "validation top-1: training.validation_fraction=0.1 (held-out training images; never the test set)"
    meta_val = "meta-validation loss (fresh optimizees, held-out meta images)"
    return [
        ["M1, M2", "—", "AdamW learning rate", "optimizer.lr", _fmt(float(adamw.lr)), "3e-4, 1e-3, 3e-3", sources, val,
         "shared by M1 and M2; weight decay stays 0 (M1 is the unregularised baseline)"],
        ["M2", "L1", "λ", "regularizer.explicit.coef", _fmt(float(regs["l1"].explicit.coef)), "1e-6, 1e-5, 1e-4, 1e-3",
         sources, val, "too large: accuracy drops with sparsity"],
        ["M2", "L1", "mode", "regularizer.explicit.mode", str(regs["l1"].explicit.mode), "penalty, proximal", sources, val,
         "proximal gives exact zeros"],
        ["M2", "L2", "λ", "regularizer.explicit.coef", _fmt(float(regs["l2"].explicit.coef)), "1e-4, 5e-4, 1e-3, 5e-3",
         sources, val, ""],
        ["M2", "L2", "mode · decoupled coefficient", "regularizer.explicit.mode, .decoupled_coef",
         f"{regs['l2'].explicit.mode} · {_fmt(float(regs['l2'].explicit.decoupled_coef))}",
         "penalty; decoupled with 0.01, 0.05, 0.1", sources, val, "decoupled = AdamW's own weight decay"],
        ["M2", "SAM", "ρ", "regularizer.rho", _fmt(float(regs["sam"].rho)), "0.01, 0.02, 0.05, 0.1, 0.2", sources, val,
         "Foret et al.'s grid"],
        ["M2", "GAM", "ρ", "regularizer.rho", _fmt(float(regs["gam"].rho)), "0.01, 0.02, 0.05, 0.1", sources, val, ""],
        ["M2", "GSAM", "ρ", "regularizer.rho", _fmt(float(regs["gsam"].rho)), "0.02, 0.05, 0.1", sources, val, ""],
        ["M2", "GSAM", "α", "regularizer.alpha", _fmt(float(regs["gsam"].alpha)), "0.01, 0.1, 0.3", sources, val,
         "Zhuang et al. tune α in [0.01, 0.3]"],
        *[["M4", REG_NAMES[r], "λ_reg (weight of L_reg)", "regularizer.meta.coef", _fmt(float(regs[r].meta.coef)),
           "0.01, 0.1, 1.0", stage1, meta_val, "1.0 = M2's own strength (explicit normalisation)" if r in ("l1", "l2") else ""]
          for r in REGULARIZERS],
        ["M4", "all", "λ_smooth (weight of L_smooth)", "smooth.coef", _fmt(float(smooth.coef)), "0.1, 1.0, 10",
         stage1, meta_val, "0 = M4 without L_smooth (ablation)"],
        ["M4", "all", "ε of L_smooth", "smooth.epsilon", _fmt(float(smooth.epsilon)), "0.003, 0.01, 0.03", stage1,
         meta_val, ""],
        ["M3, M4", "—", "meta-learning rate", "meta_training.outer.lr", _fmt(float(meta.outer.lr)), "1e-4, 3e-4, 1e-3",
         stage1, meta_val, "the same for M3 and every M4 (the manifest check enforces it)"],
        ["M3, M4", "—", "unroll length T", "meta_training.inner.unroll_steps", str(meta.inner.unroll_steps),
         "10, 20, 50", stage1, meta_val, "must divide every curriculum length (100, 200, 500, 1,000)"],
        ["M3, M4", "—", "output scale", "optimizer.output_scale", _fmt(float(lstm.output_scale)), "1e-4, 1e-3, 1e-2",
         stage1, meta_val, "the LSTM's initial step size"],
    ]


METRIC_ROWS = [
    ["Test accuracy (top-1)", "test_top1", "share of the test images classified correctly, after the last epoch",
     "%", "higher", "training/trainer.py → final.json", "mean ± sample std over seeds", "T4, Cubes, Cross-evaluation",
     "the headline performance"],
    ["Training accuracy (clean)", "train_top1", "top-1 on the run's own training images, no augmentation, after the last epoch",
     "%", "—", "training/trainer.py → final.json", "per run", "Runs", "the other half of the gap"],
    ["Generalization gap (top-1)", "gen_gap_top1", "training top-1 − test top-1 of the trained network", "pp", "lower",
     "final.json (Runs: a formula)", "mean ± sample std over seeds", "T4, Cubes, Cross-evaluation",
     "how much the network overfits — what the regularizers aim to reduce; needs no second round"],
    ["Generalization gap (loss)", "gen_gap", "test loss − training loss (cross-entropy)", "nats", "lower",
     "final.json (Runs: a formula)", "per run", "Runs; report tables", "the same, in loss"],
    ["FLOPs per training step", "flops_per_step", "forward + backward of one batch (128), + the extra passes of SAM, GAM, "
     "GSAM, + the LSTM's update for M3/M4", "GFLOPs", "lower", "diagnostics/flops.py, once per run → final.json",
     "identical across seeds", "T4, Cubes", "the training cost of each method; M4 should cost like M3, not like M2"],
    ["Wall-clock training time", "wall_clock_s", "time of the training steps (no evaluation, no checkpoints)", "h",
     "lower", "final.json", "per run", "Runs", "GPUs differ: compare methods with the same-GPU timing (Stage 4)"],
    ["Δ vs M1", "—", "mean of the method − mean of M1, same pair and round", "pp", "accuracy: higher · gap: lower",
     "T4 (formula); report: with Welch p", "of the seed means", "T4, Cubes (benchmark)",
     "the gain over unregularised AdamW"],
    ["Advantage over M1 by role", "—", "mean over the pairs of one role of (variant − M1)", "pp",
     "accuracy: higher · gap: lower", "Cross-evaluation (formula); report: transfer_summary", "of the seed means",
     "Cross-evaluation", "how the learned optimizer's advantage changes from seen to unseen tasks"],
    ["Transfer gap", "—", "on a round's source pair: its own optimizers − the other round's (unseen there)", "pp",
     "accuracy: positive = meta-training on the task helped", "Cross-evaluation (formula); report: transfer_summary",
     "of the seed means", "Cross-evaluation", "what not having meta-trained on a task costs — needs ≥ 2 rounds"],
    ["Meta-validation loss", "best_val_loss", "mean test loss of fresh optimizees trained with the frozen φ on the "
     "held-out 10% of the meta half (selects phi_best.pt)", "nats", "lower", "meta_training/meta_trainer.py → summary.json",
     "one meta-training run per arm", "Stage 1 meta-training", "how good the learned optimizer is on its own task"],
    ["L_meta · L_smooth · L_reg", "l_meta, l_smooth, l_reg", "the terms of the meta-objective (Algorithm 1), mean of the "
     "last 10% of meta-steps", "—", "lower", "meta_log.jsonl", "—", "Stage 1 meta-training",
     "whether the regularization terms are active and decreasing"],
    ["λ_max · Tr(H)", "lambda_max, hessian_trace", "top Hessian eigenvalue (power iteration) and trace (Hutchinson) "
     "at the final weights", "—", "lower", "Stage 4 audits (diagnostics/hessian.py)", "mean ± std", "report tables per pair",
     "sharpness — SAM's, GSAM's and GAM's target"],
    ["Max loss · surrogate gap · max ‖∇L‖ in an ε-ball", "max_loss_*, surrogate_gap_*, max_grad_norm_*",
     "the paper's point-of-convergence tests at convergence and at completion, ε = 0.01 by default", "—", "lower",
     "Stage 4 audits (diagnostics/sharpness.py)", "mean ± std", "report tables per pair", "flatness in the sense of each r"],
    ["Sparsity · ‖θ‖₁ · ‖θ‖₂", "sparsity, l1, l2", "share of penalised weights with |w| < 1e-3; their norms", "% · —",
     "L1: sparsity higher · L2: ‖θ‖₂ lower", "diagnostics/norms.py → final.json", "mean ± std", "report tables per pair",
     "the property L1 and L2 target"],
]


# ═════════════════════════════════════════════════════════════════ writing

class _Book:
    """openpyxl with the house style: Arial, bold headers, light grid, blue cells to fill in, no fills."""

    def __init__(self) -> None:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, Side

        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.Font, self.Alignment, self.Border, self.Side = Font, Alignment, Border, Side
        self.thin = Side(style="thin", color="BFBFBF")
        self.heavy = Side(style="medium", color="000000")

    def sheet(self, name: str):
        return self.wb.create_sheet(name)

    def put(self, ws, row: int, col: int, value: Any = None, bold: bool = False, fmt: Optional[str] = None,
            fill_in: bool = False, wrap: bool = False, italic: bool = False, size: int = 10, grid: bool = False,
            header: bool = False, align: Optional[str] = None, color: Optional[str] = None):
        cell = ws.cell(row=row, column=col)
        if value is not None:
            cell.value = value
        cell.font = self.Font(name="Arial", size=size, bold=bold or header, italic=italic,
                              color=color or ("0000FF" if fill_in else "000000"))
        cell.alignment = self.Alignment(wrap_text=wrap or header, vertical="top", horizontal=align)
        if fmt:
            cell.number_format = fmt
        if header:
            cell.border = self.Border(bottom=self.heavy, top=self.thin, left=self.thin, right=self.thin)
        elif grid:
            cell.border = self.Border(bottom=self.thin, top=self.thin, left=self.thin, right=self.thin)
        return cell

    def title(self, ws, text: str, note: str = "") -> int:
        self.put(ws, 1, 1, text, bold=True, size=14)
        if note:
            self.put(ws, 2, 1, note, italic=True, size=9, color="595959")
        return 4

    def widths(self, ws, widths: Sequence[float]) -> None:
        from openpyxl.utils import get_column_letter

        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    def fit_rows(self, ws, first: int = 1, last: Optional[int] = None) -> None:
        """Row heights for wrapped text, estimated from the column widths (Excel and LibreOffice
        both honour them; neither re-measures wrapped text written by a library)."""
        from openpyxl.utils import get_column_letter

        width_of = {}
        for merged in ws.merged_cells.ranges:
            total = sum((ws.column_dimensions[get_column_letter(c)].width or 8.43)
                        for c in range(merged.min_col, merged.max_col + 1))
            width_of[(merged.min_row, merged.min_col)] = total
        last = last or ws.max_row
        for row in ws.iter_rows(min_row=first, max_row=last):
            lines = 1
            for cell in row:
                if not isinstance(cell.value, str) or cell.value.startswith("=") or not cell.alignment.wrap_text:
                    continue
                width = width_of.get((cell.row, cell.column),
                                     ws.column_dimensions[get_column_letter(cell.column)].width or 8.43)
                per_line = max(int(width * 1.1) - 1, 4)
                needed = sum(max(1, -(-len(part) // per_line)) for part in cell.value.split("\n"))
                lines = max(lines, needed)
            if lines > 1:
                ws.row_dimensions[row[0].row].height = 13.0 * lines + 3

    def landscape(self, ws, fit_width: bool = True) -> None:
        ws.page_setup.orientation = "landscape"
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        if fit_width:
            ws.sheet_properties.pageSetUpPr.fitToPage = True
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0


def _col(index: int) -> str:
    from openpyxl.utils import get_column_letter

    return get_column_letter(index)


def _ref(sheet: str, col: str, row: int) -> str:
    return f"'{sheet}'!${col}${row}"


RUN_COLUMNS = [  # (header, width, kind) — kind: id | text | in (to fill) | formula
    ("Run ID", 8, "id"), ("Experiment file", 25, "text"), ("Study", 7, "text"), ("Optimizee", 11, "text"),
    ("Dataset", 9, "text"), ("Method", 13, "text"), ("Regularizer r", 10, "text"), ("Round of φ", 8, "text"),
    ("Seed", 6, "text"), ("r computed during training", 16, "text"), ("Optimizer", 24, "text"),
    ("Training images", 10, "text"), ("Epochs", 7, "text"), ("Steps", 9, "text"),
    ("Serves (round · stage · setting)", 34, "text"), ("Run folder (artifacts/<profile>/…)", 44, "text"),
    ("Status", 10, "in"), ("Test top-1 (%)", 9, "in"), ("Train top-1 (%)", 9, "in"), ("Gen. gap (pp)", 9, "formula"),
    ("Test loss", 9, "in"), ("Train loss", 9, "in"), ("Gen. gap (loss)", 9, "formula"), ("GFLOPs / step", 9, "in"),
    ("Wall-clock (h)", 9, "in"), ("Notes", 30, "in"),
]
RC = {name: _i for _i, (name, _, _) in enumerate(RUN_COLUMNS, start=1)}

EXP_COLUMNS = [
    ("Exp ID", 10), ("Stage", 7), ("Round", 7), ("Setting", 24), ("Role of the pair", 18), ("Optimizee", 11),
    ("Dataset", 9), ("Method", 13), ("Regularizer r", 11), ("r computed here", 20), ("Optimizer (updates θ)", 24),
    ("Trained / evaluated", 34), ("Seeds", 9), ("Runs", 14), ("Same runs as", 10), ("Status (done / runs)", 10),
    ("Test top-1 mean (%)", 10), ("± std", 7), ("Gen. gap mean (pp)", 10), ("± std ", 7), ("GFLOPs / step", 9),
    ("Δ top-1 vs M1 (pp)", 9), ("Δ gap vs M1 (pp)", 9), ("What it signifies", 46), ("Remarks", 30),
]
EC = {name: _i for _i, (name, _) in enumerate(EXP_COLUMNS, start=1)}

META_COLUMNS = [
    ("Exp ID", 10), ("Round", 6), ("Source pair", 18), ("Arm", 18), ("Serves", 8), ("Regularizer r", 10),
    ("Meta-objective", 34), ("λ_smooth", 8), ("λ_reg", 7), ("Budget", 26), ("Output φ (artifacts/<profile>/…)", 36),
    ("Status", 10), ("Trajectories done", 10), ("Meta-steps", 10), ("Best val. loss", 10), ("Best val. acc. (%)", 10),
    ("Best at trajectory", 10), ("L_meta (end)", 10), ("L_smooth (end)", 10), ("L_reg (end)", 10),
    ("Wall-clock (h)", 9), ("Notes", 30),
]
MC = {name: _i for _i, (name, _) in enumerate(META_COLUMNS, start=1)}
META_INPUTS = ("Status", "Trajectories done", "Meta-steps", "Best val. loss", "Best val. acc. (%)", "Best at trajectory",
               "L_meta (end)", "L_smooth (end)", "L_reg (end)", "Wall-clock (h)", "Notes")


def _optimizer_text(method: str, reg: str, rnd: str) -> str:
    name = REG_NAMES.get(reg, reg)
    if method == "m1_adamw":
        return "AdamW"
    if method == "m2_adamw_reg":
        return f"AdamW + {name}"
    if method == "m3_lstm":
        return f"frozen LSTM φ (round {rnd})"
    return f"frozen LSTM φ*_{name} (round {rnd})"


def _r_text(method: str, reg: str) -> str:
    if method == "m1_adamw":
        return "never"
    if method == "m2_adamw_reg":
        return "every step (+ its passes)"
    if method == "m3_lstm":
        return "never"
    return "no — only in Stage-1 meta-training"


def _signifies(exp: Exp) -> str:
    name = REG_NAMES.get(exp.reg, exp.reg)
    if exp.arm is not None:
        return ("Learns the vanilla update rule φ on the round's source task (L_meta only)." if exp.method == "m3_lstm"
                else f"Learns φ*_{name} with {name} inside the meta-objective — the regularizer is paid for here, once.")
    if exp.stage == "1":
        return ("Unregularised AdamW reference for the benchmark (the seen-setting runs)." if exp.method == "m1_adamw"
                else f"Primary benchmark: what {name} does for a conventional optimizer (against M1).")
    if exp.stage == "pilot":
        return "Early check: does the learned optimizer hold up over long training?"
    where = {"seen": "on the task it was meta-trained on", "unseen_optimizee": "on a new network (same data)",
             "unseen_dataset": "on new data (same network)", "unseen_both": "on a new network and new data"}[exp.role]
    if exp.method == "m1_adamw":
        return "AdamW reference on this task."
    if exp.method == "m2_adamw_reg":
        return f"Explicit-regularization baseline: AdamW + {name} computed at every step, {where}."
    if exp.method == "m3_lstm":
        return f"Vanilla learned optimizer {where}."
    return f"Does {name}'s effect appear without computing {name}, {where}?"


def _trained_text(exp: Exp) -> str:
    name = REG_NAMES.get(exp.reg, exp.reg)
    if exp.arm is not None:
        what = "φ (L_meta)" if exp.method == "m3_lstm" else f"φ*_{name} (L_meta + λ_smooth·L_smooth + λ_reg·L_{name})"
        return f"meta-trains the LSTM {what} on {exp.pair.label}'s meta half"
    by = _optimizer_text(exp.method, exp.reg, exp.round)
    return f"trains {exp.pair.label} with {by}; test/train top-1, FLOPs"


def _runs_text(runs: Sequence[Run]) -> str:
    if not runs:
        return ""
    return runs[0].run_id if len(runs) == 1 else f"{runs[0].run_id}–{runs[-1].run_id}"


def _contiguous(runs: Sequence[Run]) -> bool:
    rows = [r.row for r in runs]
    return rows == list(range(rows[0], rows[0] + len(rows)))


def _range(col_name: str, runs: Sequence[Run]) -> str:
    col = _col(RC[col_name])
    if _contiguous(runs):
        return f"'{RUNS}'!${col}${runs[0].row}:${col}${runs[-1].row}"
    return ",".join(f"'{RUNS}'!${col}${r.row}" for r in runs)


def build_workbook(path: PathLike, profile: str = "production", plan: Optional[Plan] = None) -> Path:
    """Write the workbook of the study at the scale of ``profile`` to ``path`` (results blank)."""
    plan = plan or plan_study_workbook(profile)
    book = _Book()
    for name in SHEETS:
        book.sheet(name)
    _layout(plan)
    _read_me(book, plan)
    _table1(book, plan)
    _table2(book)
    _table3(book, plan)
    _runs_sheet(book, plan)
    _meta_sheet(book, plan)
    _experiments_sheet(book, plan)
    _cubes_sheet(book, plan)
    _cross_sheet(book, plan)
    _metrics_sheet(book)
    book.wb.calculation.fullCalcOnLoad = True  # Excel computes every formula when the file is opened
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.wb.save(path)
    return path


# ───────────────────────────────────────────────────────────────── row numbers first: the formulas need them

RUNS_FIRST, META_FIRST = 6, 6


def _layout(plan: Plan) -> None:
    for i, run in enumerate(plan.runs):
        run.row = RUNS_FIRST + i
    for i, arm in enumerate(plan.arms):
        arm.row = META_FIRST + i
    row, section = 5, None
    for exp in plan.exps:
        if exp.section != section:
            section = exp.section
            row += 1  # the section heading
        exp.row = row
        row += 1


# ───────────────────────────────────────────────────────────────── sheets

def _read_me(book: _Book, plan: Plan) -> None:
    ws = book.wb["Read me"]
    book.widths(ws, [32, 34, 28, 26, 36, 32, 24])
    row = book.title(ws, "Learned optimizers that internalise regularization — experiment plan and results",
                     f"Generated {_dt.date.today():%d %b %Y} from the configs (profile: {plan.profile}). "
                     "Blank cells are results not computed yet.")
    text = [
        "M.Tech thesis study building on Sahoo & Krishnan, 'Learning Regularizers: Learning Optimizers that can "
        "Regularize' (arXiv:2510.08968), Algorithm 1. A coordinate-wise LSTM optimizer is meta-trained with a regularizer r "
        "inside its objective (M4) and afterwards trains new networks without computing r. Is r's effect there anyway, "
        "and at what cost? Every comparison is made one regularizer at a time: M1 · M2(r) · M3 · M4(r).",
    ]
    book.put(ws, row, 1, text[0], wrap=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 2

    def table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]], heights: int = 0) -> int:
        nonlocal row
        book.put(ws, row, 1, title, bold=True, size=11)
        row += 1
        last = len(headers)

        def widen(header: bool) -> None:  # the last column runs to the sheet's right edge (column G)
            if last < 7:
                for c in range(last + 1, 8):
                    book.put(ws, row, c, None, header=header, grid=not header)
                ws.merge_cells(start_row=row, start_column=last, end_row=row, end_column=7)

        for c, h in enumerate(headers, start=1):
            book.put(ws, row, c, h, header=True)
        widen(True)
        row += 1
        for values in rows:
            for c, v in enumerate(values, start=1):
                book.put(ws, row, c, v, wrap=True, grid=True)
            widen(False)
            row += 1
        row += 1
        return row

    table("The four methods", ["Method", "Optimizer", "Regularizer r", "When r is computed", "Stage 1", "Stages 2–3"], [
        ["M1 · Optimizee + AdamW", "AdamW", "—", "never", "reference for M2 (its seen-setting runs)", "trained, evaluated"],
        ["M2 · Optimizee + AdamW + Reg", "AdamW", "L1, L2, SAM, GAM, GSAM", "every step, every stage",
         "trained with r on each round's source pair: the primary benchmark",
         "trained with r on every pair (source pairs: the Stage-1 runs)"],
        ["M3 · Optimizee + learned optimizer", "frozen LSTM φ", "—", "never", "φ meta-trained on L_meta", "trained, evaluated"],
        ["M4 · learned optimizer + custom regularizer (proposed)", "frozen LSTM φ*_r", "L1, L2, SAM, GAM, GSAM",
         "only inside Stage-1 meta-training", "φ*_r meta-trained on L_meta + λ_smooth·L_smooth + λ_reg·L_reg",
         "trained, evaluated — no r computed"],
    ], heights=40)
    table("The stages", ["Stage", "What happens", "Output", "Sheets"], [
        ["1 · Training", "per round: meta-train φ (M3) and φ*_r for each r (M4) on the source pair's meta half; "
         "train M2 = AdamW + r on the source pair", "12 LSTM optimizers, M2's benchmark runs", "Stage 1 meta-training, T4 (E1), Runs"],
        ["2 · Optimizer evaluation", "M1, M2 (with r) and the frozen optimizers train fresh networks: the source network "
         "(seen optimizee) and other networks on the source dataset (unseen optimizee)", "test / train top-1, FLOPs",
         "T4 (E2), Runs"],
        ["3 · Optimizee evaluation", "the same on the source dataset (seen dataset) and on other "
         "datasets, with the source network or another one (unseen dataset)", "test / train top-1, FLOPs", "T4 (E3), Runs"],
        ["4 · Evaluation metrics", "test accuracy ↑, generalization gap ↓, FLOPs ↓ as 3D tables: method × regularizer "
         "× evaluation setting, per round; cross-evaluation of the rounds", "the thesis tables",
         "Cubes, Cross-evaluation, Metrics"],
    ], heights=40)
    role_rows = []
    pool = []
    for per in plan.settings.values():
        for pairs in per.values():
            for sp in pairs:
                if not sp.extended and sp.pair not in pool:
                    pool.append(sp.pair)
    order_d, order_o = list(DATASET_NAMES), list(OPTIMIZEE_NAMES)
    pool.sort(key=lambda p: (order_d.index(p.dataset), order_o.index(p.optimizee)))
    for pair in pool:
        role_rows.append([pair.label] + [ROLE_LABELS[r.role(pair.optimizee, pair.dataset)] for r in plan.rounds])
    table("The rounds — each meta-trains its optimizers on one source pair; every pair is trained with every round's "
          "optimizers", ["Optimizee / dataset"] + [f"Round {r.name}: source {Pair(r.optimizee, r.dataset).label}"
                                                    for r in plan.rounds], role_rows)
    book.put(ws, row - 1, 1, "Stage 2 = seen / unseen optimizee (the source dataset); Stage 3 = seen dataset and every "
             "other dataset (unseen dataset; 'unseen optimizee + dataset' when the network is new too).", italic=True,
             size=9, color="595959")
    row += 1
    counts = {}
    for run in plan.runs:
        counts.setdefault(run.experiment, 0)
        counts[run.experiment] += 1
    arms = len(plan.arms)
    table("Size of the study (this profile)", ["Grid", "Runs", "Note"], [
        ["Stage 1 meta-training", arms, f"{len(plan.rounds)} rounds × 6 LSTM optimizers"],
        *[[name, n, ""] for name, n in counts.items()],
        ["Total optimizee runs", len(plan.runs), "every seed counted; M2 on a source pair: the Stage-1 runs, not repeated"],
    ])
    table("The sheets", ["Sheet", "Content"], [
        ["T1 Optimizees & Optimizers", "Table 1: every optimizee, optimizer and dataset with its settings; the training protocol"],
        ["T2 Regularizers", "Table 2: L1, L2, SAM, GAM, GSAM and L_smooth — objective, settings, where each is used"],
        ["T3 Sweeps", "Table 3: hyperparameters to sweep per optimizer and regularizer, cells for the best values"],
        ["T4 Experiments", "Table 4: every experiment (Exp ID), stage by stage and round by round, with its metrics"],
        ["Cubes", "Stage 4: test accuracy, generalization gap and FLOPs as method × regularizer × setting, per round; "
                  "the Stage-1 benchmark"],
        ["Cross-evaluation", "every pair with each round's optimizers; advantage over M1 by role; the transfer gap"],
        ["Stage 1 meta-training", "one row per LSTM optimizer — cells to fill"],
        ["Runs", "one row per training run (seed) — cells to fill; everything else is computed from here"],
        ["Metrics", "what every number means and where the code computes it"],
    ])
    book.put(ws, row, 1, "How to fill it in", bold=True, size=11)
    row += 1
    for line in [
        "Blue cells are inputs: Runs (Status … Notes), Stage 1 meta-training (Status … Notes), T3 Sweeps (best values) and "
        "the Remarks columns. Black cells are design values or formulas — leave them.",
        "Status: done · running · diverged · failed. A run counts in the means once its numbers are in; T4's status "
        "column counts the runs marked done.",
        "Automatically: python scripts/make_workbook.py --fill (or --fill path\\to\\this.xlsx) writes every finished result "
        "from artifacts/<profile>/ into the blue cells of this file and keeps everything else.",
        "Accuracies are in percent (91.23 means 91.23 %); gaps in percentage points; FLOPs in GFLOPs per training step.",
    ]:
        book.put(ws, row, 1, line, wrap=True)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 1
    row += 1
    book.put(ws, row, 1, "Example of a filled Runs row (format only — not a result)", bold=True, size=11)
    row += 1
    example_h = ["Run ID", "Method", "Seed", "Status", "Test top-1 (%)", "Train top-1 (%)", "Gen. gap (pp)"]
    example_v = ["R0031", "M1 AdamW", 42, "done", 91.85, 99.10, "=F{r}-E{r}"]
    for c, h in enumerate(example_h, start=1):
        book.put(ws, row, c, h, header=True)
    row += 1
    for c, v in enumerate(example_v, start=1):
        value = v.format(r=row) if isinstance(v, str) and v.startswith("=") else v
        book.put(ws, row, c, value, grid=True, fill_in=4 <= c <= 6, fmt="0.00" if c >= 5 else None, italic=True)
    ws.freeze_panes = "A4"
    book.fit_rows(ws, 4)
    book.landscape(ws)


def _block_sheet(book: _Book, ws, row: int, groups: Sequence[Tuple[str, List[str], List[str], List[List[str]]]]) -> int:
    """Side-by-side blocks: a group heading over its entities, an attribute column, one column per entity."""
    col = 1
    tallest = 0
    for heading, heads, attributes, columns in groups:
        book.put(ws, row, col + 1, heading, bold=True, size=11)
        ws.merge_cells(start_row=row, start_column=col + 1, end_row=row, end_column=col + len(heads))
        book.put(ws, row + 1, col, "Attribute", header=True)
        for j, head in enumerate(heads, start=1):
            book.put(ws, row + 1, col + j, head, header=True)
        for i, attribute in enumerate(attributes):
            book.put(ws, row + 2 + i, col, attribute, bold=True, wrap=True, grid=True)
            for j, values in enumerate(columns, start=1):
                book.put(ws, row + 2 + i, col + j, values[i], wrap=True, grid=True)
        tallest = max(tallest, len(attributes))
        col += len(heads) + 2
    return row + 2 + tallest


def _table1(book: _Book, plan: Plan) -> None:
    ws = book.wb["T1 Optimizees & Optimizers"]
    row = book.title(ws, "Table 1 — Optimizees, optimizers and datasets",
                     "Production values (configs/); smoke-profile sizes where they differ. Read each column downwards.")
    heads, attributes, cols = _optimizee_block(plan.rounds)
    o_heads, o_attributes, o_cols = _optimizer_block()
    end = _block_sheet(book, ws, row, [("OPTIMIZEES θ — the networks being trained", heads, attributes, cols),
                                       ("OPTIMIZERS — what updates θ", o_heads, o_attributes, o_cols)])
    widths = [26] + [24] * len(heads) + [3] + [30] + [40, 52]
    book.widths(ws, widths)
    row = end + 2
    d_heads, d_attributes, d_cols = _dataset_block(plan.rounds)
    end = _block_sheet(book, ws, row, [("DATASETS", d_heads, d_attributes, d_cols)])
    row = end + 2
    book.put(ws, row, 1, "TRAINING PROTOCOL — every optimizee run (M1–M4, Stages 1–3)", bold=True, size=11)
    row += 1
    book.put(ws, row, 1, "Item", header=True)
    book.put(ws, row, 2, "Value", header=True)
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 1
    for item, value in _protocol_rows(plan):
        book.put(ws, row, 1, item, bold=True, grid=True, wrap=True)
        book.put(ws, row, 2, value, grid=True, wrap=True)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
        row += 1
    ws.freeze_panes = "B6"
    book.fit_rows(ws, 4)
    book.landscape(ws)


def _table2(book: _Book) -> None:
    ws = book.wb["T2 Regularizers"]
    row = book.title(ws, "Table 2 — Regularizers and their configuration",
                     "M2 computes r at every step wherever it trains (Stages 1–3); M4 meets r only inside its "
                     "Stage-1 meta-objective; M1 and M3 never.")
    heads, attributes, cols = _regularizer_block()
    book.widths(ws, [30] + [34] * len(heads))
    _block_sheet(book, ws, row, [("REGULARIZERS r  ·  and the meta-objective's smoothness term", heads, attributes, cols)])
    ws.freeze_panes = "B6"
    book.fit_rows(ws, 4)
    book.landscape(ws)


def _table3(book: _Book, plan: Plan) -> None:
    ws = book.wb["T3 Sweeps"]
    rounds = plan.rounds
    row = book.title(ws, "Table 3 — Sweeps: hyperparameters of every optimizer and regularizer",
                     "Current value = what the study uses now (configs/). Fill the best values; then set them in the "
                     "config files before the main study. Never select on the test set.")
    headers = ["#", "Method(s)", "Regularizer", "Hyperparameter", "Config key (override)", "Current value",
               "Suggested grid", "Sweep on", "Select by"]
    for r in rounds:
        headers += [f"Best value — round {r.name}", f"Selection metric at best — {r.name}"]
    headers += ["Remarks"]
    for c, h in enumerate(headers, start=1):
        book.put(ws, row, c, h, header=True)
    row += 1
    for i, values in enumerate(_sweep_rows(), start=1):
        cells = [i] + values[:8]
        for c, v in enumerate(cells, start=1):
            book.put(ws, row, c, v, wrap=True, grid=True)
        c = len(cells) + 1
        for _ in rounds:
            book.put(ws, row, c, None, grid=True, fill_in=True)
            book.put(ws, row, c + 1, None, grid=True, fill_in=True, fmt="0.00")
            c += 2
        book.put(ws, row, c, values[8], wrap=True, grid=True, fill_in=False)
        row += 1
    row += 1
    for note in [
        "Run each sweep value in its own study namespace, so runs do not overwrite each other, e.g.",
        "python scripts/train_optimizee.py method=m2_adamw_reg regularizer=sam regularizer.rho=0.1 optimizee=convnet4 "
        "dataset=svhn training.validation_fraction=0.1 study=sweep_sam_rho0.1",
        "python scripts/meta_train.py meta_training=m4_lstm_reg regularizer=sam regularizer.meta.coef=1.0 meta_tag=sweep_lam1 "
        "meta_training.schedule.num_trajectories=500",
        "Suggested grids follow the original papers (SAM: Foret et al.; GSAM: Zhuang et al.; L2O: Chen et al. 2020).",
    ]:
        book.put(ws, row, 1, note, italic=True, size=9, color="595959")
        row += 1
    book.widths(ws, [4, 9, 12, 22, 28, 12, 22, 30, 30] + [12, 12] * len(rounds) + [36])
    ws.freeze_panes = "E5"
    book.fit_rows(ws, 4)
    book.landscape(ws)


def _runs_sheet(book: _Book, plan: Plan) -> None:
    ws = book.wb[RUNS]
    book.title(ws, "Runs — one row per training run (seed); fill the blue cells",
               "Everything in T4, Cubes and Cross-evaluation is computed from these cells. "
               "python scripts/make_workbook.py --fill fills them from artifacts/<profile>/.")
    book.widths(ws, [w for _, w, _ in RUN_COLUMNS])
    header_row = RUNS_FIRST - 1
    for c, (name, _, _) in enumerate(RUN_COLUMNS, start=1):
        book.put(ws, header_row, c, name, header=True)
    test, train = _col(RC["Test top-1 (%)"]), _col(RC["Train top-1 (%)"])
    tl, trl = _col(RC["Test loss"]), _col(RC["Train loss"])
    for run in plan.runs:
        r = run.row
        values = {
            "Run ID": run.run_id, "Experiment file": run.experiment, "Study": run.study,
            "Optimizee": OPTIMIZEE_NAMES.get(run.pair.optimizee, run.pair.optimizee),
            "Dataset": DATASET_NAMES.get(run.pair.dataset, run.pair.dataset),
            "Method": f"{LABELS[run.method]} {'AdamW' if run.method in ('m1_adamw', 'm2_adamw_reg') else 'LSTM'}",
            "Regularizer r": REG_NAMES.get(run.reg, run.reg), "Round of φ": run.round or "—", "Seed": run.seed,
            "r computed during training": "every step" if run.method == "m2_adamw_reg" else "never",
            "Optimizer": _optimizer_text(run.method, run.reg, run.round),
            "Training images": run.images or None, "Epochs": run.epochs, "Steps": run.steps or None,
            "Serves (round · stage · setting)": "; ".join(run.serves), "Run folder (artifacts/<profile>/…)": run.folder,
            "Gen. gap (pp)": f'=IF(AND(ISNUMBER({test}{r}),ISNUMBER({train}{r})),{train}{r}-{test}{r},"")',
            "Gen. gap (loss)": f'=IF(AND(ISNUMBER({tl}{r}),ISNUMBER({trl}{r})),{tl}{r}-{trl}{r},"")',
        }
        for c, (name, _, kind) in enumerate(RUN_COLUMNS, start=1):
            fmt = {"Training images": "#,##0", "Steps": "#,##0", "Test top-1 (%)": "0.00", "Train top-1 (%)": "0.00",
                   "Gen. gap (pp)": "0.00", "Test loss": "0.0000", "Train loss": "0.0000", "Gen. gap (loss)": "0.0000",
                   "GFLOPs / step": "0.000", "Wall-clock (h)": "0.00"}.get(name)
            book.put(ws, r, c, values.get(name), grid=True, fill_in=kind == "in", fmt=fmt,
                     wrap=name in ("Serves (round · stage · setting)",))
    ws.freeze_panes = ws.cell(row=RUNS_FIRST, column=2)
    ws.auto_filter.ref = f"A{header_row}:{_col(len(RUN_COLUMNS))}{header_row + len(plan.runs)}"
    book.fit_rows(ws, header_row)
    book.landscape(ws, fit_width=False)


def _meta_sheet(book: _Book, plan: Plan) -> None:
    ws = book.wb[META]
    book.title(ws, "Stage 1 meta-training — one row per LSTM optimizer; fill the blue cells",
               "From artifacts/<profile>/meta_training/<round>/<arm>/summary.json and meta_log.jsonl "
               "(or python scripts/make_workbook.py --fill).")
    book.widths(ws, [w for _, w in META_COLUMNS])
    for c, (name, _) in enumerate(META_COLUMNS, start=1):
        book.put(ws, META_FIRST - 1, c, name, header=True)
    meta = compose_meta(["meta_training=m4_lstm_reg", "regularizer=sam", f"profile={plan.profile}"]).meta_training
    smooth = load_group_file("smooth", "smooth")
    curriculum = meta.schedule.curriculum
    budget = (f"{meta.schedule.num_trajectories:,} trajectories; lengths "
              f"{', '.join(str(n) for n in curriculum.trajectory_steps)}")
    for arm in plan.arms:
        reg = arm.reg
        if arm.entry == "m3_lstm":
            objective, lam_s, lam_r = "L_meta", "—", "—"
        else:
            objective = f"L_meta + λ_smooth·L_smooth + λ_reg·L_{REG_NAMES[reg]}"
            lam_s, lam_r = _fmt(float(smooth.coef)), _fmt(float(load_group_file("regularizer", reg).meta.coef))
        values = {
            "Exp ID": arm.exp_id, "Round": arm.round, "Source pair": arm.source.label,
            "Arm": arm.entry if reg == "none" else f"{arm.entry}-{reg}",
            "Serves": "M3" if arm.entry == "m3_lstm" else f"M4-{REG_NAMES[reg]}", "Regularizer r": REG_NAMES.get(reg, reg),
            "Meta-objective": objective, "λ_smooth": lam_s, "λ_reg": lam_r, "Budget": budget,
            "Output φ (artifacts/<profile>/…)": f"{arm.folder}/phi_best.pt",
        }
        for c, (name, _) in enumerate(META_COLUMNS, start=1):
            fmt = {"Trajectories done": "#,##0", "Meta-steps": "#,##0", "Best val. loss": "0.0000",
                   "Best val. acc. (%)": "0.00", "Best at trajectory": "#,##0", "L_meta (end)": "0.0000",
                   "L_smooth (end)": "0.0000", "L_reg (end)": "0.0000", "Wall-clock (h)": "0.00"}.get(name)
            book.put(ws, arm.row, c, values.get(name), grid=True, fill_in=name in META_INPUTS, fmt=fmt,
                     wrap=name in ("Meta-objective", "Budget"))
    ws.freeze_panes = ws.cell(row=META_FIRST, column=2)
    book.fit_rows(ws, META_FIRST - 1)
    book.landscape(ws, fit_width=False)


def _experiments_sheet(book: _Book, plan: Plan) -> None:
    ws = book.wb[T4]
    book.title(ws, "Table 4 — Experiments, stage by stage and round by round",
               "One row per (stage, round, setting, pair, method, regularizer). Metrics are seed means ± sample std, "
               "computed from the Runs sheet. M2 computes r at every step in every stage.")
    book.widths(ws, [w for _, w in EXP_COLUMNS])
    for c, (name, _) in enumerate(EXP_COLUMNS, start=1):
        book.put(ws, 4, c, name, header=True)
    section = None
    status_col = _col(MC["Status"])
    for exp in plan.exps:
        if exp.section != section:
            section = exp.section
            book.put(ws, exp.row - 1, 1, section, bold=True, size=11)
        r = exp.row
        mean_acc, mean_gap = _col(EC["Test top-1 mean (%)"]), _col(EC["Gen. gap mean (pp)"])
        values: Dict[str, Any] = {
            "Exp ID": exp.exp_id, "Stage": exp.stage, "Round": exp.round, "Setting": exp.setting,
            "Role of the pair": ROLE_LABELS.get(exp.role, exp.role) if exp.arm is None else "source (meta half)",
            "Optimizee": OPTIMIZEE_NAMES.get(exp.pair.optimizee, exp.pair.optimizee),
            "Dataset": DATASET_NAMES.get(exp.pair.dataset, exp.pair.dataset),
            "Method": {"m1_adamw": "M1 AdamW", "m2_adamw_reg": "M2 AdamW + r",
                       "m3_lstm": "M3 LSTM", "m4_lstm_reg": "M4 LSTM-r"}[exp.method],
            "Regularizer r": REG_NAMES.get(exp.reg, exp.reg),
            "r computed here": "inside the meta-objective" if exp.arm is not None and exp.method == "m4_lstm_reg"
            else ("never (L_meta only)" if exp.arm is not None else _r_text(exp.method, exp.reg)),
            "Optimizer (updates θ)": "— (trains the LSTM)" if exp.arm else _optimizer_text(exp.method, exp.reg, exp.round),
            "Trained / evaluated": _trained_text(exp), "What it signifies": _signifies(exp),
            "Same runs as": exp.same_as or None,
        }
        if exp.arm is not None:
            values["Seeds"] = "0"
            values["Runs"] = exp.arm.folder.split("/", 1)[1]
            cell = f"'{META}'!${status_col}${exp.arm.row}"
            values["Status (done / runs)"] = f'=IF({cell}="","",{cell})'
            for name in ("Test top-1 mean (%)", "± std", "Gen. gap mean (pp)", "± std ", "GFLOPs / step",
                         "Δ top-1 vs M1 (pp)", "Δ gap vs M1 (pp)"):
                values[name] = "—"
        else:
            values["Seeds"] = ", ".join(str(r_.seed) for r_ in exp.runs)
            values["Runs"] = _runs_text(exp.runs)
            values["Status (done / runs)"] = f'=COUNTIF({_range("Status", exp.runs)},"done")&" / {len(exp.runs)}"' \
                if _contiguous(exp.runs) else f'="{len(exp.runs)} runs"'
            for name, source, fn in (("Test top-1 mean (%)", "Test top-1 (%)", "AVERAGE"), ("± std", "Test top-1 (%)", "STDEV"),
                                     ("Gen. gap mean (pp)", "Gen. gap (pp)", "AVERAGE"), ("± std ", "Gen. gap (pp)", "STDEV"),
                                     ("GFLOPs / step", "GFLOPs / step", "AVERAGE")):
                values[name] = f'=IFERROR({fn}({_range(source, exp.runs)}),"")'
            if exp.m1 is not None and exp.method != "m1_adamw":
                values["Δ top-1 vs M1 (pp)"] = f'=IFERROR({mean_acc}{r}-{mean_acc}{exp.m1.row},"")'
                values["Δ gap vs M1 (pp)"] = f'=IFERROR({mean_gap}{r}-{mean_gap}{exp.m1.row},"")'
        for c, (name, _) in enumerate(EXP_COLUMNS, start=1):
            fmt = {"Test top-1 mean (%)": "0.00", "± std": "0.00", "Gen. gap mean (pp)": "0.00", "± std ": "0.00",
                   "GFLOPs / step": "0.000", "Δ top-1 vs M1 (pp)": "+0.00;-0.00;0.00",
                   "Δ gap vs M1 (pp)": "+0.00;-0.00;0.00"}.get(name)
            book.put(ws, r, c, values.get(name), grid=True, fill_in=name == "Remarks", fmt=fmt,
                     wrap=name in ("Trained / evaluated", "What it signifies", "Optimizer (updates θ)", "r computed here",
                                   "Setting", "Role of the pair"))
    ws.freeze_panes = ws.cell(row=5, column=2)
    ws.auto_filter.ref = f"A4:{_col(len(EXP_COLUMNS))}{max(e.row for e in plan.exps)}"
    book.fit_rows(ws, 4)
    book.landscape(ws, fit_width=False)


def _exp_index(plan: Plan) -> Dict[Tuple, Exp]:
    index: Dict[Tuple, Exp] = {}
    for exp in plan.exps:
        if exp.arm is None and exp.stage in ("2", "3"):
            index.setdefault((exp.round, exp.setting_key, exp.pair, exp.method, exp.reg), exp)
    return index


def _cubes_sheet(book: _Book, plan: Plan) -> None:
    ws = book.wb["Cubes"]
    row = book.title(ws, "Stage 4 — 3D tables: method × regularizer × evaluation setting, per round",
                     "Formulas over T4 (seed means). M2 trains with r at every step on every pair. "
                     "‡ = the network is unseen too. The Stage-1 benchmark compares M2 with r on against M1.")
    index = _exp_index(plan)
    from reglo.reporting.tables import settings_columns

    metrics = [("Test accuracy ↑ (%) — seed mean", "Test top-1 mean (%)", "0.00"),
               ("Generalization gap ↓ (pp, train − test top-1) — seed mean", "Gen. gap mean (pp)", "0.00"),
               ("FLOPs ↓ (GFLOPs per training step)", "GFLOPs / step", "0.000")]
    method_rows = []
    for reg in REGULARIZERS:
        name = REG_NAMES[reg]
        method_rows += [(name, "M1 AdamW", "m1_adamw", "none"), ("", f"M2 AdamW + {name}", "m2_adamw_reg", reg),
                        ("", "M3 LSTM", "m3_lstm", "none"), ("", f"M4 LSTM-{name} (ours)", "m4_lstm_reg", reg)]
    widest = 2
    for rnd in plan.rounds:
        columns = settings_columns(plan.settings.get(rnd.name, {}), extended=False)
        widest = max(widest, 2 + len(columns))
        book.put(ws, row, 1, f"ROUND {rnd.name} — optimizers meta-trained on {Pair(rnd.optimizee, rnd.dataset).label}",
                 bold=True, size=12)
        row += 2
        for title, column, fmt in metrics:
            book.put(ws, row, 1, title, bold=True, size=11)
            row += 1
            book.put(ws, row, 1, "Regularizer", header=True)
            book.put(ws, row, 2, "Method", header=True)
            start = 3
            for j, c in enumerate(columns):
                book.put(ws, row, 3 + j, c.setting.label, header=True)
            # merge consecutive identical setting labels
            j = 0
            while j < len(columns):
                k = j
                while k + 1 < len(columns) and columns[k + 1].setting.key == columns[j].setting.key:
                    k += 1
                if k > j:
                    ws.merge_cells(start_row=row, start_column=start + j, end_row=row, end_column=start + k)
                j = k + 1
            row += 1
            book.put(ws, row, 1, None, header=True)
            book.put(ws, row, 2, None, header=True)
            for j, c in enumerate(columns):
                book.put(ws, row, 3 + j, c.pair.label + (" ‡" if c.role == "unseen_both" else ""), header=True)
            row += 1
            for block, label, method, reg in method_rows:
                book.put(ws, row, 1, block, bold=bool(block), grid=True)
                book.put(ws, row, 2, label, grid=True)
                for j, c in enumerate(columns):
                    exp = index.get((rnd.name, c.setting.key, c.pair, method, reg))
                    value = f"='{T4}'!${_col(EC[column])}${exp.row}" if exp else None
                    book.put(ws, row, 3 + j, value, grid=True, fmt=fmt)
                row += 1
            row += 1
        # the Stage-1 benchmark on the source pair
        source = Pair(rnd.optimizee, rnd.dataset)
        book.put(ws, row, 1, f"Stage 1 benchmark — {source.label}: M1 against M2(r) with r on; M3 and M4(r) from the "
                             "seen setting", bold=True, size=11)
        row += 1
        heads = ["Regularizer", "Method", "Test top-1 (%)", "Δ vs M1 (pp)", "Gen. gap (pp)", "Δ vs M1 (pp) ", "GFLOPs / step"]
        for j, h in enumerate(heads, start=1):
            book.put(ws, row, j, h, header=True)
        row += 1
        bench = [e for e in plan.exps if e.stage == "1" and e.round == rnd.name and e.arm is None]
        seen = [index.get((rnd.name, "seen_optimizee", source, m, r)) for m, r in
                [("m3_lstm", "none")] + [("m4_lstm_reg", r) for r in REGULARIZERS]]
        entries = [(REG_NAMES.get(e.reg, "—"), "M1 AdamW" if e.method == "m1_adamw" else f"M2 AdamW + {REG_NAMES[e.reg]}", e)
                   for e in bench]
        entries += [("—" if e.reg == "none" else REG_NAMES[e.reg], "M3 LSTM" if e.method == "m3_lstm"
                     else f"M4 LSTM-{REG_NAMES[e.reg]} (ours)", e) for e in seen if e is not None]
        m1 = next((e for e in bench if e.method == "m1_adamw"), None)
        for reg_text, label, exp in entries:
            book.put(ws, row, 1, reg_text, grid=True)
            book.put(ws, row, 2, label, grid=True)
            acc, gap, flops = (f"'{T4}'!${_col(EC[n])}${exp.row}" for n in ("Test top-1 mean (%)", "Gen. gap mean (pp)",
                                                                               "GFLOPs / step"))
            book.put(ws, row, 3, f"={acc}", grid=True, fmt="0.00")
            book.put(ws, row, 5, f"={gap}", grid=True, fmt="0.00")
            book.put(ws, row, 7, f"={flops}", grid=True, fmt="0.000")
            if m1 is not None and exp is not m1:
                m1_acc, m1_gap = (f"'{T4}'!${_col(EC[n])}${m1.row}" for n in ("Test top-1 mean (%)", "Gen. gap mean (pp)"))
                book.put(ws, row, 4, f'=IFERROR({acc}-{m1_acc},"")', grid=True, fmt="+0.00;-0.00;0.00")
                book.put(ws, row, 6, f'=IFERROR({gap}-{m1_gap},"")', grid=True, fmt="+0.00;-0.00;0.00")
            else:
                book.put(ws, row, 4, None, grid=True)
                book.put(ws, row, 6, None, grid=True)
            row += 1
        row += 2
    book.widths(ws, [12, 26] + [15] * (widest - 2))
    ws.freeze_panes = "C4"
    book.fit_rows(ws, 4)
    book.landscape(ws)


def _cross_sheet(book: _Book, plan: Plan) -> None:
    ws = book.wb["Cross-evaluation"]
    rounds = plan.rounds
    row = book.title(ws, "Cross-evaluation — every pair with the optimizers of every round",
                     "On a round's own source pair its optimizers are seen; elsewhere they meet an unseen network, dataset "
                     "or both. The same pair under two rounds separates a harder task from a transfer gap.")
    if len(rounds) < 2:
        book.put(ws, row, 1, "One round only: nothing to cross-evaluate.", italic=True)
        return
    first: Dict[Tuple, Exp] = {}
    for exp in plan.exps:
        if exp.arm is None and exp.stage in ("2", "3"):
            key = (exp.pair, exp.method, exp.reg) if exp.method == "m1_adamw" else (exp.round, exp.pair, exp.method, exp.reg)
            first.setdefault(key, exp)
    pool: List[Pair] = []
    for per in plan.settings.values():
        for pairs in per.values():
            for sp in pairs:
                if not sp.extended and sp.pair not in pool and all(
                        any(x.pair == sp.pair and not x.extended for v in plan.settings.get(r.name, {}).values() for x in v)
                        for r in rounds):
                    pool.append(sp.pair)
    order_d, order_o = list(DATASET_NAMES), list(OPTIMIZEE_NAMES)
    pool.sort(key=lambda p: (order_d.index(p.dataset), order_o.index(p.optimizee)))
    variants = [("M3", "m3_lstm", "none")] + [(f"M4-{REG_NAMES[r]}", "m4_lstm_reg", r) for r in REGULARIZERS]

    for metric_title, column, fmt in (("Test accuracy ↑ (%)", "Test top-1 mean (%)", "0.00"),
                                      ("Generalization gap ↓ (pp)", "Gen. gap mean (pp)", "0.00")):
        book.put(ws, row, 1, f"{metric_title} — seed means", bold=True, size=12)
        row += 1
        heads = ["Optimizee / dataset"] + [f"Role for {r.name}" for r in rounds] + ["M1"] + \
            [f"{v}·{r.name}" for v, _, _ in variants for r in rounds]
        for j, h in enumerate(heads, start=1):
            book.put(ws, row, j, h, header=True)
        row += 1
        first_row = row
        value_cols: Dict[Tuple[str, str], str] = {}
        role_cols = {r.name: _col(2 + i) for i, r in enumerate(rounds)}
        m1_col = _col(2 + len(rounds))
        for i, (v, _, _) in enumerate(variants):
            for k, r in enumerate(rounds):
                value_cols[(v, r.name)] = _col(3 + len(rounds) + i * len(rounds) + k)
        for pair in pool:
            book.put(ws, row, 1, pair.label, grid=True)
            for i, r in enumerate(rounds):
                book.put(ws, row, 2 + i, ROLE_LABELS[r.role(pair.optimizee, pair.dataset)], grid=True)
            m1 = first.get((pair, "m1_adamw", "none"))
            book.put(ws, row, 2 + len(rounds), f"='{T4}'!${_col(EC[column])}${m1.row}" if m1 else None, grid=True, fmt=fmt)
            for i, (v, method, reg) in enumerate(variants):
                for k, r in enumerate(rounds):
                    exp = first.get((r.name, pair, method, reg))
                    book.put(ws, row, 3 + len(rounds) + i * len(rounds) + k,
                             f"='{T4}'!${_col(EC[column])}${exp.row}" if exp else None, grid=True, fmt=fmt)
            row += 1
        last_row = row - 1
        row += 1

        # advantage over M1, same layout without M1
        book.put(ws, row, 1, f"{metric_title}: variant − M1 on the same pair (pp)", bold=True, size=11)
        row += 1
        heads = ["Optimizee / dataset"] + [f"Role for {r.name}" for r in rounds] + [""] + \
            [f"{v}·{r.name} − M1" for v, _, _ in variants for r in rounds]
        for j, h in enumerate(heads, start=1):
            book.put(ws, row, j, h if h else None, header=True)
        row += 1
        adv_first = row
        for n, pair in enumerate(pool):
            src = first_row + n
            book.put(ws, row, 1, f"=A{src}", grid=True)
            for i, r in enumerate(rounds):
                book.put(ws, row, 2 + i, f"={role_cols[r.name]}{src}", grid=True)
            book.put(ws, row, 2 + len(rounds), None, grid=True)
            for i, (v, _, _) in enumerate(variants):
                for k, r in enumerate(rounds):
                    c = value_cols[(v, r.name)]
                    book.put(ws, row, 3 + len(rounds) + i * len(rounds) + k,
                             f'=IF(AND(ISNUMBER({c}{src}),ISNUMBER({m1_col}{src})),{c}{src}-{m1_col}{src},"")',
                             grid=True, fmt="+0.00;-0.00;0.00")
            row += 1
        adv_last = row - 1
        row += 1

        # summary: mean advantage by role, and the transfer gap on the source pairs
        book.put(ws, row, 1, f"{metric_title}: mean (variant − M1) by role of the pair, and the transfer gap", bold=True,
                 size=11)
        row += 1
        heads = ["Variant", "Round"] + [f"{ROLE_LABELS[role]} (vs M1)" for role in ROLE_ORDER] + \
            ["Source pair: own − other round(s)"]
        for j, h in enumerate(heads, start=1):
            book.put(ws, row, j, h, header=True)
        row += 1
        for v, method, reg in variants:
            for r in rounds:
                book.put(ws, row, 1, v, grid=True)
                book.put(ws, row, 2, r.name, grid=True)
                adv = value_cols[(v, r.name)]
                role_col = role_cols[r.name]
                for j, role in enumerate(ROLE_ORDER):
                    book.put(ws, row, 3 + j,
                             f'=IFERROR(AVERAGEIFS(${adv}${adv_first}:${adv}${adv_last},'
                             f'${role_col}${adv_first}:${role_col}${adv_last},"{ROLE_LABELS[role]}"),"")',
                             grid=True, fmt="+0.00;-0.00;0.00")
                source = Pair(r.optimizee, r.dataset)
                if source in pool:
                    src_row = first_row + pool.index(source)
                    own = f"{value_cols[(v, r.name)]}{src_row}"
                    others = [f"{value_cols[(v, o.name)]}{src_row}" for o in rounds if o.name != r.name]
                    book.put(ws, row, 3 + len(ROLE_ORDER), f'=IFERROR({own}-AVERAGE({",".join(others)}),"")',
                             grid=True, fmt="+0.00;-0.00;0.00")
                else:
                    book.put(ws, row, 3 + len(ROLE_ORDER), "—", grid=True)
                row += 1
        row += 2
    book.widths(ws, [22] + [26] * len(rounds) + [9] + [11] * (len(variants) * len(rounds)))
    ws.freeze_panes = "B4"
    book.fit_rows(ws, 4)
    book.landscape(ws)


def _metrics_sheet(book: _Book) -> None:
    ws = book.wb["Metrics"]
    row = book.title(ws, "Metrics — what each number means and where it comes from",
                     "Stage 4 reports test accuracy, generalization gap and FLOPs for every method and regularizer; "
                     "the rest supports them.")
    headers = ["Metric", "Key (final.json …)", "Definition", "Unit", "Better", "Computed in", "Over seeds", "Shown in",
               "What it signifies"]
    for c, h in enumerate(headers, start=1):
        book.put(ws, row, c, h, header=True)
    row += 1
    for values in METRIC_ROWS:
        for c, v in enumerate(values, start=1):
            book.put(ws, row, c, v, wrap=True, grid=True)
        row += 1
    book.widths(ws, [26, 20, 46, 8, 16, 30, 18, 22, 40])
    ws.freeze_panes = "B5"
    book.fit_rows(ws, 4)
    book.landscape(ws)


# ═════════════════════════════════════════════════════════════════ filling in results

def _header_map(ws, header_row: int) -> Dict[str, int]:
    return {str(ws.cell(row=header_row, column=c).value): c for c in range(1, ws.max_column + 1)
            if ws.cell(row=header_row, column=c).value is not None}


def _find_header(ws, first: str) -> int:
    for r in range(1, 20):
        if ws.cell(row=r, column=1).value == first:
            return r
    raise ValueError(f"sheet '{ws.title}' has no header row starting with '{first}'")


def fill_workbook(path: PathLike, artifacts_root: PathLike, out: Optional[PathLike] = None) -> Dict[str, int]:
    """Write every finished result under ``artifacts_root`` (one profile's folder) into the blue cells
    of the workbook at ``path`` — Runs and Stage 1 meta-training — and keep everything else.

    Rows are matched by their run folder / φ path, so the workbook may be sorted or annotated.
    Returns how many rows were filled per sheet.
    """
    from openpyxl import load_workbook

    from reglo.reporting.collect import collect_meta, collect_runs, discover_meta_tags, discover_studies
    from reglo.reporting.tables import _tail_mean

    root = Path(artifacts_root)
    wb = load_workbook(path)
    filled = {"runs": 0, "meta": 0, "running": 0}

    ws = wb[RUNS]
    header = _find_header(ws, "Run ID")
    cols = _header_map(ws, header)
    records = {}
    for rec in collect_runs(root, discover_studies(root)):
        rel = Path(rec["run"]).resolve().relative_to(root.resolve()).as_posix()
        records[rel] = rec
    for r in range(header + 1, ws.max_row + 1):
        folder = ws.cell(row=r, column=cols["Run folder (artifacts/<profile>/…)"]).value
        if not folder:
            continue
        rec = records.get(str(folder))
        if rec is None:
            if (root / str(folder) / "metrics.jsonl").is_file():
                ws.cell(row=r, column=cols["Status"]).value = "running"
                filled["running"] += 1
            continue
        status = "done" if rec.get("status") == "completed" else str(rec.get("status"))
        values = {
            "Status": status,
            "Test top-1 (%)": None if rec.get("test_top1") is None else 100.0 * rec["test_top1"],
            "Train top-1 (%)": None if rec.get("train_top1") is None else 100.0 * rec["train_top1"],
            "Test loss": rec.get("test_loss"), "Train loss": rec.get("train_loss"),
            "GFLOPs / step": rec.get("gflops_per_step"),
            "Wall-clock (h)": None if rec.get("wall_clock_s") is None else rec["wall_clock_s"] / 3600.0,
        }
        for name, value in values.items():
            if value is not None and not (isinstance(value, float) and not math.isfinite(value)):
                ws.cell(row=r, column=cols[name]).value = round(value, 6) if isinstance(value, float) else value
        filled["runs"] += 1

    ws = wb[META]
    header = _find_header(ws, "Exp ID")
    cols = _header_map(ws, header)
    arms = {}
    for tag in discover_meta_tags(root):
        for arm in collect_meta(root, tag):
            arms[f"meta_training/{tag}/{arm['arm']}/phi_best.pt"] = arm
    for r in range(header + 1, ws.max_row + 1):
        arm = arms.get(str(ws.cell(row=r, column=cols["Output φ (artifacts/<profile>/…)"]).value))
        if arm is None:
            continue
        s, steps = arm["summary"], arm["steps"]
        status = "done" if s.get("status") == "completed" else str(s.get("status", "running"))
        acc = s.get("best_val_acc")
        values = {
            "Status": status,
            "Trajectories done": s.get("trajectories", max((x.get("trajectory", -1) for x in steps), default=-1) + 1),
            "Meta-steps": s.get("meta_steps", len(steps)),
            "Best val. loss": s.get("best_val_loss"),
            "Best val. acc. (%)": None if acc is None else 100.0 * float(acc),
            "Best at trajectory": s.get("best_trajectory"),
            "L_meta (end)": _tail_mean(steps, "l_meta"),
            "L_smooth (end)": _tail_mean(steps, "l_smooth") if arm["meta_training"] != "m3_lstm" else None,
            "L_reg (end)": _tail_mean(steps, "l_reg") if arm["regularizer"] != "none" else None,
            "Wall-clock (h)": None if s.get("wall_clock_s") is None else float(s["wall_clock_s"]) / 3600.0,
        }
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
                if not math.isfinite(value):
                    continue
                value = round(value, 6)
            ws.cell(row=r, column=cols[name]).value = value
        filled["meta"] += 1

    wb.calculation.fullCalcOnLoad = True
    target = Path(out) if out else Path(path)
    wb.save(target)
    return filled
