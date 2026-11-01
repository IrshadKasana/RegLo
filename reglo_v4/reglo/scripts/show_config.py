#!/usr/bin/env python
"""Print the fully composed configuration of a run, and where it will write.

The quickest way to check what a run will actually use — before spending GPU hours.

Examples:
    python scripts/show_config.py --list
    python scripts/show_config.py method=m1_adamw optimizee=convnet4 dataset=svhn
    python scripts/show_config.py method=m2_adamw_reg regularizer=sam optimizee=resnet18 dataset=cifar10
    python scripts/show_config.py method=m4_lstm_reg regularizer=gsam optimizee=vit_lite7 dataset=cifar100
    python scripts/show_config.py meta_training=m4_lstm_reg regularizer=l1
    python scripts/show_config.py meta_training=m4_lstm_reg regularizer=l1 profile=smoke    # the smoke numbers
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import GROUP_DIRS, ConfigError, available, compose, experiment_names, to_yaml  # noqa: E402
from reglo.profiles import available_profiles  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import meta_run_dir, run_dir  # noqa: E402


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(
        description="Compose and print a run configuration.",
        epilog="Overrides: group=name (choose a file), key.path=value (change a value), +key.path=value (add a key).",
    )
    parser.add_argument("overrides", nargs="*", help="e.g. method=m2_adamw_reg regularizer=sam")
    parser.add_argument("--list", action="store_true", help="list the files available in every group")
    args = parser.parse_args(argv)

    if args.list:
        for group in GROUP_DIRS:
            print(f"{group:<14} {', '.join(available(group))}")
        print(f"{'experiments':<14} {', '.join(experiment_names())}")
        print(f"{'profiles':<14} {', '.join(available_profiles())}")
        return 0

    entry = "meta_training" if any(o.startswith("meta_training=") for o in args.overrides) else "method"
    try:
        cfg = compose(entry, args.overrides)
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(to_yaml(cfg), end="")
    target = meta_run_dir(cfg) if entry == "meta_training" else run_dir(cfg)
    print(f"# profile: {cfg.profile}   output folder: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
