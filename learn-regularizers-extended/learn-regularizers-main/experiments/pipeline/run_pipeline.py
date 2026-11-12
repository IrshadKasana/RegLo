"""Run the hierarchical pipeline. See src/learning_regularizers/pipeline/cli.py for options."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from learning_regularizers.pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
