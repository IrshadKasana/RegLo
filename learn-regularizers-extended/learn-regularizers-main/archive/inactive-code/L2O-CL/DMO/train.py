from pathlib import Path
import runpy
import sys


SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "learning_regularizers" / "dmo"
sys.path.insert(0, str(SRC_DIR))

runpy.run_path(str(SRC_DIR / "train.py"), run_name="__main__")
