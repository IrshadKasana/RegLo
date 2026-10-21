from pathlib import Path
import runpy
import sys


DMO_SRC = Path(__file__).resolve().parents[2] / "src" / "learning_regularizers" / "dmo"
sys.path.insert(0, str(DMO_SRC))

runpy.run_path(str(DMO_SRC / "optimizee_train.py"), run_name="__main__")
