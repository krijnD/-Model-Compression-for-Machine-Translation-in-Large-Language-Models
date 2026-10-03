"""Every path the code uses, in one place.

Large files live outside the repository. Set these environment variables to point at them
(slurm/env.sh sets them for the Snellius jobs); the defaults assume the layout in the README:

  PROJECT_DIR    folder that holds the repository           default: the repository's parent
  MODELS_DIR     fp16 ALMA-13B-R and the GPTQ checkpoints    default: $PROJECT_DIR/models
  ARTIFACTS_DIR  adapters, repacked checkpoints, LID model   default: $MODELS_DIR
  OUTPUTS_DIR    translations and segment-level scores       default: $PROJECT_DIR/outputs
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", REPO.parent))
MODELS = Path(os.environ.get("MODELS_DIR", PROJECT_DIR / "models"))
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", MODELS))
OUTPUTS = Path(os.environ.get("OUTPUTS_DIR", PROJECT_DIR / "outputs"))

# Inputs shipped with the ALMA submodule
ALMA_DATA = REPO / "third_party/ALMA/human_written_data"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"

# Committed results (small aggregates) and the paper
RESULTS = REPO / "results"
JSON = RESULTS / "json"
SCORES = RESULTS / "scores"  # one corpus-level score table per run, written by scripts/reproduce/summarize.py
DISTILL_RUNS = RESULTS / "distill_runs"
PAPER = REPO / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
