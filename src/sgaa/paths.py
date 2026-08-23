"""Central filesystem paths for the SGAA project.

Everything resolves relative to the *repository root* (the parent of ``src/``)
so that figures, results and the paper are written to stable locations no
matter from where the code is imported or invoked.
"""

from pathlib import Path

# repo root = three levels up from this file (sgaa/paths.py -> src -> root)
ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"

# Location of the (already downloaded) MNIST dataset.  Override via the
# environment variable SGAA_MNIST_ROOT if you keep the data elsewhere.
_DEFAULT_MNIST_ROOT = Path("/root/data")
import os as _os
MNIST_ROOT = Path(_os.environ.get("SGAA_MNIST_ROOT", str(_DEFAULT_MNIST_ROOT)))

for _d in (RESULTS_DIR, FIGURES_DIR, PAPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)
