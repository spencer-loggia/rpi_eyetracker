"""Run from the repository with: python pi_tracker.py config/eye_tracker.json."""

import sys
from pathlib import Path

# Also supports a source checkout before installing the console entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from macaque_tracker.pi_gui import main

if __name__ == "__main__":
    raise SystemExit(main())
