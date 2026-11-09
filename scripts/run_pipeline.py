"""Run the curation pipeline on the bundled sample corpus.

    python scripts/run_pipeline.py                       # configs/default.yaml
    python scripts/run_pipeline.py --config configs/urdu.yaml
    python scripts/run_pipeline.py --set dedup.threshold=0.8 --set dedup.workers=4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataforge.pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--config" not in argv and "-c" not in argv:
        argv = ["--config", str(ROOT / "configs" / "default.yaml")] + argv
    raise SystemExit(main(["run", "--base-dir", str(ROOT)] + argv))
