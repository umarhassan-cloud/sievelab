from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dataforge.pipeline.config import load_config
from dataforge.pipeline.runner import Pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dataforge", description="DataForge data curation pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the curation pipeline")
    run.add_argument("--config", "-c", default=None, help="YAML config (merged over defaults)")
    run.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. dedup.threshold=0.8")
    run.add_argument("--base-dir", default=".", help="directory relative paths in the config resolve against")
    run.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    if args.cmd == "run":
        cfg = load_config(args.config, args.overrides)
        result = Pipeline(cfg, base_dir=Path(args.base_dir).resolve()).run()
        print(result.manifest.to_markdown())
        if result.evaluation:
            print("Evaluation vs injected ground truth:")
            print(json.dumps({k: v for k, v in result.evaluation.items() if k not in ("final_by_gold_type", "ingested_by_gold_type")},
                             indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
