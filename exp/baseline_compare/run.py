"""Run the shared-token baseline comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

from logitfill.baselines.compare import run_compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_compare(args.config, args.out)


if __name__ == "__main__":
    main()
