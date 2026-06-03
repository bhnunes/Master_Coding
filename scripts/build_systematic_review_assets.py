from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build systematic-review assets for the thesis chapter."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("systematic_review/database/automatic_review.sqlite3"),
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("systematic_review/PAPERS"))
    parser.add_argument("--thesis-dir", type=Path, default=Path("masters_thesis/New_Thesis"))
    parser.add_argument(
        "--review-generated-dir",
        type=Path,
        default=Path("systematic_review/generated"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("systematic_review/protocol/protocol.md"),
    )
    return parser.parse_args()


def ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    ensure_repo_root_on_path()
    from helpers.systematic_review import build_assets

    args = parse_args()
    outputs = build_assets(
        database_path=args.database,
        papers_dir=args.papers_dir,
        thesis_dir=args.thesis_dir,
        review_generated_dir=args.review_generated_dir,
        protocol_path=args.protocol,
    )
    for path in outputs.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
