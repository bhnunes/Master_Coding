from __future__ import annotations


def main() -> None:
    """Stage 3 is obsolete in the metadata-first pipeline."""

    print(
        "Stage 3 is obsolete in the metadata-first pipeline. "
        "Run Stage 4.3 to update cleaning state in master_manifest.sqlite, then run Stage 5 "
        "directly against the SQLite-backed accepted-row set."
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
