"""Fetch only explicitly named PhysioNet v1.0.0 case files (maximum 20 per call)."""

import argparse
from pathlib import Path

from intraop.data.acquisition import acquire_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--destination", type=Path, default=Path("data/raw/vitaldb-1.0.0"))
    parser.add_argument("--max-total-bytes", type=int, default=1024**3)
    args = parser.parse_args()
    try:
        result = acquire_subset(
            args.destination, args.case_ids, max_total_bytes=args.max_total_bytes
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified {len(result['files'])} selected source files; no archive or directory crawl.")


if __name__ == "__main__":
    main()
