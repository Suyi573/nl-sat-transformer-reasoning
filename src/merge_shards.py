#!/usr/bin/env python3
"""Merge independently generated shards and assign globally unique IDs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", choices=("S", "W", "V"), required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.shard_dir).glob(f"{args.fragment}_*.jsonl"))
    if not paths:
        raise RuntimeError(f"No {args.fragment} shards found")
    unique: dict[tuple[str, ...], dict] = {}
    input_rows = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["fragment"] != args.fragment:
                raise AssertionError(f"Wrong fragment in {path}: {row['fragment']}")
            input_rows += 1
            unique.setdefault(tuple(row["sentences"]), row)

    rows = list(unique.values())
    for index, row in enumerate(rows, 1):
        row["metadata"]["source_shard_id"] = row["id"]
        row["id"] = f"candidate_{args.fragment}_{index:07d}"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    labels = Counter(r["label"] for r in rows)
    report = {
        "fragment": args.fragment,
        "shards": [p.name for p in paths],
        "input_rows": input_rows,
        "unique_rows": len(rows),
        "duplicates_removed": input_rows - len(rows),
        "labels": dict(labels),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
