#!/usr/bin/env python3
"""Deduplicate, balance, split, and audit the scaled S/W/V replication."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_FAMILIES = {"S": {"S"}, "W": {"S", "W"}, "V": {"S", "W", "V"}}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def summary(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "fragments": dict(Counter(row["fragment"] for row in rows)),
        "labels": dict(Counter(row["label"] for row in rows)),
        "fragment_labels": dict(Counter(f"{row['fragment']}/{row['label']}" for row in rows)),
        "families": dict(Counter(family for row in rows for family in row["families"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default="generated_data/candidates")
    parser.add_argument("--output-dir", default="generated_data/final")
    parser.add_argument("--train-per-label", type=int, default=12000)
    parser.add_argument("--validation-per-label", type=int, default=1000)
    parser.add_argument("--test-per-label", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4107)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    all_candidates = []
    candidate_reports = {}
    global_text_owner = {}
    conflicting_texts = []

    per_label = args.train_per_label + args.validation_per_label + args.test_per_label
    for fragment in ("S", "W", "V"):
        rows = load(candidate_dir / f"{fragment}.jsonl")
        unique = {}
        for row in rows:
            if row["fragment"] != fragment:
                raise AssertionError(f"Wrong fragment in {fragment} pool")
            if not set(row["families"]) <= EXPECTED_FAMILIES[fragment]:
                raise AssertionError(f"Illegal family in {row['id']}")
            if row["metadata"]["solver_status"] not in {"sat", "unsat"}:
                raise AssertionError(f"Bad solver status in {row['id']}")
            if (row["label"] == "satisfiable") != (row["metadata"]["solver_status"] == "sat"):
                raise AssertionError(f"Label/status mismatch in {row['id']}")
            key = tuple(row["sentences"])
            unique.setdefault(key, row)
            if key in global_text_owner and global_text_owner[key][1] != row["label"]:
                conflicting_texts.append((global_text_owner[key][0], row["id"]))
            global_text_owner.setdefault(key, (row["id"], row["label"]))
        candidate_reports[fragment] = {
            "input": len(rows), "unique": len(unique), "duplicates": len(rows) - len(unique),
            "labels_after_deduplication": dict(Counter(row["label"] for row in unique.values())),
        }
        all_candidates.extend(unique.values())
    if conflicting_texts:
        raise AssertionError(f"Conflicting duplicate texts: {conflicting_texts[:3]}")

    groups = defaultdict(list)
    for row in all_candidates:
        groups[(row["fragment"], row["label"])].append(row)
    rng = random.Random(args.seed)
    final = []
    for fragment in ("S", "W", "V"):
        for label in ("satisfiable", "unsatisfiable"):
            pool = groups[(fragment, label)]
            if len(pool) < per_label:
                raise RuntimeError(f"Insufficient unique {fragment}/{label}: {len(pool)}")
            for index, source in enumerate(rng.sample(pool, per_label), 1):
                row = dict(source)
                row["metadata"] = dict(row["metadata"], source_candidate_id=row["id"])
                row["id"] = f"{fragment}_{label[:3]}_{index:06d}"
                final.append(row)
    rng.shuffle(final)
    write(final, output_dir / "examples_swv_hierarchical_balanced.jsonl")

    strata = defaultdict(list)
    for row in final:
        strata[(row["fragment"], row["label"])].append(row)
    splits = {"train": [], "validation": [], "test": []}
    for rows in strata.values():
        rng.shuffle(rows)
        train_end = args.train_per_label
        validation_end = train_end + args.validation_per_label
        splits["train"].extend(rows[:train_end])
        splits["validation"].extend(rows[train_end:validation_end])
        splits["test"].extend(rows[validation_end:])
    for name, rows in splits.items():
        rng.shuffle(rows)
        write(rows, output_dir / "splits" / f"{name}.jsonl")
        for fragment in ("S", "W", "V"):
            write([row for row in rows if row["fragment"] == fragment], output_dir / "fragment_splits" / fragment / f"{name}.jsonl")

    ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    texts = {name: {tuple(row["sentences"]) for row in rows} for name, rows in splits.items()}
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}/{right}"] = {"ids": len(ids[left] & ids[right]), "texts": len(texts[left] & texts[right])}
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"Split leakage: {overlaps}")

    report = {
        "method": "Tharindu-style sampling across empirically calibrated S/W/V phase-change regions, followed by candidate-pool balancing",
        "seed": args.seed,
        "per_fragment_per_label": per_label,
        "split_per_fragment_per_label": {
            "train": args.train_per_label,
            "validation": args.validation_per_label,
            "test": args.test_per_label,
        },
        "candidate_audit": candidate_reports,
        "final": summary(final),
        "splits": {name: summary(rows) for name, rows in splits.items()},
        "cross_split_overlaps": overlaps,
    }
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
