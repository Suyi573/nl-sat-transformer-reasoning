#!/usr/bin/env python3
"""Independent integrity audit for the final data4 splits."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default="generated_data/final")
    parser.add_argument("--output", default="reports/final_integrity.json")
    args = parser.parse_args()

    root = Path(args.final_dir)
    splits = {name: load(root / "splits" / f"{name}.jsonl") for name in ("train", "validation", "test")}
    expected = {"train": 72000, "validation": 6000, "test": 6000}
    expected_per_fragment = {"train": 24000, "validation": 2000, "test": 2000}
    expected_per_label = {"train": 12000, "validation": 1000, "test": 1000}
    allowed = {"S": {"S"}, "W": {"S", "W"}, "V": {"S", "W", "V"}}

    errors: list[str] = []
    split_reports = {}
    all_rows = []
    for name, rows in splits.items():
        all_rows.extend(rows)
        fragments = Counter(row["fragment"] for row in rows)
        labels = Counter(row["label"] for row in rows)
        strata = Counter((row["fragment"], row["label"]) for row in rows)
        ids = [row["id"] for row in rows]
        texts = [tuple(row["sentences"]) for row in rows]
        if len(rows) != expected[name]:
            errors.append(f"{name}: expected {expected[name]} rows, got {len(rows)}")
        if len(set(ids)) != len(ids):
            errors.append(f"{name}: duplicate IDs")
        if len(set(texts)) != len(texts):
            errors.append(f"{name}: duplicate texts")
        for fragment in ("S", "W", "V"):
            if fragments[fragment] != expected_per_fragment[name]:
                errors.append(f"{name}/{fragment}: wrong total")
            for label in ("satisfiable", "unsatisfiable"):
                if strata[(fragment, label)] != expected_per_label[name]:
                    errors.append(f"{name}/{fragment}/{label}: wrong count")
        bad_labels = 0
        bad_families = 0
        for row in rows:
            status = row["metadata"]["solver_status"]
            if (row["label"] == "satisfiable") != (status == "sat") or status not in {"sat", "unsat"}:
                bad_labels += 1
            if not set(row["families"]) <= allowed[row["fragment"]]:
                bad_families += 1
        if bad_labels:
            errors.append(f"{name}: {bad_labels} label/status mismatches")
        if bad_families:
            errors.append(f"{name}: {bad_families} hierarchy violations")
        split_reports[name] = {
            "rows": len(rows),
            "fragments": dict(fragments),
            "labels": dict(labels),
            "strata": {f"{f}/{label}": count for (f, label), count in sorted(strata.items())},
            "unique_ids": len(set(ids)),
            "unique_texts": len(set(texts)),
            "label_status_mismatches": bad_labels,
            "hierarchy_violations": bad_families,
        }

    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_ids = {row["id"] for row in splits[left]}
        right_ids = {row["id"] for row in splits[right]}
        left_texts = {tuple(row["sentences"]) for row in splits[left]}
        right_texts = {tuple(row["sentences"]) for row in splits[right]}
        overlap = {"ids": len(left_ids & right_ids), "texts": len(left_texts & right_texts)}
        overlaps[f"{left}/{right}"] = overlap
        if overlap["ids"] or overlap["texts"]:
            errors.append(f"{left}/{right}: leakage {overlap}")

    parameters = {}
    family_counts = Counter()
    fragment_clause_counts = Counter()
    for fragment in ("S", "W", "V"):
        rows = [row for row in all_rows if row["fragment"] == fragment]
        parameters[fragment] = {
            "m": [min(r["metadata"]["m"] for r in rows), max(r["metadata"]["m"] for r in rows)],
            "n1": [min(r["metadata"]["n1"] for r in rows), max(r["metadata"]["n1"] for r in rows)],
            "n2": [min(r["metadata"]["n2"] for r in rows), max(r["metadata"]["n2"] for r in rows)],
        }
        for row in rows:
            family_counts.update(row["families"])
            fragment_clause_counts[fragment] += len(row["families"])

    family_rates = {
        "S_in_S": sum(r["families"].count("S") for r in all_rows if r["fragment"] == "S") / fragment_clause_counts["S"],
        "W_in_W": sum(r["families"].count("W") for r in all_rows if r["fragment"] == "W") / fragment_clause_counts["W"],
        "V_in_V": sum(r["families"].count("V") for r in all_rows if r["fragment"] == "V") / fragment_clause_counts["V"],
    }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "total_rows": len(all_rows),
        "splits": split_reports,
        "cross_split_overlaps": overlaps,
        "parameter_ranges": parameters,
        "family_counts": dict(family_counts),
        "key_family_rates": family_rates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
