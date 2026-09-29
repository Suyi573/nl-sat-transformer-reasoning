#!/usr/bin/env python3
"""Build data9: data8 S fixed, validation-derived W-hard2 and V-hard2."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")
LABELS = ("satisfiable", "unsatisfiable")
PER_LABEL = {"train": 12000, "validation": 1000, "test": 1000}
EXPECTED = {"S": {"S"}, "W": {"S", "W"}, "V": {"S", "W", "V"}}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def text_key(row: dict) -> tuple[str, ...]:
    sentences = row["sentences"]
    if sentences and str(sentences[0]).startswith("Fragment: "):
        sentences = sentences[1:]
    return tuple(sentences)


def validate(row: dict, fragment: str) -> None:
    if row.get("fragment") != fragment:
        raise AssertionError(f"Wrong fragment in {row.get('id')}")
    if not set(row.get("families", [])) <= EXPECTED[fragment]:
        raise AssertionError(f"Illegal family in {row.get('id')}")
    status = row.get("metadata", {}).get("solver_status")
    if status not in {"sat", "unsat"}:
        raise AssertionError(f"Missing solver status in {row.get('id')}")
    if (row["label"] == "satisfiable") != (status == "sat"):
        raise AssertionError(f"Label mismatch in {row.get('id')}")


def m_bin(row: dict) -> str:
    m = int(row["metadata"]["m"])
    return "08-12" if m <= 12 else "13-16" if m <= 16 else "17-20" if m <= 20 else "21+"


def fraction_bin(row: dict, fragment: str) -> str:
    fraction = row["families"].count(fragment) / int(row["metadata"]["m"])
    return "<0.40" if fraction < .4 else "0.40-0.49" if fraction < .5 else "0.50-0.59" if fraction < .6 else "0.60+"


def logic_negations(logic: str) -> int:
    return len(re.findall(r"\bNot\(", logic))


def signatures(row: dict, fragment: str) -> set[str]:
    result = set()
    for family, quantifier, logic in zip(row["families"], row["quantifiers"], row["logic"]):
        if family != fragment:
            continue
        q = "-".join(quantifier) if isinstance(quantifier, list) else str(quantifier)
        result.add(f"{q}|neg={logic_negations(str(logic))}")
    return result


def difficulty_weight(row: dict, fragment: str) -> float:
    mb = m_bin(row)
    fb = fraction_bin(row, fragment)
    sigs = signatures(row, fragment)
    if fragment == "W":
        m_weight = {"08-12": 0.20, "13-16": 0.40, "17-20": 1.60, "21+": 4.50}[mb]
        f_weight = {"<0.40": 1.50, "0.40-0.49": 2.00, "0.50-0.59": 0.55, "0.60+": 3.75}[fb]
        signature_weight = 1.0
        if "exists|neg=3" in sigs:
            signature_weight *= 2.50
        if "all|neg=3" in sigs:
            signature_weight *= 1.75
        if "all|neg=2" in sigs:
            signature_weight *= 1.25
        return min(m_weight * f_weight * signature_weight, 80.0)
    m_weight = {"08-12": 0.12, "13-16": 1.20, "17-20": 2.20, "21+": 3.80}[mb]
    f_weight = {"<0.40": 0.12, "0.40-0.49": 0.80, "0.50-0.59": 1.70, "0.60+": 5.50}[fb]
    signature_weight = 1.0
    for signature in ("all-exists|neg=1", "exists-all|neg=2"):
        if signature in sigs:
            signature_weight *= 2.30
    for signature in ("all-all|neg=0", "all-all|neg=1", "all-all|neg=2"):
        if signature in sigs:
            signature_weight *= 1.35
    return min(m_weight * f_weight * signature_weight, 100.0)


def weighted_without_replacement(rows: list[dict], count: int, rng: random.Random, fragment: str) -> list[dict]:
    scored = []
    for row in rows:
        weight = max(difficulty_weight(row, fragment), 1e-6)
        priority = -math.log(max(rng.random(), 1e-15)) / weight
        scored.append((priority, row))
    if len(scored) < count:
        raise RuntimeError(f"Only {len(scored)} {fragment} candidates; need {count}")
    return [row for _, row in sorted(scored, key=lambda item: item[0])[:count]]


def structure(rows: list[dict], fragment: str) -> dict:
    m_counts = Counter(m_bin(row) for row in rows)
    f_counts = Counter(fraction_bin(row, fragment) for row in rows)
    sig_counts = Counter(sig for row in rows for sig in signatures(row, fragment))
    n = len(rows)
    return {
        "rows": n,
        "m_bins": dict(sorted(m_counts.items())),
        "m_rates": {k: round(v / n, 6) for k, v in sorted(m_counts.items())},
        "target_fraction_bins": dict(sorted(f_counts.items())),
        "target_fraction_rates": {k: round(v / n, 6) for k, v in sorted(f_counts.items())},
        "signature_exposure_rates": {k: round(v / n, 6) for k, v in sig_counts.most_common()},
        "mean_sampling_weight": round(sum(difficulty_weight(row, fragment) for row in rows) / n, 6),
    }


def summary(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "fragments": dict(Counter(row["fragment"] for row in rows)),
        "labels": dict(Counter(row["label"] for row in rows)),
        "fragment_labels": dict(Counter(f"{row['fragment']}/{row['label']}" for row in rows)),
    }


def tagged(row: dict) -> dict:
    result = dict(row)
    result["sentences"] = [f"Fragment: {row['fragment']}."] + list(row["sentences"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data4", required=True)
    parser.add_argument("--data5-v", required=True)
    parser.add_argument("--data6-w", required=True)
    parser.add_argument("--data8", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9127)
    args = parser.parse_args()
    data4, data8, output = Path(args.data4), Path(args.data8), Path(args.output)
    rng = random.Random(args.seed)

    fixed_s = {}
    occupied = set()
    excluded_validation_test = {"W": set(), "V": set()}
    for split in SPLITS:
        rows = load(data8 / "generated_data" / "final" / "splits" / f"{split}.jsonl")
        fixed_s[split] = [row for row in rows if row["fragment"] == "S"]
        occupied.update(text_key(row) for row in fixed_s[split])
        if split in {"validation", "test"}:
            for fragment in ("W", "V"):
                excluded_validation_test[fragment].update(text_key(row) for row in rows if row["fragment"] == fragment)

    sources = {
        "W": [Path(args.data6_w)],
        "V": [Path(args.data5_v)],
    }
    selected = {fragment: {split: [] for split in SPLITS} for fragment in ("W", "V")}
    audits = {}
    for fragment in ("W", "V"):
        raw = []
        for split in SPLITS:
            raw.extend(row for row in load(data4 / "generated_data" / "final" / "splits" / f"{split}.jsonl") if row["fragment"] == fragment)
        for source in sources[fragment]:
            raw.extend(load(source))
        unique = {}
        for row in raw:
            validate(row, fragment)
            unique.setdefault(text_key(row), row)
        usable = [
            row for key, row in unique.items()
            if key not in occupied and key not in excluded_validation_test[fragment]
        ]
        selected_all = []
        for label_index, label in enumerate(LABELS):
            pool = [row for row in usable if row["label"] == label]
            label_rng = random.Random(args.seed + label_index + (100 if fragment == "V" else 0))
            chosen = weighted_without_replacement(pool, sum(PER_LABEL.values()), label_rng, fragment)
            label_rng.shuffle(chosen)
            start = 0
            running = 1
            for split in SPLITS:
                for source in chosen[start:start + PER_LABEL[split]]:
                    row = dict(source)
                    row["metadata"] = dict(
                        source["metadata"],
                        source_candidate_id=source.get("id"),
                        controlled_dataset="data9-hard2-first-round",
                        hard2_weighting_id=f"{fragment.lower()}-hard2-validation-profile-v1",
                        hard2_m_bin=m_bin(source),
                        hard2_fraction_bin=fraction_bin(source, fragment),
                        hard2_sampling_weight=round(difficulty_weight(source, fragment), 6),
                    )
                    row["id"] = f"{fragment}_{label[:3]}_hard2_{running:06d}"
                    running += 1
                    selected[fragment][split].append(row)
                    selected_all.append(row)
                    occupied.add(text_key(row))
                start += PER_LABEL[split]
        audits[fragment] = {
            "raw": len(raw),
            "unique": len(unique),
            "usable": len(usable),
            "usable_labels": dict(Counter(row["label"] for row in usable)),
            "selected_labels": dict(Counter(row["label"] for row in selected_all)),
            "pool_structure": structure(usable, fragment),
            "selected_structure": structure(selected_all, fragment),
        }

    split_rows = {}
    for split in SPLITS:
        rows = list(fixed_s[split]) + selected["W"][split] + selected["V"][split]
        rng.shuffle(rows)
        split_rows[split] = rows
        write(rows, output / "generated_data" / "final" / "splits" / f"{split}.jsonl")
        write([tagged(row) for row in rows], output / "generated_data" / "final" / "tagged_splits" / f"{split}.jsonl")
        for fragment in ("S", "W", "V"):
            write([row for row in rows if row["fragment"] == fragment], output / "generated_data" / "final" / "fragment_splits" / fragment / f"{split}.jsonl")

    final = [row for split in SPLITS for row in split_rows[split]]
    write(final, output / "generated_data" / "final" / "examples_swv_hard2_first_round_balanced.jsonl")
    keys = [text_key(row) for row in final]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate text remains")
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        lset = {text_key(row) for row in split_rows[left]}
        rset = {text_key(row) for row in split_rows[right]}
        overlaps[f"{left}/{right}"] = len(lset & rset)
    if any(overlaps.values()):
        raise AssertionError(f"Split leakage: {overlaps}")
    s_preserved = {}
    for split in SPLITS:
        original = fixed_s[split]
        built = [row for row in split_rows[split] if row["fragment"] == "S"]
        s_preserved[split] = {
            "count": len(built),
            "exact": {json.dumps(row, sort_keys=True) for row in original}
            == {json.dumps(row, sort_keys=True) for row in built},
        }
    if not all(item["exact"] for item in s_preserved.values()):
        raise AssertionError("S preservation failed")

    report = {
        "dataset": "data9-hard2-first-round",
        "purpose": "validation-derived structural difficulty intervention; target W 91-93%, V 84-87%",
        "design": "data8 S preserved exactly; W/V sampled by aggregate marked-large validation error profiles",
        "selection_guardrail": "data8 validation and test W/V texts excluded; no test prediction used for weighting",
        "seed": args.seed,
        "final": summary(final),
        "splits": {split: summary(rows) for split, rows in split_rows.items()},
        "s_preserved": s_preserved,
        "cross_split_text_overlap": overlaps,
        "unique_texts": len(set(keys)),
        "candidate_audit": audits,
        "final_W_structure": structure([row for row in final if row["fragment"] == "W"], "W"),
        "final_V_structure": structure([row for row in final if row["fragment"] == "V"], "V"),
    }
    report_path = output / "reports" / "final_integrity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
