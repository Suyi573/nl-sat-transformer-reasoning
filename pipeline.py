#!/usr/bin/env python3
"""Scaled S/W/V replication using the official Tharindu template implementation."""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from z3 import BoolSort, Function, IntSort, Ints, Solver

from official_fragments import (
    RelativeClausesTemplates,
    RelationalSyllogiticTemplates,
    SyllogisticTemplates,
)
from official_data_construction import nouns as OFFICIAL_NOUNS, verbs as OFFICIAL_VERBS


FAMILY_SETS = {
    "S": ("S",),
    "W": ("S", "W"),
    "V": ("S", "W", "V"),
}


def fix_indefinite_articles(text: str) -> str:
    """Correct the official generator's simple `a`/`an` surface-form errors."""
    return re.sub(r"\ba (?=[aeiouAEIOU])", "an ", text)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class HierarchicalGenerator:
    def __init__(self, config: dict):
        self.config = config
        # Preserve the author's released vocabulary, including its original
        # spellings, instead of the reduced 40/20 vocabulary used by data2.
        self.nouns = tuple(OFFICIAL_NOUNS)
        self.verbs = tuple(OFFICIAL_VERBS)
        entity = IntSort()
        boolean = BoolSort()
        self.functions = {noun: Function(noun, entity, boolean) for noun in self.nouns}
        self.functions.update({verb: Function(verb, entity, entity, boolean) for verb in self.verbs})
        self.s = SyllogisticTemplates(self.functions)
        self.w = RelativeClausesTemplates(self.functions)
        self.v = RelationalSyllogiticTemplates(self.functions)
        self.x, self.y = Ints("x y")
        self.prob = float(config.get("hierarchy_probability", 0.5))
        self.timeout = int(config.get("solver_timeout_ms", 1000))

    def choose_family(self, fragment: str) -> str:
        if fragment == "S":
            return "S"
        if fragment == "W":
            return "W" if random.random() < self.prob else "S"
        if fragment == "V":
            if random.random() < self.prob:
                return "V"
            return random.choice(("S", "W"))
        raise ValueError(fragment)

    def sentence(self, family: str, nouns: list[str], verbs: list[str]):
        if family == "S":
            formula, text, quantifiers = self.s.generate_sentence_logic_pair(nouns, verbs, self.x, self.y, negations=True)
        elif family == "W":
            formula, text, quantifiers = self.w.generate_sentence_logic_pair(nouns, verbs, self.x, self.y)
        else:
            formula, text, quantifiers = self.v.generate_sentence_logic_pair(nouns, verbs, self.x, self.y)
        return formula, fix_indefinite_articles(text) + ".", quantifiers

    def instance(self, fragment: str, m: int, n1: int, n2: int) -> dict | None:
        local_nouns = random.sample(self.nouns, n1)
        local_verbs = random.sample(self.verbs, n2) if fragment == "V" else []
        formulas, sentences, quantifiers, families = [], [], [], []
        for _ in range(m):
            family = self.choose_family(fragment)
            formula, text, qs = self.sentence(family, local_nouns, local_verbs)
            formulas.append(formula)
            sentences.append(text)
            quantifiers.append(qs)
            families.append(family)
        solver = Solver()
        solver.set("timeout", self.timeout)
        solver.add(formulas)
        status = str(solver.check())
        if status == "unknown":
            return None
        return {
            "fragment": fragment,
            "label": "satisfiable" if status == "sat" else "unsatisfiable",
            "sentences": sentences,
            "logic": [str(formula) for formula in formulas],
            "families": families,
            "quantifiers": quantifiers,
            "metadata": {"m": m, "n1": n1, "n2": n2 if fragment == "V" else 0, "solver": "z3", "solver_status": status},
        }


def parse_setting(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("setting must be m,n1,n2")
    return parts


def calibrate(args) -> None:
    config = read_json(args.config)
    generator = HierarchicalGenerator(config)
    rows = []
    for fragment in args.fragments:
        for m, n1, n2 in args.settings:
            if fragment in {"S", "W"} and not (6 <= n1 <= 16):
                continue
            if fragment == "V" and not (3 <= n1 <= 8 and 3 <= n2 <= 8):
                continue
            random.seed(args.seed + sum(map(ord, fragment)) + m * 100 + n1 * 10 + n2)
            counts = Counter()
            timeouts = 0
            for _ in range(args.samples):
                row = generator.instance(fragment, m, n1, n2)
                if row is None:
                    timeouts += 1
                else:
                    counts[row["label"]] += 1
            solved = sum(counts.values())
            result = {"fragment": fragment, "m": m, "n1": n1, "n2": n2 if fragment == "V" else 0,
                      "attempts": args.samples, "solved": solved, "timeouts": timeouts,
                      "satisfiable": counts["satisfiable"], "unsatisfiable": counts["unsatisfiable"],
                      "satisfiable_ratio": counts["satisfiable"] / solved if solved else None}
            rows.append(result)
            print(json.dumps(result), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def proposed_settings(fragment: str) -> list[tuple[int, int, int]]:
    """Return a compact grid covering the paper's predicate ranges and clause ranges."""
    settings = set()
    if fragment in {"S", "W"}:
        alpha = 1.5 if fragment == "S" else 1.75
        lower_m, upper_m = (8, 29) if fragment == "S" else (9, 32)
        for n1 in range(6, 17):
            centre = round(alpha * n1)
            for m in range(max(lower_m, centre - 3), min(upper_m, centre + 3) + 1):
                settings.add((m, n1, 0))
    else:
        # For V, use the known transition point (14,5,3) only to centre a
        # compact empirical search. Acceptance is still decided solely by the
        # observed raw SAT probability.
        for n1 in range(3, 9):
            for n2 in range(3, 9):
                centre = round((2.8 * n1 + (14 / 3) * n2) / 2)
                for m in range(max(5, centre - 3), min(25, centre + 3) + 1):
                    settings.add((m, n1, n2))
    return sorted(settings)


def calibrate_region(args) -> None:
    config = read_json(args.config)
    generator = HierarchicalGenerator(config)
    if args.timeout_ms is not None:
        generator.timeout = args.timeout_ms
    all_rows = []
    accepted = []
    for fragment in args.fragments:
        for m, n1, n2 in proposed_settings(fragment):
            random.seed(args.seed + sum(map(ord, fragment)) + m * 10000 + n1 * 100 + n2)
            counts = Counter()
            timeouts = 0
            for _ in range(args.samples):
                row = generator.instance(fragment, m, n1, n2)
                if row is None:
                    timeouts += 1
                else:
                    counts[row["label"]] += 1
            solved = sum(counts.values())
            ratio = counts["satisfiable"] / solved if solved else None
            result = {
                "fragment": fragment, "m": m, "n1": n1,
                "n2": n2 if fragment == "V" else 0,
                "attempts": args.samples, "solved": solved, "timeouts": timeouts,
                "satisfiable": counts["satisfiable"],
                "unsatisfiable": counts["unsatisfiable"],
                "satisfiable_ratio": ratio,
            }
            all_rows.append(result)
            if ratio is not None and args.min_sat <= ratio <= args.max_sat and solved >= args.min_solved:
                accepted.append(result)
                print("ACCEPT", json.dumps(result), flush=True)
    payload = {
        "method": "empirical phase-region calibration",
        "phase_band": [args.min_sat, args.max_sat],
        "samples_per_setting": args.samples,
        "seed": args.seed,
        "accepted": accepted,
        "all_results": all_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    by_fragment = Counter(row["fragment"] for row in accepted)
    print("accepted settings", dict(by_fragment), flush=True)


def choose_region_setting(fragment: str, settings: list[dict], rng: random.Random) -> dict | None:
    """Sample parameters using the algorithm in the author's released code.

    The accepted calibration rows stand in for the unpublished sampling CSV.
    """
    available = [row for row in settings if row["fragment"] == fragment]
    if not available:
        raise RuntimeError(f"No accepted phase-region settings for {fragment}")
    if fragment in {"S", "W"}:
        n1 = round(6 + rng.betavariate(2, 2) * (16 - 6))
        ratios = [row["m"] / row["n1"] for row in available]
        m = round(rng.uniform(min(ratios), max(ratios)) * n1)
        nearest = min(available, key=lambda row: abs(row["m"] / row["n1"] - m / n1))
        return {**nearest, "m": m, "n1": n1, "n2": 0}

    n1 = rng.randint(3, 8)
    ma_values = [row["m"] / row["n1"] for row in available]
    m = round(rng.uniform(min(ma_values), max(ma_values)) * n1)
    nearest_ma = min(ma_values, key=lambda value: abs(value - m / n1))
    same_ma = [row for row in available if row["m"] / row["n1"] == nearest_ma]
    mb_values = [row["m"] / row["n2"] for row in same_ma]
    n2 = round(m / rng.uniform(min(mb_values), max(mb_values)))
    if not 3 <= n2 <= 8:
        return None
    nearest = min(same_ma, key=lambda row: abs(row["m"] / row["n2"] - m / n2))
    return {**nearest, "m": m, "n1": n1, "n2": n2}


def generate_region(args) -> None:
    config = read_json(args.config)
    generator = HierarchicalGenerator(config)
    calibration = read_json(args.calibration)
    settings = calibration["accepted"]
    rng = random.Random(args.seed)
    # The generator uses the module-level RNG internally; seed it as well.
    random.seed(args.seed)
    rows, attempts = [], 0
    counts = Counter()
    setting_counts = Counter()
    while min(counts["satisfiable"], counts["unsatisfiable"]) < args.min_per_label:
        if attempts >= args.max_attempts:
            raise RuntimeError(f"Could not obtain {args.min_per_label} examples per label")
        setting = choose_region_setting(args.fragment, settings, rng)
        attempts += 1
        if setting is None:
            continue
        row = generator.instance(args.fragment, setting["m"], setting["n1"], setting["n2"])
        if row is None:
            continue
        row["id"] = f"candidate_{args.fragment}_{len(rows)+1:06d}"
        row["metadata"].update({
            "phase_region": True,
            "calibration_satisfiable_ratio": setting["satisfiable_ratio"],
            "calibration_samples": setting["solved"],
        })
        rows.append(row)
        counts[row["label"]] += 1
        setting_counts[(setting["m"], setting["n1"], setting["n2"])] += 1
        if len(rows) % 1000 == 0:
            print(args.fragment, len(rows), dict(counts), "attempts", attempts, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "fragment": args.fragment,
        "method": "sampling across empirically calibrated phase-change region",
        "phase_band": calibration["phase_band"],
        "seed": args.seed, "candidates": len(rows), "attempts": attempts,
        "timeouts": attempts - len(rows), "labels": dict(counts),
        "satisfiable_ratio": counts["satisfiable"] / len(rows),
        "parameter_ranges": {
            "m": [min(r["metadata"]["m"] for r in rows), max(r["metadata"]["m"] for r in rows)],
            "n1": [min(r["metadata"]["n1"] for r in rows), max(r["metadata"]["n1"] for r in rows)],
            "n2": [min(r["metadata"]["n2"] for r in rows), max(r["metadata"]["n2"] for r in rows)],
        },
        "unique_settings_used": len(setting_counts),
        "setting_counts": {f"m={m},n1={n1},n2={n2}": count for (m,n1,n2),count in sorted(setting_counts.items())},
        "family_counts": dict(Counter(f for row in rows for f in row["families"])),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def generate(args) -> None:
    config = read_json(args.config)
    generator = HierarchicalGenerator(config)
    m, n1, n2 = args.setting
    random.seed(args.seed)
    rows, attempts = [], 0
    counts = Counter()
    while len(rows) < args.count:
        attempts += 1
        row = generator.instance(args.fragment, m, n1, n2)
        if row is None:
            continue
        row["id"] = f"candidate_{args.fragment}_{len(rows)+1:06d}"
        rows.append(row)
        counts[row["label"]] += 1
        if len(rows) % 1000 == 0:
            print(args.fragment, len(rows), dict(counts), "attempts", attempts, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"fragment": args.fragment, "setting": {"m": m, "n1": n1, "n2": n2 if args.fragment == "V" else 0},
              "seed": args.seed, "candidates": len(rows), "attempts": attempts, "timeouts": attempts-len(rows),
              "labels": dict(counts), "satisfiable_ratio": counts["satisfiable"] / len(rows),
              "family_counts": dict(Counter(f for row in rows for f in row["families"]))}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--config", default="config/replication.json")
    calibration.add_argument("--fragments", nargs="+", choices=("S", "W", "V"), required=True)
    calibration.add_argument("--settings", nargs="+", type=parse_setting, required=True)
    calibration.add_argument("--samples", type=int, default=200)
    calibration.add_argument("--seed", type=int, default=2041)
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(func=calibrate)
    generation = sub.add_parser("generate")
    generation.add_argument("--config", default="config/replication.json")
    generation.add_argument("--fragment", choices=("S", "W", "V"), required=True)
    generation.add_argument("--setting", type=parse_setting, required=True)
    generation.add_argument("--count", type=int, default=10000)
    generation.add_argument("--seed", type=int, default=2041)
    generation.add_argument("--output", required=True)
    generation.add_argument("--report", required=True)
    generation.set_defaults(func=generate)
    region = sub.add_parser("calibrate-region")
    region.add_argument("--config", default="config/replication.json")
    region.add_argument("--fragments", nargs="+", choices=("S", "W", "V"), default=("S", "W", "V"))
    region.add_argument("--samples", type=int, default=120)
    region.add_argument("--min-solved", type=int, default=80)
    region.add_argument("--timeout-ms", type=int, default=None)
    region.add_argument("--min-sat", type=float, default=0.35)
    region.add_argument("--max-sat", type=float, default=0.65)
    region.add_argument("--seed", type=int, default=3107)
    region.add_argument("--output", required=True)
    region.set_defaults(func=calibrate_region)
    generation_region = sub.add_parser("generate-region")
    generation_region.add_argument("--config", default="config/replication.json")
    generation_region.add_argument("--calibration", required=True)
    generation_region.add_argument("--fragment", choices=("S", "W", "V"), required=True)
    generation_region.add_argument("--min-per-label", type=int, default=4500)
    generation_region.add_argument("--max-attempts", type=int, default=30000)
    generation_region.add_argument("--seed", type=int, default=3107)
    generation_region.add_argument("--output", required=True)
    generation_region.add_argument("--report", required=True)
    generation_region.set_defaults(func=generate_region)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
