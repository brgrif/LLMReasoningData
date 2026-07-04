#!/usr/bin/env python3
"""Audit curated reasoning-type splits for leakage, diversity, and coverage.

This is a read-only companion to generate.py. It does not generate data; it
summarizes the pre-rewrite health checks requested for train/eval splits.
"""
import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

REASONING_TYPES = [
    "deductive", "inductive", "abductive", "causal", "counterfactual",
    "probabilistic", "metacognitive", "moral-ethical", "analogical",
]
DEFAULT_AUDIT_TYPES = [t for t in REASONING_TYPES if t != "analogical"]


def load_domains(root: Path):
    text = (root / "DOMAINS.md").read_text(encoding="utf-8")
    return re.findall(r"^- \*\*(.+?)\.\*\*", text, re.M)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def norm_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def first_step(record):
    trace = record.get("reasoning_trace") or []
    return norm_text(trace[0].get("text", "")) if trace else ""


def structural_signature(record):
    """A conservative signature over delexicalized problem shape + trace shape."""
    problem = norm_text(record.get("problem", ""))
    problem = re.sub(r"\d+(?:\.\d+)?", "#", problem)
    problem = re.sub(r"'[^']*'|\"[^\"]*\"", "'STR'", problem)
    problem = re.sub(r"[a-z]+-[a-z]+", "word-word", problem)
    trace_labels = ",".join(step.get("label", "") for step in record.get("reasoning_trace", []))
    trace_len = len(record.get("reasoning_trace", []))
    return hashlib.sha1(f"{problem}|{trace_len}|{trace_labels}".encode()).hexdigest()[:16]


def spot_checks(records, limit=5):
    sample = records[:limit]
    out = []
    for r in sample:
        verification = r.get("verification", {})
        out.append({
            "id": r.get("id"),
            "passed": verification.get("passed"),
            "method": verification.get("method"),
            "answer": str(r.get("final_answer", ""))[:90],
            "trace_steps": len(r.get("reasoning_trace", [])),
        })
    return out


def audit_one(root: Path, reasoning_type: str):
    curated = root / "data" / "curated"
    train = load_jsonl(curated / f"{reasoning_type}.train.jsonl")
    eval_ = load_jsonl(curated / f"{reasoning_type}.eval.jsonl")
    train_problems = collections.Counter(norm_text(r.get("problem", "")) for r in train)
    eval_problems = collections.Counter(norm_text(r.get("problem", "")) for r in eval_)
    exact_overlap = sorted(set(train_problems) & set(eval_problems))

    train_first_answer = collections.Counter((first_step(r), norm_text(r.get("final_answer", ""))) for r in train)
    eval_first_answer = [(first_step(r), norm_text(r.get("final_answer", ""))) for r in eval_]
    first_answer_leaks = sum(1 for key in eval_first_answer if key in train_first_answer)

    train_struct_answer = collections.Counter((structural_signature(r), norm_text(r.get("final_answer", ""))) for r in train)
    eval_struct_answer = [(structural_signature(r), norm_text(r.get("final_answer", ""))) for r in eval_]
    struct_answer_leaks = sum(1 for key in eval_struct_answer if key in train_struct_answer)

    step_distinct = collections.defaultdict(set)
    for r in train + eval_:
        for step in r.get("reasoning_trace", []):
            step_distinct[step.get("step")].add(norm_text(step.get("text", "")))

    difficulties = collections.Counter(r.get("difficulty") for r in train + eval_)
    domains = collections.Counter(r.get("domain") for r in train + eval_)
    gen_methods = collections.Counter(r.get("generation_method") for r in train + eval_)
    provenance_sources = collections.Counter((r.get("provenance") or {}).get("source") for r in train + eval_)
    verification = collections.Counter((r.get("verification") or {}).get("method") for r in train + eval_)
    failed_verification = [r.get("id") for r in train + eval_ if not (r.get("verification") or {}).get("passed")]

    return {
        "type": reasoning_type,
        "counts": {"train": len(train), "eval": len(eval_), "total": len(train) + len(eval_)},
        "leakage": {
            "exact_problem_overlap": len(exact_overlap),
            "eval_first_step_plus_answer_in_train": first_answer_leaks,
            "eval_structural_signature_plus_answer_in_train": struct_answer_leaks,
        },
        "boilerplate": {str(k): len(v) for k, v in sorted(step_distinct.items())},
        "difficulty": {str(k): difficulties.get(k, 0) for k in range(1, 6)},
        "domains": dict(sorted(domains.items())),
        "domain_count": len(domains),
        "generation_methods": dict(gen_methods),
        "provenance_sources": dict(provenance_sources),
        "verification_methods": dict(verification),
        "failed_verification_count": len(failed_verification),
        "failed_verification_sample": failed_verification[:10],
        "spot_check_sample": spot_checks(train + eval_),
    }


def print_markdown(audits, canonical_domains):
    print("# Curated reasoning-type audit")
    print()
    print(f"Canonical domains: {len(canonical_domains)}")
    print()
    for a in audits:
        print(f"## {a['type']}")
        print(f"- Counts: train={a['counts']['train']}, eval={a['counts']['eval']}, total={a['counts']['total']}.")
        leaks = a["leakage"]
        print(f"- Leakage: exact train/eval problem overlap={leaks['exact_problem_overlap']}; eval first-step+answer in train={leaks['eval_first_step_plus_answer_in_train']}; eval structural-signature+answer in train={leaks['eval_structural_signature_plus_answer_in_train']}.")
        print(f"- Boilerplate distinct step texts by index: {a['boilerplate']}.")
        print(f"- Difficulty coverage: {a['difficulty']}.")
        missing_hard = [d for d in (4, 5) if a['difficulty'].get(str(d), 0) == 0]
        print(f"- Hard difficulties missing: {missing_hard or 'none'}.")
        print(f"- Domain coverage: {a['domain_count']}/{len(canonical_domains)} domains; {a['domains']}.")
        print(f"- Generation/provenance/verification: gen={a['generation_methods']}; source={a['provenance_sources']}; verification={a['verification_methods']}; failed_verification={a['failed_verification_count']}.")
        print("- Spot-check sample metadata:")
        for item in a["spot_check_sample"]:
            print(f"  - {item}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--types", nargs="*", default=DEFAULT_AUDIT_TYPES, choices=REASONING_TYPES)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of Markdown")
    args = parser.parse_args()
    root = args.root.resolve()
    domains = load_domains(root)
    audits = [audit_one(root, t) for t in args.types]
    if args.json:
        print(json.dumps({"canonical_domains": domains, "audits": audits}, indent=2, sort_keys=True))
    else:
        print_markdown(audits, domains)


if __name__ == "__main__":
    main()
