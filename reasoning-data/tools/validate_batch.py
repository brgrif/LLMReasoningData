#!/usr/bin/env python3
"""Validate a GENERATOR.md batch against SCHEMA.md, its dealt specs, and the
season ban list; run the end-of-batch self-audit (near-dup scan).

Usage: validate_batch.py <specs.json> <batch-dir-or-glob> [DOMAINS.md]
  specs.json    the dealer output (deal_season.py), {"salt":..,"items":[..]}
  batch-dir     directory of *.jsonl shard/type files, or a glob
  DOMAINS.md    canonical-domain source (defaults to ../DOMAINS.md)
Exit code is non-zero if any record errors."""
import json, re, sys, glob, os, collections, itertools

if len(sys.argv) < 3:
    sys.exit(__doc__)
SPECS_PATH, BATCH = sys.argv[1], sys.argv[2]
DOMAINS_MD = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DOMAINS.md")
BATCH_GLOB = BATCH if any(c in BATCH for c in "*?[") else os.path.join(BATCH, "*.jsonl")

specs = {it["id"]: it for it in json.load(open(SPECS_PATH))["items"]}

CANONICAL = set()
for line in open(DOMAINS_MD):
    m = re.match(r"- \*\*(.+?)\.?\*\*", line)
    if m: CANONICAL.add(m.group(1).strip())

PREFIX = {"deductive":"ded","inductive":"ind","abductive":"abd","analogical":"ana",
          "causal":"cau","counterfactual":"cfa","probabilistic":"prb",
          "metacognitive":"met","moral-ethical":"mor"}
REQ = {"id","reasoning_type","domain","problem","reasoning_trace","final_answer",
       "is_correct","difficulty","generation_method","verification","provenance"}
OPT = {"confidence","notes"}
VMETH = {"symbolic_solver","code_execution","answer_match","process_check","rubric_judge"}

BANS = [  # (regex, applies-to-type or None for all)
    (r"\bsocrates\b", "deductive"), (r"monty hall", None), (r"mammogram", None),
    (r"\btrolley\b", "moral-ethical"), (r"\blifeboat\b", "moral-ethical"),
    (r"\bheinz\b", None), (r"\bbutler\b", "abductive"),
    (r"ice[- ]cream.*drown|drown.*ice[- ]cream", None),
    (r"rooster.*sunrise|sunrise.*rooster", None),
    (r"smoking.*cancer", "causal"),
    (r"atom.*solar system|solar system.*atom", "analogical"),
    (r"heart.*\bpump\b", "analogical"), (r"brain.*computer", "analogical"),
    (r"which of the following", None), (r"^\s*\(?[A-D]\)\s+", None),
]

errors, warns = [], []
records = {}
for f in sorted(glob.glob(BATCH_GLOB)):
    for ln, line in enumerate(open(f), 1):
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
        except Exception as e:
            errors.append(f"{f}:{ln} parse error: {e}"); continue
        rid = r.get("id", f"{f}:{ln}")
        if rid in records:
            errors.append(f"{rid} duplicate id"); continue
        records[rid] = r

for rid, sp in specs.items():
    if rid not in records:
        errors.append(f"{rid} missing from batch")

for rid, r in records.items():
    sp = specs.get(rid)
    if sp is None:
        errors.append(f"{rid} not in specs"); continue
    e = lambda msg: errors.append(f"{rid} {msg}")
    keys = set(r)
    if not REQ <= keys: e(f"missing fields {REQ - keys}")
    extra = keys - REQ - OPT
    if extra: e(f"unexpected fields {extra}")
    if r.get("reasoning_type") != sp["reasoning_type"]: e("reasoning_type mismatch")
    if not r.get("id","").startswith(PREFIX[sp["reasoning_type"]] + "-"): e("id prefix wrong")
    if r.get("domain") not in CANONICAL: e(f"non-canonical domain {r.get('domain')!r}")
    if r.get("domain") != sp["canonical_domain"]: e("domain differs from spec")
    if r.get("difficulty") != sp["difficulty"]: e("difficulty differs from spec")
    if r.get("is_correct") is not True: e("is_correct not true")
    if r.get("generation_method") != "self_instruct": e("generation_method wrong")
    tr = r.get("reasoning_trace")
    if not isinstance(tr, list) or len(tr) < 3:
        e("trace missing or too short")
    else:
        for i, st in enumerate(tr, 1):
            if set(st) != {"step","text","label"}: e(f"trace step {i} bad keys"); break
            if st["step"] != i: e(f"trace steps not consecutive at {i}"); break
            if st["label"] not in ("valid","invalid","unverified"): e("bad step label"); break
        if any(st.get("label") != "valid" for st in tr): warns.append(f"{rid} non-valid step label in a positive")
    v = r.get("verification", {})
    if not (isinstance(v, dict) and set(v) == {"method","passed","details"}):
        e("verification object malformed")
    else:
        if v["method"] != sp["verification_method"]: e("verification method differs from spec")
        if v["passed"] is not True: e("verification.passed not true")
        if not (isinstance(v["details"], str) and len(v["details"]) > 15): e("verification details thin")
    p = r.get("provenance", {})
    if not (isinstance(p, dict) and set(p) == {"source","seed_id","created"} and
            isinstance(p.get("created"), str) and
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.get("created") or "")):
        e("provenance malformed")
    if sp["needs_confidence"]:
        c = r.get("confidence")
        if not (isinstance(c,(int,float)) and 0 <= c <= 1): e("confidence missing/bad")
    elif "confidence" in r:
        warns.append(f"{rid} confidence on a type that does not call for it")
    if not (isinstance(r.get("final_answer"), str) and r["final_answer"].strip()): e("final_answer empty")
    if not (isinstance(r.get("problem"), str) and len(r["problem"]) > 40): e("problem too short")
    text = (r.get("problem","") + " " + r.get("final_answer","")).lower()
    for pat, t in BANS:
        if (t is None or t == sp["reasoning_type"]) and re.search(pat, text, re.M):
            e(f"ban-list hit: /{pat}/")

# self-audit: near-duplicate problems (5-gram jaccard) and domain+type+structure clusters
def grams(s):
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return set(tuple(toks[i:i+5]) for i in range(len(toks)-4))
G = {rid: grams(r["problem"]) for rid, r in records.items()}
dups = []
for (a, ga), (b, gb) in itertools.combinations(G.items(), 2):
    if not ga or not gb: continue
    if specs[a]["cluster"] and specs[a]["cluster"] == specs[b]["cluster"]:
        continue  # isomorphic by design; still must not share vocabulary
    j = len(ga & gb) / len(ga | gb)
    if j > 0.35: dups.append((a, b, round(j, 2)))
for a, b, j in dups: errors.append(f"near-duplicate {a} ~ {b} (jaccard {j})")

clusters = collections.Counter((specs[r]["deck_domain"], specs[r]["reasoning_type"], specs[r]["structure"]) for r in records)
for k, n in clusters.items():
    if n > 3: warns.append(f"structure cluster over cap: {k} x{n}")

print(f"records: {len(records)}   errors: {len(errors)}   warnings: {len(warns)}")
for x in errors: print("ERROR", x)
for x in warns: print("WARN ", x)
sys.exit(1 if errors else 0)
