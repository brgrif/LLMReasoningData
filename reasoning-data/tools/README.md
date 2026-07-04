# tools/

Reusable tooling for the corpus.

- `generate.py` — procedural/parametric batch generator (verifiable types are
  executed and checked). Documented below.
- `deal_season.py` — the GENERATOR.md dealer. Shuffles the season decks and
  deals one spec per item across the nine diversity dimensions, enforcing the
  composition targets, the no-consecutive-repeat rule, the per-domain cap, and
  the isomorphism quota. The decks ship large (~115 surface domains, 26
  registers, 20 kernels, 11 twists) so per-item collisions stay rare. Emits
  `specs.json` (`{"salt":.., "items":[..]}`) that drives the authoring pass.
  Change the decks in the file to launch a new season; the run salt is drawn
  fresh each run. `python deal_season.py specs.json`. Note: this path produces
  **pre-gate** candidates for `data/raw/`, not curated records.
- `validate_batch.py` — validates an authored GENERATOR.md batch against
  SCHEMA.md and its dealt specs, scans the season ban list, and runs the
  end-of-batch near-duplicate self-audit. `python validate_batch.py specs.json
  data/raw/'*.raw.jsonl'`. Exits non-zero on any record error.

## generate.py

Produces a batch of new reasoning traces (default 500 per type - a "500-shot"
batch), verifies the verifiable types by executing their verifier, and appends
to `data/curated/` (and `data/negatives/`), continuing IDs above the global
per-type maximum and de-duplicating against whatever is already on disk. Every
emitted record validates against SCHEMA.md. This is the tool that grows
`curated/`; the GENERATOR.md path (above) grows only `raw/`.

### Usage

```
python tools/generate.py                       # dry run, 500/type, prints a report
python tools/generate.py --write               # append a 500-shot batch
python tools/generate.py --per-type 200 --seed 42 --write
python tools/generate.py --report-only         # current corpus stats, no generation
```

Safe by default: it prints a report and writes nothing unless you pass
`--write`.

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--per-type` | 500 | new records to add per reasoning type this run |
| `--seed` | 7 | RNG seed; change per season for fresh items |
| `--train-frac` | 0.8 | fraction of new records routed to train (rest to eval) |
| `--neg-frac` | 0.25 | paired negatives per verifiable type, as a fraction of its new train records |
| `--created` | today | ISO date stamped on `provenance.created` |
| `--root` | repo root | directory holding `DOMAINS.md` and `data/` |
| `--write` | off | actually append (otherwise dry run) |
| `--report-only` | off | just print current corpus stats |

### How it serves the goal

The corpus aims at a transferable logical foundation, not memorized patterns.
The generator is organized around four levers:

1. **Verifiability.** Deductive, inductive, probabilistic, counterfactual, and
   causal items are generated *and checked* by running the verifier in Python
   (truth-table entailment, executed rules, computed posteriors, run
   world-models, causal-graph analysis). An item is emitted only if it passes.
2. **Format invariance.** Each type rotates through several question formats so
   the model learns the operation, not one phrasing.
3. **Domain width.** The same logical structure is skinned across many DOMAINS.md
   domains, so structure - not vocabulary - solves the problem (RULES.md Rule 5).
4. **Contrastive signal.** Verifiable types emit paired negatives encoding the
   type's characteristic fallacy (RULES.md Rule 9), for preference data.

Difficulty is tagged 1-5 for curriculum ordering.

### Reuse across seasons

- **New season:** change `--seed`. The verifiable and metacognitive generators
  are randomized/parametric, so a new seed produces genuinely new items with no
  code change.
- **Bank-limited types:** the authored types (abductive, moral-ethical) draw
  from `BANK`-marked lists in `generate.py`. When the report shows one produced
  fewer than requested (`<-- SHORT`), extend the relevant bank; the tool never
  pads a shortfall with duplicates.
- **Analogical** is procedurally generated in three layers: (1) `STRUCTURES` --
  the cross-domain isomorphism engine: abstract relational structures each
  instantiated across many unrelated domains, generating "what plays the same
  role in system B as X in system A" items where source and target are always
  different domains (a leakage guard blocks any item whose answer appears in the
  source text); (2) parametric difficulty-4/5 trap kernels (series blockage,
  binding constraint by rate, exception-to-exception, competing relations); and
  (3) `ANA_REL` single-relation completion with explicit role labels. Train and
  eval draw from disjoint answer pools, and held-out relations / trap-skins /
  target systems are reserved for eval, so eval tests cross-domain transfer, not
  recall. Each item carries an explicit `split`. Regenerate cleanly with
  `python tools/generate.py --only analogical --replace --write` (overwrites the
  curated files, renumbers ids from 1000); fully reproducible. To scale
  cross-domain coverage, add structures to `STRUCTURES` or instances (domains) to
  existing ones -- breadth of structures x domains is what drives transfer, not
  volume of near-duplicates.
- The tool dedups against existing problems and allocates new IDs above the
  global per-type maximum (across `curated/`, `raw/`, and `negatives/`), so
  re-running is safe, additive, and never collides with pre-gate raw candidates.
- If a type's whole candidate space is already on disk, the report prints
  `ADDED NOTHING to curated` for it and writes nothing for that type - change
  `--seed` (parametric types) or extend its `BANK` (authored types) to grow it.

### Honesty

Verifiable types are genuinely checked. The rubric/ranking types
(abductive = `answer_match` by ranking, analogical = `process_check`,
moral-ethical = `rubric_judge`) are authored so the planted ground truth matches
by construction; they carry their verification method but await an independent
Solver/judge pass (PIPELINE.md). The report marks any type whose achievable
count fell short, so coverage is never overstated.
