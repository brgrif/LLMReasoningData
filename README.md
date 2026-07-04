# LLMReasoningData

Small, ruthlessly curated, verified datasets for training LLM reasoning.

The premise: quality, verifiability, and curation beat volume. A few hundred to a few thousand excellent traces outperform massive noisy corpora. Every example is treated as a claim that must survive verification, not as a pile of text.

## The core principle

The verifier is the product. Every example states how it was verified, and each reasoning type uses the strongest verification method it allows, ranked from strongest to weakest:

1. Symbolic / formal solvers
2. Code execution
3. Exact-answer matching
4. Process-level (step) scoring
5. Rubric-based LLM judging

A dataset is only as trustworthy as the weakest verifier it relies on, so the verifier choice is a design decision made per type.

## What the data serves

- **Supervised fine-tuning / distillation** on curated traces, teaching the model to produce well-formed reasoning.
- **Reinforcement learning with verifiable rewards (RLVR)**, which sharpens accuracy in checkable domains by rewarding verifiably correct outcomes.

Highly verifiable types feed both levers. Low-verifiability types feed only SFT and preference data, and are never used for verifiable-reward training.

## Reasoning types

Nine types, one recipe file each under [`reasoning-data/reasoning-types/`](reasoning-data/reasoning-types/):

| Type | Verifiability |
| --- | --- |
| deductive | high (formal / symbolic) |
| inductive | high |
| probabilistic | high |
| counterfactual | high |
| causal | high |
| abductive | rubric / ranking |
| analogical | rubric / ranking |
| metacognitive | process-level |
| moral-ethical | rubric |

## Current corpus

`reasoning-data/data/` holds **1,300 curated examples per type** (1,040 train / 260 eval, 11,700 total) plus **1,241 paired negatives** for the verifiable types, spanning many domains and several distinct question formats per type.

The verifiable types (deductive, inductive, probabilistic, counterfactual, causal) were checked by executing their verifier. The rubric / ranking types (abductive, analogical, moral-ethical) are template-authored with their verification method recorded, awaiting an independent solver / judge pass.

## Repository layout

Everything lives under [`reasoning-data/`](reasoning-data/):

- [`README.md`](reasoning-data/README.md) — project overview and status.
- [`RULES.md`](reasoning-data/RULES.md) — non-negotiable quality standards for every example.
- [`SCHEMA.md`](reasoning-data/SCHEMA.md) — the JSONL record format, every field, and validation rules.
- [`DOMAINS.md`](reasoning-data/DOMAINS.md) — subject-matter domains, grouped by verifiability.
- [`PIPELINE.md`](reasoning-data/PIPELINE.md) — the Propose-Verify-Mutate generation pipeline and deterministic gates.
- [`GENERATOR.md`](reasoning-data/GENERATOR.md) — the diversity-engineering generation engine.
- [`TRAINING.md`](reasoning-data/TRAINING.md) — how the data is consumed for SFT, distillation, RLVR, and preference optimization.
- [`ROADMAP.md`](reasoning-data/ROADMAP.md) — reasoning types deferred until the core nine prove out.
- [`DATA.md`](reasoning-data/DATA.md) — the on-disk dataset layout and the invariants each location enforces.
- [`reasoning-types/`](reasoning-data/reasoning-types/) — one self-contained recipe per reasoning type.
- [`data/`](reasoning-data/data/) — the datasets, organized into `seeds/`, `raw/`, `curated/`, and `negatives/`.
- [`tools/`](reasoning-data/tools/) — generation and validation tooling.

## Data lifecycle

A record moves left to right through the pipeline:

```
seeds/  →  raw/  →  curated/   (positives)
                 ↘  negatives/  (informative failures)
```

- **seeds/** — hand-written bootstrap problems, never drawn from eval.
- **raw/** — generated candidates awaiting the deterministic gates; may contain unverified or failing records.
- **curated/** — items that passed every gate; positives only, split into train and eval. Eval problems are never used as generation seeds and never appear in any train file.
- **negatives/** — retained incorrect traces, the contrastive signal for preference optimization. A negative inherits the split of its paired positive.

See [`DATA.md`](reasoning-data/DATA.md) for the full invariants.

## Tooling

- `tools/generate.py` — the procedural / parametric generator. Emits new traces (default 500 per type), verifies the verifiable types by executing their verifier, dedups against what is already on disk, and appends straight to `curated/` and `negatives/`. Safe by default (dry run unless `--write`).
- `tools/deal_season.py` + the GENERATOR.md engine — produce diversity-engineered candidates that land in `raw/` as pre-gate records, promoted to `curated/` only after clearing the Solver, Critic, and deterministic gates.

See [`tools/README.md`](reasoning-data/tools/README.md) for both.
