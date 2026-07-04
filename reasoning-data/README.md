# Reasoning Data

## Purpose

Build small, ruthlessly curated, verified datasets for training LLM reasoning. Quality, verifiability, and curation beat volume. A few hundred to a few thousand excellent traces outperform massive noisy corpora. This project treats every dataset as a set of claims that must survive verification, not as a pile of text.

## Core principle

The verifier is the product. Every example must state how it was verified. Verification methods rank from strongest to weakest:

1. Symbolic / formal solvers
2. Code execution
3. Exact-answer matching
4. Process-level (step) scoring
5. Rubric-based LLM judging

Always use the strongest method the reasoning type allows. A dataset is only as trustworthy as the weakest verifier it relies on, so the verifier choice is a design decision made per type, not an afterthought.

## Two training levers this data serves

1. **Supervised fine-tuning / distillation** on curated traces, which expands capability by teaching the model to produce well-formed reasoning.
2. **Reinforcement learning with verifiable rewards (RLVR)**, which sharpens accuracy in checkable domains by rewarding verifiably correct outcomes.

Highly verifiable types feed both levers. Low-verifiability types feed only SFT and preference data, and are never used for verifiable-reward training, because a reward signal you cannot check is a reward signal you cannot trust.

## Map of the repo

- **RULES.md** — the non-negotiable quality standards that apply to every example in every dataset: decontamination, verification, path-checking, generative-only answers, surface-cue scrubbing, difficulty tagging, dedup, and the defended failure modes.
- **SCHEMA.md** — the single source of truth for the data format: the JSONL record, every field, validation rules, and worked example records. All data conforms to this schema.
- **DOMAINS.md** — the subject-matter domains problems are drawn from, grouped by verifiability, so datasets stay diverse and the same structure can appear in multiple guises.
- **PIPELINE.md** — how data is generated and filtered at scale: the Propose-Verify-Mutate pipeline, the deterministic gates, and the supplementary human-sourced inputs.
- **TRAINING.md** — how the data is consumed: SFT and distillation first, the anti-collapse mixing rule, RLVR for verifiable types, and preference optimization from negatives.
- **ROADMAP.md** — reasoning types deliberately deferred until the core nine prove out, each waiting on a verifier that does not yet exist, plus the promotion criterion.
- **DATA.md** — the on-disk dataset layout: where seeds, raw candidates, curated positives, and the negatives pool live, and the invariants each location enforces.
- **reasoning-types/** — one file per reasoning type, each a self-contained recipe (definition, canonical example, what good data looks like, verification method, difficulty ladder, traps, and a sample record).
- **data/** — the datasets themselves, one file per type per split, organized into `seeds/`, `raw/`, `curated/`, and `negatives/` (see DATA.md).

## Status

Dataset available. The docs (rules, schema, domains, pipeline, per-type recipes) are in place, and `data/` holds 1,300 examples per reasoning type (1,040 train / 260 eval, 11,700 total) plus 1,241 paired negatives for the verifiable types, spanning many domains and several distinct question formats per type. Batches are produced by the reusable `tools/generate.py`, which appends 500-shot batches to `data/curated/` (and `data/negatives/`), continuing IDs above the global maximum and de-duplicating against what is already on disk. The verifiable types (deductive, inductive, probabilistic, counterfactual, causal) were checked by executing their verifier; the rubric/ranking types (abductive, analogical, moral-ethical) are template-authored with their verification method recorded, awaiting an independent Solver/judge pass.
