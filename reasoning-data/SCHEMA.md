# Schema

This file is the single source of truth for the data format. Every record in every dataset conforms to it (RULES.md, Rule 13).

Data is stored as **JSONL**: one record per line, one file per reasoning type per split, e.g. `deductive.train.jsonl`, `deductive.eval.jsonl`. JSONL is required over CSV because reasoning traces are nested structures (arrays of step objects, nested verification and provenance objects) that do not fit a flat tabular format.

## Fields

| Field | Type | Req | Description |
| --- | --- | --- | --- |
| id | string | yes | `{type-prefix}-{6 digits}`, e.g. `ded-000123`. Prefixes: ded, ind, abd, ana, cau, cfa, prb, met, mor. Negative traces append `-neg`. Example: `ded-000123` |
| reasoning_type | enum | yes | one of: deductive, inductive, abductive, analogical, causal, counterfactual, probabilistic, metacognitive, moral-ethical. Example: `deductive` |
| domain | string | yes | must match a domain listed in DOMAINS.md. Example: `logic puzzles` |
| problem | string | yes | fully self-contained problem statement including all facts, constraints, and rules needed to solve it; no external references. Example: `Every A is B. Every B is C. Is every A a C?` |
| reasoning_trace | array | yes | ordered steps, each `{"step": int, "text": string, "label": "valid" \| "invalid" \| "unverified"}`. Self-checks and considered alternatives appear as ordinary steps. Example: `[{"step":1,"text":"...","label":"valid"}]` |
| final_answer | string | yes | the concluding answer only, no reasoning. Example: `Yes` |
| is_correct | boolean | yes | true for positives; false for retained negative traces. Example: `true` |
| confidence | number | no | 0.0-1.0, the calibrated confidence expressed in the trace, where the type calls for it (probabilistic, abductive). Example: `0.82` |
| difficulty | int | yes | 1-5, per the difficulty ladder in the type's .md file. Example: `2` |
| generation_method | enum | yes | one of: procedural, teacher_distillation, self_instruct, multi_agent, human_expert. Example: `procedural` |
| verification | object | yes | `{"method": enum, "passed": boolean, "details": string}`; method is one of: symbolic_solver, code_execution, answer_match, process_check, rubric_judge. Example: `{"method":"symbolic_solver","passed":true,"details":"entailment confirmed"}` |
| provenance | object | yes | `{"source": string, "seed_id": string \| null, "created": ISO-8601 date}`. Example: `{"source":"procedural-gen","seed_id":null,"created":"2026-01-15"}` |
| notes | string | no | free text, e.g. pointer to a paired negative. Example: `paired negative: ded-000123-neg` |

## Example records

### Positive deductive record

```json
{
  "id": "ded-000123",
  "reasoning_type": "deductive",
  "domain": "logic puzzles",
  "problem": "All members of the Blue team wear a badge. Anyone wearing a badge has cleared security. Rana is a member of the Blue team. Has Rana cleared security?",
  "reasoning_trace": [
    {"step": 1, "text": "Premise: every Blue team member wears a badge.", "label": "valid"},
    {"step": 2, "text": "Premise: every badge-wearer has cleared security.", "label": "valid"},
    {"step": 3, "text": "Rana is a Blue team member, so by step 1 Rana wears a badge.", "label": "valid"},
    {"step": 4, "text": "Rana wears a badge, so by step 2 Rana has cleared security.", "label": "valid"},
    {"step": 5, "text": "Check: the conclusion uses only the given premises and valid modus ponens, no extra assumptions.", "label": "valid"}
  ],
  "final_answer": "Yes, Rana has cleared security.",
  "is_correct": true,
  "difficulty": 2,
  "generation_method": "procedural",
  "verification": {"method": "symbolic_solver", "passed": true, "details": "Encoded premises and query in first-order form; solver confirms the conclusion is entailed."},
  "provenance": {"source": "procedural-gen", "seed_id": null, "created": "2026-01-15"},
  "notes": "paired negative: ded-000123-neg"
}
```

### Negative record

```json
{
  "id": "ded-000123-neg",
  "reasoning_type": "deductive",
  "domain": "logic puzzles",
  "problem": "All members of the Blue team wear a badge. Anyone wearing a badge has cleared security. Rana has cleared security. Is Rana a member of the Blue team?",
  "reasoning_trace": [
    {"step": 1, "text": "Premise: every Blue team member wears a badge.", "label": "valid"},
    {"step": 2, "text": "Premise: every badge-wearer has cleared security.", "label": "valid"},
    {"step": 3, "text": "Rana has cleared security, so Rana wears a badge.", "label": "invalid"},
    {"step": 4, "text": "Rana wears a badge, so Rana is on the Blue team.", "label": "invalid"},
    {"step": 5, "text": "Conclude Rana is on the Blue team.", "label": "invalid"}
  ],
  "final_answer": "Yes, Rana is a member of the Blue team.",
  "is_correct": false,
  "difficulty": 2,
  "generation_method": "procedural",
  "verification": {"method": "symbolic_solver", "passed": false, "details": "Steps 3-4 affirm the consequent; clearing security does not entail badge or Blue team membership. Conclusion not entailed."},
  "provenance": {"source": "procedural-gen", "seed_id": null, "created": "2026-01-15"},
  "notes": "paired positive: ded-000123. Fallacy: affirming the consequent."
}
```

## Validation rules

- Every record must parse as JSON on its own line.
- Enums must match the allowed values exactly (case-sensitive): `reasoning_type`, `generation_method`, `verification.method`, and each step `label`.
- A record with `verification.passed` false may appear only in raw or negatives files, never in the curated positive set.
- Moral-ethical records must use `rubric_judge` as the verification method.
- Steps in `reasoning_trace` must be numbered consecutively starting from 1, with no gaps or repeats.
- `id` must carry the correct type prefix; negative traces must end in `-neg`.
- `domain` must be one of the domains listed in DOMAINS.md.

## A note on splits

Train and held-out eval live in separate files (`{type}.train.jsonl` and `{type}.eval.jsonl`). Eval problems are never used as seeds for generation. Keeping the split at the file level makes decontamination and leakage audits mechanical.

## A note on derived data

Decontamination hashes, token counts, and chat-template formatting are computed by tooling at audit or training time. They are never stored in records. The record holds the reasoning content and its verification; anything derivable from that content is regenerated on demand rather than persisted, so records stay canonical and small.
