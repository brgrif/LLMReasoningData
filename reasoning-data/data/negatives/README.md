# data/negatives/

Retained incorrect traces (`{type}.negatives.jsonl`), every record with `is_correct` false. Contrastive signal for preference optimization and source material for metacognitive data (RULES.md, Rule 9). A negative inherits the split of its paired positive, so a negative paired to an eval item never enters training. See DATA.md.

Current contents: 1,241 paired negatives across the verifiable types (deductive, probabilistic, counterfactual: 260 each; causal: 257; inductive: 204). Each encodes that type's characteristic fallacy, commits it, and reaches the wrong answer, linking to its positive via the `-neg` id and a note:

- **deductive** — poses the converse (observe the chain's last link, wrongly conclude the first) and answers "Yes", committing *affirming the consequent*; the correct answer is "No".
- **inductive** — *memorizes the first shown example* and echoes its output for every input, which already fails the next shown pair. Only pair-style problems (numeric/string) carry a negative; sequence-style items have none, so inductive holds 204 rather than 260.
- **probabilistic** — reports the sensitivity as the answer, committing *base-rate neglect*.
- **counterfactual** — reports the factual baseline and ignores the intervention.
- **causal** — concludes causation from correlation under a common cause (*confounding*). Only confounder positives get a negative; direct-cause and mediated-cause positives (where causation genuinely holds) do not, so causal holds 257.

Inductive, probabilistic, counterfactual, and causal negatives reuse the positive's problem (a paired flawed trace); deductive negatives pose the converse question. Every negative's answer differs from its paired positive's correct answer.
