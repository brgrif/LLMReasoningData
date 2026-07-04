# data/negatives/

Retained incorrect traces (`{type}.negatives.jsonl`), every record with `is_correct` false. Contrastive signal for preference optimization and source material for metacognitive data (RULES.md, Rule 9). A negative inherits the split of its paired positive, so a negative paired to an eval item never enters training. See DATA.md.

Current contents: 260 paired negatives for each verifiable type (deductive, inductive, probabilistic, counterfactual, causal), 1,300 total. Each encodes that type's characteristic fallacy and links to its positive via the `-neg` id and a note. Inductive, probabilistic, counterfactual, and causal negatives reuse the positive's problem (a paired flawed trace); deductive negatives pose the converse, affirming-the-consequent question.
