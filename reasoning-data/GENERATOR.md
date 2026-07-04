# The Reasoning Data Generator (500-shot, diversity-engineered)

This is the operating prompt for the Generator agent described in PIPELINE.md. Paste it as the generator's instructions and run it to emit one batch of 500 candidate records. Solver and Critic still run downstream; this file only governs generation.

Its one job is to produce a batch that is internally wildly diverse and, across separate runs, almost non-overlapping. Every design choice below exists to fight one enemy: mode collapse, the tendency to keep generating the same few domains, phrasings, and structures.

---

## Mission (read this, it changes how you generate)

The end goal is a logical foundation for LLMs that produces emergent, transferable reasoning. Emergence does not come from volume. It comes from the model learning each reasoning operation decoupled from any single domain. A reasoning structure that only ever appears in one costume teaches that costume. The same structure seen across dozens of unrelated domains forces the model to extract the invariant operation, and that abstraction is what composes into new behavior the training data never explicitly showed.

Therefore your success metric is not "500 correct problems." It is "500 maximally spread problems that between them tile the space of reasoning structures across the widest possible range of human and invented domains." Correctness is a floor. Spread is the goal.

---

## The one hard rule

Every single item is an independent random draw from the spec space defined below. You do not pick what feels natural. You deal from shuffled decks. If your batch looks like it came from one author with a few interests, you have failed, even if every item is correct.

---

## The diversity engine: per-item spec

Before writing any item, sample one value for each dimension. The cross-product of these dimensions is in the millions before problem content even begins, which is why runs do not repeat.

| Dimension | What it controls | Source |
| --- | --- | --- |
| reasoning_type | which operation is exercised | the 9 core types |
| domain | the field the problem lives in | DOMAIN deck (swappable) |
| scenario_kernel | the abstract structural situation, domain-independent | KERNEL deck (swappable) |
| register | the genre/voice the problem is written in | REGISTER deck (swappable) |
| entity_mode | how things are named | ENTITY-MODE deck |
| difficulty | 1-5, per the type's ladder | difficulty distribution below |
| structure | depth, variable count, distractors, hidden state, noise | STRUCTURE deck |
| length | terse to expansive | sample: terse / medium / long |
| twist | one novelty constraint that bends the obvious version | TWIST deck |

The item is built to the sampled spec. The spec is your steering, not stored data. Only schema fields end up in the record.

---

## SEED POOLS  ——  swap these every season

These are the swappable inputs. To launch a new season, replace these lists wholesale and the entire output distribution shifts, even though the machinery is unchanged. Keep each pool large. Bigger pools mean less collision.

### DOMAIN deck (starter set, replace or extend per season)
Aim for breadth that feels almost random. Starter pool:

molecular biology, epidemiology, veterinary medicine, pharmacology, structural engineering, aerospace design, naval architecture, HVAC systems, tax law, maritime law, patent law, parliamentary procedure, monetary policy, insurance underwriting, actuarial science, commodities trading, external audit, supply-chain logistics, warehouse operations, airline crew scheduling, rail network control, agriculture, viticulture, beekeeping, soil science, forestry, culinary arts, fermentation, cheesemaking, music theory, orchestration, sound engineering, historical linguistics, translation, cryptography, distributed systems, database design, compiler construction, board-game rules, tabletop RPG mechanics, video-game economies, chess composition, constructed languages, fictional magic systems (internally consistent), archaeology, paleontology, volcanology, meteorology, oceanography, orbital mechanics, telescope optics, urban planning, traffic engineering, municipal water systems, electoral system design, textile manufacturing, garment pattern-making, dye chemistry, watchmaking, locksmithing, bookbinding, glassblowing, sports analytics, tournament design, wildfire suppression tactics, emergency triage, museum conservation, perfumery, Mendelian inheritance, ecological food webs, animal behavior, factory quality control, robotics kinematics, industrial control logic, procurement, org design.

Rule: within a batch, cap any single domain at roughly 3% of items. Reach for the unusual ones, not the comfortable ones.

### KERNEL deck (abstract structural situations, domain-independent)
These are the reusable skeletons that get dressed in a sampled domain. Starter pool:

a resource is allocated under a hard constraint; a signal propagates through a network with a blockage; an agent must choose under incomplete information; a rule has an exception that itself has an exception; two conditions must jointly hold to trigger a third; a hidden state explains two divergent observations; a threshold flips a system from one regime to another; a quantity is conserved while its distribution changes; a chain of dependencies must be ordered without violating any; a feedback loop amplifies a small perturbation; evidence points at several suspects and must be narrowed; a general pattern must be inferred from a handful of instances; a mapping from one structured system to another must be completed; an intervention changes an outcome that correlation alone would mispredict; a plan must survive a change to one assumption.

Kernels are deliberately reusable across seasons. Diversity comes from the domain and register they are dressed in, not from inventing new kernels every time.

### REGISTER deck (the voice/genre the problem is written in)
formal proof, lab report, incident postmortem, courtroom exchange, customer-support ticket, recipe, assembly manual, dialogue between two specialists, child's riddle, exam question, data table with a prompt, API documentation, field notebook, diary entry, terse news brief, game rulebook, engineering spec sheet, patient chart, negotiation memo, step-by-step tutorial.

### ENTITY-MODE deck (how things are named)
abstract letters and symbols; fictional proper nouns; mundane real-world objects; invented technical jargon; numbered/coded identifiers. Vary this so the model cannot lean on familiar named entities.

### STRUCTURE deck (instantiates difficulty concretely)
shallow chain / deep chain; few variables / many variables; no distractors / several distractors; clean signal / noisy signal; single valid path / multiple valid paths; fully observed / hidden variables; consistent evidence / conflicting evidence; no exceptions / nested exceptions.

### TWIST deck (bends the obvious version so items surprise)
add a red herring that looks decisive but is not; introduce a constraint that eliminates the intuitive answer; make the shortest-looking path wrong; require noticing something absent rather than present; embed a second, distractor question; make two plausible answers hinge on one detail; invert the usual direction of the relationship.

---

## Dealing mechanic (this is how spread is guaranteed)

1. At batch start, emit a run header with a random 6-digit run salt. Use it to make your selections independent from any prior run.
2. Treat each deck as a deck of cards. Shuffle, deal without replacement until the deck is exhausted, then reshuffle. This forces even coverage instead of favorites.
3. No two consecutive items may share reasoning_type, domain, or register.
4. Enforce the per-domain 3% cap across the batch.
5. Cross-run non-overlap is guaranteed by the size of the space plus seed rotation, not by memory. Do not assume you remember previous runs. Rely on the decks.

---

## Batch composition targets (per 500)

- **Reasoning types:** all 9 present. Suggested spread, tune per season: deductive 18%, inductive 12%, probabilistic 12%, causal 12%, counterfactual 10%, abductive 10%, analogical 10%, metacognitive 10%, moral-ethical 6%. Deductive is heaviest because it is the verifiable backbone.
- **Difficulty:** L1 10%, L2 25%, L3 30%, L4 25%, L5 10%. The hard tail is where reasoning is actually taught; do not let the batch pile up at L2.
- **Registers:** at least 12 of the register deck represented.
- **Domains:** at least 40 distinct domains represented.
- **Isomorphism quota:** for roughly 10% of items, take one kernel and render it in two or three unrelated domains as separate records. This directly builds the cross-domain abstraction that drives emergence and feeds the analogical type.

---

## Anti-cliché bans (do not default to these)

The model gravitates to a handful of stock examples per type. They are banned so it explores. Refresh this list each season.

- Deductive: no Socrates-is-mortal syllogisms, no bare "all A are B" with only letters.
- Probabilistic: no Monty Hall, no mammogram/disease false-positive, no two-coin or two-child puzzles, no Bayesian cab problem.
- Causal: no ice-cream-and-drowning, no smoking-and-cancer, no rooster-and-sunrise.
- Analogical: no atom-as-solar-system, no heart-as-pump, no brain-as-computer.
- Counterfactual: no "what if a major 20th-century war went differently," no "if you traveled back and changed one thing."
- Abductive: no generic murder mystery, no "the butler did it," no "wet grass so it rained."
- Metacognitive: do not always plant an arithmetic slip; rotate the error type across false assumption, invalid step, skipped case, and over-general conclusion.
- Moral-ethical: no runaway-trolley variants, no lifeboat, no Heinz-steals-the-drug.

---

## Per-item procedure

For each of the 500:
1. Deal a spec from the decks.
2. Write the problem to that spec, fully self-contained, generative-answer only (never multiple choice), with surface vocabulary that does not leak the answer.
3. Write the step-by-step reasoning_trace, ending with a brief self-check step where the type warrants it.
4. State the final_answer as the exact target string or value.
5. Confirm the item obeys RULES.md, then emit it as one schema-valid JSONL line with generation_method "self_instruct".

Keep the sampled spec out of the record except where it maps to real schema fields (domain, difficulty). Structure, register, kernel, and twist are steering only.

---

## Output contract

- Emit exactly 500 JSONL lines, each validating against SCHEMA.md.
- Then, outside the JSONL block, emit a short diversity report: counts per reasoning_type, per difficulty level, distinct domain count, distinct register count, largest single-domain share, and isomorphism-cluster count.
- The report is how the batch is judged. If any type is missing, difficulty piles up in one band, distinct domains fall below 40, or one domain exceeds the cap, the batch is rejected and regenerated.

---

## End-of-batch self-audit

Before finalizing, scan the 500 for clusters that share domain plus type plus structure. If any cluster exceeds the cap, discard those items and redraw with fresh deals. Near-duplicate problems, even across different domains, are rejects.

---

## Seasons: what changes, what stays

Keep stable (the machinery): the nine dimensions, the dealing mechanic, the composition targets, the per-item procedure, the output contract, the self-audit.

Swap per season (the seeds): the DOMAIN deck, the KERNEL deck if desired, the REGISTER deck, the ENTITY-MODE emphasis, the TWIST deck, and the anti-cliché ban list. Also bump the run salt. Swapping the decks is what makes a new season produce a genuinely different corpus from the same engine. Treat the decks as configuration, not as part of the generator's logic.

---

## A note on the `domain` schema field

The DOMAIN deck controls the surface field the problem lives in; the record's `domain` field must still be a canonical DOMAINS.md name (SCHEMA.md validation). Each deck domain therefore maps to the canonical domain whose verification ceiling governs it (e.g. beekeeping → `biology and ecology`, airline crew scheduling → `logic puzzles`, cryptography → `formal grammars and symbol systems`). Moral-ethical items map to `ethics` regardless of their surface costume. The deck domain itself lives in the problem text and in the batch report, not in the record.
