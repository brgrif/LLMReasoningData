# Roadmap

These reasoning types are deliberately deferred until the core nine prove out. Each is deferred for the same reason: it requires a simulator, engine, or environment before its data can be verified, and this project does not ship a reasoning type without a verifier for it.

- **Temporal.** Reasoning about ordering, duration, and dependencies among events. Needs a timeline or temporal-constraint checker that can confirm an ordering or schedule satisfies the stated constraints.
- **Spatial.** Reasoning about geometry, scenes, and transformations. Needs a geometry solver or physics engine to check that a described configuration or transformation is correct.
- **Systems.** Reasoning about feedback loops and emergent behavior. Needs a system simulator that can run the described dynamics and confirm the predicted behavior.
- **Strategic.** Multi-agent planning with competing goals. Needs a game engine or payoff function that can evaluate whether a proposed strategy is actually optimal or equilibrium.
- **Social / theory of mind.** Inferring the beliefs and intentions of other agents. Needs a multi-agent simulation with known agent states, so the true beliefs can be compared against the inferred ones.
- **Decompositional.** Breaking a complex task into sub-goals. Partially covered already as a cross-cutting trace behavior that appears inside other types; a standalone type waits on a verifier that can check a decomposition is complete and correct.
- **Creative.** Producing novel ideas under constraints. Lowest verifiability of the set; needs rubric plus constraint-satisfaction checks that can confirm the output meets the hard constraints while leaving novelty to the rubric.

## Promotion criterion

A type moves from ROADMAP.md into `reasoning-types/` only when a working verifier for it exists. Until the verifier is built and tested, the type stays here, because a type without a verifier cannot meet the core principle that the verifier is the product.
