#!/usr/bin/env python3
"""Spec dealer for the GENERATOR.md 500-shot batch.

Deals one spec per item from shuffled decks (without replacement, reshuffle on
exhaustion), enforces composition targets, the no-consecutive-repeat rule, the
per-domain cap, and the isomorphism quota. Emits specs.json + a header."""
import json, random, sys, collections

OUT = sys.argv[1] if len(sys.argv) > 1 else "specs.json"

rng = random.SystemRandom()
SALT = rng.randint(100000, 999999)
R = random.Random(SALT)

TYPES = {  # type: (count, prefix, verification method)
    "deductive": (90, "ded", "symbolic_solver"),
    "inductive": (60, "ind", "code_execution"),
    "probabilistic": (60, "prb", "code_execution"),
    "causal": (60, "cau", "answer_match"),
    "counterfactual": (50, "cfa", "code_execution"),
    "abductive": (50, "abd", "answer_match"),
    "analogical": (50, "ana", "process_check"),
    "metacognitive": (50, "met", "process_check"),
    "moral-ethical": (30, "mor", "rubric_judge"),
}
NEXT_ID = 5160  # corpus max is 5159 in every type

# deck domain -> canonical DOMAINS.md domain (verbatim, lowercase)
DOMAINS = {
    "molecular biology": "biology and ecology",
    "epidemiology": "science",
    "veterinary medicine": "medicine-style diagnosis",
    "pharmacology": "medicine-style diagnosis",
    "structural engineering": "engineering and physical systems",
    "aerospace design": "engineering and physical systems",
    "naval architecture": "engineering and physical systems",
    "HVAC systems": "mechanical / systems troubleshooting",
    "tax law": "law and regulation",
    "maritime law": "law and regulation",
    "patent law": "law and regulation",
    "parliamentary procedure": "law and regulation",
    "monetary policy": "economics and markets",
    "insurance underwriting": "finance and business operations",
    "actuarial science": "mathematics",
    "commodities trading": "economics and markets",
    "external audit": "finance and business operations",
    "supply-chain logistics": "finance and business operations",
    "warehouse operations": "finance and business operations",
    "airline crew scheduling": "logic puzzles",
    "rail network control": "incident and root-cause analysis",
    "agriculture": "biology and ecology",
    "viticulture": "biology and ecology",
    "beekeeping": "biology and ecology",
    "soil science": "science",
    "forestry": "biology and ecology",
    "culinary arts": "science",
    "fermentation": "chemistry",
    "cheesemaking": "science",
    "music theory": "formal grammars and symbol systems",
    "orchestration": "logic puzzles",
    "sound engineering": "engineering and physical systems",
    "historical linguistics": "formal grammars and symbol systems",
    "translation": "formal grammars and symbol systems",
    "cryptography": "formal grammars and symbol systems",
    "distributed systems": "program behavior",
    "database design": "algorithms and program analysis",
    "compiler construction": "program behavior",
    "board-game rules": "logic puzzles",
    "tabletop RPG mechanics": "logic puzzles",
    "video-game economies": "economics and markets",
    "chess composition": "logic puzzles",
    "constructed languages": "formal grammars and symbol systems",
    "fictional magic systems": "logic puzzles",
    "archaeology": "science",
    "paleontology": "science",
    "volcanology": "science",
    "meteorology": "science",
    "oceanography": "science",
    "orbital mechanics": "engineering and physical systems",
    "telescope optics": "engineering and physical systems",
    "urban planning": "engineering and physical systems",
    "traffic engineering": "engineering and physical systems",
    "municipal water systems": "engineering and physical systems",
    "electoral system design": "mathematics",
    "textile manufacturing": "mechanical / systems troubleshooting",
    "garment pattern-making": "mathematics",
    "dye chemistry": "chemistry",
    "watchmaking": "mechanical / systems troubleshooting",
    "locksmithing": "mechanical / systems troubleshooting",
    "bookbinding": "mechanical / systems troubleshooting",
    "glassblowing": "science",
    "sports analytics": "mathematics",
    "tournament design": "mathematics",
    "wildfire suppression tactics": "incident and root-cause analysis",
    "emergency triage": "medicine-style diagnosis",
    "museum conservation": "science",
    "perfumery": "chemistry",
    "Mendelian inheritance": "biology and ecology",
    "ecological food webs": "biology and ecology",
    "animal behavior": "biology and ecology",
    "factory quality control": "incident and root-cause analysis",
    "robotics kinematics": "engineering and physical systems",
    "industrial control logic": "program behavior",
    "procurement": "finance and business operations",
    "org design": "finance and business operations",
}
DOMAIN_DECK = list(DOMAINS)

KERNELS = [
    "a resource is allocated under a hard constraint",
    "a signal propagates through a network with a blockage",
    "an agent must choose under incomplete information",
    "a rule has an exception that itself has an exception",
    "two conditions must jointly hold to trigger a third",
    "a hidden state explains two divergent observations",
    "a threshold flips a system from one regime to another",
    "a quantity is conserved while its distribution changes",
    "a chain of dependencies must be ordered without violating any",
    "a feedback loop amplifies a small perturbation",
    "evidence points at several suspects and must be narrowed",
    "a general pattern must be inferred from a handful of instances",
    "a mapping from one structured system to another must be completed",
    "an intervention changes an outcome that correlation alone would mispredict",
    "a plan must survive a change to one assumption",
]
REGISTERS = [
    "formal proof", "lab report", "incident postmortem", "courtroom exchange",
    "customer-support ticket", "recipe", "assembly manual",
    "dialogue between two specialists", "child's riddle", "exam question",
    "data table with a prompt", "API documentation", "field notebook",
    "diary entry", "terse news brief", "game rulebook",
    "engineering spec sheet", "patient chart", "negotiation memo",
    "step-by-step tutorial",
]
ENTITY_MODES = [
    "abstract letters and symbols", "fictional proper nouns",
    "mundane real-world objects", "invented technical jargon",
    "numbered/coded identifiers",
]
STRUCTURES = [
    "shallow chain", "deep chain", "few variables", "many variables",
    "several distractors", "noisy signal", "multiple valid paths",
    "hidden variables", "conflicting evidence", "nested exceptions",
]
TWISTS = [
    "add a red herring that looks decisive but is not",
    "introduce a constraint that eliminates the intuitive answer",
    "make the shortest-looking path wrong",
    "require noticing something absent rather than present",
    "embed a second, distractor question",
    "make two plausible answers hinge on one detail",
    "invert the usual direction of the relationship",
]
LENGTHS = ["terse", "medium", "long"]
MET_ERRORS = ["false assumption", "invalid step", "skipped case",
              "over-general conclusion"]


class Deck:
    def __init__(self, items):
        self.items, self.pile = list(items), []
    def deal(self, avoid=None, hard_reject=None):
        if not self.pile:
            self.pile = self.items[:]
            R.shuffle(self.pile)
        for i, c in enumerate(self.pile):
            if (avoid is None or c != avoid) and (hard_reject is None or not hard_reject(c)):
                return self.pile.pop(i)
        return self.pile.pop()


def difficulties(n):
    bands = [(1, .10), (2, .25), (3, .30), (4, .25), (5, .10)]
    out = []
    for d, frac in bands:
        out += [d] * round(n * frac)
    while len(out) < n: out.append(3)
    while len(out) > n: out.remove(3)
    R.shuffle(out)
    return out

# --- build the raw item pool ------------------------------------------------
domain_deck = Deck(DOMAIN_DECK)
register_deck = Deck(REGISTERS)
entity_deck = Deck(ENTITY_MODES)
structure_deck = Deck(STRUCTURES)
twist_deck = Deck(TWISTS)
kernel_deck = Deck(KERNELS)
length_deck = Deck(LENGTHS)
met_err_deck = Deck(MET_ERRORS)

domain_count = collections.Counter()
CAP = 15  # 3% of 500

type_pool = []
for t, (n, _, _) in TYPES.items():
    diffs = difficulties(n)
    type_pool += [(t, d) for d in diffs]
R.shuffle(type_pool)

# isomorphism clusters: 10 of size 3 + 10 of size 2 = 50 items (10%)
cluster_sizes = [3] * 10 + [2] * 10
items, cluster_id = [], 0
type_pool_iter = list(type_pool)

def take_type(pref=None):
    for i, (t, d) in enumerate(type_pool_iter):
        if pref is None or t == pref:
            return type_pool_iter.pop(i)
    return type_pool_iter.pop(0)

for size in cluster_sizes:
    cluster_id += 1
    kernel = kernel_deck.deal()
    structure = structure_deck.deal()
    # analogical feeds directly on isomorphism; give it a third of clusters
    pref = "analogical" if cluster_id % 3 == 0 else None
    t, d = take_type(pref)
    for _ in range(size - 1):  # cluster members each consume a pool slot
        take_type(t)
    doms = set()
    for j in range(size):
        dom = domain_deck.deal(hard_reject=lambda c: c in doms or domain_count[c] >= CAP)
        doms.add(dom); domain_count[dom] += 1
        # cluster members share type+difficulty+kernel+structure, differ elsewhere
        items.append(dict(
            reasoning_type=t, difficulty=d, kernel=kernel, structure=structure,
            deck_domain=dom, register=register_deck.deal(),
            entity_mode=entity_deck.deal(), twist=twist_deck.deal(),
            length=length_deck.deal(), cluster=f"iso-{cluster_id:02d}",
            cluster_size=size, cluster_index=j + 1,
        ))

for t, d in type_pool_iter:
    dom = domain_deck.deal(hard_reject=lambda c: domain_count[c] >= CAP)
    domain_count[dom] += 1
    items.append(dict(
        reasoning_type=t, difficulty=d, kernel=kernel_deck.deal(),
        structure=structure_deck.deal(), deck_domain=dom,
        register=register_deck.deal(), entity_mode=entity_deck.deal(),
        twist=twist_deck.deal(), length=length_deck.deal(),
        cluster=None, cluster_size=0, cluster_index=0,
    ))

assert len(items) == 500, len(items)

# metacognitive error rotation
mets = [it for it in items if it["reasoning_type"] == "metacognitive"]
for it in mets:
    it["planted_error"] = met_err_deck.deal()

# --- sequence with the no-consecutive-repeat rule ----------------------------
def ok(a, b):
    return (a["reasoning_type"] != b["reasoning_type"]
            and a["deck_domain"] != b["deck_domain"]
            and a["register"] != b["register"])

for attempt in range(200):
    R.shuffle(items)
    seq, rest = [items[0]], items[1:]
    stuck = False
    while rest:
        pick = next((x for x in rest if ok(seq[-1], x)), None)
        if pick is None:
            stuck = True; break
        rest.remove(pick); seq.append(pick)
    if not stuck:
        items = seq; break
else:
    sys.exit("could not sequence")

# --- assign ids (per-type counters, in batch order) --------------------------
counters = {t: NEXT_ID for t in TYPES}
for pos, it in enumerate(items, 1):
    t = it["reasoning_type"]
    it["seq"] = pos
    it["id"] = f"{TYPES[t][1]}-{counters[t]:06d}"
    counters[t] += 1
    it["canonical_domain"] = ("ethics" if t == "moral-ethical"
                              else DOMAINS[it["deck_domain"]])
    it["verification_method"] = TYPES[t][2]
    it["needs_confidence"] = t in ("probabilistic", "abductive")

# --- slice into 10 agent shards, cluster members co-located ------------------
shards = [[] for _ in range(10)]
by_cluster = collections.defaultdict(list)
singles = []
for it in items:
    (by_cluster[it["cluster"]] if it["cluster"] else singles).append(it)
groups = list(by_cluster.values())
R.shuffle(groups); R.shuffle(singles)
for g in groups:
    min(shards, key=len).extend(g)
for s in singles:
    min(shards, key=len).append(s)
for i, sh in enumerate(shards):
    for it in sh:
        it["shard"] = i

json.dump({"salt": SALT, "items": items}, open(OUT, "w"), indent=1)

# report
print("run salt:", SALT)
print("shard sizes:", [len(s) for s in shards])
tc = collections.Counter(i["reasoning_type"] for i in items)
dc = collections.Counter(i["difficulty"] for i in items)
print("types:", dict(tc))
print("difficulty:", dict(sorted(dc.items())))
print("distinct deck domains:", len({i['deck_domain'] for i in items}))
print("distinct registers:", len({i['register'] for i in items}))
print("max domain share:", max(collections.Counter(i['deck_domain'] for i in items).values()))
print("iso items:", sum(1 for i in items if i["cluster"]))
bad = sum(1 for a, b in zip(items, items[1:]) if not ok(a, b))
print("consecutive violations:", bad)
