"""
test_new_dataset.py — Synthetic generalization test.

Picks 20 random products that are NOT in the public test set, generates
fresh sessions using the exact evaluator logic, and runs them through the
agent. This simulates how the agent would perform on a completely new dataset.
"""
import random
import uuid
from collections import defaultdict

from evaluator.local_evaluator import (
    catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent

# ── config ────────────────────────────────────────────────────────────────────
N_SESSIONS  = 20          # synthetic sessions to run
MAX_TURNS   = 10
TOP_K       = 10
SEED        = 42
SCENARIOS   = ["buying", "browsing", "intent_override", "boundary"]

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading catalog and building agent index...")
catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
public_samples  = load_jsonl("data/public_set.jsonl")
public_asins    = {str(s["ground_truth"]["parent_asin"]) for s in public_samples}

# Products not seen in the public test set
unseen_asins = [a for a in catalog_ids if a not in public_asins]
print(f"  {len(catalog_ids):,} total products | {len(public_asins)} in public set | "
      f"{len(unseen_asins):,} unseen")

agent = Agent("data/catalog.jsonl")
print("  Ready.\n")

# ── build synthetic samples ───────────────────────────────────────────────────
rng = random.Random(SEED)
chosen_asins = rng.sample(unseen_asins, N_SESSIONS)

# Rotate scenarios evenly
synthetic_samples = []
for i, asin in enumerate(chosen_asins):
    scenario = SCENARIOS[i % len(SCENARIOS)]
    sample = {
        "sample_id":      f"synthetic_{i+1:04d}",
        "scenario_type":  scenario,
        "ground_truth":   {"parent_asin": asin},
        "user_profile":   {},
        "difficulty_bucket": "synthetic",
    }
    card, behavior = materialize_hidden_fields(sample, products)
    sample["intent_card"] = card
    sample["behavior"]    = behavior
    synthetic_samples.append(sample)

# ── run sessions ──────────────────────────────────────────────────────────────
results = []
print(f"{'ID':<20} {'Scenario':<16} {'Hit?':<6} {'Rank':<6} {'Turn':<6} {'RR':<6}")
print("-" * 65)

for sample in synthetic_samples:
    session_id = f"syn_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    target   = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]
    cat      = coarse_category(categories.get(target, []))

    disclosed     = set()
    boundary_used = False
    override_done = scenario != "intent_override"
    user_message  = initial_message(sample, cat, disclosed)

    hit_turn  = None
    best_rank = None

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked   = normalize_recommendations(response.get("recommendations"), catalog_ids)

        if override_done and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn  = turn
            break

        if turn == MAX_TURNS:
            break

        override = sample.get("behavior", {}).get("override") or {}
        if not override_done and turn + 1 == int(override.get("turn", 3)):
            override_done = True
            nv = str(override.get("new_value", ""))
            if nv:
                disclosed.add(nv)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                sample, response.get("ask_attribute"), disclosed, boundary_used
            )

    hit = hit_turn is not None
    rr  = (1.0 / best_rank) if best_rank else 0.0

    results.append({
        "scenario": scenario,
        "hit":      hit,
        "rank":     best_rank,
        "turn":     hit_turn,
        "rr":       rr,
    })

    status = f"rank {best_rank}" if hit else "MISS"
    print(f"{sample['sample_id']:<20} {scenario:<16} {'YES' if hit else 'NO':<6} "
          f"{str(best_rank or '-'):<6} {str(hit_turn or '-'):<6} {rr:.3f}  {status}")

# ── aggregate metrics ─────────────────────────────────────────────────────────
total = len(results)
hits  = [r for r in results if r["hit"]]
hit_rate   = len(hits) / total
mrr        = sum(r["rr"] for r in results) / total
mttc_vals  = [r["turn"] if r["turn"] else MAX_TURNS + 1 for r in results]
mttc       = sum(mttc_vals) / total
efficiency = max(0.0, (11 - mttc) / 10)
tech_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency

print("\n" + "=" * 65)
print(f"  SYNTHETIC TEST ({total} unseen products)")
print("-" * 65)
print(f"  Hit Rate @ 10   : {hit_rate:.3f}  ({len(hits)}/{total})")
print(f"  MRR             : {mrr:.3f}")
print(f"  MTTC            : {mttc:.2f} turns")
print(f"  Efficiency      : {efficiency:.3f}")
print(f"  TechnicalScore  : {tech_score:.3f}")
print("-" * 65)
by_scenario: dict = defaultdict(list)
for r in results:
    by_scenario[r["scenario"]].append(r)
for sc in SCENARIOS:
    grp = by_scenario.get(sc, [])
    if not grp:
        continue
    g_hr   = sum(1 for r in grp if r["hit"]) / len(grp)
    g_mrr  = sum(r["rr"] for r in grp) / len(grp)
    g_mttc = sum(r["turn"] if r["turn"] else MAX_TURNS+1 for r in grp) / len(grp)
    print(f"  {sc:<18}  HR {g_hr:.3f}  MRR {g_mrr:.3f}  MTTC {g_mttc:.1f}  ({sum(1 for r in grp if r['hit'])}/{len(grp)})")
print("=" * 65)
print()
print("  Compare — public set:  HR 0.995  MRR 0.844  TS 0.911")
print(f"  Compare — synthetic:   HR {hit_rate:.3f}  MRR {mrr:.3f}  TS {tech_score:.3f}")
