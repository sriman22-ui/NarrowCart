#!/usr/bin/env python3
"""
demo.py -- Conversation-style session replay for the TechJam agent.

Usage:
    python demo.py                     # interactive menu, one session
    python demo.py buying              # random buying session
    python demo.py browsing            # random browsing session
    python demo.py intent_override     # random intent_override session
    python demo.py boundary            # random boundary session
    python demo.py public_0001         # specific session by ID
    python demo.py --all               # all 200 sessions + aggregate metrics
    python demo.py buying --all        # all buying sessions + metrics
    python demo.py buying --auto       # auto-advance timing (good for recording)
"""

from __future__ import annotations

import random
import statistics
import sys
import time
import uuid

from starter.agent import Agent
from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

MAX_TURNS  = 10
TOP_K      = 10
AUTO_PAUSE = 2.5


def _safe(text: str) -> str:
    """Strip non-ASCII characters so Windows cp1252 terminal never crashes."""
    # product titles sometimes carry curly quotes / em dashes that raise
    # UnicodeEncodeError on the default Windows console codepage
    return text.encode("ascii", errors="replace").decode("ascii")


def _title(products: dict, asin: str, max_len: int = 60) -> str:
    t = _safe(str(products.get(asin, {}).get("title", "unknown")))
    return t[:max_len] + ("..." if len(t) > max_len else "")


def _divider(char: str = "-", n: int = 65) -> str:
    return char * n


def _pause(auto: bool) -> None:
    if auto:
        time.sleep(AUTO_PAUSE)
    else:
        try:
            input(f"{DIM}  [press Enter to continue]{RESET}")
        except (EOFError, KeyboardInterrupt):
            print()


def run_session(agent, sample, products, catalog_ids, categories, auto) -> dict:
    """Run one session. Returns result dict for metrics aggregation."""
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    target   = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]

    card, behavior   = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    cat_label        = coarse_category(categories.get(target, []))

    disclosed     = set()
    boundary_used = False
    override_done = scenario != "intent_override"
    user_message  = initial_message(effective_sample, cat_label, disclosed)

    # --- session header ---
    print(f"\n{BOLD}{_divider('=')}{RESET}")
    print(f"{BOLD}  SESSION  {sample['sample_id']}  |  {CYAN}{scenario.upper()}{RESET}  |  {sample['difficulty_bucket']}")
    print(f"  Category : {cat_label}")
    print(f"  Target   : {BOLD}{_safe(target)}{RESET}  {DIM}{_title(products, target)}{RESET}")
    print(f"{BOLD}{_divider('=')}{RESET}\n")

    hit_turn  = None
    best_rank = None

    for turn in range(1, MAX_TURNS + 1):

        # customer bubble
        print(f"  {CYAN}{BOLD}Customer (turn {turn}):{RESET}")
        print(f"  {CYAN}  \"{_safe(user_message)}\"{RESET}")
        print()

        response  = agent.respond(session_id, user_message, turn, TOP_K)
        ask       = response.get("ask_attribute")
        ranked    = normalize_recommendations(response.get("recommendations"), catalog_ids)
        agent_msg = response.get("message", "")

        # agent bubble
        print(f"  {YELLOW}{BOLD}Agent:{RESET}")
        print(f"  {YELLOW}  \"{_safe(agent_msg)}\"{RESET}")
        if ask:
            print(f"  {YELLOW}  [asking about: {BOLD}{ask}{RESET}{YELLOW}]{RESET}")
        print()

        # recommendations
        print(f"  {BOLD}Recommendations:{RESET}")
        for i, asin in enumerate(ranked[:TOP_K], 1):
            t = _title(products, asin)
            if asin == target:
                print(f"  {GREEN}{BOLD}  {i:2d}.  {asin}   {t}   << TARGET{RESET}")
            else:
                print(f"  {DIM}  {i:2d}.  {asin}   {t}{RESET}")

        # check hit
        if override_done and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn  = turn
            print(f"\n  {GREEN}{BOLD}  [FOUND at rank {best_rank} on turn {hit_turn}]{RESET}")
            break

        if turn == MAX_TURNS:
            print(f"\n  {RED}{BOLD}  [NOT FOUND within {MAX_TURNS} turns]{RESET}")
            break

        print()
        _pause(auto)
        print(f"  {_divider('.', 65)}")
        print()

        # next customer message
        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_done and turn + 1 == int(override.get("turn", 3)):
            override_done = True
            nv = str(override.get("new_value", ""))
            if nv:
                disclosed.add(nv)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, ask, disclosed, boundary_used
            )

    # session footer
    print(f"\n{BOLD}{_divider('=')}{RESET}")
    if hit_turn:
        rr  = 1.0 / best_rank
        eff = max(0.0, (11 - hit_turn) / 10)
        print(f"  {GREEN}{BOLD}RESULT: HIT{RESET}  --  rank {best_rank}  |  turn {hit_turn}  |  RR {rr:.3f}  |  efficiency {eff:.2f}")
    else:
        print(f"  {RED}{BOLD}RESULT: MISS{RESET}")
    print(f"{BOLD}{_divider('=')}{RESET}\n")

    return {
        "scenario": scenario,
        "hit":      hit_turn is not None,
        "rank":     best_rank,
        "turn":     hit_turn,
        "rr":       (1.0 / best_rank) if best_rank else 0.0,
    }


def print_metrics(results: list[dict]) -> None:
    """Print aggregate metrics from a list of session result dicts."""
    total = len(results)
    hits  = [r for r in results if r["hit"]]

    hit_rate = len(hits) / total
    mrr      = statistics.fmean(r["rr"] for r in results)
    mttc     = statistics.fmean(
        r["turn"] if r["turn"] is not None else MAX_TURNS + 1
        for r in results
    )
    efficiency    = max(0.0, (11 - mttc) / 10)
    tech_score    = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency

    # per-scenario breakdown
    from collections import defaultdict
    by_scenario: dict[str, list] = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    print(f"\n{BOLD}{_divider('=')}{RESET}")
    print(f"{BOLD}  AGGREGATE METRICS  ({total} sessions){RESET}")
    print(_divider())
    print(f"  Hit Rate @ 10   : {GREEN}{BOLD}{hit_rate:.3f}{RESET}   ({len(hits)}/{total} sessions found the target)")
    print(f"  MRR             : {GREEN}{BOLD}{mrr:.3f}{RESET}")
    print(f"  MTTC            : {YELLOW}{BOLD}{mttc:.2f}{RESET} turns avg to first hit")
    print(f"  Efficiency      : {YELLOW}{BOLD}{efficiency:.3f}{RESET}")
    print(f"  {BOLD}TechnicalScore  : {GREEN}{BOLD}{tech_score:.3f}{RESET}  (baseline: 0.107)")
    print(_divider())
    print(f"  {BOLD}By scenario:{RESET}")
    for scenario in ["buying", "browsing", "intent_override", "boundary"]:
        group = by_scenario.get(scenario, [])
        if not group:
            continue
        g_hits = sum(1 for r in group if r["hit"])
        g_hr   = g_hits / len(group)
        g_mrr  = statistics.fmean(r["rr"] for r in group)
        g_mttc = statistics.fmean(
            r["turn"] if r["turn"] is not None else MAX_TURNS + 1
            for r in group
        )
        hit_colour = GREEN if g_hr >= 0.95 else (YELLOW if g_hr >= 0.80 else RED)
        print(
            f"    {CYAN}{scenario:<18}{RESET}"
            f"  HR {hit_colour}{g_hr:.3f}{RESET}"
            f"  MRR {g_mrr:.3f}"
            f"  MTTC {g_mttc:.1f}"
            f"  ({g_hits}/{len(group)})"
        )
    print(f"{BOLD}{_divider('=')}{RESET}\n")


def main():
    args         = sys.argv[1:]
    auto         = "--auto" in args
    run_all      = "--all"  in args
    pos_args     = [a for a in args if not a.startswith("--")]
    scenario_arg = pos_args[0] if pos_args else None

    scenario_types = {"buying", "browsing", "intent_override", "boundary"}

    print(f"\n{BOLD}  TechJam Conversational Search -- Demo{RESET}")
    print(f"  {DIM}Loading 50,000-product catalog and building index...{RESET}")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    samples = load_jsonl("data/public_set.jsonl")
    agent   = Agent("data/catalog.jsonl")
    print(f"  {GREEN}Ready!{RESET}  {len(catalog_ids):,} products indexed.")

    if run_all:
        # `demo.py buying --all` narrows to one scenario; plain `--all` runs
        # the full 200-session public set
        pool = (
            [s for s in samples if s["scenario_type"] == scenario_arg]
            if scenario_arg in scenario_types
            else samples
        )
        label = scenario_arg.upper() if scenario_arg in scenario_types else "ALL"
        print(f"  Mode: {YELLOW}ALL sessions ({label}){RESET}  --  {len(pool)} sessions\n")

        results = []
        for i, sample in enumerate(pool, 1):
            print(f"{DIM}  --- session {i}/{len(pool)} ---{RESET}")
            result = run_session(agent, sample, products, catalog_ids, categories, auto=True)
            results.append(result)

        print_metrics(results)

    else:
        # single session mode
        print(f"  Mode: {YELLOW}{'auto-advance' if auto else 'press-Enter'}{RESET}\n")

        if scenario_arg and scenario_arg.startswith("public_"):
            pool = [s for s in samples if s["sample_id"] == scenario_arg]
            if not pool:
                print(f"Session '{scenario_arg}' not found.")
                sys.exit(1)
            sample = pool[0]
        elif scenario_arg in scenario_types:
            sample = random.choice([s for s in samples if s["scenario_type"] == scenario_arg])
        elif scenario_arg:
            print(f"Unknown argument '{scenario_arg}'.")
            sys.exit(1)
        else:
            # interactive menu
            print(f"\n{BOLD}  Pick a scenario:{RESET}")
            options = [
                ("buying",          "customer states a hard requirement upfront"),
                ("browsing",        "customer starts vague, agent unlocks info by asking"),
                ("intent_override", "customer pivots mid-session to their real need"),
                ("boundary",        "customer deflects the first question"),
                ("random",          "surprise me"),
            ]
            for i, (name, desc) in enumerate(options, 1):
                print(f"    {BOLD}{i}{RESET}.  {CYAN}{name:<18}{RESET} {DIM}{desc}{RESET}")
            print()
            choice  = input("  Enter number (1-5): ").strip()
            mapping = {"1": "buying", "2": "browsing", "3": "intent_override", "4": "boundary", "5": "random"}
            chosen  = mapping.get(choice, "random")
            sample  = (
                random.choice(samples)
                if chosen == "random"
                else random.choice([s for s in samples if s["scenario_type"] == chosen])
            )

        result = run_session(agent, sample, products, catalog_ids, categories, auto)
        print_metrics([result])


if __name__ == "__main__":
    main()
