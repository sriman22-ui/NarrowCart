#!/usr/bin/env python3
"""
demo.py -- Conversation-style session replay for the TechJam agent.

Usage:
    python demo.py                     # interactive menu
    python demo.py buying              # random buying session
    python demo.py browsing            # random browsing session
    python demo.py intent_override     # random intent_override session
    python demo.py boundary            # random boundary session
    python demo.py public_0001         # specific session by ID
    python demo.py buying --auto       # auto-advance (good for screen recording)
"""

from __future__ import annotations

import random
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


def _title(products: dict, asin: str, max_len: int = 60) -> str:
    t = str(products.get(asin, {}).get("title", "unknown"))
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


def run_session(agent, sample, products, catalog_ids, categories, auto):
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    target  = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]

    card, behavior   = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    cat_label        = coarse_category(categories.get(target, []))

    disclosed     = set()
    boundary_used = False
    override_done = scenario != "intent_override"
    user_message  = initial_message(effective_sample, cat_label, disclosed)

    # --- header ---
    print(f"\n{BOLD}{_divider('=')}{RESET}")
    print(f"{BOLD}  SESSION  {sample['sample_id']}  |  {CYAN}{scenario.upper()}{RESET}  |  {sample['difficulty_bucket']}")
    print(f"  Category : {cat_label}")
    print(f"  Target   : {BOLD}{target}{RESET}  {DIM}{_title(products, target)}{RESET}")
    print(f"{BOLD}{_divider('=')}{RESET}\n")

    hit_turn  = None
    best_rank = None

    for turn in range(1, MAX_TURNS + 1):

        # --- customer bubble ---
        print(f"  {CYAN}{BOLD}Customer (turn {turn}):{RESET}")
        print(f"  {CYAN}  \"{user_message}\"{RESET}")
        print()

        response = agent.respond(session_id, user_message, turn, TOP_K)
        ask      = response.get("ask_attribute")
        ranked   = normalize_recommendations(response.get("recommendations"), catalog_ids)

        # --- agent bubble ---
        agent_msg = response.get("message", "")
        print(f"  {YELLOW}{BOLD}Agent:{RESET}")
        print(f"  {YELLOW}  \"{agent_msg}\"{RESET}")
        if ask:
            print(f"  {YELLOW}  [asking about: {BOLD}{ask}{RESET}{YELLOW}]{RESET}")
        print()

        # --- recommendations ---
        print(f"  {BOLD}Recommendations:{RESET}")
        target_rank = None
        for i, asin in enumerate(ranked[:TOP_K], 1):
            t = _title(products, asin)
            if asin == target:
                target_rank = i
                print(f"  {GREEN}{BOLD}  {i:2d}.  {asin}   {t}   << TARGET{RESET}")
            else:
                print(f"  {DIM}  {i:2d}.  {asin}   {t}{RESET}")

        # --- check hit ---
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

    # --- footer ---
    print(f"\n{BOLD}{_divider('=')}{RESET}")
    if hit_turn:
        rr  = 1.0 / best_rank
        eff = max(0.0, (11 - hit_turn) / 10)
        print(f"  {GREEN}{BOLD}RESULT: HIT{RESET}  --  rank {best_rank}  |  turn {hit_turn}  |  RR {rr:.3f}  |  efficiency {eff:.2f}")
    else:
        print(f"  {RED}{BOLD}RESULT: MISS{RESET}")
    print(f"{BOLD}{_divider('=')}{RESET}\n")


def pick_session(samples, arg):
    scenario_types = {"buying", "browsing", "intent_override", "boundary"}

    if arg and arg.startswith("public_"):
        pool = [s for s in samples if s["sample_id"] == arg]
        if not pool:
            print(f"Session '{arg}' not found.")
            sys.exit(1)
        return pool[0]

    if arg in scenario_types:
        return random.choice([s for s in samples if s["scenario_type"] == arg])

    if arg:
        print(f"Unknown argument '{arg}'. Use a scenario type or a session ID like public_0001.")
        sys.exit(1)

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

    if chosen == "random":
        return random.choice(samples)
    return random.choice([s for s in samples if s["scenario_type"] == chosen])


def main():
    args        = sys.argv[1:]
    auto        = "--auto" in args
    pos_args    = [a for a in args if not a.startswith("--")]
    scenario_arg = pos_args[0] if pos_args else None

    print(f"\n{BOLD}  TechJam Conversational Search -- Demo{RESET}")
    print(f"  {DIM}Loading 50,000-product catalog and building index...{RESET}")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    samples = load_jsonl("data/public_set.jsonl")
    agent   = Agent("data/catalog.jsonl")
    print(f"  {GREEN}Ready!{RESET}  {len(catalog_ids):,} products indexed.")
    print(f"  Mode: {YELLOW}{'auto-advance' if auto else 'press-Enter'}{RESET}\n")

    sample = pick_session(samples, scenario_arg)
    run_session(agent, sample, products, catalog_ids, categories, auto)


if __name__ == "__main__":
    main()
