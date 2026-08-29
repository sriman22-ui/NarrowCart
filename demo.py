#!/usr/bin/env python3
"""
demo.py — Interactive session replay for the TechJam conversational search agent.

Usage:
    python demo.py                     # interactive menu
    python demo.py buying              # random buying session
    python demo.py browsing            # random browsing session
    python demo.py intent_override     # random intent_override session
    python demo.py boundary            # random boundary session
    python demo.py public_0001         # specific session by ID
    python demo.py buying --auto       # auto-advance (timed), good for recording
"""

from __future__ import annotations

import random
import sys
import time
import uuid
from pathlib import Path

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

# ── ANSI colours (work in Windows Terminal / PowerShell 7) ────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

MAX_TURNS  = 10
TOP_K      = 10
AUTO_PAUSE = 2.0   # seconds between turns in --auto mode


# ── helpers ───────────────────────────────────────────────────────────────────

def _title(products: dict, asin: str, max_len: int = 65) -> str:
    t = str(products.get(asin, {}).get("title", ""))
    return t[:max_len] + ("..." if len(t) > max_len else "")


def _bar(char: str = "-", n: int = 62) -> str:
    return char * n


def _pause(auto: bool) -> None:
    if auto:
        time.sleep(AUTO_PAUSE)
    else:
        try:
            input(f"  {DIM}[press Enter for next turn]{RESET}")
        except (EOFError, KeyboardInterrupt):
            print()


# ── session replay ────────────────────────────────────────────────────────────

def run_session(
    agent: Agent,
    sample: dict,
    products: dict,
    catalog_ids: set,
    categories: dict,
    auto: bool,
) -> None:
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    target   = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]

    card, behavior     = materialize_hidden_fields(sample, products)
    effective_sample   = {**sample, "intent_card": card, "behavior": behavior}
    cat_label          = coarse_category(categories.get(target, []))

    disclosed      = set()
    boundary_used  = False
    override_done  = scenario != "intent_override"
    user_message   = initial_message(effective_sample, cat_label, disclosed)

    # ── session header ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{_bar('=')}{RESET}")
    print(f"{BOLD}  {sample['sample_id']}  |  {CYAN}{scenario.upper()}{RESET}  |  {sample['difficulty_bucket']}")
    print(f"  Category : {cat_label}")
    print(f"  Target   : {BOLD}{target}{RESET}")
    print(f"  Title    : {DIM}{_title(products, target)}{RESET}")
    print(f"{BOLD}{_bar('=')}{RESET}")

    hit_turn  = None
    best_rank = None

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n{BOLD}  Turn {turn}{RESET}  {_bar('.', 50)}")
        print(f"  {CYAN}Customer :{RESET}  {user_message}")

        response = agent.respond(session_id, user_message, turn, TOP_K)
        ask      = response.get("ask_attribute")
        ranked   = normalize_recommendations(response.get("recommendations"), catalog_ids)

        print(f"  {YELLOW}Agent asks:{RESET}  {ask or '(no follow-up question)'}")
        print(f"\n  {BOLD}Top recommendations:{RESET}")

        for i, asin in enumerate(ranked[:TOP_K], 1):
            t = _title(products, asin)
            if asin == target:
                marker = f"{GREEN}{BOLD}>> #{i:2d}  {asin}   {t}   << TARGET{RESET}"
                print(f"    {marker}")
            else:
                print(f"    {DIM}  #{i:2d}  {asin}   {t}{RESET}")

        if override_done and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn  = turn
            print(f"\n  {GREEN}{BOLD}[HIT]  TARGET FOUND  --  rank {best_rank},  turn {hit_turn}{RESET}")
            break

        if turn == MAX_TURNS:
            print(f"\n  {RED}{BOLD}[MISS]  TARGET NOT FOUND within {MAX_TURNS} turns{RESET}")
            break

        # Prepare next user message
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

        _pause(auto)

    # ── result footer ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{_bar('=')}{RESET}")
    if hit_turn:
        rr = 1.0 / best_rank
        eff = max(0.0, (11 - hit_turn) / 10)
        print(
            f"  {GREEN}{BOLD}HIT{RESET}  "
            f"rank {best_rank}  |  turn {hit_turn}  |  "
            f"RR {rr:.3f}  |  efficiency {eff:.2f}"
        )
    else:
        print(f"  {RED}{BOLD}MISS{RESET}")
    print(f"{BOLD}{_bar('=')}{RESET}\n")


# ── interactive menu ──────────────────────────────────────────────────────────

def pick_session(samples: list[dict], arg: str | None) -> dict:
    scenario_types = {"buying", "browsing", "intent_override", "boundary"}

    if arg and arg.startswith("public_"):
        pool = [s for s in samples if s["sample_id"] == arg]
        if not pool:
            print(f"Session '{arg}' not found.")
            sys.exit(1)
        return pool[0]

    if arg in scenario_types:
        pool = [s for s in samples if s["scenario_type"] == arg]
        return random.choice(pool)

    if arg:
        print(f"Unknown argument '{arg}'. Use a scenario type or session ID.")
        sys.exit(1)

    # Interactive menu
    print(f"\n{BOLD}  Pick a scenario to demo:{RESET}")
    options = [
        ("buying",         "80 sessions - customer tells you a hard requirement up front"),
        ("browsing",       "80 sessions - customer starts vague; you unlock info by asking"),
        ("intent_override","30 sessions - customer pivots mid-session to their real need"),
        ("boundary",       "10 sessions - customer deflects your first question"),
        ("random",         "any session at random"),
    ]
    for i, (name, desc) in enumerate(options, 1):
        print(f"    {BOLD}{i}{RESET}.  {CYAN}{name:<18}{RESET}{DIM}{desc}{RESET}")
    print()

    choice = input("  Enter number (1-5): ").strip()
    mapping = {"1": "buying", "2": "browsing", "3": "intent_override", "4": "boundary", "5": "random"}
    chosen  = mapping.get(choice, "random")

    if chosen == "random":
        return random.choice(samples)
    return random.choice([s for s in samples if s["scenario_type"] == chosen])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    auto = "--auto" in args
    pos_args = [a for a in args if not a.startswith("--")]
    scenario_arg = pos_args[0] if pos_args else None

    catalog_path = Path("data/catalog.jsonl")
    dataset_path = Path("data/public_set.jsonl")

    print(f"\n{BOLD}  TechJam Conversational Search -- Demo{RESET}")
    print(f"  {DIM}Loading 50,000-product catalog and building index...{RESET}")
    catalog_ids, categories, products = catalog_index(str(catalog_path))
    samples = load_jsonl(str(dataset_path))
    agent   = Agent(str(catalog_path))
    print(f"  {GREEN}Ready!{RESET}  {len(catalog_ids):,} products indexed.\n")

    mode = "auto-advance" if auto else "press-Enter"
    print(f"  Mode: {YELLOW}{mode}{RESET}  (use --auto flag for timed replay)\n")

    sample = pick_session(samples, scenario_arg)
    run_session(agent, sample, products, catalog_ids, categories, auto)


if __name__ == "__main__":
    main()
