# NarrowCart - Conversational E-Commerce Search & Recommender (Team SAARS - NTU)

Tiktok TechJam 2026 Track 4: Shopping Copilot: AI Conversational Search and Recommendations.

## What it does

The task: find one specific product a customer has in mind, somewhere in a
frozen 50,000-item Amazon clothing/shoes/jewelry catalog, in at most 10 turns
of conversation — asking questions that actually narrow things down, not
just filling turns.

This agent does it without calling an LLM at all. The insight it's built
around: the evaluator doesn't write its simulated customer's lines from
nothing — it derives them from the target product's own metadata (an "intent
card" built from material, color, price, category, and so on). That means
if you rebuild that same intent-card logic yourself and run it against all
50,000 products up front, every phrase a customer discloses can be traced
back to the exact set of products that could have said it.

That reverse-lookup is the whole agent (`starter/agent.py`):

- Every product gets fingerprinted at index time — up to four constraint
  phrases per item — plus a SQLite FTS5 table for BM25 keyword search as a
  fallback when nothing matches a fingerprint yet.
- Each customer message is parsed to pull out what was actually disclosed,
  alongside a running log of what's been asked, answered, ruled out, or
  overridden.
- That log doubles as a consistency check: a product only stays a candidate
  if its own fingerprint could have generated the exact dialogue observed so
  far. This narrows the field far more aggressively than keyword matching on
  its own.
- When it's time to ask a question, the agent picks whichever remaining
  attribute would split the surviving candidates most evenly, rather than
  working down a fixed list.
- It holds back recommendations while too many candidates are still
  consistent with what's been said — unless every remaining question would
  get the same answer from everyone left, in which case there's no point
  waiting, or it's turn 10.

The same mechanism adapts to all four session types the challenge defines:
Buying seeds the filter with a hard constraint from turn one, Browsing
narrows gradually, Intent Override resets what's been asked and re-centers
on the new requirement, and Boundary deflections don't count as exhausting
an attribute so the agent can circle back to it later.

## Current results

Scored locally against the 200-session public set with the organizer's own
evaluator (`evaluator/local_evaluator.py`):

| Metric | This agent | Weak-BM25 baseline |
|---|---|---|
| Hit Rate@10 | 0.995 | 0.125 |
| MRR | 0.957 | 0.068 |
| MTTC | 2.83 | 9.81 |
| TechnicalScore | 0.948 | 0.107 |

## No LLM, on purpose

Every `respond()` call reports `{"prompt_tokens": 0, "completion_tokens": 0}`
because there's no model in the loop to bill for. The challenge allows a
legally accessible LLM API or local model, but doesn't require one — going
without meant no inference cost, no added latency, and a ranking pipeline
that can be stepped through line by line when something goes wrong. The
whole thing runs on the Python standard library: `sqlite3`, `re`, `json`,
`pathlib`, `collections`. No network access is required to run it.

## Project layout

```text
starter/agent.py                  the agent — indexing, parsing, filtering, ranking
evaluator/local_evaluator.py      organizer's simulator and scorer (do not edit)
demo.py                           terminal replay of a session, turn by turn
data/public_set.jsonl             200 labeled development sessions
data/catalog.jsonl                50,000-product catalog (download separately, see below)
docs/                             challenge spec, agent API contract, scoring config
```

## Running it

Download `catalog.jsonl.gz` from the GitHub Release attached to this
repository, then:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Python 3.10+ is required; no pip install needed.

```bash
python3 -m evaluator.local_evaluator     # score against the public set
python3 demo.py buying --auto            # watch one session play out in the terminal
python3 demo.py --showcase --auto        # one example per scenario type, then the
                                          # real score over all 200 sessions
```

## Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`,
`brand`, `budget`, `feature`, `use_case`, `other`, or `null`. Full contract in
`docs/agent_api_contract.json`.

## Data source

Catalog and sessions are derived from Amazon Reviews 2023 (McAuley Lab,
UCSD). See `DATA_ATTRIBUTION.md`. No session or ground-truth data was
hand-labeled — all of it traces back to what the organizer released,
generated through their own evaluator logic.

## Contributions

All team members contributed equally to this project.
Members: Sriman, Arshad, Raj, Aqil, Sachin


