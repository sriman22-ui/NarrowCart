from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


# ── constants (mirror evaluator) ──────────────────────────────────────────────

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

ASK_PRIORITY = ["material", "color", "budget", "style", "use_case", "size", "feature"]

CANDIDATE_POOL     = 500
FP_WEIGHT          = 100
CAT_WEIGHT         = 15
CONSISTENCY_BONUS  = 10_000   # ranks survivors far above non-survivors
COVERAGE_BONUS     = 1_000    # tiebreaker within survivors: prefer fewer total constraints
CONFIDENCE_THRESHOLD = 3      # withhold recs while |survivors| > this


# ── helpers (exact replicas of evaluator logic) ───────────────────────────────

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean(value: str, limit: int = 180) -> str:
    """Exact replica of evaluator's _clean_constraint."""
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        v = product.get(field)
        if isinstance(v, dict):
            parts.extend(f"{key} {item}" for key, item in v.items())
        elif isinstance(v, list):
            parts.extend(str(item) for item in v)
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts).strip()


def _intent_phrases(product: dict, limit: int = 180) -> list[str]:
    """Replicate evaluator's intent_card() — returns up to 4 constraint strings."""
    candidates: list[str] = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    mat = MATERIAL_RE.search(corpus)
    col = COLOR_RE.search(corpus)
    if mat:
        candidates.insert(0, mat.group(1).lower())
    if col:
        candidates.insert(1, f"color: {col.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean(c, limit) for c in candidates if _clean(c, limit)))
    if not cleaned:
        cleaned = [_clean(str(product.get("title") or "product"), limit)]
    return cleaned[:4]


def _classify(value: str) -> str:
    """Exact replica of evaluator's classify_constraint()."""
    lo = value.lower()
    if "budget" in lo or re.search(r"(?:\$|<=|under)\s*\d", lo):
        return "budget"
    if any(m in lo for m in MATERIALS):
        return "material"
    if any(w in lo for w in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(w in lo for w in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(w in lo for w in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(w in lo for w in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _coarse_category(categories: object) -> str:
    """Exact replica of evaluator's coarse_category()."""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in (categories or []):
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else ""


def _card_constraints(phrases: list[str]) -> tuple[list[str], list[str]]:
    """Reconstruct hard_constraints and soft_preferences from _intent_phrases output."""
    hard     = phrases[:2]
    soft_raw = phrases[2:4]
    soft     = soft_raw if soft_raw else (phrases[:1] if phrases else [])
    return hard, soft


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Conversational search agent.

    Core idea: the evaluator's customer script is generated from the target
    product's own metadata via intent_card(). Pre-computing the same fingerprint
    for every catalog product lets us reverse-map each disclosed phrase back to
    candidate products, then apply a transcript-consistency filter to narrow the
    survivor set before ranking.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._phrase_index:  dict[str, set[str]]   = defaultdict(set)
        self._asin_phrases:  dict[str, list[str]]  = {}          # asin → intent phrases
        self._cat_terms:     dict[str, frozenset]  = {}
        self._cat_bucket:    dict[str, list[str]]  = defaultdict(list)  # coarse_cat → [asin]
        self._state:         dict[str, dict]       = {}
        self._build_index()

    def _build_index(self) -> None:
        cur = self.connection.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as fh:
            for line in fh:
                p     = json.loads(line)
                asin  = str(p["parent_asin"])

                batch.append((
                    asin,
                    _text(p.get("title")),
                    _text(p.get("categories")),
                    _text(p.get("features")),
                    _text(p.get("details")),
                    _text(p.get("store")),
                    _text(p.get("description")),
                ))
                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()

                phrases = _intent_phrases(p)
                self._asin_phrases[asin] = phrases
                for phrase in phrases:
                    self._phrase_index[phrase].add(asin)

                self._cat_terms[asin] = frozenset(_terms(_text(p.get("categories"))))

                cat = _coarse_category(p.get("categories"))
                if cat:
                    self._cat_bucket[cat].append(asin)

        if batch:
            cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    # ── session lifecycle ────────────────────────────────────────────────────

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._state[session_id] = {
            "phrases":       [],    # accumulated cleaned constraint strings
            "terms":         set(), # BM25 search terms
            "asked":         set(), # attribute buckets asked this session
            "exhausted":     set(), # attributes with no remaining constraints
            "category":      None,  # coarse-category string from turn 1
            "last_ask":      None,  # attribute asked in the previous agent turn
            "observations":  [],    # [(type, ...)] for transcript-consistency filter
            "init_disclosed": set(),# phrases disclosed in the initial message
        }

    # ── per-turn helpers ─────────────────────────────────────────────────────

    def _add_phrase(self, state: dict, raw: str) -> None:
        phrase = _clean(raw)
        if not phrase or phrase in state["phrases"]:
            return
        state["phrases"].append(phrase)
        for t in _terms(phrase):
            state["terms"].add(t)

    def _parse(self, state: dict, msg: str, turn: int) -> None:
        """Extract disclosed constraint strings and BM25 terms; record observations."""

        # Intent override: keep existing phrases (old soft-pref is from the target's card).
        # Clear asked so we can ask fresh questions for the new hard constraint.
        if re.search(r"ignore my earlier preference", msg, re.I):
            state["asked"].clear()
            m = re.search(r"What I need is:\s*(.+?)\.?\s*$", msg, re.I)
            if m:
                new_val = _clean(m.group(1))
                self._add_phrase(state, new_val)
                state["observations"].append(("override", new_val))
                state["init_disclosed"].add(new_val)
            for t in _terms(msg):
                state["terms"].add(t)
            return

        # Coarse-category from "I'm looking for X"
        if state["category"] is None and "looking for" in msg.lower():
            m = re.match(r"I'm looking for ([^.,]+)", msg, re.I)
            if m:
                state["category"] = m.group(1).strip()
                for t in _terms(state["category"]):
                    state["terms"].add(t)

        # Explicit constraint reply: "For that, what matters is: X; Y."
        m = re.search(r"what matters is:\s*(.+?)\.?\s*$", msg, re.I)
        if m:
            parts = [_clean(p) for p in m.group(1).split("; ") if _clean(p)]
            for p in parts:
                self._add_phrase(state, p)
            if state["last_ask"] and parts:
                state["observations"].append(("ask", state["last_ask"], parts))
            return

        # Buying turn-1 hard constraint: "A key requirement is: X."
        m = re.search(r"key requirement is:\s*(.+?)\.?\s*$", msg, re.I)
        if m:
            phrase = _clean(m.group(1))
            self._add_phrase(state, phrase)
            state["observations"].append(("slot0", phrase))
            state["init_disclosed"].add(phrase)
            # fall through to collect remaining BM25 terms

        # Boundary deflection (one-time): "I don't have a preference for X; please use your judgment"
        # This is NOT exhaustion — discard attr from asked so it can be re-asked next turn.
        if re.search(r"don't have a preference for \w+.*use your judgment", msg, re.I):
            m2 = re.search(r"don't have a preference for (\w+)", msg, re.I)
            if m2:
                state["asked"].discard(m2.group(1).lower())
            for t in _terms(msg):
                state["terms"].add(t)
            return

        # Genuine exhaustion: "I don't have an additional preference for X"
        m = re.search(r"don't have an additional preference for (\w+)", msg, re.I)
        if m:
            attr = m.group(1).lower()
            state["exhausted"].add(attr)
            state["observations"].append(("none", attr))
            return

        # Intent-override turn-1: extract old soft pref from "I'm looking for X. {old_pref}"
        if turn == 1:
            m = re.search(r"I'm looking for [^.]+\.\s*(.+)", msg)
            if m:
                remainder = m.group(1).strip().rstrip(".")
                if remainder and "exploring" not in remainder.lower():
                    phrase = _clean(remainder)
                    self._add_phrase(state, phrase)
                    state["observations"].append(("softlast", phrase))

        for t in _terms(msg):
            state["terms"].add(t)

    # ── retrieval ────────────────────────────────────────────────────────────

    def _bm25_candidates(self, state: dict) -> list[str]:
        terms = sorted(state["terms"])  # sorted for determinism
        if not terms:
            return []
        expr = " OR ".join(f'"{t}"' for t in terms[:60])
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 3.0, 3.0, 1.5, 1.0) LIMIT ?",
            (expr, CANDIDATE_POOL),
        ).fetchall()
        return [row[0] for row in rows]

    def _get_candidates(self, state: dict) -> list[str]:
        """Category bucket when available; BM25 otherwise."""
        cat = state["category"]
        if cat and cat in self._cat_bucket:
            return self._cat_bucket[cat]
        return self._bm25_candidates(state)

    # ── transcript-consistency filter ────────────────────────────────────────

    def _consistent(self, asin: str, observations: list, init_disclosed: set) -> bool:
        """Return True if this asin's intent_card could have produced the observed dialogue."""
        phrases = self._asin_phrases.get(asin)
        if not phrases:
            return False
        hard, soft     = _card_constraints(phrases)
        all_constraints = list(dict.fromkeys([*hard, *soft]))
        disclosed       = set(init_disclosed)

        for obs in observations:
            kind = obs[0]

            if kind == "slot0":
                # hard_constraints[0] must equal the observed buying constraint
                if not hard or hard[0] != obs[1]:
                    return False
                disclosed.add(obs[1])

            elif kind == "softlast":
                # soft_preferences[-1] must equal the intent-override opener phrase
                if not soft or soft[-1] != obs[1]:
                    return False
                # NOT added to disclosed in evaluator's intent_override initial message

            elif kind == "ask":
                attr, observed_phrases = obs[1], obs[2]
                attr_norm = attr if attr in ALLOWED_ATTRIBUTES else "other"
                remaining = [v for v in all_constraints if v not in disclosed]
                expected  = [
                    v for v in remaining
                    if attr_norm == "other" or _classify(v) == attr_norm
                ][:2]
                if expected != observed_phrases:
                    return False
                disclosed.update(expected)

            elif kind == "none":
                attr      = obs[1]
                attr_norm = attr if attr in ALLOWED_ATTRIBUTES else "other"
                remaining = [v for v in all_constraints if v not in disclosed]
                has_match = any(
                    attr_norm == "other" or _classify(v) == attr_norm
                    for v in remaining
                )
                if has_match:
                    return False

            elif kind == "override":
                disclosed.add(obs[1])

        return True

    # ── information-gain ask policy ──────────────────────────────────────────

    def _shared_disclosed(self, observations: list, init_disclosed: set) -> set:
        """Disclosed set shared by all survivors (they all gave the same replies)."""
        disclosed = set(init_disclosed)
        for obs in observations:
            if obs[0] == "slot0":
                disclosed.add(obs[1])
            elif obs[0] == "ask":
                disclosed.update(obs[2])
            elif obs[0] == "override":
                disclosed.add(obs[1])
        return disclosed

    def _choose_ask_ig(self, state: dict, survivors: list[str]) -> str | None:
        """Pick the attribute that minimises expected post-reply survivor-set size."""
        skip     = state["asked"] | state["exhausted"]
        attrs    = [a for a in ASK_PRIORITY if a not in skip]
        if not attrs:
            return None
        if len(survivors) < 2:
            return attrs[0]

        disclosed = self._shared_disclosed(state["observations"], state["init_disclosed"])

        best_attr  = attrs[0]
        best_score = float("inf")

        for attr in attrs:
            attr_norm = attr if attr in ALLOWED_ATTRIBUTES else "other"
            groups: dict[str, int] = defaultdict(int)
            for asin in survivors:
                phrases = self._asin_phrases.get(asin, [])
                hard, soft = _card_constraints(phrases)
                all_c      = list(dict.fromkeys([*hard, *soft]))
                remaining  = [v for v in all_c if v not in disclosed]
                reply      = [
                    v for v in remaining
                    if attr_norm == "other" or _classify(v) == attr_norm
                ][:2]
                key = "; ".join(reply) if reply else "__none__"
                groups[key] += 1
            score = sum(n * n for n in groups.values())
            if score < best_score:
                best_score = score
                best_attr  = attr

        return best_attr

    def _choose_ask(self, state: dict, survivors: list[str]) -> str | None:
        if survivors:
            attr = self._choose_ask_ig(state, survivors)
        else:
            skip = state["asked"] | state["exhausted"]
            attr = next((a for a in ASK_PRIORITY if a not in skip), None)
        if attr:
            state["asked"].add(attr)
            state["last_ask"] = attr
        return attr

    # ── main interface ───────────────────────────────────────────────────────

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._state:
            raise RuntimeError("reset must be called before respond")
        state = self._state[session_id]

        # 1. Parse message → accumulate phrases, BM25 terms, observations.
        self._parse(state, user_message, turn)

        # 2. Retrieve candidate pool (category bucket preferred over BM25).
        candidates = self._get_candidates(state)

        # 3. Base scores: fingerprint match + category overlap + BM25 rank.
        scores: Counter = Counter()
        for phrase in state["phrases"]:
            for asin in self._phrase_index.get(phrase, ()):
                scores[asin] += FP_WEIGHT
        cat_q: frozenset = frozenset(_terms(state["category"])) if state["category"] else frozenset()
        if cat_q:
            for asin in scores:
                overlap = len(cat_q & self._cat_terms.get(asin, frozenset()))
                scores[asin] += overlap * CAT_WEIGHT
        for rank, asin in enumerate(candidates):
            scores[asin] += (CANDIDATE_POOL - rank) / CANDIDATE_POOL

        # 4. Transcript-consistency filter.
        observations   = state["observations"]
        init_disclosed = state["init_disclosed"]
        if observations:
            # Check all phrase-matched + all bucket/BM25 candidates.
            phrase_asins: set[str] = set()
            for phrase in state["phrases"]:
                phrase_asins.update(self._phrase_index.get(phrase, ()))
            pool = set(candidates) | phrase_asins
            survivors = [
                asin for asin in sorted(pool, key=lambda a: -scores.get(a, 0))
                if self._consistent(asin, observations, init_disclosed)
            ]
            # Compute how many constraint slots are already accounted for across
            # all observations (same count for every survivor — they gave identical replies).
            accounted = len(init_disclosed)
            for obs in observations:
                if obs[0] == "ask":
                    accounted += len(obs[2])
                elif obs[0] in ("none", "slot0", "override"):
                    accounted += 1

            for asin in survivors:
                phrases = self._asin_phrases.get(asin, [])
                hard, soft = _card_constraints(phrases)
                n_total = max(1, len(list(dict.fromkeys([*hard, *soft]))))
                # Coverage ratio: survivors that fully explain their card with fewer
                # total constraints rank higher — the most parsimonious survivor wins.
                scores[asin] += CONSISTENCY_BONUS + COVERAGE_BONUS * accounted / n_total
        else:
            survivors = []

        # 5. Ranked list.
        if scores:
            ranked = [asin for asin, _ in scores.most_common()]
        else:
            ranked = list(candidates)

        # 6. Confidence gate: withhold until survivors are narrow enough.
        #    Three withhold conditions (always release on turn 10):
        #    a) No observations + no phrases (browsing/boundary turn 1 — pure noise)
        #    b) Only slot0 seen so far with >1 survivor — coverage 1/n is too weak a
        #       tiebreaker; force one Q&A cycle so coverage becomes 2/n and the
        #       consistency filter can eliminate more candidates.
        #    c) Normal: survivors exist but are still too many (> CONFIDENCE_THRESHOLD)
        no_info        = not observations and not state["phrases"]
        only_slot0     = (len(observations) == 1 and observations[0][0] == "slot0"
                          and survivors and len(survivors) > 1)
        too_many       = survivors and len(survivors) > CONFIDENCE_THRESHOLD
        if turn < 10 and (no_info or only_slot0 or too_many):
            top = []
        else:
            top = [{"parent_asin": asin} for asin in ranked[:top_k]]

        # 7. Clarification question (information-gain policy when filter is active).
        ask = self._choose_ask(state, survivors)
        message = (
            f"Could you tell me your {ask.replace('_', ' ')} preference?"
            if ask
            else "Here are my best recommendations based on everything you've shared."
        )

        return {
            "message":        message,
            "ask_attribute":  ask,
            "recommendations": top,
            "usage":          {"prompt_tokens": 0, "completion_tokens": 0},
        }
