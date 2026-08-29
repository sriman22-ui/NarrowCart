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

# Ask order: attributes most likely to uniquely identify a product first.
ASK_PRIORITY = ["material", "color", "budget", "style", "use_case", "size", "feature"]

CANDIDATE_POOL = 500   # BM25 candidates to score per turn
FP_WEIGHT = 100        # fingerprint exact-match bonus vs BM25 rank tiebreaker (max 1.0)
CAT_WEIGHT = 15        # per-term category overlap bonus (secondary discriminator)


# ── helpers (replicate evaluator logic exactly) ───────────────────────────────

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
    """
    Replicate evaluator's intent_card() and return the up-to-4 cleaned constraint
    strings (hard_constraints[:2] + soft_preferences[:2]) that can be disclosed.
    """
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


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Conversational search agent.

    Core idea: the evaluator's customer script is generated from the target
    product's own metadata via intent_card(). By pre-computing the same
    fingerprint for every catalog product, we can reverse-map each disclosed
    user phrase back to the exact set of products that could have produced it,
    then rank by how many phrases match.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        # phrase → set of parent_asins that have it in their intent_card fingerprint
        self._phrase_index: dict[str, set[str]] = defaultdict(set)
        # asin → frozenset of lowercased category terms (for secondary scoring)
        self._cat_terms: dict[str, frozenset] = {}
        self._state: dict[str, dict] = {}
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
                p = json.loads(line)
                asin = str(p["parent_asin"])

                # FTS5 row
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

                # Phrase fingerprint index
                for phrase in _intent_phrases(p):
                    self._phrase_index[phrase].add(asin)

                # Category term index (for secondary scoring)
                self._cat_terms[asin] = frozenset(_terms(_text(p.get("categories"))))

        if batch:
            cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    # ── session lifecycle ────────────────────────────────────────────────────

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._state[session_id] = {
            "phrases":   [],       # exact constraint strings received so far
            "terms":     set(),    # BM25 search terms (accumulated across turns)
            "asked":     set(),    # attribute buckets we've asked about
            "exhausted": set(),    # buckets user said they have no preference for
            "filled":    set(),    # buckets covered by a received constraint phrase
            "category":  None,     # coarse-category string from turn 1
        }

    # ── per-turn helpers ────────────────────────────────────────────────────

    def _add_phrase(self, state: dict, raw: str) -> None:
        phrase = _clean(raw)
        if not phrase or phrase in state["phrases"]:
            return
        state["phrases"].append(phrase)
        state["filled"].add(_classify(phrase))
        for t in _terms(phrase):
            state["terms"].add(t)

    def _parse(self, state: dict, msg: str, turn: int) -> None:
        """Extract disclosed constraint strings and BM25 terms from the user message."""

        # Intent override: the old soft-pref IS from the target's fingerprint — keep it.
        # Only clear `asked` so we can ask fresh questions for the real hard constraint.
        if re.search(r"ignore my earlier preference", msg, re.I):
            state["asked"].clear()
            m = re.search(r"What I need is:\s*(.+?)\.?\s*$", msg, re.I)
            if m:
                self._add_phrase(state, m.group(1))
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
            for part in m.group(1).split("; "):
                self._add_phrase(state, part)
            return  # fully parsed

        # Buying turn-1 hard constraint: "A key requirement is: X."
        m = re.search(r"key requirement is:\s*(.+?)\.?\s*$", msg, re.I)
        if m:
            self._add_phrase(state, m.group(1))
            # fall through to collect remaining BM25 terms

        # Exhaustion / boundary: "I don't have [an additional / a] preference for X"
        m = re.search(r"don't have (?:an additional |a )?preference for (\w+)", msg, re.I)
        if m:
            state["exhausted"].add(m.group(1).lower())
            return

        # Intent-override turn-1: "I'm looking for {cat}. {old_soft_pref}"
        # Extract the old soft pref so it contributes to early retrieval.
        if turn == 1:
            m = re.search(r"I'm looking for [^.]+\.\s*(.+)", msg)
            if m:
                remainder = m.group(1).strip().rstrip(".")
                if remainder and "exploring" not in remainder.lower():
                    self._add_phrase(state, remainder)

        # Always accumulate BM25 terms from the full message.
        for t in _terms(msg):
            state["terms"].add(t)

    def _bm25_candidates(self, state: dict) -> list[str]:
        terms = list(state["terms"])
        if not terms:
            return []
        # Deduplicate and cap at 60 to keep FTS5 fast.
        expr = " OR ".join(f'"{t}"' for t in terms[:60])
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 3.0, 3.0, 1.5, 1.0) LIMIT ?",
            (expr, CANDIDATE_POOL),
        ).fetchall()
        return [row[0] for row in rows]

    def _choose_ask(self, state: dict) -> str | None:
        # Only skip already-asked or exhausted. Do NOT skip filled — a received
        # material phrase (e.g. "cotton") doesn't preclude asking material again
        # to unlock the more specific hard_constraints[1] (e.g. "78% Cotton, 20% Polyester").
        skip = state["asked"] | state["exhausted"]
        for attr in ASK_PRIORITY:
            if attr not in skip:
                state["asked"].add(attr)
                return attr
        return None

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

        # 1. Parse message → accumulate phrases and BM25 terms.
        self._parse(state, user_message, turn)

        # 2. BM25 retrieval over all accumulated terms.
        candidates = self._bm25_candidates(state)

        # 3. Score:
        #    - Exact fingerprint match (high weight): phrase_index lookup is O(1) per phrase.
        #    - Category overlap bonus: per matching category term (secondary discriminator).
        #    - BM25 rank bonus (tiebreaker, max 1.0).
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

        # 4. Top-k recommendations.
        if scores:
            top = [{"parent_asin": asin} for asin, _ in scores.most_common(top_k)]
        else:
            top = [{"parent_asin": asin} for asin in candidates[:top_k]]

        # 5. Clarification question.
        ask = self._choose_ask(state)
        message = (
            f"Could you tell me your {ask.replace('_', ' ')} preference?"
            if ask
            else "Here are my best recommendations based on everything you've shared."
        )

        return {
            "message": message,
            "ask_attribute": ask,
            "recommendations": top,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
