"""Submission entry point for the PTCG AI Battle Challenge (cabt engine).

The competition harness imports this file and calls ``agent(obs)`` once per
decision point. The packaged submission is a ``.tar.gz`` with this file and
``deck.csv`` at the TOP LEVEL (see scripts/build_submission.py).

Protocol (verified against kaggle_environments' bundled `cabt` env)
-------------------------------------------------------------------
Each step the engine passes an observation dict with keys:
    obs["select"]   -> the choice to make this step, or None
    obs["current"]  -> the board state (State), or None before the game starts
    obs["logs"]     -> list of event-log entries since the last step

The agent returns a ``list[int]``:
  * When ``obs["select"] is None`` the engine is asking for our DECK, so we
    return the 60 card IDs from deck.csv.
  * Otherwise we return the indices (into ``obs["select"]["option"]``) of the
    options we pick. The engine ONLY ever presents legal options, and expects
    a count in ``[select["minCount"], select["maxCount"]]``.

`select` fields: type (SelectType), context (SelectContext), minCount, maxCount,
remainDamageCounter, remainEnergyCost, option (list of Option), deck, contextCard,
effect. Each Option has at least a ``type`` (OptionType) plus context-specific
fields. See notes/cabt_api.md for the full reference.
"""

import os
import sys
from typing import Any


def _agent_dir() -> str:
    """Locate the bundle dir at runtime.

    Kaggle execs main.py with NO ``__file__`` defined, but appends the agent's
    directory to sys.path, so we find the dir that actually holds deck.csv.
    """
    candidates = list(sys.path) + ["/kaggle_simulations/agent", os.getcwd()]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "deck.csv")):
            return d
    return os.getcwd()


# deck.csv lives next to this file in the submission bundle.
_DECK_PATH = os.path.join(_agent_dir(), "deck.csv")


def load_deck(path: str = _DECK_PATH) -> list[int]:
    """Read the 60-card deck (one card ID per line)."""
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


# Loaded once at import so we don't hit the filesystem every turn.
DECK: list[int] = load_deck()


def choose(select: dict[str, Any]) -> list[int]:
    """Pick option indices for a single decision point.

    Baseline policy: take the first ``maxCount`` legal options (deterministic
    and always legal). This is intentionally simple — improving this function
    is the whole game. Hooks to build on:
      * ``select["type"]`` / ``select["context"]`` tell you WHAT is being asked
        (main action, choose a card to discard, pick an attack, etc.).
      * each ``option["type"]`` is an OptionType (ATTACK, EVOLVE, ATTACH,
        RETREAT, END, ...). Prefer ATTACK/EVOLVE over passing, etc.
      * ``obs["current"]`` gives full board state for lookahead/eval.
    """
    options = select.get("option") or []
    n = len(options)
    if n == 0:
        return []
    max_count = select.get("maxCount", 1)
    min_count = select.get("minCount", 0)
    # Clamp to what's actually available and required.
    count = max(min_count, min(max_count, n))
    return list(range(count))


def agent(obs: dict[str, Any]) -> list[int]:
    """Engine entry point. Returns the deck, or chosen option indices."""
    select = obs.get("select")
    if select is None:
        return DECK
    return choose(select)
