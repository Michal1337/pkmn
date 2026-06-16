"""Single-agent Gymnasium-style wrapper around the cabt engine for RL.

Key facts that shape this design (all verified against the engine):

* The native engine keeps ONE global battle pointer, so only one battle may be
  live per process. => use subprocess vector envs (SubprocVecEnv-style), never
  threads. ``close()`` frees the native battle.
* We drive the engine directly with ``battle_start/battle_select`` (decks given
  up front), so ``select`` is never None and there is no deck-selection step.
* The action space is dynamic. We expose a single masked ``Discrete(N_ACTIONS)``
  pick per step and BUFFER picks internally for multi-select decisions
  (``maxCount`` > 1), submitting to the engine only when the set is complete.
  Index ``SUBMIT_ACTION`` (== MAX_OPTIONS) ends an optional selection early.
* The opponent plays inside ``step``; the agent only ever sees its own turns.

Reward: +1 win / -1 loss / 0 draw at terminal (plus optional shaping hook).
"""

from __future__ import annotations

import logging
import os
import random

logging.disable(logging.CRITICAL)  # silence kaggle_environments import chatter

from .encoding import Encoder, SUBMIT_ACTION, build_mask
from .card_features import get_card_table

_AGENT_DECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "deck.csv")


def load_deck(path: str = _AGENT_DECK) -> list[int]:
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


def random_opponent(raw_obs: dict, rng: random.Random) -> list[int]:
    """Baseline opponent: a uniformly random legal selection (engine-native)."""
    sel = raw_obs["select"]
    n, k = len(sel["option"]), sel["maxCount"]
    return rng.sample(range(n), min(k, n)) if n else []


class CabtEnv:
    """Gymnasium-style env. obs is a dict of numpy arrays (see Encoder.shapes)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_deck: list[int] | None = None,
        opponent_deck: list[int] | None = None,
        agent_decks: list[list[int]] | None = None,    # POOL: sampled each episode
        opponent_decks: list[list[int]] | None = None,
        opponent_fn=None,                # (raw_obs, rng) -> list[int]; default random
        encoder: Encoder | None = None,
        randomize_side: bool = True,     # alternate which player the agent is
        reward_shaping=None,             # (prev_state, new_state, agent_idx) -> float
        max_steps: int = 4000,
        seed: int | None = None,
    ):
        # A pool of decks (each side sampled per episode). Single-deck args are a
        # convenience that wraps into a 1-element pool.
        self.agent_decks = agent_decks or ([agent_deck] if agent_deck else [load_deck()])
        self.opponent_decks = opponent_decks or ([opponent_deck] if opponent_deck else list(self.agent_decks))
        self.opponent_fn = opponent_fn or random_opponent
        self.encoder = encoder or Encoder(get_card_table())
        self.randomize_side = randomize_side
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self._obs = None            # current raw engine obs
        self._picked: list[int] = []  # buffered picks for the current decision
        self._agent_idx = 0
        self._steps = 0
        self._done = True
        self._result_override = None  # forced terminal reward (e.g. opponent forfeit)

    # -- engine handles (imported lazily so logging is disabled first) ------
    @staticmethod
    def _engine():
        from kaggle_environments.envs.cabt.cg import game
        return game

    # -- gym API ------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        game = self._engine()
        self._safe_finish()   # free any prior battle on this (global) engine

        self._agent_idx = self.rng.randint(0, 1) if self.randomize_side else 0
        agent_deck = self.rng.choice(self.agent_decks)      # sample decks this episode
        opp_deck = self.rng.choice(self.opponent_decks)
        d0 = agent_deck if self._agent_idx == 0 else opp_deck
        d1 = opp_deck if self._agent_idx == 0 else agent_deck
        obs, start = game.battle_start(d0, d1)
        if obs is None:
            raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

        self._obs = obs
        self._picked = []
        self._steps = 0
        self._done = False
        self._result_override = None
        self._advance_to_agent()
        return self._encode(), {"agent_index": self._agent_idx}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() after episode end; call reset().")
        info: dict = {}
        sel = self._obs["select"]
        mask = build_mask(sel, set(self._picked))
        action = int(action)
        if action >= len(mask) or mask[action] == 0:  # guard illegal picks
            legal = [i for i, m in enumerate(mask) if m]
            action = legal[0]
            info["illegal_action"] = True

        submit = action == SUBMIT_ACTION
        if not submit:
            self._picked.append(action)

        # auto-submit once we've reached maxCount
        if submit or len(self._picked) >= sel["maxCount"]:
            reward, terminated = self._apply_selection(sorted(set(self._picked)))
            self._picked = []
            if terminated:
                self._done = True
                return self._encode(), reward, True, False, info
            self._advance_to_agent()
            if self._done:  # opponent move ended the game
                return self._encode(), self._terminal_reward(), True, False, info
            self._steps += 1
            truncated = self._steps >= self.max_steps
            self._done = truncated
            return self._encode(), reward, False, truncated, info

        # still buffering this multi-select: same decision, updated mask
        return self._encode(), 0.0, False, False, info

    def action_masks(self):
        """Current legal-action mask (CleanRL / MaskablePPO convention)."""
        return build_mask(self._obs["select"], set(self._picked))

    def close(self):
        self._safe_finish()

    def _safe_finish(self):
        """Finish the current native battle and NULL the global pointer.

        The engine's battle_finish frees the battle but leaves Battle.battle_ptr
        dangling; calling it again (e.g. a new CabtEnv after close()) double-frees
        and crashes the process. Guarding on the pointer makes finish idempotent.
        """
        from kaggle_environments.envs.cabt.cg.sim import Battle
        if Battle.battle_ptr:
            try:
                self._engine().battle_finish()
            except Exception:
                pass
            Battle.battle_ptr = None

    # -- internals ----------------------------------------------------------
    def _encode(self):
        return self.encoder.encode(self._obs, set(self._picked))

    def _state(self):
        return self._obs["current"]

    def _apply_selection(self, indices: list[int]):
        """Submit the agent's selection; return (reward, terminated)."""
        game = self._engine()
        prev = self._state()
        try:
            self._obs = game.battle_select(indices)
        except Exception:
            # illegal selection -> agent forfeits
            return -1.0, True
        s = self._state()
        if s["result"] >= 0:
            return self._terminal_reward(), True
        shaped = 0.0
        if self.reward_shaping:
            shaped = self.reward_shaping(prev, s, self._agent_idx)
        return shaped, False

    def _advance_to_agent(self):
        """Play opponent decisions until it's the agent's turn or the game ends."""
        game = self._engine()
        while True:
            s = self._state()
            if s["result"] >= 0:
                self._done = True
                return
            if s["yourIndex"] == self._agent_idx:
                return  # agent's decision
            picks = self.opponent_fn(self._obs, self.rng)
            try:
                self._obs = game.battle_select(picks)
            except Exception:
                # opponent made an illegal move -> agent wins (result stays -1,
                # so force the reward rather than reading the unfinished state)
                self._result_override = 1.0
                self._done = True
                return

    def _terminal_reward(self) -> float:
        if self._result_override is not None:
            return self._result_override
        r = self._state()["result"]
        if r == 2 or r < 0:
            return 0.0
        return 1.0 if r == self._agent_idx else -1.0


def prize_diff_shaping(scale: float = 0.1):
    """Optional dense reward based on prize cards taken.

    In PTCG you take a card from YOUR OWN prize pile when you KO an opponent's
    Pokemon, and you win when your pile is empty. So MY pile shrinking is good,
    the OPPONENT's pile shrinking is bad.
    """
    def shape(prev, new, agent_idx):
        opp = 1 - agent_idx
        def prizes(state, i):
            return len(state["players"][i].get("prize") or [])
        d_me = prizes(prev, agent_idx) - prizes(new, agent_idx)   # prizes I took
        d_opp = prizes(prev, opp) - prizes(new, opp)              # prizes opp took
        return scale * (d_me - d_opp)
    return shape
