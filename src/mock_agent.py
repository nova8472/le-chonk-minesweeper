"""Reference agents that need no API. Used for pipeline testing and as
non-LLM baselines in the benchmark."""

from __future__ import annotations

import random

from .metrics import deduce


class RandomAgent:
    """Uniformly random legal reveals. The floor baseline."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env, hint=None):
        g = env.game
        cells = [
            (r, c)
            for r in range(g.rows)
            for c in range(g.cols)
            if not g.revealed[r][c] and not g.flagged[r][c]
        ]
        r, c = self.rng.choice(cells)
        return {
            "action": {"name": "reveal_cell",
                       "arguments": {"row": r + 1, "col": chr(ord("A") + c)}},
            "reasoning": "random move",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def remember(self, *args): ...


class SolverAgent:
    """Classic single-cell deduction solver using visible info only.

    Plays safe cells, flags newly deduced mines, guesses randomly when stuck.
    This is the non-LLM baseline that mistral-large-4 is compared against.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env, hint=None):
        g = env.game
        forced_mines, safe_cells = deduce(g)

        revealable_safe = sorted(s for s in safe_cells if not g.flagged[s[0]][s[1]])
        if revealable_safe:
            r, c = self.rng.choice(revealable_safe)
            return self._reveal(r, c)

        flaggable = sorted(m for m in forced_mines if not g.flagged[m[0]][m[1]])
        if flaggable:
            r, c = self.rng.choice(flaggable)
            return {
                "action": {"name": "toggle_flag",
                           "arguments": {"row": r + 1, "col": chr(ord("A") + c)}},
                "reasoning": "flag deduced mine",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        cells = [
            (r, c)
            for r in range(g.rows)
            for c in range(g.cols)
            if not g.revealed[r][c] and not g.flagged[r][c]
        ]
        r, c = self.rng.choice(cells)
        return self._reveal(r, c)

    @staticmethod
    def _reveal(r: int, c: int) -> dict:
        return {
            "action": {"name": "reveal_cell",
                       "arguments": {"row": r + 1, "col": chr(ord("A") + c)}},
            "reasoning": "reveal",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def remember(self, *args): ...
