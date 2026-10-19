"""Shared game loop used by the benchmark and the windowed demo."""

from __future__ import annotations

import json
import time
from collections import Counter

from .metrics import deduce


def play_game(env, agent, seed=None, max_moves=150, on_move=None, classify=True):
    """Run one full game. Returns a record dict.

    Per move we classify reveals against what a perfect player could have
    deduced from the visible board (conservative, flag-free deduction):
      - "opening": the guaranteed-safe first click
      - "deduced": a reveal that was logically forced
      - "risky":   a reveal that was NOT forced while safe moves existed
      - "guess":   a reveal when no safe move existed at all
    """
    env.reset(seed)
    game = env.game
    move_log = []
    noop_streak = 0
    repeat_streak = 0
    last_key = None
    start = time.time()

    while not game.done and game.moves < max_moves:
        hint = None
        if noop_streak >= 3:
            hint = ("Your last 3+ actions had no effect. Re-read the feedback above: "
                    "check the coordinates, whether the cell is already revealed or "
                    "flagged, and do not repeat rejected moves.")
        elif repeat_streak >= 3:
            hint = "You keep repeating the same action. Change your approach."

        first = not game.first_click_done
        safe_cells = set()
        if classify and not first:
            _, safe_cells = deduce(game)

        t0 = time.time()
        resp = agent.act(env, hint=hint)
        latency = time.time() - t0
        action = resp.get("action")

        classification = None
        if classify and action and action.get("name") == "reveal_cell":
            coord = env.parse_coord(action.get("arguments") or {})
            if coord:
                r, c = coord
                if (0 <= r < game.rows and 0 <= c < game.cols
                        and not game.revealed[r][c] and not game.flagged[r][c]):
                    if first:
                        classification = "opening"
                    elif (r, c) in safe_cells:
                        classification = "deduced"
                    elif safe_cells:
                        classification = "risky"
                    else:
                        classification = "guess"

        result, done = env.step(action or {"name": "", "arguments": {}})
        feedback = result.detail

        if result.outcome == "no_op":
            noop_streak += 1
        else:
            noop_streak = 0
        key = None
        if action:
            key = (action.get("name"),
                   json.dumps(action.get("arguments") or {}, sort_keys=True))
        repeat_streak = repeat_streak + 1 if (key and key == last_key) else 0
        last_key = key

        agent.remember(resp.get("reasoning", ""), action, feedback)

        move_log.append({
            "move": game.moves,
            "action": action,
            "cell": result.cell,
            "reasoning": resp.get("reasoning", ""),
            "feedback": feedback,
            "outcome": result.outcome,
            "classification": classification,
            "latency_s": round(latency, 2),
            "prompt_tokens": resp.get("prompt_tokens", 0),
            "completion_tokens": resp.get("completion_tokens", 0),
        })
        if on_move:
            on_move(env, move_log[-1])

    outcome = "win" if game.won else ("loss" if game.done else "stalled")
    return {
        "seed": seed,
        "difficulty": game.difficulty,
        "won": game.won,
        "outcome": outcome,
        "moves": game.moves,
        "wasted_moves": sum(1 for m in move_log if m["outcome"] == "no_op"),
        "move_classifications": dict(Counter(
            m["classification"] for m in move_log if m["classification"])),
        "prompt_tokens": sum(m["prompt_tokens"] for m in move_log),
        "completion_tokens": sum(m["completion_tokens"] for m in move_log),
        "game_seconds": round(time.time() - start, 1),
        "move_log": move_log,
    }
