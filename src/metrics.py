"""Deduction analysis and benchmark aggregation.

The deduction engine uses only the visible board (what a perfect player
could derive), never the ground truth. It applies the two classic
single-cell rules to a fixpoint:

  rule 1: number N with exactly N unrevealed neighbors  -> those are mines
  rule 2: number N with all N mines already deduced      -> rest are safe

Flags are deliberately ignored by this engine (flags are the player's
beliefs, not facts), which makes the metrics conservative and objective.
"""

from __future__ import annotations

from collections import Counter


def deduce(game) -> tuple[set, set]:
    """Fixpoint deduction over the visible board.

    Returns (forced_mines, safe_cells) as sets of 0-based (row, col).
    """
    K: set = set()  # forced mines
    S: set = set()  # forced safe cells
    numbered = [
        (r, c, game.counts[r][c])
        for r in range(game.rows)
        for c in range(game.cols)
        if game.revealed[r][c] and game.counts[r][c] > 0
    ]
    changed = True
    while changed:
        changed = False
        for r, c, n in numbered:
            U = [
                (nr, nc)
                for nr, nc in game.neighbors(r, c)
                if not game.revealed[nr][nc]
            ]
            unknown = [u for u in U if u not in K]
            if not unknown:
                continue
            if len(U) == n:  # every unrevealed neighbor is a mine
                for u in unknown:
                    if u not in K:
                        K.add(u)
                        changed = True
            known_mines = len(U) - len(unknown)
            if known_mines == n:  # mines fully accounted for -> rest safe
                for u in unknown:
                    if u not in S:
                        S.add(u)
                        changed = True
    return K, S


def summarize(records: list) -> dict:
    """Aggregate per-game records into a summary dict."""
    games = len(records)
    wins = [r for r in records if r["won"]]
    losses = [r for r in records if r["outcome"] == "loss"]
    stalled = [r for r in records if r["outcome"] == "stalled"]
    errored = [r for r in records if r["outcome"] == "error"]

    class_counts = Counter()
    for r in records:
        class_counts.update(r.get("move_classifications") or {})

    loss_reasons = Counter()
    for r in losses:
        last = (r.get("move_log") or [{}])[-1]
        loss_reasons[last.get("classification") or "unknown"] += 1

    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 2) if values else None

    return {
        "games": games,
        "wins": len(wins),
        "win_rate": round(len(wins) / games, 3) if games else None,
        "losses": len(losses),
        "loss_reasons": dict(loss_reasons),
        "stalled": len(stalled),
        "errored": len(errored),
        "avg_moves": avg([r.get("moves") for r in records]),
        "avg_wasted_moves": avg([r.get("wasted_moves") for r in records]),
        "avg_game_seconds": avg([r.get("game_seconds") for r in records]),
        "avg_prompt_tokens": avg([r.get("prompt_tokens") for r in records]),
        "avg_completion_tokens": avg([r.get("completion_tokens") for r in records]),
        "move_classifications": dict(class_counts),
    }


def format_summary(summary: dict, label: str) -> str:
    """Human-readable benchmark report."""
    lines = [
        f"=== {label} ===",
        f"Games:      {summary['games']}",
        f"Wins:       {summary['wins']}/{summary['games']}"
        + (f" ({summary['win_rate'] * 100:.1f}%)" if summary["win_rate"] is not None else ""),
        f"Losses:     {summary['losses']}  by reason: {summary['loss_reasons']}",
        f"Stalled:    {summary['stalled']}   Errored: {summary['errored']}",
        f"Avg moves:  {summary['avg_moves']}  (wasted/no-op: {summary['avg_wasted_moves']})",
        f"Avg secs:   {summary['avg_game_seconds']}",
        f"Avg tokens: prompt {summary['avg_prompt_tokens']} / completion {summary['avg_completion_tokens']}",
        f"Reveal classes: {summary['move_classifications']}",
    ]
    return "\n".join(lines)
