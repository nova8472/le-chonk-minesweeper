"""Pure, seeded Minesweeper logic. No rendering, no I/O, no model code.

Cells are stored 0-based internally. Model-facing coordinates use column
letters (A..) plus 1-based row numbers, e.g. E5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

DIFFICULTIES = {
    "beginner": {"rows": 9, "cols": 9, "mines": 10},
    "intermediate": {"rows": 16, "cols": 16, "mines": 40},
    "expert": {"rows": 16, "cols": 30, "mines": 99},
}


@dataclass
class MoveResult:
    outcome: str                    # "ok" | "win" | "loss" | "no_op"
    kind: str                       # "reveal" | "flag" | "unflag" | "no_op"
    cell: tuple[int, int] | None    # 0-based (row, col)
    detail: str                     # feedback shown to the model


def cell_name(row: int, col: int) -> str:
    """Model-facing coordinate, e.g. (4, 4) -> 'E5'."""
    return f"{chr(ord('A') + col)}{row + 1}"


class Minesweeper:
    """Classic Minesweeper with deterministic seeded boards.

    The first reveal is always safe: mines are placed after the first click,
    avoiding the clicked cell and its neighbors (so the first click opens a
    region instead of a single number).
    """

    def __init__(self, difficulty: str = "beginner", seed: int | None = None):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty: {difficulty!r}")
        cfg = DIFFICULTIES[difficulty]
        self.difficulty = difficulty
        self.rows: int = cfg["rows"]
        self.cols: int = cfg["cols"]
        self.n_mines: int = cfg["mines"]
        self.rng = random.Random(seed)
        self.reset(seed)

    # ------------------------------------------------------------------ state

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self.revealed = [[False] * self.cols for _ in range(self.rows)]
        self.flagged = [[False] * self.cols for _ in range(self.rows)]
        self.mines = [[False] * self.cols for _ in range(self.rows)]
        self.counts = [[0] * self.cols for _ in range(self.rows)]
        self.first_click_done = False
        self.done = False
        self.won = False
        self.moves = 0
        self.hit_cell: tuple[int, int] | None = None

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    yield nr, nc

    def _place_mines(self, safe_r: int, safe_c: int) -> None:
        excluded = {(safe_r, safe_c), *self.neighbors(safe_r, safe_c)}
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in excluded
        ]
        if len(candidates) < self.n_mines:
            # Tiny boards: at least keep the clicked cell itself safe.
            excluded = {(safe_r, safe_c)}
            candidates = [
                (r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in excluded
            ]
        for r, c in self.rng.sample(candidates, self.n_mines):
            self.mines[r][c] = True
        for r in range(self.rows):
            for c in range(self.cols):
                self.counts[r][c] = sum(
                    1 for nr, nc in self.neighbors(r, c) if self.mines[nr][nc]
                )

    # ---------------------------------------------------------------- actions

    def reveal(self, r: int, c: int) -> MoveResult:
        name = cell_name(r, c)
        if self.done:
            return MoveResult("no_op", "no_op", (r, c), "The game is already over.")
        if not self._in_bounds(r, c):
            return MoveResult("no_op", "no_op", (r, c), f"{name} is outside the board.")
        if self.flagged[r][c]:
            return MoveResult("no_op", "no_op", (r, c),
                              f"{name} is FLAGGED (dark square with a red flag). "
                              f"Unflag it with toggle_flag first if you want to reveal it.")
        if self.revealed[r][c]:
            shown = "empty" if self.counts[r][c] == 0 else f"number {self.counts[r][c]}"
            return MoveResult(
                "no_op", "no_op", (r, c),
                f"{name} is ALREADY REVEALED (light gray square showing {shown}) -- "
                f"it cannot be revealed again. Pick a dark unrevealed cell.")
        if not self.first_click_done:
            self._place_mines(r, c)
            self.first_click_done = True
        self.moves += 1
        if self.mines[r][c]:
            self.done = True
            self.hit_cell = (r, c)
            return MoveResult("loss", "reveal", (r, c),
                              f"BOOM. {name} was a mine. You lose.")
        opened = self._flood(r, c)
        if all(
            self.revealed[r2][c2] or self.mines[r2][c2]
            for r2 in range(self.rows)
            for c2 in range(self.cols)
        ):
            self.done = True
            self.won = True
            return MoveResult(
                "win", "reveal", (r, c),
                f"Revealed {name} and its region ({opened} cells opened). "
                f"Every safe cell is now revealed -- you WIN!",
            )
        n = self.counts[r][c]
        detail = f"Revealed {name}: " + ("empty" if n == 0 else f"number {n}")
        if opened > 1:
            detail += f" (opened {opened} cells)"
        return MoveResult("ok", "reveal", (r, c), detail + ".")

    def toggle_flag(self, r: int, c: int) -> MoveResult:
        name = cell_name(r, c)
        if self.done:
            return MoveResult("no_op", "no_op", (r, c), "The game is already over.")
        if not self._in_bounds(r, c):
            return MoveResult("no_op", "no_op", (r, c), f"{name} is outside the board.")
        if self.revealed[r][c]:
            return MoveResult(
                "no_op", "no_op", (r, c),
                f"{name} is already revealed; flags only go on unrevealed cells.",
            )
        self.moves += 1
        self.flagged[r][c] = not self.flagged[r][c]
        if self.flagged[r][c]:
            return MoveResult("ok", "flag", (r, c), f"Flag placed on {name}.")
        return MoveResult("ok", "unflag", (r, c), f"Flag removed from {name}.")

    def _flood(self, r: int, c: int) -> int:
        """Reveal the cell and flood-open the connected zero-count region.

        Flagged cells are skipped (a wrong flag never triggers an accidental
        loss). Returns the number of cells opened.
        """
        stack = [(r, c)]
        opened = 0
        while stack:
            cr, cc = stack.pop()
            if self.revealed[cr][cc]:
                continue
            self.revealed[cr][cc] = True
            opened += 1
            if self.counts[cr][cc] == 0:
                stack.extend(
                    (nr, nc)
                    for nr, nc in self.neighbors(cr, cc)
                    if not self.revealed[nr][nc] and not self.flagged[nr][nc]
                )
        return opened

    # ---------------------------------------------------------------- queries

    @property
    def safe_cells_total(self) -> int:
        return self.rows * self.cols - self.n_mines

    @property
    def revealed_count(self) -> int:
        return sum(map(sum, self.revealed))

    @property
    def flags_placed(self) -> int:
        return sum(map(sum, self.flagged))

    def is_mine(self, r: int, c: int) -> bool:
        return self.mines[r][c]

    def visible_grid(self):
        """What the player sees: None (unrevealed), 'flag', or an int count."""
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if self.flagged[r][c]:
                    row.append("flag")
                elif self.revealed[r][c]:
                    row.append(self.counts[r][c])
                else:
                    row.append(None)
            grid.append(row)
        return grid

    def ground_truth(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "rows": self.rows,
            "cols": self.cols,
            "mines": self.n_mines,
            "mine_cells": [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if self.mines[r][c]
            ],
            "won": self.won,
            "done": self.done,
            "moves": self.moves,
        }
