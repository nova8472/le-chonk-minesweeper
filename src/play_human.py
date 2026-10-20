"""Play Minesweeper yourself, on the exact same rendering the model sees.

Run: .venv/Scripts/python -m src.play_human [--seed 7] [--difficulty beginner]

Left-click a cell to reveal it, right-click to place or remove a flag.
"""

from __future__ import annotations

import argparse
import random
import time
import tkinter as tk

from PIL import Image, ImageTk

from .cat import cat_image
from .env import CELL, HEADER_H, LABEL_H, LABEL_W, MinesweeperEnv


def pixel_to_cell(x: int, y: int) -> tuple[int, int] | None:
    """Translate a click position on the rendered board to a 0-based cell.

    Returns None for clicks in the header or the coordinate-label margins.
    Bounds beyond the last column/row are the caller's problem.
    """
    if x < LABEL_W or y < HEADER_H + LABEL_H:
        return None
    col = (x - LABEL_W) // CELL
    row = (y - HEADER_H - LABEL_H) // CELL
    if col < 0 or row < 0:
        return None
    return row, col


class HumanGame:
    def __init__(self, root: tk.Tk, env: MinesweeperEnv, seed: int):
        self.root = root
        self.env = env
        self.seed = seed
        self.start = time.time()
        self.photo = None
        self.message = "Left-click to reveal, right-click to flag. Good luck!"
        self.timer_running = True

        self.board = tk.Label(root, bd=0)
        self.board.pack(padx=12, pady=12)
        self.board.bind("<Button-1>", self.on_reveal)
        self.board.bind("<Button-2>", self.on_flag)
        self.board.bind("<Button-3>", self.on_flag)

        self.status = tk.Label(root, anchor="w", font=("Consolas", 10, "bold"))
        self.status.pack(fill="x", padx=12)
        self.info = tk.Label(root, anchor="w", font=("Consolas", 10),
                             wraplength=560, justify="left")
        self.info.pack(fill="x", padx=12, pady=(2, 6))

        controls = tk.Frame(root)
        controls.pack(pady=(0, 12))
        tk.Button(controls, text="New game (random seed)",
                  command=self.new_game).pack(side="left", padx=4)
        tk.Button(controls, text="Restart same board",
                  command=self.restart).pack(side="left", padx=4)

        self.redraw()
        self.tick()

    # ------------------------------------------------------------------ input

    def on_reveal(self, event):
        cell = pixel_to_cell(event.x, event.y)
        if cell:
            r, c = cell
            self.step("reveal_cell", r, c)

    def on_flag(self, event):
        cell = pixel_to_cell(event.x, event.y)
        if cell:
            r, c = cell
            self.step("toggle_flag", r, c)

    def step(self, name: str, r: int, c: int):
        g = self.env.game
        if not (0 <= r < g.rows and 0 <= c < g.cols):
            return
        result, _ = self.env.step({
            "name": name,
            "arguments": {"row": r + 1, "col": chr(ord("A") + c)},
        })
        self.message = result.detail
        self.redraw()

    def new_game(self):
        self.seed = random.randint(0, 999_999)
        self.restart()

    def restart(self):
        self.env.reset(self.seed)
        self.start = time.time()
        self.timer_running = True
        self.message = "New board. Left-click to reveal, right-click to flag."
        self.redraw()

    # ------------------------------------------------------------------ output

    def redraw(self):
        self.photo = ImageTk.PhotoImage(self.env.render())
        self.board.configure(image=self.photo)
        self.status.configure(text=self.env.status_line())
        self.info.configure(text=self.message)

    def tick(self):
        if self.timer_running:
            g = self.env.game
            elapsed = int(time.time() - self.start)
            if g.done:
                verb = "WON" if g.won else "LOST"
                self.info.configure(
                    text=f"{self.message}  [game over: {verb} in {elapsed}s]")
                self.timer_running = False
            else:
                self.info.configure(text=f"{self.message}  [{elapsed}s]")
            self.root.after(1000, self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", default="beginner",
                        choices=["beginner", "intermediate"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = MinesweeperEnv(args.difficulty, seed=args.seed)
    root = tk.Tk()
    root.title("Le Chonk Minesweeper - human mode")
    icon = ImageTk.PhotoImage(cat_image(4))
    root.iconphoto(True, icon)
    HumanGame(root, env, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
