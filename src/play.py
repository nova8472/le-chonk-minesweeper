"""Interactive windowed demo: watch the agent play with live reasoning.

Examples:
  python -m src.play --seed 7
  python -m src.play --agent solver --seed 42 --delay 0.05

Requires tkinter (bundled with python.org Windows builds) and Pillow.
Record the window with OBS Studio for the submission video.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
from dotenv import load_dotenv

from .agent import MinesweeperAgent
from .cat import cat_image
from .env import MinesweeperEnv
from .mock_agent import RandomAgent, SolverAgent
from .runner import play_game

MAX_LOG_LINES = 400


class Demo:
    def __init__(self, env, agent, seed, delay, max_moves):
        self.env = env
        self.agent = agent
        self.seed = seed
        self.delay = delay
        self.max_moves = max_moves
        self.q: queue.Queue = queue.Queue()
        self.finished = False

        self.root = tk.Tk()
        self.root.title("Le Chonk plays Minesweeper")
        icon = ImageTk.PhotoImage(cat_image(4))
        self.root.iconphoto(True, icon)
        self._icon_ref = icon  # keep alive
        self.board_label = tk.Label(self.root)
        self.board_label.pack(padx=10, pady=10)

        self.thinking_frame = tk.Frame(self.root)
        self.thinking_frame.pack(fill="x", padx=10, pady=(0, 4))
        self.cat_photo = ImageTk.PhotoImage(cat_image(2))
        tk.Label(self.thinking_frame, image=self.cat_photo).pack(side="left")
        self.thinking_label = tk.Label(self.thinking_frame, text="Ready.",
                                       anchor="w", font=("Consolas", 10))
        self.thinking_label.pack(side="left", padx=8)
        self.bar = ttk.Progressbar(self.thinking_frame, mode="indeterminate",
                                   length=240)
        self.bar.pack(side="left", padx=8)
        self.thinking = False
        self.dots = 0

        self.log = tk.Text(self.root, width=96, height=20, state="disabled",
                           font=("Consolas", 10), wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.status = tk.Label(self.root, text="Starting...", anchor="w",
                               font=("Consolas", 10, "bold"))
        self.status.pack(fill="x", padx=10, pady=(0, 10))
        self.photo = None  # must outlive the Label update

    # ---- worker thread ---------------------------------------------------

    def _worker(self):
        try:
            record = play_game(self.env, self.agent, seed=self.seed,
                               max_moves=self.max_moves, on_move=self._emit,
                               on_turn_start=self._think)
            self.q.put(("done", record))
        except Exception as err:
            self.q.put(("error", str(err)))

    def _think(self, move_no: int):
        self.q.put(("thinking", move_no))

    def _emit(self, env, move):
        if self.delay:
            time.sleep(self.delay)
        self.q.put(("move", (env.render(), move)))

    # ---- UI thread ---------------------------------------------------------

    def _start_thinking(self, move_no: int):
        self.thinking = True
        self.bar.start(14)
        self._animate_dots(move_no)

    def _animate_dots(self, move_no: int):
        if not self.thinking:
            return
        self.dots = (self.dots + 1) % 4
        self.thinking_label.configure(
            text=f"mistral-large-4 is thinking about move {move_no}"
                 + "." * self.dots)
        self.root.after(400, lambda: self._animate_dots(move_no))

    def _stop_thinking(self):
        self.thinking = False
        self.bar.stop()
        self.thinking_label.configure(text="Move played.")

    def _show_move(self, image, move):
        self._stop_thinking()
        self.photo = ImageTk.PhotoImage(image)
        self.board_label.configure(image=self.photo)
        header = f"[move {move['move']}] {move['outcome'].upper()}"
        if move["classification"]:
            header += f" ({move['classification']})"
        header += f"  {move['latency_s']}s  tokens {move['prompt_tokens']}+{move['completion_tokens']}"
        self.log.configure(state="normal")
        self.log.insert("end", header + "\n")
        thought = (move.get("thinking") or "").strip()
        if thought:
            excerpt = " ".join(thought.split())[:220]
            self.log.insert("end", "  thinking: " + excerpt + "\n")
        if move["reasoning"]:
            self.log.insert("end", "  thinking: " + move["reasoning"].strip() + "\n")
        self.log.insert("end", "  action:   " + str(move["action"]) + "\n")
        self.log.insert("end", "  result:   " + move["feedback"] + "\n\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES:
            self.log.delete("1.0", f"{lines - MAX_LOG_LINES}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _show_final(self, record):
        self.photo = ImageTk.PhotoImage(self.env.render())
        self.board_label.configure(image=self.photo)
        if "error" in record:
            self.status.configure(text=f"ERROR: {record['error']}")
            return
        verb = "WON" if record["won"] else record["outcome"].upper()
        self.status.configure(
            text=f"Game {verb} | moves: {record['moves']} | "
                 f"tokens: {record['prompt_tokens']}+{record['completion_tokens']} | "
                 f"time: {record['game_seconds']}s")

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "move":
                    self._show_move(*payload)
                elif kind == "thinking":
                    self._start_thinking(payload)
                elif kind == "done":
                    self._show_final(payload)
                    self.finished = True
                elif kind == "error":
                    self._show_final({"error": payload})
                    self.finished = True
        except queue.Empty:
            pass
        if not self.finished:
            self.root.after(150, self._poll)

    def run(self):
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(150, self._poll)
        self.root.mainloop()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", default="beginner",
                        choices=["beginner", "intermediate"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--agent", default="model",
                        choices=["model", "solver", "random"])
    parser.add_argument("--model", default="mistral-large-4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning", default="none",
                        choices=["none", "high"])
    parser.add_argument("--delay", type=float, default=0.6,
                        help="pause between moves, seconds")
    parser.add_argument("--max-moves", type=int, default=150)
    args = parser.parse_args()

    env = MinesweeperEnv(args.difficulty)
    if args.agent == "model":
        agent = MinesweeperAgent(model=args.model, temperature=args.temperature,
                                  reasoning_effort=args.reasoning)
    elif args.agent == "solver":
        agent = SolverAgent(seed=args.seed)
    else:
        agent = RandomAgent(seed=args.seed)

    Demo(env, agent, args.seed, args.delay, args.max_moves).run()


if __name__ == "__main__":
    main()
