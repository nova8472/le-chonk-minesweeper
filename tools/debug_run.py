"""Debug probe: play a few moves with per-move verbosity.

Run: .venv/Scripts/python tools/debug_run.py [seed] [max_moves]
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from src.agent import MinesweeperAgent
from src.env import MinesweeperEnv
from src.runner import play_game

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
max_moves = int(sys.argv[2]) if len(sys.argv) > 2 else 8
effort = sys.argv[3] if len(sys.argv) > 3 else "high"

env = MinesweeperEnv("beginner")
agent = MinesweeperAgent(reasoning_effort=effort)


def show(env, move):
    print(f"move {move['move']}: {move['action']} -> {move['feedback'][:120]} "
          f"[{move['latency_s']}s, {move['prompt_tokens']}+{move['completion_tokens']} tok]",
          flush=True)
    if move["reasoning"]:
        print("  thinking:", move["reasoning"][:200], flush=True)


record = play_game(env, agent, seed=seed, max_moves=max_moves, on_move=show)
print("outcome:", record["outcome"], "| moves:", record["moves"],
      "| wasted:", record["wasted_moves"],
      "| tokens:", record["prompt_tokens"], "+", record["completion_tokens"])
