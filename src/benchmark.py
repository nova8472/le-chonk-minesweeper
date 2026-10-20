"""Seeded headless benchmark.

Examples:
  python -m src.benchmark --agent model --difficulty beginner --games 20
  python -m src.benchmark --agent solver --games 100
  python -m src.benchmark --agent random --games 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .agent import MinesweeperAgent
from .env import MinesweeperEnv
from .metrics import format_summary, summarize
from .mock_agent import RandomAgent, SolverAgent
from .runner import play_game


def make_agent(kind: str, args) -> object:
    if kind == "model":
        return MinesweeperAgent(model=args.model, temperature=args.temperature,
                                reasoning_effort=args.reasoning)
    if kind == "solver":
        return SolverAgent(seed=args.seed)
    if kind == "random":
        return RandomAgent(seed=args.seed)
    raise ValueError(f"unknown agent {kind!r}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", default="beginner",
                        choices=["beginner", "intermediate"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000,
                        help="first seed; game i uses seed + i")
    parser.add_argument("--agent", default="model",
                        choices=["model", "solver", "random"])
    parser.add_argument("--model", default="mistral-large-4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning", default="none",
                        choices=["none", "high"])
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    env = MinesweeperEnv(args.difficulty)
    records = []
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    for i in range(args.games):
        game_seed = args.seed + i
        agent = make_agent(args.agent, args)
        t0 = time.time()

        def report(env, move, _idx=i, _seed=game_seed):
            act = move.get("action") or {}
            args_str = act.get("arguments") or {}
            print(f"[game {_idx + 1}/{args.games} seed={_seed}] move {move['move']}: "
                  f"{act.get('name', '?')} {args_str} -> {move['outcome']}",
                  flush=True)

        print(f"[game {i + 1}/{args.games}] seed={game_seed} starting", flush=True)
        try:
            record = play_game(env, agent, seed=game_seed,
                               max_moves=args.max_moves, on_move=report)
        except Exception as err:
            record = {
                "seed": game_seed, "difficulty": args.difficulty,
                "won": False, "outcome": "error", "error": str(err),
                "moves": None, "wasted_moves": None, "move_classifications": {},
                "prompt_tokens": None, "completion_tokens": None,
                "game_seconds": None, "move_log": [],
            }
        record["wall_s"] = round(time.time() - t0, 1)
        records.append(record)
        status = "WIN" if record["won"] else record["outcome"].upper()
        print(f"[{i + 1}/{args.games}] seed={game_seed} -> {status} "
              f"moves={record['moves']} ({record['wall_s']}s)")

    summary = summarize(records)
    summary["agent"] = args.agent
    summary["model"] = args.model if args.agent == "model" else args.agent
    summary["difficulty"] = args.difficulty
    summary["first_seed"] = args.seed
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    tag = f"{args.agent}-{args.difficulty}-s{args.seed}-n{args.games}"
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "games": records}, f, indent=2)

    print()
    print(format_summary(summary, f"{summary['model']} on {args.difficulty}"))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
