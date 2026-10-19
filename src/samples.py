"""Generate sample screenshots for the README / media folder.

Usage: python -m src.samples
"""

from __future__ import annotations

from pathlib import Path

from .env import MinesweeperEnv
from .mock_agent import RandomAgent, SolverAgent
from .runner import play_game


def main():
    media = Path("media")
    media.mkdir(exist_ok=True)

    env = MinesweeperEnv("beginner")

    env.reset(7)
    env.render().save(media / "sample_start.png")

    env.reset(7)
    agent = SolverAgent(seed=7)
    for _ in range(6):
        resp = agent.act(env)
        env.step(resp["action"])
    env.render().save(media / "sample_mid_game.png")

    record = None
    for s in range(3, 30):
        record = play_game(env, RandomAgent(seed=s), seed=s)
        if record["outcome"] == "loss":
            break
    env.render().save(media / "sample_end.png")

    print(f"saved samples to media/ (end board outcome: {record['outcome']})")


if __name__ == "__main__":
    main()
