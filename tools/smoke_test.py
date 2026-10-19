"""One-shot smoke test: key loading, vision input, and tool calling.

Run: .venv/Scripts/python tools/smoke_test.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.agent import MinesweeperAgent
from src.env import MinesweeperEnv

env = MinesweeperEnv("beginner")
env.reset(7)
agent = MinesweeperAgent()
resp = agent.act(env)
print("model:      ", agent.model)
print("reasoning:  ", (resp["reasoning"] or "(none)")[:300])
print("action:     ", resp["action"])
print("tokens:     ", resp["prompt_tokens"], "prompt /", resp["completion_tokens"], "completion")
