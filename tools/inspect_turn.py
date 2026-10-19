"""Inspect the raw model response on move 2 to debug the rambling.

Run: .venv/Scripts/python tools/inspect_turn.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.agent import MinesweeperAgent
from src.env import MinesweeperEnv

env = MinesweeperEnv("beginner")
env.reset(1000)
agent = MinesweeperAgent()

r1 = agent.act(env)
print("move1 action:", r1["action"], "| reasoning:", (r1["reasoning"] or "")[:150])
result, done = env.step(r1["action"])
agent.remember(r1, result.detail)

messages = agent._build_messages(env, None)
print("--- move 2 request structure ---")
for m in messages:
    role = m["role"]
    if role == "assistant":
        print("assistant: content=", (m.get("content") or "")[:80],
              "| tool_calls=", m.get("tool_calls"))
    elif role == "tool":
        print("tool:", m.get("tool_call_id"), "->", (m.get("content") or "")[:60])
    else:
        kind = type(m.get("content")).__name__
        print(f"{role}: ({kind})", str(m.get("content"))[:80].replace("\n", " "))

resp = agent._complete(messages)
choice = resp.choices[0]
print("--- move 2 raw response ---")
print("finish_reason:", getattr(choice, "finish_reason", None))
print("content:", repr(choice.message.content)[:3000])
print("tool_calls:", choice.message.tool_calls)
