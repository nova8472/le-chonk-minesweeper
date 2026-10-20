"""Rendering + pipeline tests. Run: python tests/test_env.py"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.env import MinesweeperEnv
from src.metrics import deduce
from src.mock_agent import RandomAgent, SolverAgent
from src.runner import play_game


def test_render_size():
    env = MinesweeperEnv("beginner")
    img = env.render()
    assert img.size == (474, 514)  # 34+9*48+8 by 44+30+9*48+8


def test_render_pixels():
    env = MinesweeperEnv("beginner")
    env.reset(7)
    img = env.render()
    assert img.getpixel((58, 104)) == (106, 124, 141)  # hidden cell fill
    assert img.getpixel((470, 40)) == (248, 249, 250)  # header background
    env.step({"name": "reveal_cell", "arguments": {"row": 5, "col": "E"}})
    img = env.render()
    assert img.getpixel((250, 289)) == (233, 236, 239)  # E5 now revealed
    # base64 export is a valid PNG
    import base64
    raw = base64.b64decode(env.render_base64())
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_cat_art():
    from src.cat import CAT_PIXELS, cat_image
    assert all(len(row) == 12 for row in CAT_PIXELS)
    img = cat_image(3)
    assert img.size == (36, 36) and img.mode == "RGBA"
    # the cat's body is orange, not transparent
    assert img.getpixel((18, 16)) == (244, 157, 55, 255)


def test_parse_coord():
    pc = MinesweeperEnv.parse_coord
    assert pc({"row": 5, "col": "E"}) == (4, 4)
    assert pc({"row": "E", "col": 5}) == (4, 4)
    assert pc({"row": 5, "col": "E5"}) == (4, 4)
    assert pc({"row": "5", "col": "e"}) == (4, 4)
    assert pc({"row": 1, "col": "I"}) == (0, 8)
    assert pc({"row": 5, "col": ""}) is None
    assert pc({"row": "", "col": ""}) is None


def test_env_step_feedback():
    env = MinesweeperEnv("beginner")
    env.reset(7)
    result, done = env.step({"name": "reveal_cell", "arguments": {"row": 5, "col": "E"}})
    assert result.outcome == "ok" and not done
    result, _ = env.step({"name": "reveal_cell", "arguments": {"row": 5, "col": "E"}})
    assert result.outcome == "no_op" and "already revealed" in result.detail.lower()
    result, _ = env.step({"name": "fly", "arguments": {}})
    assert result.outcome == "no_op" and "Unknown action" in result.detail


def test_deduction_engine():
    env = MinesweeperEnv("beginner")
    env.reset(7)
    env.step({"name": "reveal_cell", "arguments": {"row": 5, "col": "E"}})
    K, S = deduce(env.game)
    assert isinstance(K, set) and isinstance(S, set)
    for r, c in K:
        assert env.game.is_mine(r, c), "forced mine must actually be a mine"
    for r, c in S:
        assert not env.game.is_mine(r, c), "forced safe cell must not be a mine"


def test_solver_pipeline():
    env = MinesweeperEnv("beginner")
    wins = 0
    for seed in range(10):
        rec = play_game(env, SolverAgent(seed=seed), seed=seed)
        assert rec["outcome"] in ("win", "loss")
        assert rec["moves"] > 0
        wins += int(rec["won"])
    assert wins >= 5, f"solver should win most beginner games, got {wins}/10"


def test_random_pipeline():
    env = MinesweeperEnv("beginner")
    rec = play_game(env, RandomAgent(seed=0), seed=0, max_moves=100)
    assert rec["outcome"] in ("win", "loss", "stalled")
    assert rec["moves"] > 0


def test_pixel_to_cell():
    from src.play_human import pixel_to_cell
    assert pixel_to_cell(58, 104) == (0, 0)     # cell A1
    assert pixel_to_cell(250, 289) == (4, 4)    # cell E5
    assert pixel_to_cell(10, 10) is None        # header band
    assert pixel_to_cell(20, 120) is None       # row-number margin
    assert pixel_to_cell(400, 60) is None       # still in the header band


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("ALL ENV TESTS PASSED")
