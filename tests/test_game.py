"""Unit tests for the pure game logic. Run: python tests/test_game.py"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.game import Minesweeper, cell_name


def test_first_click_safe():
    for seed in range(50):
        g = Minesweeper("beginner", seed)
        res = g.reveal(4, 4)
        assert res.outcome != "loss", f"seed {seed}: first click hit a mine"
        assert g.counts[4][4] == 0, f"seed {seed}: first click had a neighboring mine"
        assert sum(map(sum, g.revealed)) > 1, f"seed {seed}: first click did not open a region"


def test_determinism():
    a = Minesweeper("beginner", 7)
    a.reveal(0, 0)
    b = Minesweeper("beginner", 7)
    b.reveal(0, 0)
    assert a.mines == b.mines
    assert a.counts == b.counts


def test_flag_blocks_and_toggles():
    g = Minesweeper("beginner", 1)
    assert g.toggle_flag(0, 0).outcome == "ok" and g.flagged[0][0]
    assert g.reveal(0, 0).outcome == "no_op"
    assert g.toggle_flag(0, 0).outcome == "ok" and not g.flagged[0][0]
    assert g.reveal(0, 0).outcome == "ok"


def test_noop_on_revealed():
    g = Minesweeper("beginner", 2)
    g.reveal(4, 4)
    assert g.reveal(4, 4).outcome == "no_op"
    assert g.toggle_flag(4, 4).outcome == "no_op"


def test_loss():
    g = Minesweeper("beginner", 3)
    g.reveal(0, 0)
    mine = next((r, c) for r in range(g.rows) for c in range(g.cols) if g.mines[r][c])
    res = g.reveal(*mine)
    assert res.outcome == "loss"
    assert g.done and not g.won
    assert g.hit_cell == mine
    assert g.reveal(0, 1).outcome == "no_op"  # frozen after game over


def test_win_detection():
    g = Minesweeper("beginner", 3)
    g.reveal(0, 0)
    last = None
    for r in range(g.rows):
        for c in range(g.cols):
            if not g.revealed[r][c] and not g.mines[r][c]:
                last = g.reveal(r, c)
    assert g.done and g.won
    assert last.outcome == "win"


def test_counts_consistent():
    g = Minesweeper("beginner", 11)
    g.reveal(4, 4)
    for r in range(g.rows):
        for c in range(g.cols):
            assert g.counts[r][c] == sum(
                1 for nr, nc in g.neighbors(r, c) if g.mines[nr][nc])


def test_cell_name():
    assert cell_name(0, 0) == "A1"
    assert cell_name(4, 4) == "E5"
    assert cell_name(8, 8) == "I9"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("ALL GAME TESTS PASSED")
