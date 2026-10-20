"""GameEnv interface and the Minesweeper environment with vision-friendly rendering.

The model sees only rendered pixels (plus coordinate labels drawn into the
image). Ground truth is never rendered or sent to the model; it is used
exclusively for logging and metrics.
"""

from __future__ import annotations

import base64
import io
import re
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw, ImageFont

from .cat import cat_image
from .game import Minesweeper, MoveResult

CELL = 48            # px per board cell
LABEL_W = 34         # left margin for row numbers
LABEL_H = 30         # top margin for column letters
HEADER_H = 44        # branding band above the board
BORDER = 8
TITLE = "LE CHONK PLAYS MINESWEEPER"

PALETTE = {
    "background": (248, 249, 250),
    "hidden": (106, 124, 141),
    "hidden_top": (146, 162, 178),    # bevel highlight
    "hidden_bottom": (82, 98, 114),   # bevel shadow
    "revealed": (233, 236, 239),
    "grid": (205, 210, 214),
    "label": (55, 60, 65),
    "flag_cloth": (214, 45, 32),
    "flag_pole": (120, 90, 60),
    "hit_background": (196, 60, 50),
    "mine": (15, 15, 15),
}

NUMBER_COLORS = {
    1: (36, 84, 214),
    2: (27, 127, 59),
    3: (214, 48, 36),
    4: (24, 24, 142),
    5: (140, 43, 30),
    6: (22, 128, 140),
    7: (26, 26, 26),
    8: (128, 128, 128),
}

_FONT_CACHE: dict = {}


def _load_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


class GameEnv(ABC):
    """Minimal interface shared by every game the harness supports."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None: ...

    @abstractmethod
    def step(self, action: dict) -> tuple[MoveResult, bool]: ...

    @abstractmethod
    def render(self): ...

    @abstractmethod
    def status_line(self) -> str: ...

    @abstractmethod
    def ground_truth(self) -> dict: ...


class MinesweeperEnv(GameEnv):
    """Minesweeper + Pillow rendering. Headless by default (no window)."""

    def __init__(self, difficulty: str = "beginner", seed: int | None = None):
        if difficulty == "expert":
            raise ValueError("expert boards need >26 column labels; not supported yet")
        self.difficulty = difficulty
        self.seed = seed
        self.game = Minesweeper(difficulty, seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.game.reset(self.seed)

    # ---------------------------------------------------------------- actions

    def step(self, action: dict) -> tuple[MoveResult, bool]:
        name = action.get("name", "")
        args = action.get("arguments") or {}
        if name not in ("reveal_cell", "toggle_flag"):
            result = MoveResult(
                "no_op", "no_op", None,
                f"Unknown action {name!r}. Use reveal_cell or toggle_flag.",
            )
        else:
            coord = self.parse_coord(args)
            if coord is None:
                result = MoveResult(
                    "no_op", "no_op", None,
                    "Could not parse the coordinate. Pass row as a 1-based number "
                    "and col as a letter, e.g. row=5, col='E'.",
                )
            else:
                r, c = coord
                if name == "reveal_cell":
                    result = self.game.reveal(r, c)
                else:
                    result = self.game.toggle_flag(r, c)
        return result, self.game.done

    @staticmethod
    def parse_coord(args: dict) -> tuple[int, int] | None:
        """Tolerant coordinate parsing -> 0-based (row, col), or None.

        Accepts swapped fields ("E" in row, 5 in col) and combined strings
        like "E5" in either field.
        """
        row = args.get("row")
        col = args.get("col")

        col_s = str(col).strip().upper() if col is not None else ""
        row_s = str(row).strip().upper() if row is not None else ""

        col_l = None
        row_n = None

        m = re.match(r"^([A-Z]+)(\d+)$", col_s)
        if m:
            col_l, row_n = m.group(1)[0], int(m.group(2))
        if col_l is None:
            m = re.match(r"^([A-Z]+)(\d+)$", row_s)
            if m:
                col_l, row_n = m.group(1)[0], int(m.group(2))
        if col_l is None:
            if len(col_s) == 1 and "A" <= col_s <= "Z":
                col_l = col_s
            elif len(row_s) == 1 and "A" <= row_s <= "Z":
                col_l = row_s  # model swapped the fields
        if row_n is None:
            def _int(v):
                try:
                    return int(str(v).strip())
                except (TypeError, ValueError):
                    return None
            row_n = _int(row)
            if row_n is None:
                row_n = _int(col)  # model swapped the fields
            if row_n is None:
                return None
        if col_l is None:
            return None
        return row_n - 1, ord(col_l) - ord("A")

    # ---------------------------------------------------------------- rendering

    def render(self) -> Image.Image:
        g = self.game
        w = LABEL_W + g.cols * CELL + BORDER
        h = HEADER_H + LABEL_H + g.rows * CELL + BORDER
        img = Image.new("RGB", (w, h), PALETTE["background"])
        d = ImageDraw.Draw(img)
        number_font = _load_font(30)
        label_font = _load_font(16)
        title_font = _load_font(18)

        # Branding header: pixel cat + title, model tag right-aligned.
        # The tag layout adapts (smaller font, or skipped) so the two never
        # overlap regardless of which system font Pillow picks up.
        cat = cat_image(3)
        img.paste(cat, (8, (HEADER_H - cat.height) // 2), cat)
        title_font = _load_font(18)
        title_x = 8 + cat.width + 10
        d.text((title_x, HEADER_H // 2), TITLE,
               font=title_font, fill=PALETTE["label"], anchor="lm")
        try:
            title_end = title_x + title_font.getlength(TITLE)
            tag = "mistral-large-4"
            for size in (16, 14, 12):
                tag_font = _load_font(size)
                tag_x = w - BORDER - 4 - tag_font.getlength(tag)
                if tag_x >= title_end + 16:
                    d.text((tag_x, HEADER_H // 2), tag,
                           font=tag_font, fill=PALETTE["label"], anchor="lm")
                    break
        except (AttributeError, OSError):
            pass  # exotic font fallbacks: skip the tag rather than overlap
        d.line([0, HEADER_H - 1, w, HEADER_H - 1], fill=PALETTE["grid"], width=1)

        for c in range(g.cols):
            x = LABEL_W + c * CELL + CELL // 2
            d.text((x, HEADER_H + LABEL_H // 2), chr(ord("A") + c),
                   font=label_font, fill=PALETTE["label"], anchor="mm")
        for r in range(g.rows):
            y = HEADER_H + LABEL_H + r * CELL + CELL // 2
            d.text((LABEL_W // 2, y), str(r + 1),
                   font=label_font, fill=PALETTE["label"], anchor="mm")

        for r in range(g.rows):
            for c in range(g.cols):
                x0 = LABEL_W + c * CELL
                y0 = HEADER_H + LABEL_H + r * CELL
                x1, y1 = x0 + CELL - 2, y0 + CELL - 2
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                if g.revealed[r][c]:
                    d.rectangle([x0, y0, x1, y1],
                                fill=PALETTE["revealed"], outline=PALETTE["grid"], width=1)
                    n = g.counts[r][c]
                    if n > 0:
                        d.text((cx, cy), str(n), font=number_font,
                               fill=NUMBER_COLORS[n], anchor="mm")
                    continue
                show_mine = g.done and not g.won and g.mines[r][c]
                if show_mine:
                    fill = PALETTE["hit_background"] if g.hit_cell == (r, c) else PALETTE["revealed"]
                    d.rectangle([x0, y0, x1, y1], fill=fill, outline=PALETTE["grid"], width=1)
                    self._draw_mine(d, cx, cy)
                    continue
                d.rectangle([x0, y0, x1, y1], fill=PALETTE["hidden"])
                d.line([x0, y0, x1, y0], fill=PALETTE["hidden_top"], width=2)
                d.line([x0, y0, x0, y1], fill=PALETTE["hidden_top"], width=2)
                d.line([x1, y0, x1, y1], fill=PALETTE["hidden_bottom"], width=2)
                d.line([x0, y1, x1, y1], fill=PALETTE["hidden_bottom"], width=2)
                if g.flagged[r][c]:
                    self._draw_flag(d, cx, cy)
        return img

    def render_base64(self) -> str:
        """Current board as a base64 PNG (for the Mistral vision API)."""
        buf = io.BytesIO()
        self.render().save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def status_line(self) -> str:
        g = self.game
        state = "WON" if g.won else ("LOST" if g.done else "in progress")
        return (
            f"Board: {g.difficulty} ({g.rows}x{g.cols}, {g.n_mines} mines) | "
            f"Move: {g.moves + 1} | Flags placed: {g.flags_placed}/{g.n_mines} | "
            f"Cells revealed: {g.revealed_count}/{g.safe_cells_total} | "
            f"Game state: {state}"
        )

    def ground_truth(self) -> dict:
        return self.game.ground_truth()

    @staticmethod
    def _draw_flag(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
        d.line([cx - 3, cy - 14, cx - 3, cy + 13], fill=PALETTE["flag_pole"], width=3)
        d.polygon(
            [(cx - 3, cy - 14), (cx + 13, cy - 8), (cx - 3, cy - 2)],
            fill=PALETTE["flag_cloth"],
        )
        d.line([cx - 10, cy + 13, cx + 8, cy + 13], fill=PALETTE["flag_pole"], width=3)

    @staticmethod
    def _draw_mine(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
        mine = PALETTE["mine"]
        d.line([cx - 14, cy, cx + 14, cy], fill=mine, width=3)
        d.line([cx, cy - 14, cx, cy + 14], fill=mine, width=3)
        d.line([cx - 10, cy - 10, cx + 10, cy + 10], fill=mine, width=3)
        d.line([cx - 10, cy + 10, cx + 10, cy - 10], fill=mine, width=3)
        d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=mine)
        d.ellipse([cx - 4, cy - 5, cx - 1, cy - 2], fill=(255, 255, 255))
