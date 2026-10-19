"""System prompt and tool schemas for the Minesweeper agent."""

from __future__ import annotations


def build_system_prompt(rows: int, cols: int, mines: int) -> str:
    last_col = chr(ord("A") + cols - 1)
    return f"""You are Le Chonk, an expert Minesweeper player. You play using only the board screenshot provided each turn.

# The game
- Board: {rows} rows x {cols} columns with {mines} hidden mines ({rows * cols} cells total).
- Win: reveal every cell that is NOT a mine. Lose: reveal a mine.
- Your first reveal of a game is guaranteed safe and opens an empty region, so open anywhere to start.
- A revealed number N means exactly N of its (up to 8) adjacent cells are mines. A revealed 0 shows as empty.
- Flag cells you are certain are mines (toggle_flag places or removes a flag). Flags are not required to win, but they help you track deductions.
- The game ends automatically; you just keep making moves.

# How the board looks (read this carefully)
- UNREVEALED cells are DARK slate-blue squares with a subtle 3D bevel. These are the only cells you can reveal or flag.
- REVEALED cells are flat LIGHT GRAY squares. They are done -- never click them again. A revealed count shows as a large colored digit; a 0 shows as an empty light square.
- FLAGGED cells are dark squares with a red flag icon.
- Column letters run A..{last_col} left-to-right along the top margin; row numbers 1..{rows} run top-to-bottom along the left margin.
- Before every move, verify the target cell in the screenshot is DARK. If it is light gray, it is already revealed.

# Coordinates
- Columns are labeled A (left) to {last_col} (right); rows are labeled 1 (top) to {rows} (bottom). The labels are drawn on the screenshot margins.
- Refer to cells as column letter + row number, e.g. E5. In tool calls pass row as a number (1 = top row) and col as a letter, e.g. row=5, col="E".

# Strategy
1. A number N with exactly N unrevealed neighbors: all those neighbors are mines -> flag them.
2. A number N with N flagged neighbors: every other unrevealed neighbor is safe -> reveal it.
3. Work the frontier (numbers touching unrevealed cells) and chain rules 1 and 2.
4. If no certain move exists, make the lowest-risk guess you can justify. Heuristics: cells far from any number are usually safer; frontier 50/50 guesses are the worst; a fresh corner or edge region is often a good way to unlock progress when stuck.
5. Keep your scratchpad up to date with write_note: track deduced mines, deduced safe cells, and open questions. You see your notes every turn.

# Discipline
- Every turn: read the screenshot carefully, reason briefly, then call exactly ONE action tool (reveal_cell or toggle_flag). You may also call write_note in the same turn.
- Keep your visible reply text under 4 short sentences. Any longer analysis must go into your write_note call instead -- long replies get cut off and waste turns.
- Never repeat a move that was rejected or had no effect; the feedback tells you why it failed.
- The screenshot is the ground truth for the board; trust it over your memory.
- Check coordinates twice: row 1 is the TOP row, column A is the LEFTMOST column."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reveal_cell",
            "description": "Reveal (left-click) the cell at the given coordinate. "
                           "Pass row as a 1-based number and col as a letter, e.g. row=5, col='E'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "description": "Row number, 1 at the top"},
                    "col": {"type": "string", "description": "Column letter, A at the left"},
                },
                "required": ["row", "col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_flag",
            "description": "Place a flag on an unrevealed cell you are certain is a mine, "
                           "or remove an existing flag. Pass row as a 1-based number and col as a letter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "description": "Row number, 1 at the top"},
                    "col": {"type": "string", "description": "Column letter, A at the left"},
                },
                "required": ["row", "col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Overwrite your persistent scratchpad (max ~500 chars), shown to you "
                           "every turn. Track deduced mines, deduced safe cells, and plans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The full new content of your scratchpad"},
                },
                "required": ["note"],
            },
        },
    },
]
