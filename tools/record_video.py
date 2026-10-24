"""Record the demo video by composing frames and piping them to ffmpeg.

Runs a live game with mistral-large-4, captures every move as two frame
groups -- a "thinking" group (board state + spinner, while the model
processes) and a "move" group (board + reasoning + feedback) -- plus title
and end cards. Composes everything at 1920x1080 and streams raw frames to
ffmpeg. No screen capture needed.

Requires ffmpeg (system install, or: pip install imageio-ffmpeg).

Usage:
  .venv/Scripts/python tools/record_video.py --seed 1000
Exits 0 and writes the mp4 only if the model WINS the game (a win makes a
far better demo video); exits 1 on a loss so a wrapper can retry a seed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from src.agent import MinesweeperAgent
from src.cat import cat_image
from src.env import MinesweeperEnv, _load_font

W, H = 1920, 1080
FPS = 10

BG = (24, 26, 31)
CARD = (248, 249, 250)
TEXT = (232, 234, 236)
DIM = (150, 156, 162)
ACCENT = (244, 157, 55)  # le chonk orange
CLASS_COLORS = {
    "deduced": (76, 175, 80),
    "risky": (255, 152, 0),
    "guess": (171, 71, 188),
    "opening": (3, 169, 244),
}

PANEL_X = 1040
PANEL_W = 850
TEXT_W = PANEL_W - 60


def wrap(draw, text, font, max_width, max_lines=None):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + " ..."
    return lines


class Composer:
    def __init__(self):
        from PIL import Image, ImageDraw
        self.Image = Image
        self.Draw = ImageDraw
        self.f_title = _load_font(64)
        self.f_sub = _load_font(30)
        self.f_h = _load_font(30)
        self.f_panel = _load_font(26)
        self.f_small = _load_font(22)
        self.f_tiny = _load_font(20)

    def new_canvas(self):
        img = self.Image.new("RGB", (W, H), BG)
        return img, self.Draw.Draw(img)

    def paste_board(self, img, board):
        # Left card with the board centered.
        d = self.Draw.Draw(img)
        d.rounded_rectangle([30, 40, 1010, 1040], radius=18, fill=CARD)
        scale = min(940 / board.width, 960 / board.height)
        board = board.resize((int(board.width * scale), int(board.height * scale)),
                             self.Image.LANCZOS)
        x = 30 + (980 - board.width) // 2
        y = 40 + (1000 - board.height) // 2
        img.paste(board, (x, y))

    def _panel_header(self, img, d, title="mistral-large-4", subtitle=""):
        cat = cat_image(2)
        d.rounded_rectangle([PANEL_X, 44, PANEL_X + 780, 96], radius=12, fill=(38, 41, 47))
        chip = self.Image.new("RGBA", (44, 44), (0, 0, 0, 0))
        chip.paste(cat, (4, 4), cat)
        img.paste(chip, (PANEL_X + 6, 48), chip)
        d.text((PANEL_X + 60, 70), title, font=self.f_h, fill=TEXT, anchor="lm")
        if subtitle:
            d.text((PANEL_X + 60 + self.f_h.getlength(title) + 18, 70),
                   subtitle, font=self.f_small, fill=DIM, anchor="lm")

    def thinking_frame(self, board, move_no, status, dots):
        img, d = self.new_canvas()
        self.paste_board(img, board)
        self._panel_header(img, d)
        d.text((PANEL_X + 30, 130), f"mistral-large-4 is thinking about move {move_no}"
               + "." * dots, font=self.f_panel, fill=TEXT)
        # animated bar
        bar_w = 60 + (dots * 140)
        d.rounded_rectangle([PANEL_X + 30, 190, PANEL_X + 30 + bar_w, 206],
                            radius=8, fill=ACCENT)
        d.rounded_rectangle([PANEL_X + 30, 190, PANEL_X + 30 + 620, 206],
                            radius=8, outline=DIM, width=2)
        d.text((PANEL_X + 30, 240), status, font=self.f_tiny, fill=DIM)
        d.text((PANEL_X + 30, H - 60), "vision-only: pixels in, tool calls out",
               font=self.f_tiny, fill=DIM)
        return img

    def move_frame(self, board, mv):
        img, d = self.new_canvas()
        self.paste_board(img, board)
        self._panel_header(img, d, subtitle=f"move {mv['move']}")
        cls = mv.get("classification") or ""
        if cls:
            d.text((PANEL_X + 30, 130), cls.upper(),
                   font=self.f_panel, fill=CLASS_COLORS.get(cls, TEXT))
        # reasoning block
        d.text((PANEL_X + 30, 180), "THINKING", font=self.f_tiny, fill=DIM)
        y = 212
        for line in wrap(d, mv.get("reasoning") or "(no visible reasoning)",
                         self.f_small, TEXT_W, max_lines=8):
            d.text((PANEL_X + 30, y), line, font=self.f_small, fill=TEXT)
            y += 30
        # action + feedback
        y += 16
        d.text((PANEL_X + 30, y), "MOVE", font=self.f_tiny, fill=DIM)
        y += 32
        act = mv.get("action") or {}
        act_args = act.get("arguments") or {}
        d.text((PANEL_X + 30, y), f"{act.get('name', '?')}({act_args})",
               font=self.f_small, fill=ACCENT)
        y += 34
        for line in wrap(d, mv.get("feedback") or "", self.f_small, TEXT_W,
                         max_lines=4):
            d.text((PANEL_X + 30, y), line, font=self.f_small, fill=DIM)
            y += 30
        d.text((PANEL_X + 30, H - 60),
               f"{mv.get('latency_s', 0)}s | {mv.get('prompt_tokens', 0)}"
               f"+{mv.get('completion_tokens', 0)} tokens",
               font=self.f_tiny, fill=DIM)
        return img

    def title_card(self):
        img, d = self.new_canvas()
        cat = cat_image(8)
        img.paste(cat, ((W - cat.width) // 2, 250), cat)
        t = "LE CHONK PLAYS MINESWEEPER"
        d.text((W // 2, 480), t, font=self.f_title, fill=ACCENT, anchor="mm")
        s = "mistral-large-4, vision-only: pixels in, tool calls out"
        d.text((W // 2, 560), s, font=self.f_sub, fill=TEXT, anchor="mm")
        s2 = "Mistral AI community challenge - Le Chonk wants to play (October 2026)"
        d.text((W // 2, 615), s2, font=self.f_small, fill=DIM, anchor="mm")
        return img

    def end_card(self, record):
        img, d = self.new_canvas()
        cat = cat_image(8)
        img.paste(cat, ((W - cat.width) // 2, 200), cat)
        verb = "VICTORY" if record["won"] else "GAME OVER"
        d.text((W // 2, 420), verb, font=self.f_title,
               fill=CLASS_COLORS["deduced"] if record["won"] else CLASS_COLORS["risky"],
               anchor="mm")
        lines = [
            f"mistral-large-4 won in {record['moves']} moves"
            if record["won"] else
            f"mistral-large-4 lost on move {record['moves']}",
            f"prompt {record['prompt_tokens']} + completion {record['completion_tokens']} tokens",
            f"classification: {record['move_classifications']}",
            "github.com/nova8472/le-chonk-minesweeper",
        ]
        y = 520
        for line in lines:
            d.text((W // 2, y), line, font=self.f_sub, fill=TEXT, anchor="mm")
            y += 48
        return img


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install it, or: pip install imageio-ffmpeg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default="media/le-chonk-minesweeper.mp4")
    parser.add_argument("--max-seconds", type=float, default=112.0)
    args = parser.parse_args()

    load_dotenv()
    env = MinesweeperEnv("beginner")
    agent = MinesweeperAgent()

    moves = []

    def on_turn_start(move_no):
        moves.append({
            "move_no": move_no,
            "thinking_board": env.render().copy(),
            "status": env.status_line(),
            "latency": 0.0,
        })

    def on_move(env_, move):
        mv = dict(move)
        mv["board"] = env_.render().copy()
        if moves:
            moves[-1].update({
                "latency": move.get("latency_s", 0.0),
                "latency_s": move.get("latency_s", 0.0),
                "reasoning": move.get("reasoning", ""),
                "thinking": move.get("thinking", ""),
                "feedback": move.get("feedback", ""),
                "action": move.get("action"),
                "classification": move.get("classification"),
                "prompt_tokens": move.get("prompt_tokens", 0),
                "completion_tokens": move.get("completion_tokens", 0),
                "board": mv["board"],
                "move": move.get("move"),
                "outcome": move.get("outcome"),
            })
        print(f"move {move.get('move')}: {move.get('outcome')}", flush=True)

    from src.runner import play_game
    print(f"playing game, seed={args.seed} ...", flush=True)
    record = play_game(env, agent, seed=args.seed,
                       on_move=on_move, on_turn_start=on_turn_start)
    record["seed"] = args.seed

    pathlib.Path("results").mkdir(exist_ok=True)
    with open(f"results/video-game-s{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    if not record["won"]:
        print(f"GAME LOST (seed {args.seed}) - no video written; try another seed",
              flush=True)
        sys.exit(1)

    # ---- compose and encode ---------------------------------------------
    comp = Composer()
    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(pathlib.Path(args.out).resolve())]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def emit(img, seconds):
        n = max(1, round(seconds * FPS))
        raw = img.convert("RGB").tobytes()
        for _ in range(n):
            proc.stdin.write(raw)

    # durations
    title_s, end_s = 3.0, 6.0
    total = title_s + end_s
    per_move = []
    for mv in moves:
        td = min(max(mv.get("latency", 0) * 0.35, 0.8), 2.0)
        rd = 2.4
        per_move.append((td, rd))
        total += td + rd
    scale = 1.0
    if total > args.max_seconds:
        scale = (args.max_seconds - title_s - end_s) / (total - title_s - end_s)

    emit(comp.title_card(), title_s)
    dots = 0
    for mv, (td, rd) in zip(moves, per_move):
        td, rd = td * scale, rd * scale
        n_think = max(1, round(td * FPS))
        for i in range(n_think):
            dots = (i % 3) + 1
            emit(comp.thinking_frame(mv["thinking_board"], mv["move_no"],
                                     mv["status"], dots), 1 / FPS)
        emit(comp.move_frame(mv["board"], mv), rd)
    emit(comp.end_card(record), end_s)

    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        print(f"ffmpeg exited with {rc}", flush=True)
        sys.exit(1)
    size = pathlib.Path(args.out).stat().st_size / 1e6
    real_seconds = title_s + end_s + scale * (total - title_s - end_s)
    print(f"video written: {args.out} ({size:.1f} MB, {real_seconds:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
