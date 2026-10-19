# Le Chonk plays Minesweeper

A vision-agent harness that lets **Mistral Large 4 Preview** (`mistral-large-4`) play Minesweeper using nothing but screenshots.

Built for the Mistral AI community challenge **"Le Chonk wants to play"** (October 2026).

## What it does

- A from-scratch, seeded Minesweeper implementation rendered to images with Pillow. No game engine, no scraping, no external assets.
- Every turn, `mistral-large-4` receives the current board screenshot (with coordinate labels drawn into the margins), its own scratchpad notes, and a transcript of recent turns. It reasons briefly and answers with tool calls: `reveal_cell`, `toggle_flag`, or `write_note`.
- The model **never** sees the grid as text, the ground truth, or the mine layout — pixels only.
- A seeded headless benchmark measures win rate, guess rate, wasted moves, and token usage. Two non-LLM baselines are included: a classic constraint solver and a random agent.

## Setup

```bash
git clone <repo-url>
cd le-chonk-minesweeper
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # then paste your MISTRAL_API_KEY into .env
python tests\test_game.py         # sanity check, no API needed
python tests\test_env.py          # rendering + pipeline check, no API needed
```

Requirements: Python 3.10+, a Mistral Studio API key (https://console.mistral.ai), tkinter (bundled with python.org Windows builds) for the demo window.

> Note: the challenge requires the model name `mistral-large-4` (not `mistral-large-latest`); that is the default everywhere in this repo.

## Usage

### Watch it play (windowed demo)

```bash
python -m src.play --seed 7
```

Opens a window showing the board and, below it, the model's live reasoning, tool call, and the game's feedback for every move. Record this window with OBS Studio (or any screen recorder) for the demo video.

Options: `--agent solver` (non-LLM baseline), `--agent random`, `--difficulty beginner|intermediate`, `--delay 0.6` (pause between moves), `--seed N`.

### Benchmark

```bash
python -m src.benchmark --agent model --difficulty beginner --games 20 --seed 1000
python -m src.benchmark --agent solver --games 100   # baseline, no API
python -m src.benchmark --agent random --games 100  # floor baseline, no API
```

Runs seeded games headless and writes a full JSON report to `results/`, then prints a summary: wins, loss reasons, average moves, token usage, and a classification of every reveal (`opening` / `deduced` / `risky` / `guess`).

## How it works

```
                 ┌────────────────────────────────────────────────┐
                 │  mistral-large-4 (vision + tool calling)       │
                 │  in:  system prompt, notes, transcript, photo  │
                 │  out: reasoning + tool call                     │
                 └───────────────▲───────────────────┬────────────┘
                                 │                   │ reveal_cell / toggle_flag
             screenshot (PNG)    │                   ▼
      ┌──────────────────────┐  │          ┌────────────────────┐
      │ MinesweeperEnv        │──┘          │ runner (game loop) │
      │ - seeded board        │◄────────────│ - hints on loops  │
      │ - PIL rendering        │   feedback  │ - move metrics    │
      │ - ground truth (log)  │             └────────────────────┘
      └──────────────────────┘
```

Design choices:

- **Pixels only.** The model plays exactly like a human looking at the screen. Ground truth is used exclusively for logging and metrics, never shown to the model.
- **Tool calling instead of text parsing.** Actions are function calls with a JSON schema; coordinates are tolerant (row/col swaps and combined strings like `E5` are handled).
- **A scratchpad (`write_note`).** The model keeps persistent notes (deduced mines, safe cells, plans) that are pinned into its context every turn.
- **Loop protection.** No-op actions and repeated actions trigger escalating hints injected into the prompt.
- **Honest metrics.** Every reveal is classified against what a perfect player could have deduced from the visible board (conservative, flag-free deduction), so losses are attributed to forced guesses vs actual mistakes.

## Results

Benchmark: 20 seeded games, beginner (9x9, 10 mines), temperature 0.

*(to be filled after the benchmark run)*

| Agent | Win rate | Avg moves | Losses by guess | Losses by risky move |
|-------|----------|-----------|-----------------|----------------------|
| mistral-large-4 | TBD | TBD | TBD | TBD |
| Constraint solver | TBD | TBD | TBD | TBD |
| Random | TBD | TBD | TBD | TBD |

## Project layout

```
src/
  game.py        pure seeded Minesweeper logic (no I/O)
  env.py         GameEnv interface + PIL rendering + coordinate parsing
  prompts.py     system prompt + tool schemas
  agent.py       mistral-large-4 agent (vision + tool calling + notes)
  mock_agent.py  solver + random baselines (no API)
  runner.py      shared game loop, hints, move classification
  metrics.py     deduction engine + benchmark aggregation
  benchmark.py   seeded headless benchmark CLI
  play.py        windowed demo (tkinter)
  samples.py     generates the sample screenshots in media/
tests/
  test_game.py   game logic unit tests
  test_env.py    rendering + pipeline tests
```

## How it was developed

Solo project, built with Mistral Vibe (agentic CLI coding) pair-programming in the evenings of the challenge week. Stack: Python 3.10, `mistralai` SDK, Pillow, tkinter. Based on no existing game-playing framework — the game, rendering, agent loop, and metrics are written from scratch for this challenge.

## Limitations

- Beginner boards only in the current benchmark; intermediate (16x16, 40 mines) is supported but not yet benchmarked.
- Minesweeper sometimes forces a guess; win rates below 100% are partially structural, which the guess/risky classification makes visible.
- The deduction engine used for metrics applies only the two classic single-cell rules (conservative): some "guess" moves would be resolvable with full constraint solving.
