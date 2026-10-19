"""mistral-large-4 vision agent for Minesweeper.

Each turn the agent receives: the system prompt (rules + its own notes),
a text transcript of recent turns (screenshots omitted to save tokens),
and the current board screenshot. It answers with brief reasoning plus
tool call(s): reveal_cell, toggle_flag, write_note.
"""

from __future__ import annotations

import json
import os
import time

from .prompts import TOOLS, build_system_prompt

ACTION_TOOLS = {"reveal_cell", "toggle_flag"}
MAX_API_RETRIES = 4


class MinesweeperAgent:
    """Stateful vision agent. Create one per game (fresh notes/transcript)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-large-4",
        temperature: float = 0.0,
        max_tokens: int = 500,
        history_turns: int = 12,
    ):
        from mistralai import Mistral

        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Put it in .env (see .env.example) "
                "or export it as an environment variable."
            )
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.notes = ""
        self.transcript: list[dict[str, str]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._system_base: str | None = None

    # ------------------------------------------------------------------- turn

    def act(self, env, hint: str | None = None) -> dict:
        """Look at the board and decide the next move.

        Returns {"action": {name, arguments} | None, "reasoning": str,
                 "prompt_tokens": int, "completion_tokens": int}.
        """
        game = env.game
        if self._system_base is None:
            self._system_base = build_system_prompt(game.rows, game.cols, game.n_mines)

        messages = self._build_messages(env, hint)
        response = self._complete(messages)
        parsed = self._extract(response)

        if parsed["action"] is None:
            # The model only wrote a note (or nothing). Ask once for an action.
            messages = messages + [
                {"role": "assistant", "content": parsed["reasoning"] or ""},
                {"role": "user", "content": (
                    "You did not call an action tool. Call exactly one of "
                    "reveal_cell or toggle_flag now."
                )},
            ]
            response = self._complete(messages)
            parsed = self._extract(response)

        return parsed

    def remember(self, reasoning: str, action: dict | None, feedback: str) -> None:
        """Record one completed turn in the transcript (text only, no images)."""
        if action:
            args = action.get("arguments") or {}
            call = f"{action['name']}(row={args.get('row')}, col={args.get('col')})"
        else:
            call = "(no action)"
        self.transcript.append({
            "assistant": f"{reasoning.strip()}\nTool call: {call}",
            "user": feedback,
        })

    # ------------------------------------------------------------------ plumbing

    def _build_messages(self, env, hint: str | None) -> list[dict]:
        notes = self.notes.strip() or "(empty)"
        system = self._system_base + f"\n\n# Your scratchpad notes\n{notes}\n"

        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in self.transcript[-self.history_turns:]:
            messages.append({"role": "assistant", "content": turn["assistant"]})
            messages.append({"role": "user", "content": turn["user"]})

        status = env.status_line()
        if hint:
            status += "\nNOTE: " + hint
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Current board screenshot below.\n" + status},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{env.render_base64()}"}},
            ],
        })
        return messages

    def _complete(self, messages: list[dict]):
        last_err = None
        for attempt in range(MAX_API_RETRIES):
            try:
                return self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as err:  # network / rate limit / transient 5xx
                last_err = err
                time.sleep(min(2 ** attempt * 2, 30))
        raise RuntimeError(f"Mistral API failed after {MAX_API_RETRIES} attempts") from last_err

    def _extract(self, response) -> dict:
        choice = response.choices[0]
        reasoning = (choice.message.content or "").strip()
        action = None
        note = None
        for tc in (choice.message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            name = tc.function.name
            if name in ACTION_TOOLS and action is None:
                action = {"name": name, "arguments": args}
            elif name == "write_note" and note is None:
                note = (args.get("note") or "").strip()
        if note:
            self.notes = note[-1000:]
        usage = getattr(response, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        return {
            "action": action,
            "reasoning": reasoning,
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }
