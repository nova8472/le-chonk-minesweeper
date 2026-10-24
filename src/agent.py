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
        max_tokens: int = 8000,
        history_turns: int = 12,
        reasoning_effort: str = "none",  # mistral-large-4 supports "high" or "none"; "high" thinks so long per move it truncates
    ):
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral  # mistralai>=3 package layout

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
        self.reasoning_effort = reasoning_effort
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
            # The model only wrote a note (or nothing). Ask once for an action,
            # keeping the proper assistant/tool message structure.
            if parsed["tool_calls"]:
                messages.append({"role": "assistant",
                                 "content": parsed["reasoning"] or None,
                                 "tool_calls": parsed["tool_calls"]})
                for call in parsed["tool_calls"]:
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": ("Scratchpad updated."
                                    if call["function"]["name"] == "write_note"
                                    else "No effect."),
                    })
            elif parsed["reasoning"]:
                messages.append({"role": "assistant", "content": parsed["reasoning"]})
            messages.append({
                "role": "user",
                "content": ("You did not call an action tool. Call exactly one of "
                            "reveal_cell or toggle_flag now."),
            })
            response = self._complete(messages)
            parsed = self._extract(response)
            # If it still refuses, the runner records a wasted move instead of
            # killing the game (loop hints will nudge it next turn).

        return parsed

    def remember(self, turn: dict, feedback: str) -> None:
        """Record one completed turn using the proper tool-call message format.

        Past turns are replayed as: assistant message with real tool_calls,
        followed by tool-role responses with the env feedback. This keeps the
        model calling tools instead of writing them as text.
        Turns without any tool call are not stored; the runner's hint system
        covers those failures.
        """
        tool_calls = turn.get("tool_calls") or []
        if not tool_calls:
            return
        short_reasoning = " ".join((turn.get("reasoning") or "").split())[:400] or None
        responses = []
        for call in tool_calls:
            if call["function"]["name"] == "write_note":
                responses.append((call["id"], "Scratchpad updated."))
            else:
                responses.append((call["id"], feedback))
        self.transcript.append({
            "assistant": {"content": short_reasoning, "tool_calls": tool_calls},
            "tool_responses": responses,
        })

    # ------------------------------------------------------------------ plumbing

    def _build_messages(self, env, hint: str | None) -> list[dict]:
        notes = self.notes.strip() or "(empty)"
        system = self._system_base + f"\n\n# Your scratchpad notes\n{notes}\n"

        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in self.transcript[-self.history_turns:]:
            t = turn["assistant"]
            messages.append({"role": "assistant",
                             "content": t["content"],
                             "tool_calls": t["tool_calls"]})
            for tid, txt in turn["tool_responses"]:
                messages.append({"role": "tool", "tool_call_id": tid, "content": txt})

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
                    top_p=1.0,  # required: top_p must be 1 for greedy sampling
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
            except Exception as err:  # network / rate limit / transient 5xx
                last_err = err
                status = getattr(err, "status_code", None)
                msg = str(err)
                if status is None and "Status 4" in msg and "Status 429" not in msg:
                    status = 400
                # Client errors (except rate limiting) will not fix themselves.
                if isinstance(status, int) and status != 429 and 400 <= status < 500:
                    raise RuntimeError(f"Mistral API client error: {err}") from err
                time.sleep(min(2 ** attempt * 2, 30))
        raise RuntimeError(
            f"Mistral API failed after {MAX_API_RETRIES} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _split_chunks(content) -> tuple[str, str]:
        """Split content chunks into (visible_text, thinking_text).

        Handles both dicts and mistralai>=3 SDK objects (TextChunk, ThinkChunk).
        """
        if not isinstance(content, list):
            return (content or "").strip(), ""
        visible, thinking = [], []
        for part in content:
            cls = type(part).__name__
            if isinstance(part, dict) or cls == "TextChunk":
                if isinstance(part, dict):
                    if part.get("type") in (None, "text"):
                        visible.append(part.get("text") or "")
                else:
                    visible.append(getattr(part, "text", "") or "")
            elif cls == "ThinkChunk" or isinstance(part, dict) and part.get("thinking"):
                subs = (part.get("thinking") if isinstance(part, dict)
                        else getattr(part, "thinking", None)) or []
                for sub in subs:
                    if isinstance(sub, dict):
                        thinking.append(sub.get("text") or "")
                    else:
                        thinking.append(getattr(sub, "text", "") or "")
        return " ".join(visible).strip(), "\n".join(thinking).strip()

    def _extract(self, response) -> dict:
        choice = response.choices[0]
        reasoning, thinking = self._split_chunks(choice.message.content)
        action = None
        note = None
        raw_calls = []
        for tc in (choice.message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            name = tc.function.name
            tc_id = getattr(tc, "id", None) or f"call_{len(raw_calls) + 1}"
            raw_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": tc.function.arguments or "{}"},
            })
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
            "thinking": thinking,
            "tool_calls": raw_calls,
            "model": getattr(response, "model", None),  # server-echoed model name
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }
