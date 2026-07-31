from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from typing import Any

from shared.messages import VisualObservation

SYSTEM_PROMPT = """\
You control a simulated Unitree G1 humanoid from third-person camera images.
At each turn, choose one motion command for the robot to execute next.
Return only the command text, without JSON, punctuation, or explanation.
"""

COMMAND_PROMPT = """\
Choose the next command. Prefer one of these basic commands:
- stand
- walk forward 0.4
- walk backward 0.3
- walk left 0.3 facing=forward
- walk right 0.3 facing=forward
"""


@dataclass(frozen=True)
class _ConversationTurn:
    observation: VisualObservation
    assistant: str


class LlamaServerClient:
    def __init__(self, *, base_url: str, timeout: float) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.timeout = timeout
        self._history: list[_ConversationTurn] = []

    def complete(
        self,
        observation: VisualObservation,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for turn in self._history:
            messages.append(_user_message(turn.observation))
            messages.append({"role": "assistant", "content": turn.assistant})
        messages.append(_user_message(observation, retry_feedback=retry_feedback))

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {"model": "", "messages": messages, "temperature": 0},
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            document = json.loads(response.read().decode("utf-8"))

        try:
            content = document["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama-server returned no assistant message") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("llama-server returned an empty assistant message")
        return content.strip()

    def commit(self, observation: VisualObservation, command: str) -> None:
        self._history.append(_ConversationTurn(observation, command))


def _user_message(
    observation: VisualObservation,
    *,
    retry_feedback: str | None = None,
) -> dict[str, Any]:
    completed = observation.completed_command or "none (initial observation)"
    text = f"Completed command: {completed}\n\n{COMMAND_PROMPT}"
    if retry_feedback is not None:
        text = f"{retry_feedback}\n\n{text}"
    image_url = (
        "data:image/jpeg;base64," + b64encode(observation.jpeg).decode("ascii")
    )
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
