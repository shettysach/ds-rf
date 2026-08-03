from __future__ import annotations

import time

from dora import Node

from agent.vlm import LlamaServerClient
from shared.config import AgentConfig
from shared.messages import (
    AgentCommand,
    PipelineError,
    VisualObservation,
    agent_command_to_arrow,
    observation_from_arrow,
    pipeline_error_from_arrow,
)

MAX_INVALID_RESPONSES = 3
FALLBACK_COMMAND = '{"motion":"stand","direction":"forward"}'


class AgentLoop:
    def __init__(self, node: Node, client: LlamaServerClient) -> None:
        self.node = node
        self.client = client
        self.observation: VisualObservation | None = None
        self.pending_command: str | None = None
        self.invalid_responses = 0

    def run(self) -> None:
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT":
                continue
            if event["id"] == "observation":
                self._accept_observation(
                    observation_from_arrow(
                        event["value"], dict(event.get("metadata") or {})
                    )
                )
            elif event["id"] in {"planning_error", "encoding_error", "sonic_error"}:
                self._accept_error(pipeline_error_from_arrow(event["value"]))

    def _accept_observation(self, observation: VisualObservation) -> None:
        if self.observation is None:
            if observation.observation_id != 0:
                raise RuntimeError(
                    f"First observation must be 0, got {observation.observation_id}"
                )
            if observation.completed_command is not None:
                raise RuntimeError("Initial observation has a completed command")
        else:
            expected_id = self.observation.observation_id + 1
            if observation.observation_id != expected_id:
                raise RuntimeError(
                    f"Expected observation {expected_id}, got "
                    f"{observation.observation_id}"
                )
            if observation.completed_command != self.pending_command:
                raise RuntimeError(
                    "Completed command does not match the command sent for the "
                    "previous observation"
                )
            assert self.pending_command is not None
            self.client.commit(self.observation, self.pending_command)

        self.observation = observation
        self.pending_command = None
        self.invalid_responses = 0
        self._query_and_send()

    def _accept_error(self, error: PipelineError) -> None:
        if self.observation is None:
            raise RuntimeError(f"{error.source} failed before the first observation")
        if error.observation_id != self.observation.observation_id:
            raise RuntimeError(
                f"Stale {error.source} error for observation {error.observation_id}"
            )
        if error.source not in {"motion-gen", "text-encoder"}:
            self.node.log(
                "error",
                f"[OBS {error.observation_id}] {error.source} error: {error.detail}",
                target="dsrf.agent",
                fields={
                    "event": "pipeline_error",
                    "observation_id": str(error.observation_id),
                    "source": error.source,
                    "detail": error.detail,
                },
            )
            raise RuntimeError(f"{error.source}: {error.detail}")

        self.invalid_responses += 1
        previous = self.pending_command or ""
        self.node.log(
            "warn",
            f"[OBS {error.observation_id}] invalid command: "
            f"{previous!r} error={error.detail!r}",
            target="dsrf.agent",
            fields={
                "event": "invalid_command",
                "observation_id": str(error.observation_id),
                "command": previous,
                "detail": error.detail,
                "attempt": str(self.invalid_responses),
            },
        )
        if self.invalid_responses >= MAX_INVALID_RESPONSES:
            self.node.log(
                "warn",
                f"[OBS {error.observation_id}] fallback command: "
                f"{FALLBACK_COMMAND!r} after {self.invalid_responses} invalid responses",
                target="dsrf.agent",
                fields={
                    "event": "fallback_command",
                    "observation_id": str(error.observation_id),
                    "command": FALLBACK_COMMAND,
                    "invalid_responses": str(self.invalid_responses),
                },
            )
            self._send(FALLBACK_COMMAND)
            return
        feedback = f"Your previous response {previous!r} was invalid: {error.detail}"
        self._query_and_send(retry_feedback=feedback)

    def _query_and_send(self, *, retry_feedback: str | None = None) -> None:
        assert self.observation is not None
        observation_id = self.observation.observation_id
        attempt = self.invalid_responses
        fields = {
            "event": "vlm_request",
            "observation_id": str(observation_id),
            "attempt": str(attempt),
        }
        self.node.log(
            "debug",
            f"[OBS {observation_id}] VLM request started retry={attempt}",
            target="dsrf.agent.vlm",
            fields=fields,
        )
        started_at = time.perf_counter()
        try:
            command = self.client.complete(
                self.observation,
                retry_feedback=retry_feedback,
            )
        except Exception as exc:
            vlm_ms = (time.perf_counter() - started_at) * 1000.0
            detail = f"{type(exc).__name__}: {exc}"
            self.node.log(
                "error",
                f"[OBS {observation_id}] VLM request failed: {detail}",
                target="dsrf.agent.vlm",
                fields={
                    "event": "vlm_error",
                    "observation_id": str(observation_id),
                    "attempt": str(attempt),
                    "vlm_ms": f"{vlm_ms:.1f}",
                    "detail": detail,
                },
            )
            raise

        vlm_ms = (time.perf_counter() - started_at) * 1000.0
        self.node.log(
            "info",
            f"[OBS {observation_id}] VLM command: {command!r} "
            f"vlm_ms={vlm_ms:.1f} retry={attempt}",
            target="dsrf.agent.vlm",
            fields={
                "event": "vlm_response",
                "observation_id": str(observation_id),
                "command": command,
                "vlm_ms": f"{vlm_ms:.1f}",
                "attempt": str(attempt),
                "jpeg_kb": f"{len(self.observation.jpeg) / 1024.0:.1f}",
            },
        )
        self._send(command)

    def _send(self, command_text: str) -> None:
        assert self.observation is not None
        command = AgentCommand(self.observation.observation_id, command_text)
        data, metadata = agent_command_to_arrow(command)
        self.node.send_output("command", data, metadata=metadata)
        self.pending_command = command.text


def main() -> None:
    cfg = AgentConfig.from_env()
    node = Node()
    client = LlamaServerClient(
        base_url=cfg.vlm_url,
        timeout=cfg.vlm_timeout,
        system_prompt=cfg.system_prompt.read_text(encoding="utf-8"),
        user_prompt=cfg.user_prompt.read_text(encoding="utf-8"),
    )
    AgentLoop(node, client).run()


if __name__ == "__main__":
    main()
