from __future__ import annotations

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
FALLBACK_COMMAND = "stand"


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
            elif event["id"] in {"planning_error", "sonic_error"}:
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
        if error.source != "motion-gen":
            raise RuntimeError(f"{error.source}: {error.detail}")

        self.invalid_responses += 1
        if self.invalid_responses >= MAX_INVALID_RESPONSES:
            self._send(FALLBACK_COMMAND)
            return
        previous = self.pending_command or ""
        feedback = f"Your previous response {previous!r} was invalid: {error.detail}"
        self._query_and_send(retry_feedback=feedback)

    def _query_and_send(self, *, retry_feedback: str | None = None) -> None:
        assert self.observation is not None
        command = self.client.complete(
            self.observation,
            retry_feedback=retry_feedback,
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
    client = LlamaServerClient(base_url=cfg.vlm_url, timeout=cfg.vlm_timeout)
    AgentLoop(node, client).run()


if __name__ == "__main__":
    main()
