import json
from typing import Any, cast

from agent.vlm import LlamaServerClient
from nodes.agent import AgentLoop
from shared.messages import (
    PipelineError,
    VisualObservation,
    agent_command_from_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)


class _Response:
    def __init__(self, command: str) -> None:
        self.payload = json.dumps(
            {"choices": [{"message": {"content": command}}]}
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def read(self) -> bytes:
        return self.payload


def test_llama_client_uses_blank_model_and_replays_history(monkeypatch) -> None:
    posted: list[dict[str, Any]] = []
    responses = iter(["stand", "walk forward 0.4"])

    def urlopen(request, timeout):
        posted.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data),
            }
        )
        return _Response(next(responses))

    monkeypatch.setattr("agent.vlm.urllib.request.urlopen", urlopen)
    client = LlamaServerClient(
        base_url="http://127.0.0.1:8080/",
        timeout=12.0,
        system_prompt="System file prompt.\n",
        user_prompt="User file prompt.\n",
    )
    first = VisualObservation(0, None, b"first")
    assert client.complete(first) == "stand"
    client.commit(first, "stand")

    second = VisualObservation(1, "stand", b"second")
    assert client.complete(second) == "walk forward 0.4"

    assert posted[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert posted[0]["timeout"] == 12.0
    assert posted[0]["payload"]["model"] == ""
    messages = posted[1]["payload"]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == "System file prompt.\n"
    assert messages[1]["content"][0]["text"].endswith("User file prompt.\n")
    assert messages[2]["content"] == "stand"
    assert messages[1]["content"][1]["image_url"]["url"].endswith("Zmlyc3Q=")
    assert messages[3]["content"][1]["image_url"]["url"].endswith("c2Vjb25k")


class _Node:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = iter(events)
        self.outputs: list[tuple[str, object, dict[str, object]]] = []
        self.logs: list[tuple[str, str, dict[str, object]]] = []

    def __iter__(self):
        return self.events

    def send_output(self, output_id, value, **kwargs) -> None:
        self.outputs.append((output_id, value, kwargs))

    def log(self, level, message, **kwargs) -> None:
        self.logs.append((level, message, kwargs))


class _Client:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.feedback: list[str | None] = []
        self.commits: list[tuple[int, str]] = []

    def complete(self, observation, *, retry_feedback=None) -> str:
        self.feedback.append(retry_feedback)
        return next(self.responses)

    def commit(self, observation, command) -> None:
        self.commits.append((observation.observation_id, command))


def _observation_event(observation: VisualObservation) -> dict[str, object]:
    value, metadata = observation_to_arrow(observation)
    return {
        "type": "INPUT",
        "id": "observation",
        "value": value,
        "metadata": metadata,
    }


def _error_event(observation_id: int) -> dict[str, object]:
    return {
        "type": "INPUT",
        "id": "planning_error",
        "value": pipeline_error_to_arrow(
            PipelineError("motion-gen", observation_id, "bad command")
        ),
    }


def test_agent_retries_three_invalid_responses_then_stands() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            _error_event(0),
            _error_event(0),
            _error_event(0),
            {"type": "STOP"},
        ]
    )
    client = _Client(["invalid one", "invalid two", "invalid three"])

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    commands = [
        agent_command_from_arrow(value, cast(Any, kwargs["metadata"]))
        for output_id, value, kwargs in node.outputs
        if output_id == "command"
    ]
    assert [command.text for command in commands] == [
        "invalid one",
        "invalid two",
        "invalid three",
        '{"motion":"stand","direction":"forward"}',
    ]
    assert client.feedback[0] is None
    assert all(feedback is not None for feedback in client.feedback[1:])
    vlm_messages = [message for _, message, _ in node.logs if "VLM command" in message]
    assert len(vlm_messages) == 3
    assert "[OBS 0] VLM command: 'invalid one'" in vlm_messages[0]
    assert vlm_messages[0].endswith("retry=0")
    assert "[OBS 0] VLM command: 'invalid two'" in vlm_messages[1]
    assert vlm_messages[1].endswith("retry=1")
    assert "[OBS 0] VLM command: 'invalid three'" in vlm_messages[2]
    assert vlm_messages[2].endswith("retry=2")
    assert any("fallback command: '{\"motion\":\"stand\",\"direction\":\"forward\"}'" in message for _, message, _ in node.logs)


def test_agent_commits_exact_completed_command() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"first")),
            _observation_event(VisualObservation(1, "walk forward 0.4", b"second")),
            {"type": "STOP"},
        ]
    )
    client = _Client(["walk forward 0.4", "stand"])

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    assert client.commits == [(0, "walk forward 0.4")]
