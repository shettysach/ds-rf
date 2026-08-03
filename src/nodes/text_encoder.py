from __future__ import annotations

import time

from dora import Node

from shared.config import TextEncoderConfig
from shared.messages import (
    EncodedCommand,
    agent_command_from_arrow,
    encoded_command_to_arrow,
)
from text_encoder import TextEncoder


def main() -> None:
    cfg = TextEncoderConfig.from_env()
    node = Node()
    encoder = TextEncoder(cfg.model, device=cfg.device)

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "command":
            continue

        metadata = dict(event.get("metadata") or {})
        request = agent_command_from_arrow(event["value"], metadata)
        started_at = time.perf_counter()
        embedding = encoder.encode(request.text)
        encoded = EncodedCommand(
            observation_id=request.observation_id,
            text=request.text,
            embedding=embedding,
        )
        value, output_metadata = encoded_command_to_arrow(encoded)
        node.send_output("encoded_command", value, metadata=output_metadata)

        encode_ms = (time.perf_counter() - started_at) * 1000.0
        node.log(
            "info",
            f"[OBS {request.observation_id}] command encoded: "
            f"text={request.text!r} encode_ms={encode_ms:.1f}",
            target="dsrf.text_encoder",
            fields={
                "event": "command_encoded",
                "observation_id": str(request.observation_id),
                "command": request.text,
                "encode_ms": f"{encode_ms:.1f}",
            },
        )


if __name__ == "__main__":
    main()
