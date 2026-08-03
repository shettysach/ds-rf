from __future__ import annotations

from dora import Node

from shared.messages import AgentCommand, agent_command_to_arrow


def main() -> None:
    node = Node()
    sent = False

    for event in node:
        if event["type"] == "STOP":
            break
        if sent or event["type"] != "INPUT" or event["id"] != "observation":
            continue

        metadata = dict(event.get("metadata") or {})
        observation_id = int(metadata["observation_id"])
        command = AgentCommand(observation_id, "ardy smoke test")
        data, command_metadata = agent_command_to_arrow(command)
        node.send_output("command", data, metadata=command_metadata)
        sent = True


if __name__ == "__main__":
    main()
