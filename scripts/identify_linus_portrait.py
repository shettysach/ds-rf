#!/usr/bin/env python3
"""Ask the local vision-language model which image shows Linus."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import urllib.request
from pathlib import Path

DEFAULT_IMAGES = (
    Path(__file__).resolve().parents[1]
    / "tasks/portrait_corridors/images/karpathy.png",
    Path(__file__).resolve().parents[1] / "tasks/portrait_corridors/images/linus.png",
    Path(__file__).resolve().parents[1] / "tasks/portrait_corridors/images/nolan.png",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify which of three images is a portrait of Linus."
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        metavar="IMAGE",
        help="three images; defaults to Karpathy, Linus, then Bugs",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("VLM_URL", "http://127.0.0.1:8080"),
        help="llama-server base URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    images = list(args.images) or list(DEFAULT_IMAGES)
    if len(images) != 3:
        parser.error("provide exactly three images")
    random.shuffle(images)
    real_answer = next(
        index for index, path in enumerate(images) if path.stem.lower() == "linus"
    )

    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Three images follow, labeled Image 0, Image 1, and Image 2. "
                "Which image contains a portrait of Linus? Reply with only the "
                "single image number: 0, 1, or 2."
            ),
        }
    ]
    for index, path in enumerate(images):
        if not path.is_file():
            raise SystemExit(f"Image does not exist: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"Image {index}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )

    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "",
                "messages": [
                    {
                        "role": "system",
                        "content": "You classify images accurately and follow output formats exactly.",
                    },
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    try:
        answer = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected llama-server response: {result!r}") from exc
    print(f"REAL ANSWER: {real_answer}")
    print(f"VLM ANSWER: {answer.strip()}")


if __name__ == "__main__":
    main()
