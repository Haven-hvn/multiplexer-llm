#!/usr/bin/env python3
"""Example: routing Jev /v1/systemone requests through the multiplexer.

Speaks the TypeSafe Jev "System One" /v1/systemone contract across multiple
endpoints, with the same weighted selection, capacity slots, and automatic
failover as chat completions.
"""

import asyncio
import os

from multiplexer_llm import Multiplexer
from openai import AsyncOpenAI


async def main() -> None:
    # Endpoint clients for servers speaking the /v1/systemone contract.
    # base_url includes the OpenAI-style /v1 prefix, so path "/systemone"
    # resolves to "{base_url}/systemone" (i.e. /v1/systemone).
    primary = AsyncOpenAI(
        api_key=os.getenv("SYSTEMONE_API_KEY", "not-needed"),
        base_url=os.getenv("SYSTEMONE_PRIMARY_URL", "http://127.0.0.1:8000/v1"),
    )
    backup = AsyncOpenAI(
        api_key=os.getenv("SYSTEMONE_API_KEY", "not-needed"),
        base_url=os.getenv("SYSTEMONE_BACKUP_URL", "http://127.0.0.1:8001/v1"),
    )

    async with Multiplexer() as multiplexer:
        multiplexer.add_model(primary, 5, "systemone-latest")
        # Fallbacks are only used when every primary is rate limited/unavailable
        multiplexer.add_fallback_model(backup, 1, "systemone-backup")

        result = await multiplexer.systemone.create(
            state="refund please",
            questions={
                "greeting": {
                    "type": "noul",
                    "instructions": "Is this a greeting?",
                },
                "topic": {
                    "type": "choice",
                    "instructions": "Pick a category.",
                    "criteria": {
                        "billing": "billing issue",
                        "other": None,
                    },
                },
                "sentiment": {
                    "type": "score",
                    "instructions": "Rate the sentiment.",
                    "criteria": [
                        "very negative",
                        "negative",
                        "neutral",
                        "positive",
                        "very positive",
                    ],
                },
            },
        )

        # Answers are keyed by question id; usage reports token accounting
        print("model:", result["model"])
        print("greeting probability:", result["answers"]["greeting"]["noul"])
        print("chosen topic:", result["answers"]["topic"]["choice"])
        print("sentiment score:", result["answers"]["sentiment"]["score"])
        print("usage:", result["usage"])
        print("stats:", multiplexer.get_stats())


if __name__ == "__main__":
    asyncio.run(main())
