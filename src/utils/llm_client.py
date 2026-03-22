"""
Vertex AI Gemini client.
Authentication: Application Default Credentials (gcloud auth application-default login).
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
        )
    return _client


async def complete(
    model: str,
    messages: list[dict],
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    json_mode: bool = False,
) -> str:
    """Call a Gemini model on Vertex AI and return the text response."""
    contents = [
        types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])])
        for msg in messages
    ]

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_prompt,
    )
    if json_mode:
        config.response_mime_type = "application/json"

    response = await get_client().aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response.text
