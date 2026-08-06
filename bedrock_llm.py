import os
from google import genai
from pydantic import BaseModel
from google.genai import types
from datetime import datetime
import os

LOG_FILE = "logs/gemini.log"

def log(message: str):
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),
]

def embed_text(text: str):
    """
    Returns a list[float] embedding.
    """

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )

    embedding = response.embeddings[0].values

    log(f"Embedding Success | Dimension={len(embedding)}")

    return embedding


def generate_answer(prompt: str):

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=safety_settings,
            temperature=0.2,
        ),
    )
    log("Generation Success")

    if response.candidates:
        for i, candidate in enumerate(response.candidates):
            if hasattr(candidate, "safety_ratings"):
                log(f"Safety Ratings Candidate {i}: {candidate.safety_ratings}")

    log(response.text)
    return response.text


if __name__ == "__main__":
    print(generate_answer("Hello"))
