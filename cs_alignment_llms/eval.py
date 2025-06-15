import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class LikertRating(int, Enum):
    COMPLETELY_UNACCEPTABLE = 1
    MOSTLY_UNACCEPTABLE = 2
    SOMEWHAT_UNACCEPTABLE = 3
    UNSURE = 4
    SOMEWHAT_ACCEPTABLE = 5
    MOSTLY_ACCEPTABLE = 6
    COMPLETELY_ACCEPTABLE = 7


class ModelResponse(BaseModel):
    rating: LikertRating


# Initialize client

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


# TODO: Should we somehow acount for the "calibration" and task specification
# that K&D perform, to ensure that the LLM understands how to rate the utterances?


def get_acceptability_score(client: OpenAI, model: str, utterance: str) -> LikertRating:
    resp = (
        client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "user", "content": "How acceptable is this sentence?"},
                {"role": "user", "content": utterance},
            ],
            response_format=ModelResponse,
        )
        .choices[0]
        .message.parsed
    )
    if resp is None:
        raise ValueError("Got None as response from OpenRouter")
    return resp.rating


def main():
    pass


if __name__ == "__main__":
    main()
