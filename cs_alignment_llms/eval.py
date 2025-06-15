import os
import pandas as pd
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


RESULTS_FILE = "results.csv"


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
    # Initialize client
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    all_models = []
    all_utterances = []

    prev_data = (
        pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame()
    )

    data = []
    for model in all_models:
        for utterance in all_utterances:
            if not prev_data[
                (prev_data["model"] == model) & (prev_data["utterance"] == utterance)
            ].empty:
                continue  # skip model/utterance combinations we have already processed

            score = get_acceptability_score(client, model, utterance)
            data.append(
                {"model": model, "utterance": utterance, "acceptability_rating": score}
            )

    data = pd.DataFrame(data)
    data = pd.concat([prev_data, data])
    data.to_csv("results.csv")


if __name__ == "__main__":
    main()
