import os
from enum import Enum

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm


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
                {
                    "role": "user",
                    "content": "How acceptable is this sentence?"
                    + "Rate the acceptability of this sentence on a 7-point Likert scale, "
                    + "where 1 is completely unacceptable and 7 is completely acceptable.",
                },
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

    all_models = [
        # OpenAI
        "openai/gpt-4.1-2025-04-14",
        "openai/gpt-3.5-turbo-0125",
        # Google
        "google/gemini-2.5-flash-preview-05-20",
        "google/gemma-3-27b-it",
        "google/gemma-3-12b-it",
        # Anthropic
        "anthropic/claude-4-sonnet-20250522",
        "anthropic/claude-3-7-sonnet-20250219:thinking",
        # DeepSeek
        "deepseek/deepseek-chat-v3-0324",
        "deepseek/deepseek-r1-0528",
        # Meta
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        # Mistral
        "mistralai/mistral-nemo",
        # Qwen
        "qwen/qwen-2.5-72b-instruct",
    ]
    stimuli_data = pd.read_csv("stimuli.csv")
    stimuli_data = stimuli_data[stimuli_data["stimulus_number"] >= 79]

    prev_data = (
        pd.read_csv(RESULTS_FILE)
        if os.path.exists(RESULTS_FILE)
        else pd.DataFrame(columns=["model", "stimulus_number"])
    )

    data = []
    for model in tqdm(all_models):
        for _, row in tqdm(stimuli_data.iterrows()):
            utterance = str(row["text"])
            stim_num = int(row["stimulus_number"])
            if not prev_data[
                (prev_data["model"] == model)
                & (prev_data["stimulus_number"] == stim_num)
            ].empty:
                continue  # skip model/utterance combinations we have already processed

            score = get_acceptability_score(client, model, utterance)
            data.append(
                {
                    "model": model,
                    "utterance": utterance,
                    "stimulus_number": stim_num,
                    "acceptability_rating": score,
                }
            )

    data = pd.DataFrame(data)
    data = pd.concat([prev_data, data])
    data.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
