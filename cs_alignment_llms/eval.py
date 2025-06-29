import os
import re
from enum import Enum

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

MAX_ATTEMPTS = 5
NUM_TRIALS = 3
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
    msg = (
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Rate the acceptability of the following sentence on a 7-point Likert scale, "
                    'where 1 means "completely unacceptable" and 7 means "completely acceptable".\n\n'
                    "Respond ONLY in this JSON format:\n"
                    "{\n"
                    '  "rating": 5\n'
                    "}\n\n"
                    "The sentence is:\n"
                    f'"{utterance}"',
                }
            ],
        )
        .choices[0]
        .message.content
    )
    try:
        resp = ModelResponse.model_validate(msg)
    except ValidationError:
        # Attempt to extract via regex
        rating_match = re.search(r'"rating"\s*:\s*([1-7])', msg)
        rating = int(rating_match.group(1)) if rating_match else None
        resp = ModelResponse(rating=LikertRating(rating)) if rating else None

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
        "openai/gpt-4o-mini",
        "openai/o4-mini-2025-04-16",
        "openai/o4-mini-high-2025-04-16",
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

    data = (
        pd.read_csv(RESULTS_FILE)
        if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE)
        else pd.DataFrame(columns=["model", "stimulus_number", "trial"])
    )

    for model in (model_bar := tqdm(all_models)):
        model_bar.set_description(f"Model {model}")
        for _, row in (
            stim_bar := tqdm(
                stimuli_data.iterrows(), total=len(stimuli_data.index), leave=False
            )
        ):
            utterance = str(row["text"])
            stim_num = int(row["stimulus_number"])
            stim_bar.set_description(f"Stimulus #{(stim_num)}")
            for trial_num in range(1, NUM_TRIALS + 1):
                if not data[
                    (data["model"] == model)
                    & (data["stimulus_number"] == stim_num)
                    & (data["trial"] == trial_num)
                ].empty:
                    continue  # skip model/utterance combinations we have already processed

                n_tries = 0
                while True:
                    try:
                        n_tries += 1
                        score = get_acceptability_score(client, model, utterance)
                        break
                    except Exception as err:
                        if n_tries > MAX_ATTEMPTS:
                            raise RuntimeError("Max tries reached") from err
                        print(f"Got error {err}.\nTrying again (n={n_tries})")

                # Save to file after every trial
                new_data = pd.DataFrame(
                    {
                        "model": [model],
                        "utterance": [utterance],
                        "stimulus_number": [stim_num],
                        "acceptability_rating": [score],
                        "trial": trial_num,
                    }
                )
                data = pd.concat([data, new_data])
                data.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
