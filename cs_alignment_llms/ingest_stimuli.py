import re

import pandas as pd


def main():
    with open("materials.txt", "r") as f:
        materials_doc = f.read()

    pat = re.compile(r"Q\d+\.\d+\s+(.+?)(?=\s+•|\s+End)", re.DOTALL)
    matches = pat.findall(materials_doc)

    # Valid stimuli should have "acceptable"/"aceptable"
    matches = [m for m in matches if re.search(r"[Aa]cc?eptable", m)]
    matches = [str(m).split("\n")[0].strip() for m in matches]

    first_stim_idx = -1
    for idx, stim in enumerate(matches):
        if stim == "Su hermano has trained at the gym every day.":
            first_stim_idx = idx
            break

    if first_stim_idx == -1:
        raise ValueError("Could not find first stimulus")

    matches = matches[first_stim_idx:]

    match_info = [
        {"stimulus_number": idx + 1, "text": stimulus}
        for idx, stimulus in enumerate(matches)
    ]
    pd.DataFrame(match_info).to_csv("stimuli.csv", index=False)


if __name__ == "__main__":
    main()
