"""
Script to extract demographics data.
"""

import os
import pandas as pd

RAW_CSV_DIR = "../data/raw-csv/"
DEMOGRAPHICS_CSV_DIR = "../data/ramai-human/"

events = ["mpd", "mlinpl"]

id2name = {
    "sex": {"m": "male", "f": "female"},
    "age": {0: "0-18", 1: "19-26", 2: "27-39", 3: "40+"},
    "education": {
        0: "< h. school",
        1: "h. school",
        2: "bachelor",
        3: "master+",
    },
    "group": {
        "mpd": "MPD",
        "mlinpl": "ML in PL",
    },
}


def main():
    games_mpd = pd.read_csv(os.path.join(RAW_CSV_DIR, f"mpd_games.csv"))
    games_mpd["group"] = "mpd"
    games_mlinpl = pd.read_csv(os.path.join(RAW_CSV_DIR, f"mlinpl_games.csv"))
    games_mlinpl["group"] = "mlinpl"

    games = pd.concat([games_mpd, games_mlinpl], ignore_index=True)
    demographics = games.loc[
        (~games["sex"].isna()) & (~games["age"].isna()) & (~games["education"].isna()),
        ["group", "sex", "age", "education"],
    ]
    demographics.reset_index(drop=True, inplace=True)
    demographics["age"] = demographics["age"].apply(lambda x: min(x, 3))
    demographics["education"] = demographics["education"].apply(lambda x: min(x, 3))

    demographics["sex"] = demographics["sex"].apply(lambda x: id2name["sex"][x])
    demographics["age"] = demographics["age"].apply(lambda x: id2name["age"][x])
    demographics["education"] = demographics["education"].apply(
        lambda x: id2name["education"][x]
    )
    demographics["group"] = demographics["group"].apply(lambda x: id2name["group"][x])

    demographics.to_csv(
        os.path.join(DEMOGRAPHICS_CSV_DIR, "demographics.csv"), index=False
    )


if __name__ == "__main__":
    main()
