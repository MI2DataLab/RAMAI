"""
Script processing the RAMAI-Human data to prepare it for the linear mixed-effects models.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math

DATA_DIR = ""
RAW_CSV_DIR = os.path.join(DATA_DIR, "raw-csv")
RAMAI_HUMAN_DIR = os.path.join(DATA_DIR, "ramai-human")

events = ["event1", "event2"]
df_hint_trusted = pd.DataFrame(
    columns=[
        "game_id",
        "last_hint_correct",
        "history_hint_correct",
        "group",
        "sex",
        "age",
        "education",
        "hint_density",
        "hint_trusted",
    ]
)
df_manipulation_detected = pd.DataFrame(
    columns=[
        "game_id",
        "last_hint_correct",
        "history_hint_correct",
        "group",
        "sex",
        "age",
        "education",
        "hint_density",
        "manipulation_detected",
    ]
)


def main():
    for event in events:
        games = pd.read_csv(os.path.join(RAW_CSV_DIR, f"{event}_games.csv"))
        answers = pd.read_csv(os.path.join(RAW_CSV_DIR, f"{event}_answers.csv"))

        for game_id in games["id"]:
            answers_tmp = answers.loc[answers["game_id"] == game_id, :]
            answers_tmp = answers_tmp.sort_values("question_count")
            prev = None
            sex = games.loc[games["id"] == game_id, "sex"].values[0]
            age = games.loc[games["id"] == game_id, "age"].values[0]
            education = games.loc[games["id"] == game_id, "education"].values[0]
            if type(sex) != str or math.isnan(age) or math.isnan(education):
                continue
            age = min(age, 3)
            education = min(education, 3)
            history_hint_incorrect = 0
            history_hint_correct = 0
            question_num = 0
            hint_num = 0
            for ix, question in answers_tmp.iterrows():
                question_num += 1
                if question["hint_used"] == 0:
                    continue
                hint_num += 1
                if question["hint_correct"] == 0:
                    if prev is not None:
                        df_hint_trusted.loc[len(df_hint_trusted)] = {
                            "game_id": game_id,
                            "last_hint_correct": prev,
                            "history_hint_correct": history_hint_correct
                            / (history_hint_correct + history_hint_incorrect),
                            "group": event,
                            "sex": sex,
                            "age": age,
                            "education": education,
                            "hint_density": hint_num / question_num,
                            "hint_trusted": question["hint_trusted"],
                        }
                        df_manipulation_detected.loc[len(df_manipulation_detected)] = {
                            "game_id": game_id,
                            "last_hint_correct": prev,
                            "history_hint_correct": history_hint_correct
                            / (history_hint_correct + history_hint_incorrect),
                            "group": event,
                            "sex": sex,
                            "age": age,
                            "education": education,
                            "hint_density": hint_num / question_num,
                            "manipulation_detected": int(
                                question["hint_ans"] != question["answer"]
                            ),
                        }
                    history_hint_incorrect += 1
                else:
                    if prev is not None:
                        df_hint_trusted.loc[len(df_hint_trusted)] = {
                            "game_id": game_id,
                            "last_hint_correct": prev,
                            "history_hint_correct": history_hint_correct
                            / (history_hint_correct + history_hint_incorrect),
                            "group": event,
                            "sex": sex,
                            "age": age,
                            "education": education,
                            "hint_density": hint_num / question_num,
                            "hint_trusted": question["hint_trusted"],
                        }
                    history_hint_correct += 1
                prev = question["hint_correct"]

    scaler = StandardScaler()
    df_hint_trusted[["history_hint_correct", "hint_density"]] = scaler.fit_transform(
        df_hint_trusted[["history_hint_correct", "hint_density"]]
    )
    df_hint_trusted.to_csv(
        os.path.join(RAMAI_HUMAN_DIR, "hint_trusted.csv"), index=False
    )
    scaler = StandardScaler()
    df_manipulation_detected[["history_hint_correct", "hint_density"]] = (
        scaler.fit_transform(
            df_manipulation_detected[["history_hint_correct", "hint_density"]]
        )
    )
    df_manipulation_detected.to_csv(
        os.path.join(RAMAI_HUMAN_DIR, "manipulation_detected.csv"), index=False
    )


if __name__ == "__main__":
    main()
