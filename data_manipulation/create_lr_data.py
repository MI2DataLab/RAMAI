import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math

RAW_CSV_DIR = "../data/raw_csv/"
LR_DIR = "../data/lr/"

events = ["dpm", "mlinpl"]
df_followed = pd.DataFrame(
    columns=[
        "last_hint_correct",
        "history_hint_correct",
        "group",
        "sex",
        "age",
        "education",
        "hint_density",
        "followed",
    ]
)
df_detection = pd.DataFrame(
    columns=[
        "last_hint_correct",
        "history_hint_correct",
        "group",
        "sex",
        "age",
        "education",
        "hint_density",
        "lie_detected",
    ]
)

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
                    df_followed.loc[len(df_followed)] = {
                        "last_hint_correct": prev,
                        "history_hint_correct": history_hint_correct
                        / (history_hint_correct + history_hint_incorrect),
                        "group": event,
                        "sex": sex,
                        "age": age,
                        "education": education,
                        "hint_density": hint_num / question_num,
                        "followed": question["followed"],
                    }
                    df_detection.loc[len(df_detection)] = {
                        "last_hint_correct": prev,
                        "history_hint_correct": history_hint_correct
                        / (history_hint_correct + history_hint_incorrect),
                        "group": event,
                        "sex": sex,
                        "age": age,
                        "education": education,
                        "hint_density": hint_num / question_num,
                        "lie_detected": int(question["hint_ans"] != question["answer"]),
                    }
                history_hint_incorrect += 1
            else:
                if prev is not None:
                    df_followed.loc[len(df_followed)] = {
                        "last_hint_correct": prev,
                        "history_hint_correct": history_hint_correct
                        / (history_hint_correct + history_hint_incorrect),
                        "group": event,
                        "sex": sex,
                        "age": age,
                        "education": education,
                        "hint_density": hint_num / question_num,
                        "followed": question["followed"],
                    }
                history_hint_correct += 1
            prev = question["hint_correct"]

scaler = StandardScaler()
df_followed[["history_hint_correct", "hint_density"]] = scaler.fit_transform(
    df_followed[["history_hint_correct", "hint_density"]]
)
df_followed.to_csv(os.path.join(LR_DIR, "followed.csv"), index=False)
scaler = StandardScaler()
df_detection[["history_hint_correct", "hint_density"]] = scaler.fit_transform(
    df_detection[["history_hint_correct", "hint_density"]]
)
df_detection.to_csv(os.path.join(LR_DIR, "detection.csv"), index=False)
