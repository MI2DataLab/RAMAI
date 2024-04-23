"""
Script for extracting raw data from the SQLite databases and saving them as CSV files.
"""

import os
import sqlite3
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")


DATABASE_DIR = "../data/raw/"
OUTPUT_CSV_DIR = "../data/raw-csv/"


def get_date(date):
    if pd.isna(date):
        return None
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")


id2name = {}
order = {}

id2name["sex"] = {"m": "male", "f": "female"}
order["sex"] = ["male", "female"]

id2name["age"] = {0: "0-18", 1: "19-26", 2: "27-39", 3: "40-59", 4: "60+"}
order["age"] = ["0-18", "19-26", "27-39", "40-59", "60+"]

id2name["education"] = {
    0: "below high school",
    1: "high school",
    2: "bachelor",
    3: "master",
    4: "phd",
}
order["education"] = ["below high school", "high school", "bachelor", "master", "phd"]


class Event:
    """
    Class to store event meta-information

    Attributes:
    name (str): Event name
    start_date (datetime): Event start date
    end_date (datetime): Event end date
    """

    def __init__(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime,
    ):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date

    def __repr__(self):
        return f"Event: {self.name} ({self.start_date} - {self.end_date})"


def main():
    mpd = Event(
        "MPD",
        get_date("2023-09-14 00:00:00.000"),
        get_date("2023-09-20 23:59:59.999"),
    )
    mlinpl = Event(
        "MLinPL",
        get_date("2023-10-26 00:00:00.000"),
        get_date("2023-10-29 23:59:59.999"),
    )

    events = [mpd, mlinpl]

    for event in events:
        cnx = sqlite3.connect(
            os.path.join(DATABASE_DIR, f"{event.name.lower()}.sqlite3")
        )
        games = pd.read_sql_query("SELECT * FROM main_game", cnx)
        answers = pd.read_sql_query("SELECT * FROM main_answer", cnx)
        questions = pd.read_sql_query("SELECT * FROM main_question", cnx)

        games.loc[:, "timestamp_start"] = games.loc[:, "timestamp_start"].apply(
            get_date
        )
        games.loc[:, "timestamp_2"] = games.loc[:, "timestamp_2"].apply(get_date)
        games.loc[:, "timestamp_7"] = games.loc[:, "timestamp_7"].apply(get_date)
        games.loc[:, "timestamp_end"] = games.loc[:, "timestamp_end"].apply(get_date)
        games = games.loc[~games["timestamp_start"].isna(), :]
        games = games.loc[games["timestamp_start"] > event.start_date, :]
        games = games.loc[games["timestamp_start"] < event.end_date, :]

        games["time_to_2"] = games["timestamp_2"] - games["timestamp_start"]
        games["time_to_7"] = games["timestamp_7"] - games["timestamp_start"]
        games["time_to_end"] = (
            games.loc[games["game_won"] == 1, "timestamp_end"]
            - games.loc[games["game_won"] == 1, "timestamp_start"]
        )
        games["time_total"] = games["timestamp_end"] - games["timestamp_start"]

        games.drop(
            columns=[
                "username",
                "email",
                "is_mobile",
                "is_tablet",
                "is_touch_capable",
                "is_pc",
                "is_bot",
                "browser_family",
                "browser_version",
                "os_family",
                "os_version",
                "device_family",
                "code",
            ],
            inplace=True,
        )

        games.to_csv(os.path.join(OUTPUT_CSV_DIR, f"{event.name.lower()}_games.csv"))

        answers = answers.loc[answers["game_id"].isin(games["id"]), :]
        answers["hint_used"] = answers["hint_ans"].apply(lambda x: 1 if x else 0)

        # changed_to_hint: participant changed their answer after the AI suggestion
        answers["changed_answer"] = answers["answer_before_prompt"] != answers["answer"]
        answers["changed_answer"] = [
            int(x[0]) if x[1] is not None else None
            for x in zip(answers["changed_answer"], answers["answer_before_prompt"])
        ]

        # changed_to_hint: participant changed their answer to the AI suggestion
        answers["changed_to_hint"] = (answers["changed_answer"] == 1) & (
            answers["answer"] == answers["hint_ans"]
        )
        answers["changed_to_hint"] = [
            int(x[0]) if x[1] is not None else None
            for x in zip(answers["changed_to_hint"], answers["answer_before_prompt"])
        ]

        # changed_to_hint: participant selected the suggested answer
        answers["hint_trusted"] = answers["answer"] == answers["hint_ans"]
        answers["hint_trusted"] = [
            int(x[0]) if x[1] else None
            for x in zip(answers["hint_trusted"], answers["hint_used"])
        ]

        answers["question_number"] = answers["question_number"] + 1
        answers["correct_ans"] = answers["question_id"].apply(
            lambda x: questions.loc[questions["id"] == x, "correct_ans"].values[0]
        )

        # wrong_hint_trusted: participant selected the suggested answer, but it was manipualative
        answers["wrong_hint_trusted"] = (answers["hint_trusted"] == 1) & (
            answers["answer"] != answers["correct_ans"]
        )
        answers["wrong_hint_trusted"] = [
            int(x[0]) if x[1] else None
            for x in zip(answers["wrong_hint_trusted"], answers["hint_used"])
        ]

        # decepted: participant answered the question correctly at first, but changed the answer to the AI suggestion
        answers["decepted"] = (answers["answer"] == answers["hint_ans"]) & (
            answers["answer_before_prompt"] == answers["correct_ans"]
        )
        answers["decepted"] = [
            int(x[0]) if x[1] is not None else None
            for x in zip(answers["decepted"], answers["answer_before_prompt"])
        ]

        # question_count: the total number of questions answered by the participant before the current question
        answers["question_count"] = None
        for ix, row in answers.iterrows():
            answers_before = len(
                answers.loc[
                    (answers["game_id"] == row["game_id"])
                    & (answers["timestamp"] < row["timestamp"]),
                    :,
                ]
            )
            answers.loc[ix, "question_count"] = answers_before

        answers.to_csv(
            os.path.join(OUTPUT_CSV_DIR, f"{event.name.lower()}_answers.csv")
        )
        questions.to_csv(
            os.path.join(OUTPUT_CSV_DIR, f"{event.name.lower()}_questions.csv")
        )

        cnx.close()


if __name__ == "__main__":
    main()
