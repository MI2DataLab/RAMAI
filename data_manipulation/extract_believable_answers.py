"""
Script to extract believable answers from the RAMAI-Human experiment.
"""

import os
import pandas as pd
import sqlite3
import warnings

warnings.filterwarnings("ignore")


DATA_DIR = "../data"
DATABASE_DIR = os.path.join(DATA_DIR, "raw")
RAW_CSV_DIR = os.path.join(DATA_DIR, "raw-csv")
RAMAI_LLM_DIR = os.path.join(DATA_DIR, "ramai-llm")


def main():
    cnx = sqlite3.connect(os.path.join(DATABASE_DIR, f"mlinpl.sqlite3"))
    questions = pd.read_sql_query("SELECT * FROM main_question", cnx)
    answers_mlinpl = pd.read_csv(
        os.path.join(RAW_CSV_DIR, "mlinpl_answers.csv"), index_col=0
    )
    answers_mpd = pd.read_csv(os.path.join(RAW_CSV_DIR, "mpd_answers.csv"), index_col=0)
    answers = pd.concat([answers_mlinpl, answers_mpd], ignore_index=True)
    answers_grouped = (
        answers.groupby(["question_id", "hint_ans"])
        .agg(
            {
                "decepted": "sum",
                "decepted": "sum",
            }
        )
        .sort_values("question_id", ascending=True)
    )
    believable_answers = answers_grouped.loc[answers_grouped["decepted"] >= 2, :]
    believable_answers["question"] = believable_answers.index.map(
        lambda x: questions.loc[questions["id"] == x[0], "content"].values[0]
    )
    believable_answers["answer"] = believable_answers.index.map(
        lambda x: questions.loc[questions["id"] == x[0], f"ans_{x[1]}"].values[0]
    )
    believable_answers[f"hint"] = believable_answers.index.map(
        lambda x: questions.loc[questions["id"] == x[0], f"hint_{x[1]}"].values[0]
    )
    believable_answers.reset_index(inplace=True)
    believable_answers["answer_correct"] = believable_answers["question_id"].map(
        lambda x: questions.loc[questions["id"] == x, "correct_ans"].values[0]
    )
    for letter in ["A", "B", "C", "D"]:
        believable_answers[letter] = believable_answers["question_id"].map(
            lambda x: questions.loc[questions["id"] == x, f"ans_{letter}"].values[0]
        )
    believable_answers = believable_answers.loc[
        :,
        ["question_id", "question", "A", "B", "C", "D", "hint_ans", "answer_correct"],
    ]
    believable_answers.rename(columns={"hint_ans": "answer_LLM"}, inplace=True)
    believable_answers.to_csv(os.path.join(RAMAI_LLM_DIR, "believable_answers.csv"))


if __name__ == "__main__":
    main()
