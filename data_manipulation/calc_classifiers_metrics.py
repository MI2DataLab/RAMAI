"""
Script to extract precision and recall and confusion matrix values of the classifiers
"""

import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix


DATA_DIR = ""
RAMAI_LLM_DIR = os.path.join(DATA_DIR, "ramai-llm")
MANIPULATION_FUSE_DIR = os.path.join(DATA_DIR, "manipulation-fuse")
CLASSIFIERS_DIR = os.path.join(MANIPULATION_FUSE_DIR, "classifiers")
CONFUSION_DIR = os.path.join(MANIPULATION_FUSE_DIR, "confusion")

model_names = {
    "mixtral-8x7b": "Mixtral-8x7B",
    "dolphin-2.5": "Dolphin",
    "gpt-3.5-turbo": "GPT-3.5-turbo",
    "gpt-4": "GPT-4",
    "gemini-pro": "Gemini-Pro",
}


def main():
    model_answers, human_answers = {}, {}
    for type in ["manipulative", "truthful"]:
        model_answers[type] = {
            model_names[name]: pd.read_csv(
                os.path.join(CLASSIFIERS_DIR, f"{name}_{type}.csv"), index_col=0
            )
            for name in model_names
        }
        human_answers[type] = pd.read_csv(
            os.path.join(RAMAI_LLM_DIR, f"ramai_llm_{type}.csv"), index_col=0
        )

    y_manipulative = list(human_answers["manipulative"]["manipulative"].values)
    y_truthful = [0 for _ in range(len(human_answers["truthful"]))]
    y = y_manipulative + y_truthful

    # calculate precision and recall
    results = pd.DataFrame(columns=["model", "setting", "precision", "recall"])
    for template in ["low-context", "high-context"]:
        for model_name in model_names:
            y_pred = (
                model_answers["manipulative"][model_names[model_name]][
                    template
                ].values.tolist()
                + model_answers["truthful"][model_names[model_name]][
                    template
                ].values.tolist()
            )
            results.loc[len(results)] = [
                model_names[model_name],
                template,
                precision_score(y, y_pred),
                recall_score(y, y_pred),
            ]
    results.to_csv(
        os.path.join(MANIPULATION_FUSE_DIR, "classifier_pr.csv"), index=False
    )

    # calculate FP, FN, TP, TN
    for template in ["low-context", "high-context"]:
        results = {
            x: pd.DataFrame(
                index=list(model_names.values()),
                columns=list(model_names.values()),
            )
            for x in ["FP", "FN", "TP", "TN"]
        }
        for model_judge in list(model_names.values()):
            for model_respondent in list(model_names.values()):
                manipulative_indexes = list(
                    human_answers["manipulative"]
                    .loc[human_answers["manipulative"]["model"] == model_respondent, :]
                    .index
                )
                truthful_indexes = list(
                    human_answers["truthful"]
                    .loc[human_answers["truthful"]["model"] == model_respondent, :]
                    .index
                )
                y_manipulative = list(
                    human_answers["manipulative"]
                    .loc[manipulative_indexes, "manipulative"]
                    .values
                )
                y_truthful = [0 for _ in range(len(truthful_indexes))]
                y = y_manipulative + y_truthful
                y_pred = (
                    model_answers["manipulative"][model_judge]
                    .loc[manipulative_indexes, template]
                    .values.tolist()
                    + model_answers["truthful"][model_judge]
                    .loc[truthful_indexes, template]
                    .values.tolist()
                )
                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                results["TN"].loc[model_judge, model_respondent] = tn
                results["FP"].loc[model_judge, model_respondent] = fp
                results["FN"].loc[model_judge, model_respondent] = fn
                results["TP"].loc[model_judge, model_respondent] = tp
        results["TN"].to_csv(
            os.path.join(CONFUSION_DIR, f"{template}_tn.csv"), index=True
        )
        results["FP"].to_csv(
            os.path.join(CONFUSION_DIR, f"{template}_fp.csv"), index=True
        )
        results["FN"].to_csv(
            os.path.join(CONFUSION_DIR, f"{template}_fn.csv"), index=True
        )
        results["TP"].to_csv(
            os.path.join(CONFUSION_DIR, f"{template}_tp.csv"), index=True
        )


if __name__ == "__main__":
    main()
