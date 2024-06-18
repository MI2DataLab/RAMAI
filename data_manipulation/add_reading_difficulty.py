"""
Script to add reading difficulty to the LIWC results (RAMAI-LLM).
"""

import os
import pandas as pd
import textstat


DATA_DIR = ""
RAMAI_LLM_DIR = os.path.join(DATA_DIR, "ramai-llm")


def main():
    ramai_llm_data = {}
    for type in ["manipulative", "truthful"]:
        ramai_llm_data[type] = pd.read_csv(
            os.path.join(RAMAI_LLM_DIR, f"ramai_llm_{type}.csv"),
            index_col=0,
        )
        docs = list(ramai_llm_data[type]["response"])
        somg_index = [textstat.smog_index(doc) for doc in docs]
        liwc_results = pd.read_csv(
            os.path.join(RAMAI_LLM_DIR, f"liwc_{type}.csv"), index_col=0
        )
        liwc_results["Reading Difficulty"] = somg_index
        liwc_results.to_csv(os.path.join(RAMAI_LLM_DIR, f"liwc_{type}.csv"))


if __name__ == "__main__":
    main()
