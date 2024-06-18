"""
Script to generate hints fo the ramai game.
"""

import re
import os
import pandas as pd
from llama2 import LLaMA2


# Set configuration
LLAMA2_DIR = ""
DATA_DIR = ""

MODEL_NAME = f"llama-2-70b"
CKPT_DIR = os.path.join(LLAMA2_DIR, MODEL_NAME)
TOKENIZER_PATH = os.path.join(LLAMA2_DIR, "tokenizer.model")
MAX_SEQ_LEN = 4096
TEMPERATURE = 1.35
TOP_P = 0.6
MAX_GEN_LEN = 256
MAX_BATCH_SIZE = 4

MILLIONAIR_QUESTIONS_PATH = os.path.join(DATA_DIR, "raw-csv", "questions.csv")
RAMAI_GAME_DIR = os.path.join(DATA_DIR, "ramai-game")


def extract_answer(answer: str) -> str:
    breakers = [i.start() for i in re.finditer(str("#" * 30), answer)]
    question_start = breakers[3]
    question_only = answer[question_start + 30 :]
    hint_start = question_only.find("justification: ") + len("justification: ")
    hint_end = question_only.find("#") - 1
    return question_only[hint_start:hint_end]


def main():
    # Build model
    generator = LLaMA2.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
    )

    # Load templates
    template = {}
    with open(
        os.path.join(RAMAI_GAME_DIR, "template_manipulative.txt"), "r", encoding="utf8"
    ) as f:
        template["manipulative"] = f.read()
    with open(
        os.path.join(RAMAI_GAME_DIR, "template_truthful.txt"), "r", encoding="utf8"
    ) as f:
        template["truthful"] = f.read()

    # Iterate over questions
    questions = pd.read_csv(MILLIONAIR_QUESTIONS_PATH, index_col=0)
    for ix, question in questions.iterrows():
        prompts = []
        for letter in ["A", "B", "C", "D"]:
            # Choose a template
            type = (
                "truthful"
                if bool(letter == question["correct_ans"])
                else "manipulative"
            )
            # Fill the template
            prompt = template[type].format(
                question_content=question["content"],
                ans_A=question["ans_A"],
                ans_B=question["ans_B"],
                ans_C=question["ans_C"],
                ans_D=question["ans_D"],
                letter=letter,
                letter_answer=question[f"ans_{letter}"],
            )
            prompts.append(prompt)

        # Generate hints
        answers = [
            generator.text_completion(
                prompts,
                max_gen_len=MAX_GEN_LEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                echo=True,
            )[i]["generation"]
            for i in range(len(prompts))
        ]
        hints = [extract_answer(answer) for answer in answers]

        questions.loc[ix, f"hint_A"] = hints[0]
        questions.loc[ix, f"hint_B"] = hints[1]
        questions.loc[ix, f"hint_C"] = hints[2]
        questions.loc[ix, f"hint_D"] = hints[3]

        # Save hints
        questions.to_csv(os.path.join(RAMAI_GAME_DIR, f"ramai_game_hints.csv"))


if __name__ == "__main__":
    main()
