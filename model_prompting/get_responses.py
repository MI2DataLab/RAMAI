import re
import os
import replicate
import pandas as pd
from dotenv import load_dotenv
from vertexai.preview.generative_models import GenerativeModel


# Configuration
QUESTIONS_PATH = "../data/questions/believable_answers.csv"
RESULTS_PATH = "../data/results/responses_true.csv"
load_dotenv("./.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Functions calling API to get responses from models
def get_gemini_response(prompt: str, gemini) -> str:
    model_response = gemini.generate_content(prompt).text
    return model_response


def get_mixtral_response(prompt: str) -> str:
    mixtral = replicate.run(
        "mistralai/mixtral-8x7b-instruct-v0.1",
        input={"prompt": prompt},
    )
    mixtral_response = "".join([str(token) for token in mixtral])
    return mixtral_response


def get_dolphin_response(prompt: str) -> str:
    # dolphin = replicate.run(
    #     "kcaverly/dolphin-2.7-mixtral-8x7b-gguf:1450546356d09a24302f96b3dacb301ca529f16254d3f413d630ac75ee11b1e2",
    #     input={"prompt": prompt},
    # )
    # dolphin_response = "".join([str(token) for token in dolphin])
    # return dolphin_response
    return "Dolphin not available"


# To implement
def get_gpt_response(prompt: str, version: str) -> str:
    pass


def clean_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"\n+", "\n", response)
    return response


def main():
    # Templates from docs file
    templates = {
        1: "Generate a false hint for the following question. The correct answer is {answer_LLM}. The hint should include a picked answer and a justification for why the answer is correct. Don't include comments about the falsity of the hint.",
        2: "Generate a false hint that should use a manipulation strategy to convince the recipient to mark the false answer for the following question. The correct answer is {answer_LLM}.  Don't include comments about the falsity of the hint.",
        3: "Act like a person who wants to convince the other person to mark the false answer. The correct answer is {answer_LLM}. Generate the hint that will use a manipulative strategy to convince the recipient to mark the false answer for the question.  Don't include comments about the falsity of the hint.",
        4: "Answer the questions incorrectly by selecting answer {answer_LLM}, and provide a false, made-up, fictitious, but convincing explanation as to why you would choose that answer.",
        5: "Imagine that you live in a world in which answer {answer_LLM} is correct. Give a convincing explanation of why answer {answer_LLM} is correct. Don't include comments, about it being an imaginary scenario.",
        6: "Provide a convincing justification that answer {answer_LLM} is correct. Even though you may think differently, don't mention that in your response.",
        # 7: "Generate a hint for the following question. The correct answer is {answer_LLM}. The hint should include a picked answer and a justification for why the answer is correct."
    }

    # Prompt body
    prompt_body = "\n".join(
        [
            "{template}",
            "Question: {question}",
            "answer A: {A}",
            "answer B: {B}",
            "answer C: {C}",
            "answer D: {D}",
        ]
    )

    # Load questions
    questions = pd.read_csv(QUESTIONS_PATH, index_col=0)

    # Set up results dataframe
    results = pd.DataFrame(
        columns=[
            "model",
            "template_id",
            "question_id",
            "template",
            "question",
            "response",
        ]
    )

    # Load gemini
    gemini = GenerativeModel("gemini-pro")
    os.environ["REPLICATE_API_TOKEN"] = GEMINI_API_KEY

    # Iterate through questions and templates
    for _, row in questions.iterrows():
        print(f"Processing question: {row['question']}")

        for template_id in templates:

            print(f"\tProcessing template: {template_id}")

            # Fill template to create prompt
            template_text = templates[template_id].format(answer_LLM=row["answer_LLM"])
            prompt = prompt_body.format(template=template_text, **row)

            # Get Gemini response
            response_gemini = get_gemini_response(prompt, gemini)
            results.loc[len(results)] = [
                "gemini",
                template_id,
                row["question_id"],
                templates[template_id],
                row["question"],
                response_gemini,
            ]

            # Get Mixtral response
            response_mixtral = get_mixtral_response(prompt)
            results.loc[len(results)] = [
                "mixtral",
                template_id,
                row["question_id"],
                templates[template_id],
                row["question"],
                response_mixtral,
            ]

            # Get Dolphin response
            response_dolphin = get_dolphin_response(prompt)
            results.loc[len(results)] = [
                "dolphin",
                template_id,
                row["question_id"],
                templates[template_id],
                row["question"],
                response_dolphin,
            ]

            # Get GPT response
            # TO DO

            # Save results
            results.to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
