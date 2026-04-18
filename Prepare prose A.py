import openai
import pandas as pd
import time
import os

# Resolve all file paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────
# Configuration — fill in credentials
# ─────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o-mini"
N_TEST = None  # Set to an integer to process only the first N rows (saves to answers_prose_test.csv)

# ─────────────────────────────────────────
# Set up OpenAI client pointing to OpenRouter
# ─────────────────────────────────────────
client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL,
)

# ─────────────────────────────────────────
# Prompt template for gold answer generation
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable and reliable assistant.

Your task is to produce a high-quality reference answer to the given question, suitable for evaluation purposes.

Requirements:

- Write in clear, fluent prose paragraphs (no bullet points, no headers)
- Be informative, accurate, and appropriately detailed (100–200 words)
- Maintain a neutral, objective, and informative tone
- Structure the answer logically: briefly introduce the topic, provide key explanations, and conclude if appropriate
- Avoid unnecessary stylistic variation or conversational language
- Do not include any formatting symbols such as *, #, or -
- Do not fabricate facts; if the question is ambiguous or uncertain, provide the most widely accepted explanation

Write for an educated general audience.
"""

def generate_prose_answer(question: str) -> str:
    """Call GPT-4o-mini via OpenRouter to generate a prose gold answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0.0,  # fixed for reproducibility
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# Main: load questions, generate answers, save
# ─────────────────────────────────────────
def main():
    # Determine output file and load data (paths resolved relative to script location)
    out_file = os.path.join(SCRIPT_DIR, "data", "answers_prose_test.csv" if N_TEST is not None else "answers_prose.csv")

    if os.path.exists(out_file):
        df = pd.read_csv(out_file, index_col="id")
        print(f"Resuming from {out_file} ({len(df)} rows).")
    else:
        df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "questions_60.csv"), index_col="id")
        df["answer_prose"] = None
        print(f"Loaded {len(df)} questions from questions_60.csv.")

    # Limit rows if N_TEST is set
    if N_TEST is not None:
        df = df.head(N_TEST)

    for idx, row in df.iterrows():
        # Skip if already generated (useful if script was interrupted)
        if pd.notna(row.get("answer_prose")):
            print(f"[{idx}] Already done, skipping.")
            continue

        print(f"[{idx}] Generating answer for: {row['question'][:60]}...")

        try:
            answer = generate_prose_answer(row["question"])
            df.at[idx, "answer_prose"] = answer

            # Save after every answer in case of interruption
            df.to_csv(out_file, index=True)
            print(f"[{idx}] Done. ({len(answer.split())} words)")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"[{idx}] ERROR: {e}")
            time.sleep(5)  # wait longer on error before continuing
            continue

    print(f"\nAll done! Saved to {out_file}")
    print(df[["our_category", "question", "answer_prose"]].head())


if __name__ == "__main__":
    main()