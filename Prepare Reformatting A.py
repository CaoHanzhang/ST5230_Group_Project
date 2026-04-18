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
N_TEST = None  # Set to an integer to process only the first N rows (saves to answers_all_180_test.csv)

client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL,
)

# ─────────────────────────────────────────
# Prompt templates for each reformat style
# ─────────────────────────────────────────
BULLET_PROMPT = """You are a precise and faithful formatting assistant.

You will be given a prose answer. Your task is to reformat it into bullet points.

Requirements:
- Convert each sentence or minimal semantic unit from the original text into a separate bullet point using "- " as the prefix
- Preserve all original information exactly; do NOT add, remove, merge, split, or paraphrase any content
- Maintain the original wording as much as possible (only minimal changes for formatting)
- Preserve the original order of information
- The number of bullet points should correspond closely to the number of distinct information units in the original text
- Avoid over-fragmentation or merging of content
- Do NOT use any markdown headers (no # symbols)
- Do not add any introduction or conclusion outside the bullet list

If any conflict arises, prioritize exact content preservation over readability.

The output should be a faithful structural transformation of the original text only.
"""

HEADER_PROMPT = """You are a precise and faithful formatting assistant.

You will be given a prose answer. Your task is to reformat it using markdown headers and paragraphs.

Requirements:
- Organize the content into 2–4 sections using markdown headers (## )
- Each section should group related sentences from the original text; do NOT invent new categories or concepts
- Preserve all original information exactly; do NOT add, remove, merge, split, or paraphrase any content
- Maintain the original wording as much as possible (only minimal changes for formatting)
- Preserve the original order of information within each section
- The number of sections should reflect the natural grouping of the original content rather than an arbitrary split
- Do NOT use bullet points or numbered lists
- Do not add any introduction or conclusion outside the sections

If any conflict arises, prioritize exact content preservation over readability.

The output should be a faithful structural transformation of the original text only.
"""


def reformat_answer(prose: str, style: str) -> str:
    """Reformat a prose answer into bullet points or markdown headers."""
    assert style in ("bullet", "header"), "style must be 'bullet' or 'header'"

    system_prompt = BULLET_PROMPT if style == "bullet" else HEADER_PROMPT

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original prose answer:\n\n{prose}"}
        ],
        temperature=0.0,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# Main: load prose answers, reformat, save
# ─────────────────────────────────────────
def main():
    # Determine output file and load data (paths resolved relative to script location)
    out_file = os.path.join(SCRIPT_DIR, "data", "answers_all_180_test.csv" if N_TEST is not None else "answers_all_180.csv")
    prose_src = os.path.join(SCRIPT_DIR, "data", "answers_prose_test.csv" if N_TEST is not None else "answers_prose.csv")

    if os.path.exists(out_file):
        df = pd.read_csv(out_file, index_col="id")
        print(f"Resuming from {out_file} ({len(df)} rows).")
    else:
        df = pd.read_csv(prose_src, index_col="id")
        for col in ["answer_bullet", "answer_header"]:
            if col not in df.columns:
                df[col] = None
        df.to_csv(out_file, index=True)
        print(f"Loaded {len(df)} questions from {prose_src}. Saved initial file to {out_file}.")

    # Limit rows if N_TEST is set
    if N_TEST is not None:
        df = df.head(N_TEST)

    for idx, row in df.iterrows():
        prose = row["answer_prose"]

        # ── Bullet reformat ──
        if pd.isna(df.at[idx, "answer_bullet"]):
            print(f"[{idx}] Reformatting to bullet...")
            try:
                df.at[idx, "answer_bullet"] = reformat_answer(prose, "bullet")
                df.to_csv(out_file, index=True)
                time.sleep(0.5)
            except Exception as e:
                print(f"[{idx}] ERROR (bullet): {e}")
                time.sleep(5)
        else:
            print(f"[{idx}] Bullet already done, skipping.")

        # ── Header reformat ──
        if pd.isna(df.at[idx, "answer_header"]):
            print(f"[{idx}] Reformatting to header...")
            try:
                df.at[idx, "answer_header"] = reformat_answer(prose, "header")
                df.to_csv(out_file, index=True)
                time.sleep(0.5)
            except Exception as e:
                print(f"[{idx}] ERROR (header): {e}")
                time.sleep(5)
        else:
            print(f"[{idx}] Header already done, skipping.")

    print(f"\nAll done! Saved to {out_file}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()