"""
Judge.py — Pairwise format preference evaluation for formatting bias study.
v2: adds judge reasoning, format-leakage detection, and answer-length metadata.

Protocol (per proposal):
  - 60 questions × 3 format pairs × 2 swap orderings × 2 judge models = 720 API calls
  - Format pairs: (prose vs bullet), (prose vs header), (bullet vs header)
  - Swap protocol: present (A,B) then (B,A); if preference flips → randomly break tie
  - Judge models: GPT-4o-mini and Claude Haiku — both via OpenRouter (single API key)
  - Results saved incrementally after every call for resume support

Output columns (judgements.csv):
  id, question, our_category,
  format_a, format_b,                     # which two formats are compared
  len_a, len_b, len_diff,                 # word-count lengths (free, local)
  judge,                                  # "gpt4o_mini" | "claude_haiku"
  winner_ab,  winner_ba,  final_winner,   # swap-resolved verdict (A or B only)
  reason_ab,  reason_ba,                  # judge's one-sentence rationale
  format_leakage_ab, format_leakage_ba,   # True if reason mentions formatting terms
  any_format_leakage,                     # True if either direction leaked
  swap_consistent,                        # True if both orderings agree
  raw_ab, raw_ba                          # full raw model output for audit

Three analytical dimensions added vs v1
───────────────────────────────────────
1. Judge reasoning  (reason_ab / reason_ba)
   Captures *why* the judge preferred a format. Enables qualitative coding of
   whether wins are driven by content attributes ("more accurate") vs. structural
   ones ("easy to scan", "well-organized"). Directly answers the TA question
   "why does the bias become larger?"

2. Format-leakage flag  (format_leakage_ab / _ba / any_format_leakage)
   Automatically detects reasoning that names formatting cues using a keyword
   list. A high leakage rate in a task-type or format-pair means the judge is
   rewarding presentation over substance — the core finding the TA wants.
   Example report sentence: "In procedural questions, 62 % of structured-format
   wins were accompanied by format-leakage reasoning, vs 18 % for factual."

3. Answer-length metadata  (len_a / len_b / len_diff)
   Free to compute (local, no API calls). Allows correlating bias direction with
   length asymmetry — e.g., do judges systematically favour longer responses
   regardless of format? Disentangles length bias from pure formatting bias.
"""

import openai
import pandas as pd
import random
import re
import time
import os

random.seed(42)

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GPT_MODEL   = "openai/gpt-4o-mini"
HAIKU_MODEL = "anthropic/claude-haiku-4-5"

OUTPUT_FILE = "judgements.csv"

# ─────────────────────────────────────────
# Format pairs to compare
# ─────────────────────────────────────────
FORMAT_PAIRS = [
    ("answer_prose",  "answer_bullet"),
    ("answer_prose",  "answer_header"),
    ("answer_bullet", "answer_header"),
]

FORMAT_LABEL = {
    "answer_prose":  "prose",
    "answer_bullet": "bullet",
    "answer_header": "header",
}

# ─────────────────────────────────────────
# Format-leakage keyword list
# Words that suggest the judge is reasoning about *presentation*, not content.
# ─────────────────────────────────────────
FORMAT_LEAKAGE_KEYWORDS = {
    # Structural labels
    "bullet", "bullets", "header", "headers", "prose",
    "paragraph", "paragraphs", "list", "lists", "numbered",
    # Presentational adjectives
    "structured", "unstructured", "formatted", "formatting",
    "layout", "visual", "scannable", "skimmable",
    # Readability framing
    "readable", "readability", "easy to read", "easier to read",
    "easy to follow", "easier to follow", "well-organized",
    "organized", "organizes", "organising", "organizing",
    "presented", "presentation", "concise", "conciseness",
    "clear structure", "clearly structured", "broken down",
    "broken into", "sectioned",
}


def has_format_leakage(reason: str) -> bool:
    """Return True if the reason text contains any format-leakage keyword."""
    if not reason:
        return False
    lower = reason.lower()
    return any(kw in lower for kw in FORMAT_LEAKAGE_KEYWORDS)


def word_count(text: str) -> int:
    return len(str(text).split())


# ─────────────────────────────────────────
# Judge prompts
# ─────────────────────────────────────────
JUDGE_SYSTEM = """You are an impartial and careful evaluator.

You will be given a question and two answers (Answer A and Answer B). Both answers contain the same information but may differ in formatting.

Your task is to decide which answer is better for a reader.

Evaluation criteria:
- Clarity: How easy is the answer to understand?
- Readability: Is the structure helpful and easy to follow?
- Organization: Is the information presented in a well-organized way?

Important constraints:
- The two answers are guaranteed to be factually equivalent in content
- Do NOT judge based on correctness, completeness, or factual accuracy
- Focus only on presentation quality and user experience
- Do not assume that any format (e.g., bullet points or headers) is inherently better
- Make your decision based only on how effectively the format supports understanding

Respond using EXACTLY this two-line format, nothing else:
VERDICT: [[A]] or [[B]]
REASON: <two to three sentences explaining what content difference drove your decision>"""

JUDGE_USER_TEMPLATE = """Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Evaluate the responses and reply with VERDICT and REASON."""


# ─────────────────────────────────────────
# Single unified API client (both models via OpenRouter)
# ─────────────────────────────────────────
client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


# ─────────────────────────────────────────
# Parse verdict + reason from raw output
# ─────────────────────────────────────────
def _parse_output(text: str) -> tuple[str, str]:
    """
    Extract verdict and reason from the model's two-line response.
    Returns (verdict, reason) where verdict in {A, B, UNKNOWN}.
    """
    verdict = "UNKNOWN"
    reason  = ""

    # Primary: look for VERDICT: [[A]] or [[B]]
    m = re.search(r"VERDICT\s*:\s*\[\[(A|B)\]\]", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
    else:
        # Fallback 1: any [[A]] or [[B]] token
        m2 = re.search(r"\[\[(A|B)\]\]", text, re.IGNORECASE)
        if m2:
            verdict = m2.group(1).upper()
        else:
            # Fallback 2: bare A or B on its own
            for token in ("A", "B"):
                if re.search(rf"\b{token}\b", text.upper()):
                    verdict = token
                    break

    # Reason: everything after "REASON:"
    m_r = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m_r:
        reason = m_r.group(1).strip().split("\n")[0].strip()

    return verdict, reason


# ─────────────────────────────────────────
# Single API call helpers
# ─────────────────────────────────────────
def call_gpt(question: str, response_a: str, response_b: str) -> tuple[str, str, str]:
    """Returns (verdict, reason, raw_response)."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question, response_a=response_a, response_b=response_b
    )
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    raw = resp.choices[0].message.content.strip()
    verdict, reason = _parse_output(raw)
    return verdict, reason, raw


def call_haiku(question: str, response_a: str, response_b: str) -> tuple[str, str, str]:
    """Returns (verdict, reason, raw_response)."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question, response_a=response_a, response_b=response_b
    )
    resp = client.chat.completions.create(
        model=HAIKU_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    raw = resp.choices[0].message.content.strip()
    verdict, reason = _parse_output(raw)
    return verdict, reason, raw


JUDGE_CALLABLES = {
    "gpt4o_mini":   call_gpt,
    "claude_haiku": call_haiku,
}


# ─────────────────────────────────────────
# Swap-based evaluation for one (question, pair, judge)
# ─────────────────────────────────────────
def evaluate_pair(
    question: str,
    text_a: str,
    text_b: str,
    judge_name: str,
    retries: int = 3,
) -> dict:
    """
    Call the judge twice with swapped orderings.
    TIE is not a valid verdict — the judge is forced to pick A or B.

    Swap resolution:
      - Both orderings agree → final = that winner
      - Disagree (position bias present) → final = random.choice([v_ab, v_ba])
      - Either side is UNKNOWN/ERROR → final = the valid one, or ERROR if both fail
    """
    call_fn = JUDGE_CALLABLES[judge_name]

    def _call_with_retry(qa, qb):
        for attempt in range(retries):
            try:
                return call_fn(question, qa, qb)
            except Exception as e:
                print(f"    [{judge_name}] ERROR attempt {attempt+1}: {e}")
                time.sleep(5 * (attempt + 1))
        return "ERROR", "", "ERROR"

    # ── Ordering 1: A shown first ──
    v_ab, reason_ab, raw_ab = _call_with_retry(text_a, text_b)
    time.sleep(0.4)

    # ── Ordering 2: B shown first — flip verdict back to A/B perspective ──
    v_ba_raw, reason_ba, raw_ba = _call_with_retry(text_b, text_a)
    if   v_ba_raw == "A": v_ba = "B"      # model said "first" (=B) is better
    elif v_ba_raw == "B": v_ba = "A"      # model said "second" (=A) is better
    else:                 v_ba = v_ba_raw  # UNKNOWN or ERROR — passes through
    time.sleep(0.4)

    # ── Resolve ──
    # Both valid and agree → take the consensus
    # Both valid but disagree → position bias, randomly break the tie
    # One side errored → take the valid one
    if v_ab in ("A", "B") and v_ba in ("A", "B"):
        swap_consistent = (v_ab == v_ba)
        final_winner    = v_ab if swap_consistent else random.choice([v_ab, v_ba])
    elif v_ab in ("A", "B"):
        swap_consistent = False
        final_winner    = v_ab
    elif v_ba in ("A", "B"):
        swap_consistent = False
        final_winner    = v_ba
    else:
        swap_consistent = False
        final_winner    = "ERROR"

    # ── Format-leakage flags ──
    leak_ab = has_format_leakage(reason_ab)
    leak_ba = has_format_leakage(reason_ba)

    return {
        "winner_ab":           v_ab,
        "winner_ba":           v_ba,
        "final_winner":        final_winner,
        "reason_ab":           reason_ab,
        "reason_ba":           reason_ba,
        "format_leakage_ab":   leak_ab,
        "format_leakage_ba":   leak_ba,
        "any_format_leakage":  leak_ab or leak_ba,
        "swap_consistent":     swap_consistent,
        "raw_ab":              raw_ab,
        "raw_ba":              raw_ba,
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(N_test: int | None = None):
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "answers_all_180.csv"), index_col="id")
    print(f"Loaded {len(df)} questions.\n")

    if N_test is not None:
        df = df.iloc[:N_test]
        print(f"Test mode: running on first {N_test} question(s).\n")

    # ── Load or init output file ──
    try:
        results_df = pd.read_csv(OUTPUT_FILE)
        print(f"Resuming: {len(results_df)} judgements already recorded.")
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "id", "question", "our_category",
            "format_a", "format_b",
            "len_a", "len_b", "len_diff",
            "judge",
            "winner_ab", "winner_ba", "final_winner",
            "reason_ab", "reason_ba",
            "format_leakage_ab", "format_leakage_ba", "any_format_leakage",
            "swap_consistent",
            "raw_ab", "raw_ba",
        ])

    done_keys = set(
        zip(results_df["id"], results_df["format_a"],
            results_df["format_b"], results_df["judge"])
    )

    total_calls = len(df) * len(FORMAT_PAIRS) * len(JUDGE_CALLABLES)
    print(f"Progress: {len(results_df)}/{total_calls} judgement rows\n")

    new_rows = []

    for idx, row in df.iterrows():
        question = row["question"]
        category = row["our_category"]

        for col_a, col_b in FORMAT_PAIRS:
            fmt_a  = FORMAT_LABEL[col_a]
            fmt_b  = FORMAT_LABEL[col_b]
            text_a = str(row[col_a])
            text_b = str(row[col_b])

            # ── Length metadata (free — local computation) ──
            len_a    = word_count(text_a)
            len_b    = word_count(text_b)
            len_diff = len_a - len_b   # positive → A is longer

            for judge_name in JUDGE_CALLABLES:
                key = (idx, fmt_a, fmt_b, judge_name)
                if key in done_keys:
                    continue

                print(f"[{idx:2d}] {fmt_a:6s} vs {fmt_b:6s} | {judge_name} ...")
                result = evaluate_pair(question, text_a, text_b, judge_name)

                new_rows.append({
                    "id":           idx,
                    "question":     question,
                    "our_category": category,
                    "format_a":     fmt_a,
                    "format_b":     fmt_b,
                    "len_a":        len_a,
                    "len_b":        len_b,
                    "len_diff":     len_diff,
                    "judge":        judge_name,
                    **result,
                })

                # Save after every row (resume support)
                save_df = pd.concat(
                    [results_df, pd.DataFrame(new_rows)], ignore_index=True
                )
                save_df.to_csv(OUTPUT_FILE, index=False)
                done_keys.add(key)

                print(
                    f"       → final={result['final_winner']} "
                    f"consistent={result['swap_consistent']} "
                    f"leakage={result['any_format_leakage']}\n"
                    f"         reason_ab: {result['reason_ab'][:90]}"
                )

    print(f"\nAll done! {OUTPUT_FILE} has "
          f"{len(results_df) + len(new_rows)} rows.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_test", type=int, default=None,
                        help="Number of questions to judge (default: all)")
    args = parser.parse_args()
    main(N_test=args.N_test)
