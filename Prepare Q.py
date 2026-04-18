from datasets import load_dataset
import pandas as pd
import random

random.seed(42)

# ─────────────────────────────────────────
# 1. TruthfulQA (for factual questions)
# ─────────────────────────────────────────
# TruthfulQA has two configs: 'generation' and 'multiple_choice'
# We use 'generation' which contains open-ended questions
ds_truthful = load_dataset("truthful_qa", "generation", split="validation")

# Inspect the fields
print(ds_truthful[0])
# Key fields: 'question', 'best_answer', 'correct_answers', 'category'

# Convert to DataFrame, keep only relevant columns
df_truthful = pd.DataFrame({
    "question": ds_truthful["question"],
    "category_original": ds_truthful["category"],
})

# Randomly sample 20 questions for the factual category
factual_sample = df_truthful.sample(n=20, random_state=42).reset_index(drop=True)
factual_sample["our_category"] = "factual"
print(f"Factual question count: {len(factual_sample)}")


# ─────────────────────────────────────────
# 2. CommonsenseQA (for opinion questions)
# ─────────────────────────────────────────
ds_commonsense = load_dataset("commonsense_qa", split="train")

# Inspect the fields
print(ds_commonsense[0])
# Key fields: 'id', 'question', 'question_concept', 'choices', 'answerKey'

df_commonsense = pd.DataFrame({
    "question": ds_commonsense["question"],
    "question_concept": ds_commonsense["question_concept"],
})

# Randomly sample 20 questions for the opinion category
opinion_sample = df_commonsense.sample(n=20, random_state=42).reset_index(drop=True)
opinion_sample["our_category"] = "opinion"
print(f"Opinion question count: {len(opinion_sample)}")


# ─────────────────────────────────────────
# 3. Procedural category (manually written)
# ─────────────────────────────────────────
procedural_questions = [
    # Daily life
    "How do you make a cup of pour-over coffee?",
    "How do you do laundry properly, including sorting and washing?",
    "How do you cook a basic omelette?",
    "How do you change a flat tyre on a car?",
    "How do you create and stick to a personal monthly budget?",
    "How do you properly clean and maintain a kitchen knife?",
    "How do you pack efficiently for a one-week trip?",

    # Technical operations
    "How do you set up a basic Python virtual environment?",
    "How do you perform a clean installation of an operating system on a laptop?",
    "How do you back up your smartphone data before switching to a new phone?",
    "How do you set up two-factor authentication on an online account?",
    "How do you troubleshoot a slow internet connection at home?",
    "How do you create a pivot table in a spreadsheet application?",
    "How do you compress and send a large folder of files to someone?",

    # Academic / workplace
    "How should you prepare for a job interview?",
    "How do you write an effective cover letter for a job application?",
    "How do you take effective notes during a lecture or meeting?",
    "How do you plan and write a research essay from scratch?",
    "How do you give constructive feedback to a colleague?",
    "How do you set up a productive daily study routine?",
]

procedural_sample = pd.DataFrame({
    "question": procedural_questions,
    "our_category": "procedural"
})


# ─────────────────────────────────────────
# 4. Merge and save
# ─────────────────────────────────────────
all_questions = pd.concat([
    factual_sample[["question", "our_category"]],
    opinion_sample[["question", "our_category"]],
    procedural_sample[["question", "our_category"]],
], ignore_index=True)

all_questions.index.name = "id"
all_questions.to_csv("questions_60.csv", index=True)
print(all_questions)
print(f"\nTotal question count: {len(all_questions)}")