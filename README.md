# ST5230 Group Project — Formatting Bias in LLM-as-a-Judge

## Repository Structure
ST5230_Group_Project/
├── data/
│   ├── questions.csv          # 60 questions across 3 categories
│   └── responses/             # Generated and reformatted answers
├── prompts/
│   ├── generation_prompt.txt  # Prompt for generating gold answers
│   ├── reformat_prompt.txt    # Prompt for reformatting answers
│   └── judge_prompt.txt       # Pairwise judging prompt
├── scripts/
│   ├── generate_answers.py    # Answer generation via API
│   ├── reformat_answers.py    # Reformatting via API
│   └── run_judge.py           # Pairwise evaluation via API
├── results/
│   └── judgments.csv          # Raw judge outputs
├── analysis/
│   └── analysis.ipynb         # Statistics and figures
├── requirements.txt
├── .env.example
└── README.md

## Environment Setup
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and add your OpenRouter API key
- Random seeds are fixed in all scripts and reported in the paper

## How to Reproduce
1. Clone the repo
2. Set up environment as above
3. Run `generate_answers.py` to produce gold answers
4. Run `reformat_answers.py` to produce formatted variants
5. Run `run_judge.py` to collect pairwise judgments
6. Open `analysis.ipynb` to reproduce all statistics and figures
