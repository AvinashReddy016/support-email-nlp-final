# Support Email NLP

This repository contains a small project that analyzes customer support emails and auto-generates suggested replies.

## Contents

- `data/` - original dataset (CSV)
- `src/main.py` - processing script (heuristics + optional OpenAI)
- `output/processed_emails.csv` - processed output with added columns
- `notebooks/demo.ipynb` - demonstration notebook (open in GitHub or Jupyter)
- `requirements.txt` - Python dependencies
- `docs_screenshot.png` - screenshot provided by the user (for README)

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the processing script:

```bash
python src/main.py
```

3. Inspect results:

- `output/processed_emails.csv` contains `likely_urgency`, `subject_topic`, `sentiment`, `summary`, and `auto_reply` columns.
- `notebooks/demo.ipynb` demonstrates loading the output and showing sample results.

## Notes

- If you want higher-quality summaries/replies, set `OPENAI_API_KEY` in your environment before running. The script will attempt to use OpenAI when available; otherwise it falls back to heuristics.
- This repo is ready to be pushed to GitHub as-is for submission.

---

![screenshot](docs_screenshot.png)
