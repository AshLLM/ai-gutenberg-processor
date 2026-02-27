# ai-gutenberg-processor

A Jupyter notebook pipeline that takes a Project Gutenberg ebook URL and outputs:

- a metadata JSON sidecar (`metadata/<id>.metadata.json`)
- a clean core text file (`core_txts/<id>_clean.txt`) with Gutenberg boilerplate stripped out

## How it works

```
URL → scrape metadata → fetch plain text → AI boundary detection → extract & save → NLTK demo
```

Boundary detection uses 6 chained GPT calls (map → select → extract, for both the start and end of
the literary text) to find reliable anchor strings, which are then used to slice out the core text.

## Project files

```
ai-gutenberg-processor/
├── ai_gutenberg_cleaner.ipynb   ← main notebook
├── helper.py                    ← metadata scraping, text utilities, OpenAI setup
├── requirements.txt
├── OPENAI_KEY.env.example
├── metadata/                    ← generated metadata JSON files (git-ignored)
└── core_txts/                   ← generated clean text files (git-ignored)
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AshLLM/ai-gutenberg-processor.git
cd ai-gutenberg-processor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
cp OPENAI_KEY.env.example OPENAI_KEY.env
```

Then edit `OPENAI_KEY.env` and replace the placeholder with your key:

```
OPENAI_API_KEY=sk-...
```

`OPENAI_KEY.env` is git-ignored — don't commit it.

## Usage

Open `ai_gutenberg_cleaner.ipynb`, set `gutenberg_url` in the first code cell, and run all cells:

```python
gutenberg_url = "https://www.gutenberg.org/ebooks/84"  # Frankenstein
```

## Notes

- Each run makes 6 OpenAI API calls, so it'll cost a small amount depending on your plan.
- `helper.py` can be imported into other notebooks if you want to reuse the utilities.
- Learning/portfolio project — not production-ready.
