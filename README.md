# ai-gutenberg-processor

A Jupyter notebook pipeline that takes a Project Gutenberg ebook URL and outputs:

- a metadata JSON sidecar (`metadata/<id>.metadata.json`)
- a clean core text file (`core_txts/<id>_clean.txt`) with Gutenberg boilerplate stripped out

Raw Project Gutenberg texts contain inconsistent boilerplate headers and footers that interfere with downstream text processing. Manual cleaning is tedious and error-prone; regex-based approaches are brittle and break across editions. This pipeline offers a more robust alternative, using chained LLM calls to reliably detect and extract the core literary text — producing clean, ready-to-use output suitable for corpus construction, NLP analysis, or machine learning pipelines.

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

## Sample Output

Running the pipeline on [Alice's Adventures in Wonderland by Lewis Carroll](https://www.gutenberg.org/ebooks/11) produces:

**`metadata/11.metadata.json`**
```json
{
  "title": "Alice's Adventures in Wonderland",
  "author": "Carroll, Lewis",
  "language": "English",
  "publication_date": "1865",
  "ebook_no": "11",
  "subjects": [
    "Fantasy fiction",
    "Children's stories",
    "Imaginary places -- Juvenile fiction",
    "Alice (Fictitious character from Carroll) -- Juvenile fiction"
  ],
  "genre": "Fantasy fiction",
  "source_url": "https://www.gutenberg.org/ebooks/11"
}
```

**`core_txts/11_clean.txt`** (opening lines)
```
CHAPTER I.

Down the Rabbit-Hole

Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into
the book her sister was reading, but it had no pictures or
conversations in it, "and what is the use of a book," thought Alice
"without pictures or conversations?"
```

All Project Gutenberg header and footer boilerplate has been stripped. The core text begins at the first line of the work and ends at the last.

## Notes

- Each run makes 6 OpenAI API calls, so it'll cost a small amount depending on your plan.
- `helper.py` can be imported into other notebooks if you want to reuse the utilities.
- This is a learning/portfolio project for demo purposes — not production ready.
