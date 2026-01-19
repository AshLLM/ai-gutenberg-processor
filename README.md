# Slicing Text with LLMs

This project demonstrates how to slice and process text using Python and an LLM (OpenAI API), with basic natural language processing using NLTK. It is implemented as a Jupyter Notebook and is intended as a beginner-friendly example.

---

## 📁 Project Structure

```text
.
├── slicing_text_llm.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required NLTK data

This project uses NLTK's tokenizer, which requires an additional download:

```python
import nltk
nltk.download("punkt")
```

You only need to do this once.

---

## OpenAI API Key Setup

This project uses the OpenAI API. **Do not hardcode your API key.**

Create a `.env` file in the project root with the following content:

```env
OPENAI_API_KEY=your_api_key_here
```

The notebook loads this key using `python-dotenv`. The `.env` file is ignored by Git and should never be committed.

---

## Running the Notebook

After completing the setup steps above, launch Jupyter:

```bash
jupyter notebook
```

Then open `slicing_text_llm.ipynb` and run the cells in order.

---

## Notes

* This project is intended for learning and experimentation.
* API usage may incur costs depending on your OpenAI account.
* Make sure your API key remains private.

---

## 📄 License

This project is provided for educational purposes. Add a license if you plan to share or reuse it publicly.
