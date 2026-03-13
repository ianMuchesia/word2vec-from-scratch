# word2vec-from-scratch

A pure NumPy implementation of **Word2Vec Skip-Gram** with **Negative Sampling**, trained on IMDB movie reviews. No deep learning framework, just the core math and arrays.

---

## Overview

This project builds Word2Vec from scratch and keeps the pipeline explicit:

- Custom tokenizer and vocabulary mapping (`<PAD>`, `<UNK>`)
- Corpus preparation from raw IMDB reviews
- Unigram-table negative sampling with $p(w)^{0.75}$
- Skip-Gram training loop with manual gradient updates
- Notebook-based embedding evaluation (neighbors, analogy, TSNE)

---

## Project Structure

```text
word2vec-from-scratch/
├── src/
│   ├── tokenizer.py
│   ├── prepare_data.py
│   ├── negative_sampling.py
│   └── skipgram.py
├── notebooks/
│   └── embedding_evaluation.ipynb
├── data/
│   ├── IMDB Dataset.csv
│   ├── corpus.txt
│   └── embeddings.npy
├── requirements.txt
└── README.md
```

---

## How It Works

### 1) Tokenization (`src/tokenizer.py`)
Lowercases text, removes punctuation, builds a capped vocabulary (10,000 words), and encodes reviews into word IDs.

### 2) Data Preparation (`src/prepare_data.py`)
Reads `data/IMDB Dataset.csv`, tokenizes all reviews, and writes a flattened integer corpus to `data/corpus.txt`.

### 3) Negative Sampling (`src/negative_sampling.py`)
Builds a large unigram table using smoothed frequencies and samples negative word IDs efficiently.

### 4) Training (`src/skipgram.py`)
Initializes `W1` and `W2`, generates target-context pairs with a window, applies sigmoid + negative sampling loss, and saves learned embeddings to `data/embeddings.npy`.

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_data
```

Then open `notebooks/embedding_evaluation.ipynb` and run cells for training + evaluation.

---

## Current Status

This is an active work-in-progress, but the end-to-end path is already in place: preprocess data → build sampling table → train embeddings → evaluate in notebook.

---

## Notes

- `experiments/` and `math-notes/` are currently placeholders.
- Workflow is currently script + notebook (no CLI yet).
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
