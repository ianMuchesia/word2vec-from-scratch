# word2vec-from-scratch

A pure NumPy implementation of the **Word2Vec Skip-Gram** model with **Negative Sampling**, trained on the IMDB movie-review dataset. No PyTorch, no TensorFlow — just math and arrays.

---

## Overview

Word2Vec is a shallow neural network that learns dense vector representations (embeddings) of words by predicting surrounding context words. This project builds that algorithm from the ground up, implementing every piece by hand:

- Custom tokenizer with vocabulary capping
- Unigram-distribution negative sampling
- Skip-gram forward pass with sigmoid activation
- Backpropagation and gradient updates
- Embedding persistence and evaluation utilities

---

## Project Structure

```
word2vec-from-scratch/
├── data/
│   ├── IMDB Dataset.csv       # Raw training data (50k reviews)
│   └── corpus.txt             # Encoded integer corpus (generated)
├── notebooks/
│   └── embedding_evaluation.ipynb   # Cosine similarity & analogy tests
├── src/
│   ├── tokenizer.py           # Text cleaning, vocab building, encoding
│   ├── prepare_data.py        # CSV → encoded corpus pipeline
│   ├── negative_sampling.py   # Unigram table + negative sampler
│   └── skipgram.py            # Model, training loop, save utilities
└── README.md
```

---

## How It Works

### 1. Tokenization (`tokenizer.py`)
Cleans raw text (lowercase, strip punctuation), builds a frequency-capped vocabulary of up to 10,000 words, and encodes sentences to integer ID sequences. Unknown words fall back to a special `<UNK>` token.

### 2. Data Preparation (`prepare_data.py`)
Reads the IMDB CSV, runs all 50,000 reviews through the tokenizer, and writes a single flat integer corpus to `data/corpus.txt` for efficient training.

### 3. Negative Sampling (`negative_sampling.py`)
Builds a large unigram table (100M slots) where each word's presence is proportional to its frequency raised to the power of **0.75** — the same smoothing used in the original Word2Vec paper — then samples negative words efficiently from it.

### 4. Skip-Gram Model (`skipgram.py`)
Two embedding matrices `W1` (input) and `W2` (output), each of shape `(vocab_size, embedding_dim)`, initialised with small random values. For each `(target, context)` pair the model:
- Looks up the target vector from `W1` and context vector from `W2`
- Computes a dot-product score and passes it through sigmoid
- Applies the **Negative Sampling loss**: maximise $\log\sigma(v_t \cdot v_c)$ for positive pairs, minimise $\log\sigma(v_t \cdot v_n)$ for negatives
- Accumulates gradients across all negative samples before updating `W1`

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/ianMuchesia/word2vec-from-scratch.git
cd word2vec-from-scratch
python -m venv venv && source venv/bin/activate
pip install numpy pandas

# 2. Build the corpus
python -m src.prepare_data

# 3. Train (see skipgram.py for full training script)
python -m src.skipgram
```

---

## Status

> 🚧 Work in progress — embedding evaluation notebook is actively being built.

---

## References

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)
- Mikolov et al., *Distributed Representations of Words and Phrases* (2013)
