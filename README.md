# word2vec-from-scratch

Word2Vec (Skip-Gram + Negative Sampling) implemented from scratch with NumPy.

## Current project state (reviewed)
- Core training code exists in `src/` and runs with manual gradients.
- Data preparation is script-based and currently tied to `data/IMDB Dataset.csv`.
- Evaluation is notebook-first in `notebooks/embedding_evaluation.ipynb`.
- `experiments/` and `math-notes/` folders are currently empty placeholders.

## Implemented modules
- `src/tokenizer.py`: lowercasing, punctuation stripping, vocab build (max 10,000), encode/decode with `<PAD>` and `<UNK>`.
- `src/prepare_data.py`: loads IMDB CSV, tokenizes reviews, writes flattened integer corpus to `data/corpus.txt`.
- `src/negative_sampling.py`: builds unigram table using $p(w)^{0.75}$ and samples negatives.
- `src/skipgram.py`: defines `SkipGramModel`, training pair generation, per-step loss/updates, and embedding save.

## Local data observed
- `data/IMDB Dataset.csv` (~64MB)
- `data/corpus.txt` (~37MB)
- `data/embeddings.npy` (~7.7MB)

## Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run pipeline
```bash
# from project root
python -m src.prepare_data
```

Then open `notebooks/embedding_evaluation.ipynb` and run cells to:
1) load corpus IDs, 2) build vocab, 3) build unigram table, 4) train model, 5) evaluate neighbors/analogies/TSNE.

## Known limitations
- No CLI entrypoint yet (workflow is script + notebook).
- No automated tests yet.
- Import paths assume running from project root / notebook root setup.

## Dependencies
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## References
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*
- Mikolov et al. (2013), *Distributed Representations of Words and Phrases*
