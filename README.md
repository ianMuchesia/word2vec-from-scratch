# word2vec-from-scratch
Minimal Word2Vec (Skip-Gram + Negative Sampling) built from scratch with NumPy.

## What this project includes
- Text cleaning and tokenization
- Vocabulary building with `<PAD>` and `<UNK>`
- IMDB review preprocessing into integer corpus IDs
- Unigram-table negative sampling (`p(w)^0.75`)
- Skip-Gram training loop with manual gradient updates
- Basic embedding evaluation in a notebook

## Project layout
```text
src/tokenizer.py
src/prepare_data.py
src/negative_sampling.py
src/skipgram.py
notebooks/embedding_evaluation.ipynb
data/
```

## Quick start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_data
```

Train from notebook or script by importing:
- `SkipGramModel`, `train_model`
- `build_unigram_table`

## Notes
- This repo is educational and intentionally simple.
- Large data artifacts are excluded via `.gitignore`.

## References
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*
- Mikolov et al. (2013), *Distributed Representations of Words and Phrases*
