# Financial PhraseBank — Dataset Reference

The training set behind the **News Sentiment Lab** tab and
`notebooks/liquidAI_prep.ipynb`. This is our **TRNA substitute**: a finance-specific
sentiment classifier trained here later scores our own news corpus (RavenPack /
Refinitiv), in place of the proprietary Thomson Reuters News Analytics feed the
original paper used.

## What it is

Financial PhraseBank is a benchmark dataset of **English financial-news sentences**,
each labeled with **sentence-level sentiment from an investor's point of view**
(`negative` / `neutral` / `positive`). It is sentence-level, not document-level, and
the sentiment is about **likely effect on the stock price**, not general tone.

## Who created it, and why

- **Authors:** Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry
  Takala (Aalto University).
- **Paper:** *Good debt or bad debt: Detecting semantic orientations in economic
  texts*, JASIST 65(4), 2014. [arXiv:1307.5336](https://arxiv.org/abs/1307.5336).
- **Why:** to provide a **human-annotated benchmark** for training and evaluating
  financial sentiment models, because the overall sentiment of a sentence often
  differs from the prior polarity of its individual words (e.g. "cost reduction" is
  positive; "dividend cut" is negative). Generic lexicons miss this.

## What it contains

- ~**4,840 sentences** drawn from English financial news.
- **3 classes:** `negative` (0), `neutral` (1), `positive` (2).
- Released in **four agreement-based subsets** (how strongly annotators agreed on the
  majority label):

| Config | Rule | Sentences |
| --- | --- | ---: |
| `sentences_50agree` | ≥50% annotator agreement | **4,846** ← used here |
| `sentences_66agree` | ≥66% agreement | 4,217 |
| `sentences_75agree` | ≥75% agreement | 3,453 |
| `sentences_allagree` | 100% agreement | 2,264 |

We use **`sentences_50agree`** — the most inclusive (most data, noisiest labels).

### Schema

Two columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `sentence` | string | one financial-news sentence |
| `label` | ClassLabel | `0=negative`, `1=neutral`, `2=positive` |

Sample rows (decoded):

| sentence (truncated) | label |
| --- | --- |
| Pretax profit rose to EUR 0.6 mn from EUR 0.4 mn … | positive |
| As a result of these negotiations the company … | negative |
| The total headcount reduction will be 50 persons … | negative |
| Investment management and investment advisory … | neutral |
| The company reported net sales of 302 mln euro … | neutral |

### Splits (Parquet mirror used here)

The mirror ships pre-defined splits totaling 4,846:

| Split | Rows |
| --- | ---: |
| train | 3,100 |
| validation | 776 |
| test | 970 |

### Class balance (train split)

Heavily imbalanced — neutral dominates, which is why we track **macro-F1**, not just
accuracy:

| Label | Count | % |
| --- | ---: | ---: |
| neutral | 1,852 | 59.7% |
| positive | 866 | 27.9% |
| negative | 382 | 12.3% |

Sentence length: ~23 words mean, ~21 median (short single sentences; `max_length=128`
tokens is ample).

## How the labels were constructed

1. **Annotators:** 16 people with finance backgrounds — 3 researchers + 13 Aalto
   master's students (finance / accounting / economics).
2. **Task:** classify each sentence as positive / neutral / negative **using only the
   information in the sentence**, judged **from an investor's perspective** (would the
   news likely move the stock price up, down, or not at all?). Sentences with no
   economically relevant sentiment are **neutral**.
3. **Overlap:** each sentence received **5–8 independent annotations**.
4. **Gold label:** **majority vote** across annotators.
5. **Agreement:** ~74.9% overall; ~98.7% on the easier positive-vs-negative
   distinction. The four subsets above are thresholds on this agreement.

## Where to find it

- **Original (canonical):**
  [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank)
  — ⚠️ script-based; **no longer loads on `datasets` v4/v5**.
- **Used here (Parquet mirror):**
  [`atrost/financial_phrasebank`](https://huggingface.co/datasets/atrost/financial_phrasebank)
  — `ClassLabel` names + ready-made train/val/test splits.
- **Fallback mirror:**
  [`warwickai/financial_phrasebank_mirror`](https://huggingface.co/datasets/warwickai/financial_phrasebank_mirror)
  — single train split, integer labels.

```python
from datasets import load_dataset
raw = load_dataset("atrost/financial_phrasebank")  # what this repo uses
```

## How we use it in this project

- **Now:** supervised training set for a DistilBERT 3-way sentiment baseline
  (see the Sentiment Lab tab and `notebooks/liquidAI_prep.ipynb`).
- **Later (Iteration 4 of `docs/news_sentiment_finetuning_plan.md`):** the trained
  model batch-scores our cached RavenPack / Refinitiv headlines, then we aggregate to
  weekly per-stock sentiment **shock** and **trend** features for the learning-to-rank
  backtest.

**Caveat — domain shift:** PhraseBank sentences come from curated financial news and
are investor-oriented at the sentence level. Our real corpus (headlines, different
vendors, entity linking across mergers) differs, so monitor performance when applying
the model beyond PhraseBank.

## Read more

- Dataset card: <https://huggingface.co/datasets/takala/financial_phrasebank>
- Original paper (arXiv): <https://arxiv.org/abs/1307.5336>
- Fine-tuning plan: [`docs/news_sentiment_finetuning_plan.md`](news_sentiment_finetuning_plan.md)
- Notebook walkthrough: `notebooks/liquidAI_prep.ipynb`
