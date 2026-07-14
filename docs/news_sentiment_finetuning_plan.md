# Fine-Tuning a News-Sentiment Model (TRNA Substitute) — Plan & Tracker

**Purpose (two birds, one stone):**
1. **Interview prep** — rehearse the exact skills the Stage-2 coding exercise evaluates
   (Hugging Face / PyTorch navigation, fine-tuning intuition, data-first debugging,
   clean code under time pressure).
2. **Paper replication** — produce a *custom* financial-news sentiment model as the
   legitimate substitute for the paper's proprietary **Thomson Reuters News Analytics
   (TRNA)** sentiment feed. Its per-article scores become the weekly sentiment feature
   for the RankNet / ListNet learning-to-rank backtest.

> Guiding principle (straight from the interview brief): **Start simple and iterate.
> A working baseline you improve beats an ambitious approach that never runs.**

---

## How To Use This Doc

- Work top-to-bottom through the **Iterations**. Each is independently runnable.
- Track progress on the **Kanban board** below — move a card's checkbox and cut/paste
  the line between sections (`Backlog → To Do → In Progress → Done`) as you go.
- For interview prep, prioritize **Iteration 0–3** in a throwaway Colab notebook.
- For the paper, the repo integration (**Iteration 4–5**) comes *after* the interview.

---

## Kanban Board

### 🗂️ Backlog (later / nice-to-have)
- [ ] Try LoRA / PEFT fine-tune and compare to full fine-tune (param count, speed, accuracy)
- [ ] Try a larger backbone (`roberta-base`, `deberta-v3-base`) and compare
- [ ] Calibrate output probabilities (reliability curve) before using as a feature
- [ ] Domain-adaptive pretraining (MLM on unlabeled financial headlines) before classification
- [ ] Package the inference path into `src/sentiment_ltr/` with tests

### 📋 To Do
- [ ] **4.1.5a** Pick 5 pilot tickers with rich RavenPack exports (e.g. MSFT, GOOG, AMZN, JPM, XOM) — confirm ≥20k labeled headlines each via `discover_ravenpack_article_files` + `ravenpack_split_summary`
- [ ] **4.1.5b** Run `train_ravenpack(tickers=[...5 tickers...])` — single multi-ticker training call; verify test macro-F1 ≥ AAPL-only baseline
- [ ] **4.1.5c** Per-ticker breakdown — report test macro-F1 for each ticker individually to confirm generalisation isn't driven by one stock
- [ ] **4.1.5d** Scale to all ~500 universe tickers once 5-ticker run validates the pipeline; gate on coverage (skip tickers with < N labeled rows)
- [ ] **0.2** Skim the HF docs you'll touch: `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, `datasets.load_dataset`
- [ ] **2.4** Swap in `ProsusAI/finbert`; compare metrics; read its model card for label order
- [ ] **3.1** Plot class balance + confusion matrix; write down what each error type means
- [ ] **3.2** Debug drill: list "what I'd check if val acc were stuck at chance" (label dtype, mapping, LR, logits shape, leakage)
- [ ] **3.3** Re-implement the **same fine-tune as a raw PyTorch loop** (DataLoader, optimizer, `loss.backward()`) — highest-leverage rep
- [ ] **4.1** Add a headline/text column to the RavenPack (or Refinitiv) pull so raw text is cached
- [ ] **4.2** Batch-score cached news with the fine-tuned model → per-article sentiment
- [ ] **4.3** Aggregate to **weekly per-stock** sentiment (shock + trend), matching the paper's feature design
- [ ] **5.1** Join weekly sentiment to market features; create quartile rank labels
- [ ] **5.2** Feed into RankNet / ListNet; run rolling 2006–2014 backtest

### 🚧 In Progress
- _(move the card you're actively working on here)_

### ✅ Done
- [x] **Iteration 4 — AAPL fine-tune complete** — `train_ravenpack(tickers=["AAPL"])` trained on ~38k headlines (2003–2011), validated on 2012, tested on 2013–2014; test macro-F1 improved from **27.5% → 82.1%** (+54.6 pp); checkpoint saved to `data/models/ravenpack_distilbert_best/`; full before/after comparison in `notebooks/finetune_on_ravenpack.ipynb` and **RavenPack Fine-Tuning** tab in `app.py`.
- [x] **0.1** Env ready — `torch`/`transformers`/`datasets`/`evaluate`/`accelerate`/`ipywidgets` installed into the `sentiment-ltr-paper` conda env (clean `pip check`); pinned in `requirements-finetuning.txt` + `environment.yml`/`environment.lock.yml`; notebook auto-detects and runs on Apple **MPS**
- [x] **1.1** Load Financial PhraseBank via Parquet mirror `atrost/financial_phrasebank` (datasets-v5 safe); inspect schema, labels, samples, class balance (neutral 59.7% / positive 27.9% / negative 12.3%), token-length stats — `notebooks/liquidAI_prep.ipynb`
- [x] **1.1b** Load + inspect tokenizer & `distilbert-base-uncased` head (66.9M params, `768→3`), wire label maps into config, and run a one-batch untrained **forward-pass sanity check** — `notebooks/liquidAI_prep.ipynb`
- [x] **1.2** Tokenize full train/val/test splits (`max_length=128`, fixed-length padding) — `notebooks/liquidAI_prep.ipynb`
- [x] **1.3** First `Trainer` baseline — 1 epoch, `lr=2e-5`, `batch=16`, `eval_strategy=epoch` — **val acc 78.9%**, **test acc 80.6%**, train loss 0.67 (~41s on MPS) — `notebooks/liquidAI_prep.ipynb`
- [x] **2.1** Document split choice — use `atrost/financial_phrasebank` pre-defined train/val/test (3100/776/970, `sentences_50agree`; no re-split) — `phrasebank_sentiment.py`, notebook §5, `docs/financial_phrasebank.md`
- [x] **2.2** Add **macro-F1** to `compute_metrics` (accuracy + `eval_f1`) — `phrasebank_sentiment.build_compute_metrics()`, notebook, Sentiment Lab metrics
- [x] **2.3** 3-epoch training with `load_best_model_at_end=True`, `metric_for_best_model=f1` — **val F1 83.5% / acc 85.2%**, **test F1 82.1% / acc 83.9%** (~125s on MPS); checkpoint `data/models/phrasebank_distilbert_best/`
- [x] **Notebook executes end-to-end** with zero error outputs (verified via `nbconvert --execute`)
- [x] **Reproducibility committed** — README "Optional extras" section + dataset-mirror caveat; pushed to GitHub (`e399e7e`)

---

## Step-by-Step Plan (the Iterations)

### Iteration 0 — Environment & doc reconnaissance (~15 min)
**Goal:** A runnable notebook and a mental map of the APIs.
- Fresh Colab with GPU. Keep it **separate from this repo** during interview prep so you
  optimize for reps, not plumbing.
- Install: `transformers datasets evaluate accelerate scikit-learn`.
- Open (don't memorize) the docs for `Trainer`, `TrainingArguments`,
  `AutoModelForSequenceClassification`, `load_dataset`. Practice *finding* answers — the
  interview explicitly rewards "I don't know this API, here's how I'd look it up."

### Iteration 1 — Minimal working baseline (~30 min)
**Goal:** *Anything* trains end-to-end and prints a number. Do **not** optimize.
- Data: `load_dataset("financial_phrasebank", "sentences_50agree")` — standard finance
  benchmark, 3 labels (negative / neutral / positive).
- Tokenize, build `Trainer`, train **1 epoch**, print accuracy.
- Success = no crashes + a baseline accuracy you can improve from.

### Iteration 2 — Make it a real workflow (~30 min)
**Goal:** Trustworthy evaluation + model comparison.
- Stratified train/val/test split; fixed seed.
- `compute_metrics`: accuracy **and macro-F1** (neutral class dominates → accuracy lies).
- `TrainingArguments` with per-epoch eval and `load_best_model_at_end`.
- Defensible hyperparameters: `lr≈2e-5`, `batch_size=16/32`, `epochs=2–4`, watch overfit.
- Swap `distilbert` → `ProsusAI/finbert`; compare. Mind each model's **label ordering**.

### Iteration 3 — Debugging muscle + raw PyTorch (~45 min)
**Goal:** Show the "check data before model" reflex and that you can drop below `Trainer`.
- Confusion matrix + class-balance plot; interpret the errors.
- Write the diagnostic checklist (stuck loss → data/label/LR/shape, not architecture).
- **Re-implement the fine-tune as a plain PyTorch loop** (`DataLoader`,
  `AdamW`, `zero_grad → forward → loss → backward → step`, manual eval). This is the
  single best drill since the brief names PyTorch *and* HF.

### Iteration 4 — Wire sentiment into the repo (post-interview)
**Goal:** Turn the model into the TRNA-substitute feature.
- Cache raw news text (today only scores are pulled; `live_data.py` ~line 594 omits
  `headline`/`event_text` to save space — add them back for a text corpus).
- Batch-score cached headlines → per-article sentiment probabilities.
- Aggregate to weekly per-stock sentiment **shock** and **trend** (paper's design).

**Status:** AAPL fine-tune ✅ complete (see progress log). Multi-ticker generalisation is Iteration 4.1.5 below.

---

### Iteration 4.1.5 — Multi-ticker generalisation (5-ticker pilot → full universe)

**Goal:** Confirm that a single RavenPack-fine-tuned checkpoint generalises across stocks
before scaling to all ~500 universe tickers. The AAPL-only run is a proof-of-concept;
a robust TRNA substitute needs to score *any* stock, not just Apple.

#### Why this step matters
- A model trained only on AAPL headlines may learn Apple-specific vocabulary (product
  launches, Tim Cook quotes, iPhone cycle language) that doesn't transfer.
- A multi-ticker checkpoint trained on diverse stocks is far more likely to generalise
  to unseen tickers in the full universe.
- Validating on 5 tickers first is cheap insurance before a multi-hour full-universe run.

#### Pilot ticker selection criteria
Pick 5 tickers that are:
1. **Coverage-rich** — ≥20k labeled RavenPack headlines each (check with `ravenpack_split_summary`)
2. **Sector-diverse** — at least 3 different GICS sectors to avoid sector overfitting
3. **Already in the batch cache** — avoid needing a fresh WRDS pull

Suggested pilot set (adjust based on actual coverage):

| Ticker | Sector | Rationale |
| --- | --- | --- |
| MSFT | Technology | Large-cap tech, different product cycle than AAPL |
| JPM | Financials | Macro-sensitive; earnings/credit news differs structurally |
| XOM | Energy | Commodity-driven; very different headline vocabulary |
| JNJ | Healthcare | Regulatory/FDA news cadence |
| WMT | Consumer Staples | Retail; volume-driven, low headline volatility |

#### Implementation steps

**Step 1 — Confirm coverage**
```python
from sentiment_ltr.models.ravenpack_sentiment import (
    discover_ravenpack_article_files, load_ravenpack_labeled_frame, ravenpack_split_summary
)
PILOT_TICKERS = ["MSFT", "JPM", "XOM", "JNJ", "WMT"]
for t in PILOT_TICKERS:
    paths = discover_ravenpack_article_files([t])
    if paths:
        labeled = load_ravenpack_labeled_frame([t])
        print(t, ravenpack_split_summary(labeled))
    else:
        print(t, "— no export found")
```

**Step 2 — Train on pilot set (including AAPL)**
```python
from sentiment_ltr.models.ravenpack_sentiment import train_ravenpack
metrics = train_ravenpack(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT"],
    init_from_phrasebank=True,   # warm start
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    seed=42,
)
```
Checkpoint saved to `data/models/ravenpack_distilbert_best/` (same path — overwrites AAPL-only run).

**Step 3 — Per-ticker evaluation**
After training, re-evaluate on each ticker's test split individually to confirm the
checkpoint doesn't sacrifice single-ticker performance for average performance:
```python
from sentiment_ltr.models.ravenpack_sentiment import evaluate_phrasebank_baseline_on_ravenpack
from sentiment_ltr.models.phrasebank_sentiment import resolve_model_dir
for t in ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT"]:
    result = evaluate_phrasebank_baseline_on_ravenpack(
        [t], model_dir=DEFAULT_RAVENPACK_MODEL_DIR, eval_split="test"
    )
    print(f"{t:6s}  macro-F1={result['macro_f1']:.1%}  acc={result['accuracy']:.1%}")
```

**Step 4 — Full universe (gate on coverage)**
Once the 5-ticker pilot validates:
- Loop over all tickers in `data/raw/news/ravenpack/` with `discover_ravenpack_article_files()`
- Skip tickers with < 5,000 labeled rows (insufficient training signal)
- Train a single checkpoint on the full pooled dataset
- Store per-ticker test metrics in `metrics.json` under `per_ticker_test`

#### Success criteria
- [ ] All 5 pilot tickers have test macro-F1 ≥ 60% (well above the 27.5% zero-shot baseline)
- [ ] No single ticker's F1 collapses below 50% (would indicate negative transfer)
- [ ] Checkpoint loads and scores an unseen ticker's headlines without error

#### Data leakage check
The same time-based split (`TRAIN_END = 2011-12-31`, `TEST_START = 2013-01-01`) is applied
independently per ticker within `ravenpack_to_hf_dataset()`. There is no cross-ticker
leakage risk — each ticker's articles are entirely independent time series.

---
**Goal:** Close the loop to the paper.
- Join weekly sentiment + market features; build quartile rank labels.
- Train RankNet / ListNet; rolling 2006–2014 backtest; compare to paper Table 3.

---

## Skill → Rubric Mapping (what to say out loud)

| Interview criterion | Practice it here by… |
| --- | --- |
| **Framework navigation** | Doing the loop once with `Trainer`, once in raw PyTorch; reading model cards to pick abstractions |
| **Fine-tuning intuition** | Justifying lr / batch / epochs / `max_length` / freeze vs full vs LoRA; why pretrained + small LR works |
| **Debugging approach** | Data first: class balance, label mapping, truncation, split leakage, loss-not-decreasing triage |
| **Code quality** | Named helpers (`tokenize_fn`, `compute_metrics`), no dead cells, clarity over cleverness |

---

## Quick Reference — Defensible Defaults

- **Backbone:** `distilbert-base-uncased` (fast baseline) → `ProsusAI/finbert` (domain).
- **Learning rate:** `2e-5` full fine-tune; `1e-4`+ if using LoRA adapters.
- **Batch size:** 16–32 (drop if OOM; or use grad accumulation).
- **Epochs:** 2–4 on small data; rely on best-val checkpoint, not the last epoch.
- **Metric:** macro-F1 primary (imbalanced), accuracy secondary.
- **Repro:** set seeds (`set_seed`), log config, fixed splits.

## Datasets to Know
- **Financial PhraseBank** — labeled finance sentences (primary practice set).
  - ⚠️ The canonical `financial_phrasebank` repo is **script-based** and no longer loads on
    `datasets` v4/v5 (`RuntimeError: Dataset scripts are no longer supported`;
    `trust_remote_code` removed). Use a **Parquet mirror** instead:
    - `atrost/financial_phrasebank` — has `ClassLabel` names + train/val/test splits
      (3100/776/970 = 4846). **Used in `notebooks/liquidAI_prep.ipynb`.**
    - `warwickai/financial_phrasebank_mirror` — single train split, int labels (fallback).
- `zeroshot/twitter-financial-news-sentiment` — alt finance set if you want variety.
- Your own cached RavenPack / Refinitiv headlines — the real target corpus (Iteration 4).

## Environment (verified compatible with `sentiment-ltr-paper`)
Installed and `pip check`-clean alongside the existing app deps (numpy 2.4, pandas 2.2,
pyarrow 21, scikit-learn 1.9, fsspec 2026.4 — **none downgraded**):

| Package | Version |
| --- | --- |
| torch | 2.12.1 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| evaluate | 0.4.6 |
| accelerate | 1.14.0 |
| tokenizers | 0.22.2 |

Install: `pip install torch transformers datasets evaluate accelerate` (scikit-learn already present).
The setup notebook runs end-to-end locally on Apple **MPS**; Colab GPU also works.
(Cosmetic: `pip install ipywidgets` silences a tqdm progress-bar warning in Jupyter.)

---

## Progress Log

| Date | Milestone |
| --- | --- |
| 2026-06-28 | **Plan + tracker created** (`docs/news_sentiment_finetuning_plan.md`). |
| 2026-06-28 | **Env set up & verified** — fine-tuning stack installed into `sentiment-ltr-paper` (clean `pip check`, no downgrades); pinned in `requirements-finetuning.txt` + `environment.yml`/`environment.lock.yml`; README replication notes added. |
| 2026-06-28 | **Setup notebook built** (`notebooks/liquidAI_prep.ipynb`) — imports/device check, PhraseBank load+inspect (datasets-v5 Parquet mirror), tokenizer+model load+inspect, forward-pass sanity check. Runs end-to-end on MPS with zero errors. |
| 2026-06-28 | **Committed & pushed** to GitHub (`e399e7e`). |
| 2026-06-28 | **First training baseline (1.2 + 1.3)** — full-dataset tokenization + 1-epoch `Trainer` on DistilBERT; **val acc 78.9%**, **test acc 80.6%** (~41s on MPS). |
| 2026-06-29 | **Iteration 2 complete (2.1–2.3)** — macro-F1 in `compute_metrics`, 3 epochs, `load_best_model_at_end` on val F1; **val F1 83.5% / acc 85.2%**, **test F1 82.1% / acc 83.9%** (~125s MPS); saved to `phrasebank_distilbert_best/`. |

| 2026-07-06 | **Iteration 4 AAPL fine-tune complete** — `train_ravenpack(["AAPL"])` test macro-F1 **82.1%** (+54.6 pp vs PhraseBank zero-shot); RavenPack Fine-Tuning tab added to `app.py`; colour-coded mismatch tables in notebook. |
| 2026-07-14 | **Iteration 4.1.5 planned** — multi-ticker generalisation (5-ticker pilot → full universe); section added to this doc. |

**Next up:** **4.1.5a** — confirm RavenPack coverage for MSFT / JPM / XOM / JNJ / WMT; then **4.1.5b** multi-ticker training run.

---

_Last updated: 2026-07-14_
