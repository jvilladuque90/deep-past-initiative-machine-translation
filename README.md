# Akkadian → English Machine Translation (Deep Past Initiative)

> Fine-tuning an 11-billion-parameter language model with **QLoRA** to translate **Old Assyrian Akkadian** (cuneiform transliteration) into English — a low-resource, ancient-language translation task — on a single consumer-grade 24 GB GPU.

---

## The problem

Akkadian is an extinct Semitic language written in cuneiform. Hundreds of thousands of Old Assyrian clay tablets (trade letters, contracts, accounts) survive, but only a small fraction are translated, and doing so by hand requires rare expertise. This makes it a textbook **low-resource machine-translation** problem:

- **Tiny, noisy parallel corpus.** Few aligned Akkadian↔English pairs exist, and the source is *transliteration* full of editorial marks (brackets for breaks, `<gap>`, half-brackets for uncertain signs, Sumerograms written in capitals).
- **Granularity mismatch.** Training translations are **document-level**, but the test set is **sentence-level** — so pairs must be aligned/segmented before training.
- **Specialized vocabulary.** Logograms/Sumerograms (e.g. `KÙ.BABBAR` = silver, `ANŠE` = donkey) and syllabic spellings need to be learned, not just memorized.

The competition metric is the **geometric mean of BLEU and chrF++** (SacreBLEU): `score = √(BLEU × chrF++)`.

## The approach

A single notebook ([`translator_train.ipynb`](translator_train.ipynb)) implements a **dual-environment pipeline**: train on Colab, run inference on Kaggle.

### 1. Model: flan-t5-xxl (11B) + QLoRA — why quantization
`google/flan-t5-xxl` is an instruction-tuned encoder-decoder with strong language understanding, but 11B parameters in fp16 need ~22 GB *just for weights* — too large to fine-tune on a single 24 GB GPU. To fit:

- **4-bit NF4 quantization** (`bitsandbytes`) shrinks the frozen base model to ~5.5 GB VRAM, with double (nested) quantization for extra savings.
- **LoRA adapters** (`peft`, rank 16, on the T5 `q`/`v` attention projections) make only ~0.x% of parameters trainable (~20 MB) — this is the "QLoRA" recipe: train low-rank adapters on top of a 4-bit-quantized base.
- **Gradient checkpointing** + **8-bit paged AdamW** keep the optimizer/activation memory low enough to train on a **Colab L4 (24 GB)**.

### 2. Multi-source data pipeline (≈7 sources → aligned training pairs)
The notebook fuses several inputs into one tagged pair list `(source, target, task_type)`:

| Source | Role |
|---|---|
| `train.csv` | Document-level transliteration ↔ translation |
| `Sentences_Oare_FirstWord_LinNum.csv` | Sentence boundaries (first-word positions) used to **segment** documents into sentence-aligned pairs |
| `OA_Lexicon_eBL.csv` | Form → lexeme lookup (word-level pairs) |
| `eBL_Dictionary.csv` | Lexeme → definition (word-level pairs) |
| `published_texts.csv` | Additional published texts |
| `test.csv` | Held-out sentence-level inputs for submission |
| Built-in `SUMEROGRAM_MAP` | Curated logogram → English gloss table |

Preprocessing normalizes transliteration (strips editorial marks, maps `Ḫ/ḫ→H/h`, collapses breaks to `<gap>`/`<big_gap>`). Sentence alignment uses word-position overlap between `Sentences_Oare` and `train.csv`; long documents are chunked. Each pair is tagged `[WORD]`, `[SENT]`, or `[DOC]` so the model learns the task from the prompt.

### 3. Three-phase training curriculum
A curriculum that walks the model from vocabulary to fluent sentence translation:

1. **Phase 1 — Vocabulary pre-training** (`syllable/logogram → meaning`): train on short `[WORD]` pairs so the model internalizes Akkadian morphology and Sumerograms first (5 epochs, LR 2e-4).
2. **Phase 2 — Sentence translation**: train on `[SENT]`/`[DOC]` pairs with upsampling and **data augmentation** (syllable dropout, shuffling), dedup, label smoothing (15 epochs, LR 5e-5).
3. **Phase 3 — Multi-task fine-tuning with metric-aware eval**: generation-based evaluation that directly optimizes the competition metric (`√(BLEU × chrF++)`), beam search, early stopping on `geo_mean` (10 epochs, LR 1e-5).

Each phase checkpoints to `OUTPUT_DIR` and **resumes from the latest checkpoint** — important for surviving Colab disconnects.

### 4. Merge → inference
After training, LoRA adapters are **merged** into the base model (`merge_and_unload`) and saved. For submission the merged model is loaded in **8-bit** with `device_map="auto"`, splitting across **2× T4 (Kaggle, ~32 GB total)** for beam-search generation, then post-processed into `submission.csv`.

## Tech stack

`PyTorch` · `Hugging Face Transformers / Datasets / PEFT / Accelerate` · `bitsandbytes` (4-bit/8-bit) · `SacreBLEU` · `scikit-learn` · `kagglehub` · `pandas` / `numpy` · `matplotlib`

## Setup & run

> **Hardware:** training needs a CUDA GPU with ≥24 GB VRAM (Colab L4). Inference as configured assumes 2× T4 (Kaggle). `bitsandbytes` quantization is CUDA-only.

```bash
git clone https://github.com/jvilladuque90/deep-past-initiative-machine-translation.git
cd deep-past-initiative-machine-translation
pip install -r requirements.txt

# Optional: point the notebook at your data/output dirs
cp .env.example .env        # edit DATA_DIR / OUTPUT_DIR
```

Then open `translator_train.ipynb`. The first code cell **auto-detects** Kaggle / Colab / Local and sets `DATA_DIR`/`OUTPUT_DIR` accordingly (override via `.env`). Run cells top to bottom:

- **On Colab** → runs the 3-phase QLoRA training, merges LoRA, saves `final_merged/`.
- **On Kaggle** (set `KAGGLE = True` in the setup cell) → skips training, loads the pre-merged model in 8-bit, and writes `submission.csv`.

### Data

The CSV datasets are **not included** in this repository — they are competition / third-party–licensed material (Kaggle "Deep Past Initiative" competition data and the **electronic Babylonian Library (eBL)** lexicon & dictionary). Obtain them from their original sources and place them in `DATA_DIR`. They are git-ignored by default; please respect their licenses and do not redistribute.

## Results / Limitations / Next steps

### Results
- **Validation `√(BLEU × chrF++)`: _TODO — fill in from your Phase 3 best checkpoint_** (`eval_geo_mean` printed by the Training Curves cell, with the component `eval_bleu` / `eval_chrf`).

*No metrics are claimed here until they're filled in from an actual run.*

### Limitations
- **Very low-resource:** the parallel corpus is small and noisy; sentence alignment is heuristic (word-position overlap + chunking), so some training targets are approximate.
- **Document→sentence mismatch:** training is largely document-level while evaluation is sentence-level, an inherent source of error.
- **Compute-bound choices:** 4-bit/8-bit quantization and tiny batch sizes (down to 1 with grad-accum) are memory-driven and may cost some quality versus full-precision training.
- **Environment coupling:** the pipeline is tailored to Colab (train) + Kaggle (infer); the `final_merged` model must be moved between them manually.
- A `MINE` (data-mining) task tag is referenced in the pipeline but no mining source is currently implemented.

### Next steps
- Fill in and track the BLEU/chrF++ score; add a small qualitative sample table of translations.
- Improve sentence alignment (e.g. length-ratio or embedding-based alignment instead of word-position heuristics).
- Try larger LoRA rank / more target modules, and back-translation or monolingual pre-training on `published_texts`.
- Refactor the notebook into a small package (`src/`) with a CLI for reproducible runs (see structure notes).

---

*Educational / research project for the Deep Past Initiative Akkadian translation challenge.*
