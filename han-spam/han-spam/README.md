# Email Spam Detection — FT+HAN Practice Implementation

> **Course**: AI in Cybersecurity — Final Project
>
> **Paper**: Email spam detection using hierarchical attention hybrid deep learning method
> Sultan Zavrak, Seyhmus Yilmaz — *Expert Systems With Applications 233 (2023)*

This project is a practice implementation of the FT+HAN architecture described
in the paper above. It reproduces the paper's experimental results using:

- **Architecture**: FastText embeddings + multi-filter CNN + Bidirectional GRU
  + dual attention (word-level and sentence-level) for binary spam/ham classification
- **Datasets**: Enron-Spam and SpamAssassin — the same corpora used in the paper
- **Training**: Google Colab (free T4 GPU) — preprocessing runs locally in Docker
- **Evaluation**: 10-fold stratified cross-validation (same-dataset) and
  cross-dataset evaluation, reproducing Tables 3 & 4 from the paper

---

## Architecture overview

```
Local Docker (CPU)                    Google Colab (T4 GPU)
──────────────────                    ─────────────────────
Preprocess raw emails                 Train 10-fold CV on Enron
  │ clean + tokenize                  Train 10-fold CV on SpamAssassin
  │ FastText embeddings               Cross-dataset evaluation
  │ encode to tensors                 Save checkpoint per fold → Drive
  ▼                                          │
Google Drive ◄─────────────────────────────────┘
  han-spam/
    data/processed/    ← pushed from local
    checkpoints/       ← saved by Colab, pulled to local
    results/           ← saved by Colab, pulled to local
```

---

## Model details (paper Section 4.3)

| Hyperparameter | Value |
|---|---|
| Word embedding | FastText, dim=200, skip-gram |
| CNN filters | [64, 128], kernels [3, 5] |
| BiGRU units | 50 per direction (100 concat) |
| Attention dim | 100 (word-level and sentence-level) |
| Dropout | 0.5 |
| Optimizer | Adam, lr=0.001 |
| Batch size | 64 |
| Max epochs | 20 (early stopping patience=3) |
| CV protocol | 10-fold stratified cross-validation |
| Primary metric | AUC |

---

## Datasets

| Dataset | Ham | Spam | Total | Source |
|---|---|---|---|---|
| Enron-Spam (EN) | 16,545 | 17,171 | 33,716 | [GitHub CSV mirror](https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip) |
| SpamAssassin (SA) | 4,144 | 1,892 | 6,036 | [Apache public corpus](https://spamassassin.apache.org/old/publiccorpus/) |

---

## Requirements

- Docker Desktop ≥ 24 with Compose v2 (must be running before any `docker compose` command)
- `rclone` installed on host — [download here](https://rclone.org/downloads/)
- A Google account (for Drive access — free tier is sufficient)
- ~2 GB free space on Google Drive

---

## For original developers

### One-time setup

**1. Connect Google Drive via rclone**

Run on your Windows host (not inside Docker):

```powershell
rclone config
```

Follow the prompts:
- `n` → new remote → name: `gdrive`
- Storage: `drive` (Google Drive)
- Leave client ID and secret blank → scope: `1` (full access)
- Complete the browser OAuth flow → confirm with `y`

Copy the generated config into the project:

```powershell
copy %APPDATA%\rclone\rclone.conf rclone\rclone.conf
```

Verify it works:

```powershell
rclone lsd gdrive:
```

> `rclone.conf` is git-ignored — it contains your OAuth token, never commit it.

**2. Build the Docker image**

```powershell
docker compose build
```

**3. Verify the environment**

```powershell
docker compose run --rm train python scripts/verify_env.py
```

All 9 checks should pass.

---

### Phase 2 — Preprocess data

**Step 1: Download datasets manually**

Download and place files as shown below:

```
data/raw/
  enron_spam_data.csv            ← unzip from GitHub link above
  spamassassin/
    easy_ham/                    ← extract 20030228_easy_ham.tar.bz2
    easy_ham_2/                  ← extract 20030228_easy_ham_2.tar.bz2
    hard_ham/                    ← extract 20030228_hard_ham.tar.bz2
    spam/                        ← extract 20030228_spam.tar.bz2
    spam_2/                      ← extract 20050311_spam_2.tar.bz2
```

Use [7-Zip](https://www.7-zip.org/) on Windows to extract `.tar.bz2` files.

**Step 2: Copy datasets into Docker volume**

```powershell
docker compose cp data\raw\enron_spam_data.csv train:/workspace/data/raw/
docker compose cp data\raw\spamassassin train:/workspace/data/raw/
```

**Step 3: Run preprocessing**

```powershell
docker compose up -d train
docker compose exec train bash
python src/data/preprocess.py
```

Expected runtime: ~15–20 minutes on a 12-thread CPU.

**Step 4: Push processed data to Google Drive**

```bash
python scripts/drive_sync.py push-data
```

---

### Phase 3 — Train on Google Colab

1. Upload `notebooks/train_colab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Run All

The notebook trains 10-fold CV on both datasets and saves every fold's
checkpoint to Drive immediately — crash-resilient, resumable on reconnect.

Expected runtime: ~1–1.5 hours total on a free T4 GPU.

**After training, pull results to local:**

```bash
docker compose exec train bash
python scripts/drive_sync.py pull-checkpoints
python scripts/drive_sync.py pull-results
```

---

### Results analysis

Open Jupyter at `http://localhost:8888` (token: `hanspam`) and run
`notebooks/results_analysis.ipynb`. This reproduces Tables 3 & 4 from
the paper and compares your AUC numbers against the paper's reported values.

### Phase 5 — FastAPI serving (`/predict`)

After you have pulled processed data + checkpoints from Drive, start the API:

```powershell
docker compose up -d api
```

Health check:

```powershell
curl http://localhost:8000/health
```

Prediction request:

```powershell
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Congratulations! You won a free prize. Click now.\"}"
```

Example response:

```json
{
  "label": "spam",
  "spam_probability": 0.9821,
  "ham_probability": 0.0179,
  "threshold": 0.5
}
```

Checkpoint selection:
- If `HAN_WEIGHTS_PATH` is set, the API uses that exact checkpoint.
- Otherwise it auto-selects the first `*.weights.h5` found under `checkpoints/`.

### Phase 6 — Retrain with synthetic augmentation

If you generated `data/raw/synthetic_spam_ham_dataset.csv`, you can include it as
training augmentation (synthetic is added to train splits only; test splits remain real).

Copy raw files into Docker volumes (one-time per fresh volume):

```powershell
docker compose up -d --no-deps train
docker compose cp data/raw/enron_spam_data.csv train:/workspace/data/raw/
docker compose cp data/raw/spamassassin train:/workspace/data/raw/
docker compose cp data/raw/synthetic_spam_ham_dataset.csv train:/workspace/data/raw/
```

Run preprocessing with synthetic enabled:

```powershell
docker compose run --no-deps --rm -e HAN_INCLUDE_SYNTHETIC=true train python src/data/preprocess.py
```

> Important: set the flag with `-e HAN_INCLUDE_SYNTHETIC=true` on `docker compose run`.
> Setting `$env:HAN_INCLUDE_SYNTHETIC` on the host shell alone does not always propagate.

Push updated processed data to Drive:

```powershell
docker compose run --no-deps --rm train python scripts/drive_sync.py push-data
```

Then rerun Colab training to produce new checkpoints on augmented data.

---

## For collaborators (getting trained models without retraining)

The trained checkpoints and results are shared via Google Drive.
Ask the project owner to share the `han-spam` Drive folder with your Google account.

Once you have access, follow these steps:

**1. Clone the repo**

```bash
git clone https://github.com/yourname/han-spam.git
cd han-spam
```

**2. Set up rclone to point to the shared folder**

Install rclone from [rclone.org/downloads](https://rclone.org/downloads/), then:

```powershell
rclone config
```

- `n` → new remote → name: `gdrive`
- Storage: `drive` (Google Drive)
- Leave client ID and secret blank → scope: `1` (full access)
- Complete the browser OAuth flow → confirm with `y`

Copy the config into the project:

```powershell
copy %APPDATA%\rclone\rclone.conf rclone\rclone.conf
```

**3. Build Docker**

```powershell
docker compose build
docker compose run --rm train python scripts/verify_env.py
```

**4. Pull everything from the shared Drive**

```powershell
docker compose up -d train
docker compose exec train bash
python scripts/drive_sync.py pull-data
python scripts/drive_sync.py pull-checkpoints
python scripts/drive_sync.py pull-results
```

**5. Run results analysis**

Open `http://localhost:8888` (token: `hanspam`) and run
`notebooks/results_analysis.ipynb` — no retraining needed.

> **Note:** `rclone.conf` is git-ignored. Every collaborator generates their
> own via `rclone config` and authenticates with their own Google account.
> As long as the project owner has shared the `han-spam` Drive folder with
> your Google account, rclone will be able to read from it.

---

## Project structure

```
han-spam/
├── Dockerfile                        # CPU-only Python 3.10 + rclone
├── docker-compose.yml                # train, jupyter, mlflow, tensorboard
├── requirements.txt                  # pinned dependencies
├── configs/
│   └── config.yaml                   # all hyperparameters
├── rclone/
│   └── rclone.conf.template          # Drive setup instructions (conf is git-ignored)
├── scripts/
│   ├── verify_env.py                 # environment smoke test (9 checks)
│   └── drive_sync.py                 # push/pull data, checkpoints, results
├── notebooks/
│   ├── train_colab.ipynb             # upload to Colab for training
│   └── results_analysis.ipynb       # run locally to compare vs paper
├── src/
│   ├── data/
│   │   ├── clean_text.py             # HTML strip, normalise, tokenize
│   │   ├── load_datasets.py          # parse Enron CSV + SpamAssassin files
│   │   ├── hierarchical_tokenizer.py # doc → [sentences × words] tensors
│   │   ├── embeddings.py             # FastText training + embedding matrix
│   │   └── preprocess.py            # orchestrates the full pipeline
│   ├── models/                       # HAN architecture (Phase 3)
│   ├── training/                     # evaluation helpers (Phase 4)
│   ├── serving/                      # FastAPI inference app (Phase 5)
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── seed.py
├── data/
│   ├── raw/                          # place datasets here (git-ignored)
│   └── processed/                    # generated by preprocess.py (git-ignored)
├── checkpoints/                      # pulled from Drive (git-ignored)
└── outputs/                          # results + plots (git-ignored)
```

---

## Services

```powershell
docker compose up -d
```

| Service | URL | Purpose |
|---|---|---|
| `api` | http://localhost:8000 | FastAPI `/predict` serving endpoint |
| `train` | — | Preprocessing / sync shell |
| `jupyter` | http://localhost:8888 | Notebooks (token: `hanspam`) |
| `mlflow` | http://localhost:5000 | Experiment tracking |
| `tensorboard` | http://localhost:6006 | Training curves |
