# Mac Mini migration guide

Move this project (code **and** cached data) onto a Mac Mini so it can act as
the always-on server. You can SSH / Tailscale in from another Mac and drive the
same webapp and story pulls.

**Audience:** Cursor (or you) on the **destination** Mac Mini, after the project
folder has arrived (AirDrop, rsync, or git + data sync).

**Source machine (example):**  
`/Users/armandoordoricadelatorre/Documents/U of T/PhD/PhD Research/Sentiment_learn_to_rank_paper`

**GitHub:** `git@github.com:armandordorica/sentiment_learn_to_rank_paper.git`  
**Conda env name:** `sentiment-ltr-paper` (see `environment.yml`)

---

## What AirDrop / copying the folder already gives you

If you AirDropped or copied the **whole project folder**, you typically already
have:

| Item | Location |
| --- | --- |
| Code, scripts, webapp | repo root |
| Git history | `.git/` (if included) |
| Secrets | `.env` (if included — **verify**) |
| Batch + story caches | `data/raw/` (~several GB; gitignored) |
| Models | `data/models/` (gitignored) |
| Queue + quota settings | `app_data/` |
| Env spec | `environment.yml`, `requirements-finetuning.txt` |

Repo size on the source machine was on the order of **~9 GB** total, mostly
`data/`.

### What is **not** in the folder (you must recreate)

1. **Conda/Miniconda install** and the `sentiment-ltr-paper` environment  
2. **LSEG Workspace** app + signed-in desktop session (needed for live Refinitiv
   `get_story` / headlines)  
3. **crontab** entries (story-quota hourly watchdog)  
4. **SSH / Tailscale** for remote access from your laptop  
5. Optional: **GitHub SSH key** on the Mini if you will `git push` from there  

Do **not** copy `~/miniconda3/envs/sentiment-ltr-paper` between Macs. Recreate
it from `environment.yml`.

---

## Checklist (do in order)

### 0. Put the folder somewhere stable

Suggested:

```bash
mkdir -p "$HOME/PhD"
# If AirDrop landed in Downloads/Desktop, move it:
# mv ~/Downloads/Sentiment_learn_to_rank_paper "$HOME/PhD/"
cd "$HOME/PhD/Sentiment_learn_to_rank_paper"
pwd
```

Confirm data arrived:

```bash
du -sh data data/raw data/raw/data_explorer_top1k data/raw/data_explorer_full_stories 2>/dev/null
ls -la .env environment.yml app_data/story_quota_settings.json
```

If `.env` is missing, copy it from the source Mac before continuing.

### 1. System basics

- Install **Xcode Command Line Tools** if prompted (`xcode-select --install`)
- Install **Homebrew** (optional but useful)
- Install **Miniconda** or **Mambaforge** (Apple Silicon / arm64)
- Enable **Remote Login** (System Settings → General → Sharing → Remote Login)
  if you want SSH from the other Mac
- Optional: install **Tailscale** on both Macs for easy remote access

### 2. Create the conda environment

From the repo root:

```bash
cd "$HOME/PhD/Sentiment_learn_to_rank_paper"   # or your actual path
conda env create -f environment.yml
conda activate sentiment-ltr-paper
python -c "import pandas, fastapi, plotly; print('imports ok', pandas.__version__)"
```

If the env already exists and is broken:

```bash
conda env update -f environment.yml --prune
# or: conda env remove -n sentiment-ltr-paper && conda env create -f environment.yml
```

Fine-tuning extras are pulled via `requirements-finetuning.txt` referenced from
`environment.yml`. First PyTorch / HF install can take a while.

### 3. LSEG Workspace (for live news / story bodies)

Refinitiv full-story pulls use **desktop Workspace auth**, not only `.env`.

1. Install **LSEG Workspace** on the Mac Mini  
2. Sign in and leave it running while pulls / cron run  
3. Smoke-test (optional):

```bash
conda activate sentiment-ltr-paper
PYTHONPATH=src:. python scripts/test_refinitiv_connection.py
```

WRDS / RavenPack / Yahoo can work with credentials in `.env` without Workspace.
Wire bodies (`get_story`) need Workspace on **this** machine.

### 4. Install the hourly story-quota cron

Crontab does not travel with AirDrop. From the repo root, with the project env’s
Python preferred:

```bash
conda activate sentiment-ltr-paper
export SENTIMENT_LTR_PYTHON="$(which python)"
bash scripts/install_story_quota_cron.sh
crontab -l | grep run_daily_story_quota
```

Or start the webapp and use **Many stocks → 2F · Story quota automation → Edit
settings → Save & apply**.

Settings live in `app_data/story_quota_settings.json` (max/day, min-sleep, cron
minute). Queue: `app_data/story_pull_queue.txt`.

On macOS, cron may need **Full Disk Access** for `/usr/sbin/cron` (System
Settings → Privacy & Security).

### 5. Start the webapp (server mode)

```bash
conda activate sentiment-ltr-paper
cd "$HOME/PhD/Sentiment_learn_to_rank_paper"
caffeinate -is python -m uvicorn webapp.main:app --host 0.0.0.0 --port 8001
```

- Local on Mini: http://127.0.0.1:8001  
- From another Mac on the LAN / Tailscale: `http://<mini-ip>:8001`  
- Or SSH tunnel from the laptop:

```bash
ssh -L 8001:127.0.0.1:8001 <user>@<mac-mini-hostname-or-ip>
# then open http://127.0.0.1:8001 on the laptop
```

Useful routes:

| Route | Purpose |
| --- | --- |
| `/data-explorer` | One-ticker inspect + story pull status |
| `/batch` | Many stocks; **2F** story quota automation + queue |

Keep the Mac awake (`caffeinate`, or Energy Saver settings) so overnight pulls
and cron are not suspended.

### 6. Verify story automation

```bash
conda activate sentiment-ltr-paper
PYTHONPATH=src:. python scripts/run_daily_story_quota.py --dry-run
ls data/raw/data_explorer_full_stories/MSFT/_pull_progress.json 2>/dev/null
tail -n 30 data/raw/data_explorer_full_stories/_daily_quota.log 2>/dev/null
```

If a pull is already mid-run on the **old** Mac, do not start a second competing
story job on the Mini until you stop the old one (shared Workspace quota is per
login / desktop session — usually one machine at a time).

---

## Alternative: git clone + rsync (if not AirDropping)

On the Mini:

```bash
git clone git@github.com:armandordorica/sentiment_learn_to_rank_paper.git
cd sentiment_learn_to_rank_paper
```

From the **source** Mac (data is gitignored):

```bash
SRC="/Users/armandoordoricadelatorre/Documents/U of T/PhD/PhD Research/Sentiment_learn_to_rank_paper"
DST="<user>@<mac-mini>:~/PhD/Sentiment_learn_to_rank_paper"

rsync -aH --info=progress2 --partial \
  "$SRC/data/" "$DST/data/"

rsync -aH --info=progress2 --partial \
  "$SRC/app_data/" "$DST/app_data/"

scp "$SRC/.env" "$DST/.env"
```

Then continue from **§2 Create the conda environment**.

Re-run the same `rsync` later to sync new story bodies without re-copying
everything.

---

## Ongoing sync (laptop ↔ Mini)

| Direction | Typical use |
| --- | --- |
| Mini → GitHub | `git push` code changes made on the Mini |
| Laptop → Mini | `rsync` of `data/` when you pull more bodies on one machine |
| Prefer one live story worker | Only one Mac should run `cache_refinitiv_full_stories` / overnight pulls against the same Workspace login |

Absolute paths in an old crontab from the laptop will be wrong on the Mini —
always reinstall cron on the Mini (`install_story_quota_cron.sh` or 2F).

---

## Cursor agent prompt (paste on the Mini)

```text
Read migration.md in this repo and finish Mac Mini setup:
1) Confirm repo path, .env, and data/ sizes
2) Create/activate conda env from environment.yml
3) Install story-quota cron via scripts/install_story_quota_cron.sh
4) Start uvicorn on port 8001 with caffeinate
5) Report any missing pieces (Workspace, .env, disk, cron)
Do not start a second story pull if one is already running elsewhere on the same LSEG login.
```

---

## Quick smoke checklist

- [ ] `conda activate sentiment-ltr-paper` works  
- [ ] `.env` present with WRDS (and any other) credentials  
- [ ] `data/raw/data_explorer_top1k` and `data_explorer_full_stories` look populated  
- [ ] http://127.0.0.1:8001/batch loads; **2F** shows queue + quota  
- [ ] `crontab -l` contains `run_daily_story_quota.py`  
- [ ] LSEG Workspace signed in (if pulling stories)  
- [ ] SSH or Tailscale from the other Mac works  

---

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| `ModuleNotFoundError` | Wrong Python; use conda env, not system Python |
| Webapp 500 / old UI | Restart uvicorn from the new path; prefer `--reload` while debugging |
| Story pull 429 / scope errors | Workspace not signed in on Mini, or another machine is burning the same news quota |
| Cron never runs | Full Disk Access for cron; check `_daily_quota.log`; confirm `SENTIMENT_LTR_PYTHON` |
| Empty Data Explorer caches | `data/raw/` missing — AirDrop incomplete or need rsync |
| Git “untracked data” | Normal — `data/raw` and `data/models` are gitignored |

---

## References in-repo

- `environment.yml` — conda env  
- `app_data/story_quota_settings.json` — max/day, sleep, cron minute  
- `app_data/story_pull_queue.txt` — multi-ticker story order  
- `scripts/run_daily_story_quota.py` — hourly watchdog  
- `scripts/install_story_quota_cron.sh` — install/refresh crontab  
- `scripts/cache_refinitiv_full_stories.py` — one-ticker body pull  
- `docs/data_pull_validation.md` — data inventory  
