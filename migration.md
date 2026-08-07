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

## Install LSEG Workspace + verify credentials & transfer

Do this **on the Mac Mini**, in Terminal or Cursor, from the project root.
Activate the env first:

```bash
cd /path/to/Sentiment_learn_to_rank_paper
conda activate sentiment-ltr-paper
```

### A. Install and sign into LSEG Workspace

1. On the Mini, open a browser and download **LSEG Workspace** for Mac from your
   firm/school portal or [https://www.lseg.com/en/data-analytics/products/workspace](https://www.lseg.com/en/data-analytics/products/workspace)
   (use whatever download link your account normally uses on the laptop).
2. Install the `.dmg` / app into **Applications**.
3. Launch **Workspace**, sign in with the **same account** you use on the laptop.
4. Leave Workspace **running and unlocked** while testing (desktop session mode
   talks to the local Workspace process).
5. Optional but recommended: in Workspace, confirm you can open a news/quote
   panel for `AAPL.O` so the login is fully live.

Do **not** keep an active overnight story pull on the laptop and the Mini at the
same time against this login (shared ~10k/day news quota).

### B. Confirm `.env` and LSEG Python package

```bash
# Must exist after AirDrop (dotfile — verify explicitly)
ls -la .env
grep -E '^(WRDS_|LSEG_|HF_)' .env | sed 's/=.*/=***/'   # names only, no secrets

# Config file Workspace/desktop often uses (may be empty/placeholder)
ls -la lseg-data.config.json 2>/dev/null || echo "(no lseg-data.config.json yet — desktop mode may still work)"

# Ensure lseg-data is installed in this env
python -c "import lseg.data as ld; print('lseg-data ok', getattr(ld, '__version__', '?'))" \
  || pip install -r requirements-refinitiv.txt
```

`.env` fields (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `WRDS_USERNAME` / `WRDS_PASSWORD` | WRDS / RavenPack-on-WRDS |
| `LSEG_APP_KEY` / `LSEG_USERNAME` / `LSEG_PASSWORD` | Cloud platform fallback (optional if desktop works) |
| `LSEG_CONFIG_PATH` | Usually `./lseg-data.config.json` |
| `LSEG_SESSION_MODE` | Optional force; leave blank for auto desktop→platform |
| `HF_TOKEN` | Hugging Face (fine-tunes) |

### C. Test Refinitiv / Workspace (live)

With Workspace signed in:

```bash
PYTHONPATH=src:. python scripts/test_refinitiv_connection.py
```

**Pass:** prints a small table for `AAPL.O` / `MSFT.O` (BID/ASK/Revenue) and exits 0.  
**Fail:** session/config/scope error — fix Workspace login first; only then touch cloud keys in `.env`.

Optional news/story smoke (uses the same desktop session; costs a few API calls):

```bash
PYTHONPATH=src:. python - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.env'))
import lseg.data as ld
from sentiment_ltr.data.refinitiv_session import open_refinitiv_session
from sentiment_ltr.data.refinitiv_queries import fetch_refinitiv_story

root = Path('.').resolve()
open_refinitiv_session(root, ld)
# One known-good path: headlines month sample via news helper is heavier;
# just confirm get_data already worked above. Close cleanly:
ld.close_session()
print('session open/close ok')
PY
```

### D. Test WRDS credentials

```bash
PYTHONPATH=src:. python - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.env'))
from sentiment_ltr.data.live_data import wrds_credentials_available, test_wrds_connection

assert wrds_credentials_available(), 'WRDS_USERNAME/PASSWORD missing in .env'
info = test_wrds_connection()
print('WRDS ok — latest CRSP date:', info['latest_crsp_date'])
print(info['sample_rows'])
PY
```

### E. Verify the AirDrop / folder transfer

```bash
# Sizes should be in the same ballpark as the laptop (~9GB repo, multi-GB data)
du -sh . data data/raw data/raw/data_explorer_top1k data/raw/data_explorer_full_stories

# Top-1k batch cache present
test -d data/raw/data_explorer_top1k/by_ticker && \
  echo "top1k dirs:" $(find data/raw/data_explorer_top1k/by_ticker -maxdepth 1 -type d | wc -l)

# Story bodies (AAPL should be ~27k files; MSFT partial OK)
echo "AAPL bodies:" $(find data/raw/data_explorer_full_stories/AAPL -maxdepth 1 -name '*.txt' ! -name '_*' 2>/dev/null | wc -l)
echo "MSFT bodies:" $(find data/raw/data_explorer_full_stories/MSFT -maxdepth 1 -name '*.txt' ! -name '_*' 2>/dev/null | wc -l)

# Queue + settings
wc -l app_data/story_pull_queue.txt
cat app_data/story_quota_settings.json

# Logs travelled with the folder
ls -lh data/raw/data_explorer_full_stories/MSFT/_pull.log \
      data/raw/data_explorer_full_stories/_daily_quota.log \
      data/raw/data_explorer_top1k/batch_runner.log 2>/dev/null
```

Rough expectations from the laptop snapshot:

| Check | Healthy look |
| --- | --- |
| `data/raw/data_explorer_top1k/by_ticker` | ~1000 `rank_*` dirs |
| AAPL `*.txt` bodies | ~27,000 |
| MSFT `*.txt` bodies | thousands (in progress on laptop; partial OK) |
| `app_data/ravenpack_news_threshold_universe.csv` | ~537 rows |
| `app_data/story_pull_queue.txt` | ~517 ticker lines (+ comments) |

### F. Webapp smoke (optional but good)

```bash
caffeinate -is python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8001
# browser: http://127.0.0.1:8001/batch  and  /data-explorer
```

On **2F**, confirm queue counts and quota status load. On Data Explorer, load
cached AAPL without hitting live APIs first (“cache” path).

### G. Pass / fail summary

| Test | Command / check | Pass means |
| --- | --- | --- |
| Transfer | `du` + AAPL/MSFT file counts | Caches present, not empty |
| `.env` | `ls .env` + WRDS/LSEG keys set | Secrets arrived |
| WRDS | `test_wrds_connection()` | Latest CRSP date prints |
| Workspace + LSEG | `scripts/test_refinitiv_connection.py` | Price table prints |
| Webapp | `/batch` loads | UI + 2F queue visible |

If WRDS works but Refinitiv fails → Workspace install/login problem, not AirDrop.  
If both fail → check `.env` and conda env.  
If APIs work but counts are tiny → AirDrop missed `data/`; re-copy or rsync `data/`.

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
