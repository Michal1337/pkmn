"""Re-acquire PTCG episode replays from a daily Kaggle dataset into _kaggle_scout/ep/.

Reproducible (committed) so the BC corpus can be regenerated — the prior _kaggle_scout was an
un-committed local dir that got wiped. Each daily dataset is ~21GB, so we download only N
individual per-episode JSONs (each ~3-8MB). Episodes carry rewards=[r0,r1] (winner) + the cabt
obs/action traces; "top" is approximated by the highest-median day + winning-side cloning downstream.

  python scripts/scout_episodes.py [daily_slug] [N]
  python scripts/scout_episodes.py kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20 300
"""
import os
import sys
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

SLUG = sys.argv[1] if len(sys.argv) > 1 else "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = "_kaggle_scout/ep"
os.makedirs(OUT, exist_ok=True)

api = KaggleApi()
api.authenticate()

# --- collect up to N episode file names (paginated) ---
names, token = [], None
while len(names) < N:
    res = api.dataset_list_files(SLUG, page_token=token, page_size=100)
    batch = [f.name for f in res.files if f.name.endswith(".json")]
    if not batch:
        break
    names.extend(batch)
    token = res.nextPageToken
    if not token:
        break
names = names[:N]
print(f"[scout] {SLUG}: collected {len(names)} episode names -> downloading to {OUT}", flush=True)

# --- download each (skip existing); unzip if Kaggle delivers a .zip ---
ok = 0
for i, nm in enumerate(names):
    dst = os.path.join(OUT, nm)
    if os.path.exists(dst):
        ok += 1
        continue
    try:
        api.dataset_download_file(SLUG, nm, path=OUT, force=False, quiet=True)
        z = dst + ".zip"
        if os.path.exists(z):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(OUT)
            os.remove(z)
        if os.path.exists(dst):
            ok += 1
    except Exception as e:
        print(f"[scout] FAIL {nm}: {str(e)[:90]}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"[scout]   {i + 1}/{len(names)} ({ok} ok)", flush=True)

print(f"[scout] DONE: {ok}/{len(names)} episodes in {OUT}", flush=True)
