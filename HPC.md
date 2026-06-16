# Eden cluster cheat sheet

*Verified 2026-06-16. Earlier versions of this file were stale (stud nodes,
pyenv, hopper=H200) — corrected below.*

## Login

```bash
ssh eden.mini.pw.edu.pl     # uses ~/.ssh/config ProxyJump via ssh.mini.pw.edu.pl
```

**Don't spam SSH.** The jump host rate-limits rapid repeated connections — symptom
is `kex_exchange_identification: Connection closed` then `Permission denied
(publickey,password)` even though the key is fine. Batch work into a **single**
`ssh host '...'` call, run long jobs with `nohup … > log 2>&1 &` and **poll the
log infrequently**, and move files in few/large transfers. If refused, back off
~2–3 min. Optional multiplexing to reuse one connection:
`-o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=600`.

## Environment (no pyenv)

There is **no `~/.pyenv`**. Use the system `python3.12` (`/usr/bin/python3`) + **uv**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/<repo> && uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e .                               # + torch via the CUDA wheel index
```

- **Pin `numpy<2.0`.** numpy 2.1+ wheels need the x86-64-v2 CPU baseline, which the
  **login node lacks** (`RuntimeError: machine doesn't support X86_V2`). torch pulls
  numpy 2.x, so `uv pip install "numpy>=1.26,<2.0"` **last**. (Compute nodes are fine,
  but preprocessing on the login node will crash without this.)
- **Ship shell scripts as LF**, not CRLF. Windows CRLF breaks bash on the cluster
  (`set: pipefail: invalid option name`, tainted `PATH`, `source` failures). Use
  `tr -d '\r' < f | ssh host 'cat > f'` or a `.gitattributes` with `*.sh text eol=lf`.
- Project repos live in `~/src/<repo>`.

## Account + resource budget

All jobs **must use `-A re-com`**. The "**≤4 GPUs / 600 GB concurrent**" cap is a
**soft community guideline**, not a hard SLURM limit (sacctmgr shows no GrpTRES) —
over-cap jobs just pend with `Reason=Priority`. Leave headroom for others.

## Nodes (live state via `sfree`)

`sfree` is the best capacity check — it prints **Free CPUs / Free GPUs / Free MEM**
per node. Use it before every submit.

| node | partition | GPUs | notes |
| --- | --- | --- | --- |
| **dgx-1…4** | `short`, `long` | 8× **A100** each | workhorse; stable DDP; usually some A100s free |
| **hopper** | `hopper` | 8× **H100** | fast; partition cap 5d |
| **hopper-2** | `hopper-2` | 4× **H200** + 4× **H200 MIG `3g.71gb`** + 4× **MIG `4g.71gb`** | fastest; MIG slices (71 GB) often free; cap 24h. DDP on H200 has SIGSEGV'd — use 1-GPU. |
| **sr-1…3** | `short`, `long` | **none** (48 CPU) | CPU-only jobs. **Often RAM-starved (~10–20 GiB free)** — check `sfree` before `--mem`. |
| **pascal** | `experimental` | 4× Tesla | old/slow but frequently fully **IDLE** |

GRES names: `gpu:a100:N`, `gpu:h100:N`, `gpu:h200:N`, `gpu:h200_3g.71gb:N`,
`gpu:h200_4g.71gb:N`, `gpu:tesla:N`. (There are **no `stud` nodes**.)

Partition walltime caps: `short` 24h · `long` 5d · `hopper` 5d · `hopper-2` 24h ·
`experimental` 5d. Need >24h → use `long`/`experimental`, not `short`.

## Submitting jobs

`sbatch --wrap` for one-liners (runs in **`/bin/sh`** — POSIX only: `.` not
`source`, no `for…do…done`), or a script file for anything multi-line. **Always log
to `$HOME/<name>.log`.**

### Single-GPU (most training jobs)
```bash
sbatch -A re-com -p short --gres=gpu:a100:1 -t 24:00:00 \
  --cpus-per-task=16 --mem=200G --output=$HOME/my_job.log \
  --wrap='cd ~/src/<repo> && . .venv/bin/activate && PYTHONUNBUFFERED=1 python -m <module>'
# fast paths: -p hopper --gres=gpu:h100:1   |   -p hopper-2 --gres=gpu:h200_3g.71gb:1 (71GB MIG, often free)
```

### CPU-only job (e.g. heavy pandas / LightGBM / numba)
```bash
# no --gres. Pick a node with free CPUs AND free MEM from sfree (sr is RAM-starved;
# hopper/dgx-1/hopper-2 usually have hundreds of GiB free).
sbatch -A re-com -p hopper --cpus-per-task=16 --mem=96G -t 02:00:00 \
  --output=$HOME/job.log <script.sh>
```

### Multi-GPU DDP (A100/dgx is the safe choice; avoid H200)
```bash
sbatch -A re-com -p short --gres=gpu:a100:2 --cpus-per-task=16 --mem=300G \
  --output=$HOME/my_job.log \
  --wrap='cd ~/src/<repo> && . .venv/bin/activate && PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
    torchrun --standalone --nproc-per-node=2 -m <module> [args]'
```

### Chained jobs
```bash
sbatch ... --dependency=afterany:$PARENT_JID --wrap='...'   # afterok / afternotok too
```
Running jobs **can't** have their walltime extended — plan it upfront (24h is fine
even if you expect <6h).

## Always-needed flags
- `PYTHONUNBUFFERED=1` — live logs.
- `--cpus-per-task=16` — SLURM defaults to 1 CPU → DataLoader workers starve.
- `NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo` — DDP only, prevents NCCL hangs.

## Monitoring
```bash
sfree                                            # free CPU/GPU/MEM per node (use first!)
squeue -u $USER -o '%i %T %P %j %M %N %R'        # my jobs (%R = pending reason)
sacct  -j JOBID -o JobID,State,Elapsed,ExitCode  # finished-job info
scontrol show job JOBID | grep -oE 'Reason=[^ ]+'
```

## Common pitfalls
1. **`--wrap` runs in `/bin/sh`** → `for…do…done` and `source` fail. Use a script file
   (shebang `#!/bin/bash`) for loops; use `.` instead of `source` in `--wrap`.
2. **CRLF / numpy≥2** — see Environment above (both will silently break the run).
3. **`sr` nodes are RAM-starved.** A `--mem=128G` job pends forever there; check `sfree`.
4. **`short` is often deeply backlogged** (tens of pending jobs) → `Reason=Priority`.
   Don't pin a busy node; use `sfree` and submit where there's real free capacity
   (hopper-2 MIG, hopper, dgx all commonly have room).
5. **Reusing `--output` across reruns**: a poll that greps the log can match the
   *previous* run's content. Truncate (`: > log`) before resubmitting, or wait for
   the job to leave the queue before reading.
6. **Stale `.pyc`**: after editing a module, `rm -rf …/__pycache__` if behavior
   doesn't change. Keep job-script args in sync with the script's argparse.
7. **Output-dir collisions** overwrite checkpoints — rename when changing the recipe.
8. **H200 + multi-GPU DDP = SIGSEGV.** Use 1-GPU on H200, or DDP on dgx/A100.
9. **Don't delete others' files**; **don't train on the login node** (sbatch + monitor only).

## When nothing's free
Run `sfree`. The `short`/A100 queue is the most contended; **hopper-2 MIG slices
(`gpu:h200_3g.71gb:1`, 71 GB) and idle `pascal` are frequent fast paths**, and a
shorter `-t` walltime backfills sooner. Else dep-chain behind your own running jobs
(`--dependency=afterany:<jid>`) so they auto-start as resources free.
```
