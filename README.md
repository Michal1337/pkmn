# PTCG AI Battle Challenge

Agent + tooling for the Kaggle
[Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).
Two agents play Pokémon TCG on the **cabt** engine (runs on
`kaggle-environments`); each turn an agent gets an observation and returns the
indices of the legal options it picks. Submissions are a `.tar.gz` containing
`main.py` and `deck.csv` at the top level.

## Layout

```
agent/
  main.py              # submission entry point — defines agent(obs)
  deck.csv             # 60-card deck (known-legal starter)
scripts/
  run_battle.py        # play one game, write an HTML replay
  evaluate.py          # play N games, report win rate
  build_submission.py  # package agent/ into submission.tar.gz (files at top level)
  _common.py           # shared helpers (loads agents, silences env import noise)
notes/
  cabt_api.md          # observation / action / enum reference (verified vs source)
requirements.txt
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install --no-deps "git+https://github.com/Kaggle/kaggle-environments.git"
pip install numpy jsonschema
```

Why `--no-deps`: the `cabt` env lives only in the GitHub (master) build, and
the full dependency set pulls heavy ML packages (gymnax/flax/orbax) that cabt
does not need — and that fail to install on Windows without long-path support.
The cabt engine itself ships as a native lib (`cg.dll` / `libcg.so`) inside the
package, so no separate SDK download is required to run battles locally.

Sanity check:

```bash
python -c "from kaggle_environments import make; e=make('cabt'); print('ok')"
```

## Develop

```bash
# Watch our agent play the built-in random bot, then open the replay.
python scripts/run_battle.py
start result.html                      # Windows (xdg-open / open elsewhere)

# Measure a change (sides alternate to cancel first-player advantage).
python scripts/evaluate.py -n 50       # agent/main.py vs random
python scripts/evaluate.py agent/main.py first -n 50
```

The policy lives in one place: **`choose()` in `agent/main.py`**. The baseline
just takes the first `maxCount` legal options — correct but naive. Improving
`choose()` (and the deck) is the competition. See `notes/cabt_api.md` for the
observation/option schema and the OptionType / SelectContext enums to branch on.

## Submit

```bash
python scripts/build_submission.py     # -> submission.tar.gz (validates 60-card deck)
```

Upload `submission.tar.gz` on the competition's Submit page. The builder packs
every file under `agent/` flat, so `main.py` and `deck.csv` sit at the archive
root as required.

## Notes

- Reward: win `+1`, loss `-1`, draw `0`.
- `agent(obs)` returns the 60-card deck when `obs["select"] is None`, otherwise
  a `list[int]` of chosen option indices (length within `[minCount, maxCount]`).
- The engine only ever offers legal options, so a returned index is always safe;
  the strategy is *which* legal option to take.
