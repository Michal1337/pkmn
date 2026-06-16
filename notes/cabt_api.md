# cabt engine — API reference

Notes on the **cabt** Pokémon TCG battle engine used by the
[PTCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).
Verified against the engine source bundled in `kaggle_environments/envs/cabt`
(build 1.30.1) and the official docs at <https://matsuoinstitute.github.io/cabt/>.

## How a battle runs

Battles run on `kaggle-environments`. Two agents play; each is a function
`agent(obs: dict) -> list[int]`. The engine is a native library (`cg.dll` /
`libcg.so`) driven from `cg/game.py`.

```python
from kaggle_environments import make
env = make("cabt", configuration={})
env.run([agent_a, agent_b])
env.render(mode="html")   # full replay
```

Spec (`cabt.json`): 2 agents, reward ∈ {-1 loss, 0 draw, 1 win},
`episodeSteps=10000`, `runTimeout=3000`.

## The observation

Each step the active agent receives `obs` with keys:

| key                  | meaning |
|----------------------|---------|
| `select`             | the decision to make this step, or `None` |
| `current`            | board state (`State`), or `None` before the game starts |
| `logs`               | event-log entries since the last step |
| `search_begin_input` | opaque payload for the search/simulation API |

### `select`

```
{
  "type": <SelectType>, "context": <SelectContext>,
  "minCount": int, "maxCount": int,
  "remainDamageCounter": int, "remainEnergyCost": int,
  "option": [ <Option>, ... ],
  "deck": null, "contextCard": null, "effect": null
}
```

The agent returns the **indices into `option`** that it picks, as a
`list[int]` whose length is in `[minCount, maxCount]`. The engine only ever
lists legal options.

When `select is None` the engine is asking for the **deck**: return the 60
card IDs.

**Option** (fields present depend on `type`):
`type` (OptionType, required), `number`, `area`, `index`, `playerIndex`,
`toolIndex`, `energyIndex`, `count`, `inPlayArea`, `inPlayIndex`, `attackId`,
`cardId`, `serial`, `specialConditionType`.

### `current` → `State`

`turn`, `turnActionCount`, `yourIndex`, `firstPlayer`, `supporterPlayed`,
`stadiumPlayed`, `energyAttached`, `retreated`, `result` (−1 ongoing, else
winner index / draw), `stadium`, `looking`, `players` (list of 2 PlayerState).

**PlayerState**: `active` (0–1 Pokémon), `bench` (≤ `benchMax`, usually 5),
`benchMax`, `deckCount`, `discard`, `prize` (first = bottom, last = top),
`handCount`, `hand` (own cards only; opponent shows count), and status flags
`poisoned`, `burned`, `asleep`, `paralyzed`, `confused`.

**Pokemon**: `id`, `serial`, `hp`, `maxHp`, `appearThisTurn`, `energies`,
`energyCards`, `tools`, `preEvolution`.

## Enums

**OptionType**: PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END,
CARD, TOOL_CARD, ENERGY_CARD, ENERGY, YES, NO, SKILL, SPECIAL_CONDITION.

**SelectType**: MAIN, CARD, ATTACHED_CARD, ENERGY, SKILL, ATTACK, EVOLVE,
COUNT, YES_NO, SPECIAL_CONDITION.

**SelectContext**: e.g. SETUP_ACTIVE_POKEMON, SWITCH, DISCARD, DAMAGE_COUNTER…

> Integer values for these enums are defined in the full SDK's `api` module
> (`matsuoinstitute.github.io/cabt/api.html`), which is **not** part of the
> kaggle-environments bundle. Confirm the exact ints there before branching on
> specific option types.

## Card data (CardData)

`cardId`, `name`, `cardType`, `retreatCost`, `hp`, `weakness`, `resistance`,
`energyType`, `basic`/`stage1`/`stage2`, `ex`/`megaEx`/`tera`/`aceSpec`,
`evolvesFrom`, `skills`, `attacks`. **Attack**: `attackId`, `name`, `text`,
`damage`, `energies`.

## Engine entry points (`cg/game.py`)

- `battle_start(deck0, deck1) -> (obs|None, StartData)` — raises if a deck != 60.
- `battle_select(select_list: list[int]) -> obs`
- `battle_finish()`
- `visualize_data() -> str`

## Deck

`deck.csv`: one card ID per line, exactly 60. Standard construction limits
apply (≤ 4 copies of a card except basic Energy; must include a basic Pokémon).
A known-legal starter deck ships in `agent/deck.csv` (copied from the engine's
built-in sample).

## Search / simulation API

The full SDK exposes `SearchBegin/SearchStep/SearchEnd` plus `AllCard` /
`AllAttack` for lookahead and card lookups (useful for MCTS or an RL
environment). These are part of the documented SDK but not the minimal
kaggle-environments bundle.
