"""Web UI to play the cabt TCG against a trained checkpoint, in the browser.

Same engine + AI as scripts/play.py, exposed over Flask: the board + your legal options
render as a clickable page; clicks POST your pick, the AI plays its turns, the board
refreshes. v1/v2, greedy/MCTS auto-detected from the ckpt.

  PYTHONPATH=. python scripts/serve_play.py --ckpt _tourney/v2_latest/latest.pt --mode mcts --n-sims 80
  then open http://127.0.0.1:8000
"""
from __future__ import annotations
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # import sibling play.py
from play import make_ai, describe, _name, _cid, ENERGY          # noqa: E402

from flask import Flask, jsonify, request                        # noqa: E402
from rl.card_features import get_card_table                      # noqa: E402
from rl.decks import DECKS                                       # noqa: E402
from sdk_cg.game import battle_start, battle_select, battle_finish  # noqa: E402

CARDS = get_card_table()

# sdk_cg.api.SelectType -> readable label for the decision header
SELTYPE = {0: "MAIN", 1: "CARD", 2: "ATTACHED", 3: "CARD/ATTACHED", 4: "ENERGY", 5: "SKILL",
           6: "ATTACK", 7: "EVOLVE", 8: "COUNT", 9: "YES/NO", 10: "SPECIAL"}


def _pk(pk):
    if not pk:
        return None
    return {"name": _name(CARDS, pk.get("id")), "hp": pk.get("hp"), "maxHp": pk.get("maxHp"),
            "energy": "".join(ENERGY[e] for e in (pk.get("energies") or []) if 0 <= e < len(ENERGY)),
            "tools": [_name(CARDS, _cid(t)) for t in (pk.get("tools") or [])],
            "new": bool(pk.get("appearThisTurn"))}


def _side(w):
    act = (w.get("active") or [None])
    return {"prizes": len(w.get("prize") or []), "deck": w.get("deckCount"),
            "hand": w.get("handCount", len(w.get("hand") or [])), "discard": len(w.get("discard") or []),
            "active": _pk(act[0] if act else None),
            "bench": [_pk(b) for b in (w.get("bench") or [])]}


class Game:
    def __init__(self, ckpt, mode, n_sims, n_det, your_deck, ai_deck, you):
        self.lock = threading.Lock()
        self.your_deck = your_deck; self.ai_deck = ai_deck; self.you = you
        self.ai, self.arch = make_ai(ckpt, mode, n_sims, n_det, {"deck": ai_deck})
        self.mode = mode; self.obs = None
        self.new()

    def _d(self, who):
        return self.your_deck if who == self.you else self.ai_deck

    def _advance(self):
        # play deck-steps + the AI's turns until it's the human's decision or game over
        o = self.obs
        while o["current"]["result"] < 0:
            who = o["current"]["yourIndex"]; sel = o.get("select")
            if sel is None:
                o = battle_select([int(c) for c in self._d(who)]); continue
            if who == self.you:
                break
            o = battle_select(self.ai(o))
        self.obs = o

    def new(self):
        with self.lock:
            try: battle_finish()
            except Exception: pass
            d0 = self.your_deck if self.you == 0 else self.ai_deck
            d1 = self.ai_deck if self.you == 0 else self.your_deck
            self.obs = battle_start(d0, d1)[0]
            self._advance()
        return self.state()

    def pick(self, picks):
        with self.lock:
            if self.obs["current"]["result"] < 0 and self.obs.get("select") is not None:
                self.obs = battle_select([int(p) for p in picks])
                self._advance()
        return self.state()

    def state(self):
        o = self.obs; cur = o["current"]; r = cur["result"]; me = self.you
        st = {"over": r >= 0, "result": r, "turn": cur.get("turn"), "you": me,
              "opp": _side(cur["players"][1 - me]), "me": _side(cur["players"][me]),
              "hand": [_name(CARDS, _cid(c)) for c in (cur["players"][me].get("hand") or [])]}
        if r >= 0:
            st["msg"] = "DRAW" if r == 2 else ("YOU WIN!" if r == me else "You lose.")
            st["decision"] = None
        else:
            sel = o["select"]
            eff = sel.get("effect") or sel.get("contextCard")
            st["decision"] = {"type": sel.get("type"),
                              "typeName": SELTYPE.get(sel.get("type"), f"type {sel.get('type')}"),
                              "effect": (_name(CARDS, _cid(eff)) if eff else ""),
                              "maxCount": sel.get("maxCount", 1),
                              "options": [{"i": i, "desc": describe(CARDS, op, cur, me, self.your_deck)}
                                          for i, op in enumerate(sel["option"])]}
        return st


GAME: Game | None = None
app = Flask(__name__)


@app.route("/")
def index():
    return HTML


@app.route("/api/state")
def api_state():
    return jsonify(GAME.state())


@app.route("/api/new", methods=["POST"])
def api_new():
    return jsonify(GAME.new())


@app.route("/api/pick", methods=["POST"])
def api_pick():
    return jsonify(GAME.pick(request.get_json(force=True).get("picks", [])))


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>cabt vs AI</title>
<style>
body{font-family:system-ui,Arial;margin:0;background:#0f1420;color:#e6e9ef}
.wrap{max-width:1000px;margin:0 auto;padding:14px}
h2{margin:6px 0}
.side{border:1px solid #2a3450;border-radius:10px;padding:10px;margin:8px 0}
.opp{background:#1a1320}.me{background:#13201a}
.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.card{background:#222b42;border:1px solid #3a4a6a;border-radius:8px;padding:6px 8px;min-width:120px}
.card.act{border-color:#7aa2ff;background:#26345c}
.nm{font-weight:600}.hp{font-size:12px;color:#9fb3d1}
.bar{height:6px;background:#3a2030;border-radius:3px;margin-top:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:#4ade80}
.eng{font-size:12px;color:#ffd27a}.tool{font-size:11px;color:#a0e0ff}
.meta{font-size:12px;color:#9fb3d1}
.hand{font-size:13px;color:#cfe0ff;margin-top:4px}
.opts{margin-top:10px}
.opt{display:inline-block;margin:4px;padding:8px 12px;background:#2b3a64;border:1px solid #4a6;
     border-radius:8px;cursor:pointer;color:#fff}
.opt:hover{background:#3a4f88}.opt.sel{background:#1f7a4d;border-color:#4ade80}
button{padding:8px 14px;border-radius:8px;border:0;background:#3a4f88;color:#fff;cursor:pointer;margin-right:6px}
#banner{font-size:20px;font-weight:700;margin:8px 0}
.dim{opacity:.6}
</style></head><body><div class="wrap">
<h2>cabt — you vs the checkpoint</h2>
<div id="banner"></div>
<div id="opp" class="side opp"></div>
<div id="me" class="side me"></div>
<div id="hand" class="hand"></div>
<div id="opts" class="opts"></div>
<div style="margin-top:10px"><button onclick="newGame()">New game</button>
<span id="info" class="meta"></span></div>
</div>
<script>
let S=null, sel=[];
function pk(p,act){ if(!p) return '<div class="card dim">(empty)</div>';
  let f=p.maxHp?Math.round(100*p.hp/p.maxHp):0;
  return `<div class="card ${act?'act':''}"><div class="nm">${p.name}${p.new?' •':''}</div>
   <div class="hp">${p.hp}/${p.maxHp}</div><div class="bar"><i style="width:${f}%"></i></div>
   ${p.energy?`<div class="eng">E: ${p.energy}</div>`:''}
   ${p.tools.length?`<div class="tool">tool: ${p.tools.join(',')}</div>`:''}</div>`;}
function side(d){ return `<div class="meta">prizes ${d.prizes} · deck ${d.deck} · hand ${d.hand} · discard ${d.discard}</div>
  <div class="row">${pk(d.active,true)}${d.bench.map(b=>pk(b,false)).join('')}</div>`;}
function render(s){ S=s; sel=[];
  document.getElementById('opp').innerHTML='<b>OPPONENT (P'+(1-s.you)+')</b>'+side(s.opp);
  document.getElementById('me').innerHTML='<b>YOU (P'+s.you+')</b>'+side(s.me);
  document.getElementById('hand').innerHTML='HAND: '+(s.hand.join(' · ')||'(empty)');
  document.getElementById('info').textContent='turn '+s.turn;
  let b=document.getElementById('banner'), o=document.getElementById('opts');
  if(s.over){ b.textContent=s.msg; o.innerHTML=''; return; }
  b.textContent='';
  let d=s.decision, h=`<div class="meta">choose ${d.maxCount} of ${d.options.length} — ${d.typeName}${d.effect?' for '+d.effect:''}</div>`;
  h+=d.options.map(op=>`<span class="opt" data-i="${op.i}" onclick="clk(${op.i})">[${op.i}] ${op.desc}</span>`).join('');
  if(d.maxCount>1) h+='<div style="margin-top:8px"><button onclick="confirmPicks()">Confirm ('+d.maxCount+')</button></div>';
  o.innerHTML=h;}
function clk(i){ let d=S.decision;
  if(d.maxCount===1){ send([i]); return; }
  let e=[...document.querySelectorAll('.opt')].find(x=>+x.dataset.i===i);
  if(sel.includes(i)){ sel=sel.filter(x=>x!==i); e.classList.remove('sel'); }
  else if(sel.length<d.maxCount){ sel.push(i); e.classList.add('sel'); }}
function confirmPicks(){ if(sel.length) send(sel); }
async function send(picks){ render(await (await fetch('/api/pick',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({picks})})).json()); }
async function newGame(){ render(await (await fetch('/api/new',{method:'POST'})).json()); }
async function load(){ render(await (await fetch('/api/state')).json()); }
load();
</script></body></html>"""


def main():
    global GAME
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["greedy", "mcts"], default="mcts")
    p.add_argument("--n-sims", type=int, default=80)
    p.add_argument("--n-det", type=int, default=2)
    p.add_argument("--your-deck", default=None)
    p.add_argument("--ai-deck", default=None)
    p.add_argument("--side", type=int, default=0, choices=[0, 1])
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    ai_deck = DECKS.get(a.ai_deck) if a.ai_deck else list(DECKS.values())[0]
    your_deck = DECKS.get(a.your_deck) if a.your_deck else ai_deck
    GAME = Game(a.ckpt, a.mode, a.n_sims, a.n_det, your_deck, ai_deck, a.side)
    print(f"[serve] opponent={GAME.arch} {a.mode}; open http://127.0.0.1:{a.port}", flush=True)
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
