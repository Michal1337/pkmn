#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate rl/effect_data.py: a FROZEN effect-category multi-hot lookup.

Loads ALL card / attack text from the cabt SDK (sdk_cg.api), repairs the cp1252
mojibake, and assigns FINAL multi-hot functional categories with deterministic
keyword/regex classifiers ("tag all that apply"). Emits a stdlib-only frozen
lookup module rl/effect_data.py.

Scope (by cardType):
  0 = Pokemon  -> ABILITY (skills[0]) classified into ABILITY_CATS
                  ATTACKS (via card.attacks ids, from all_attack()) into ATTACK_CATS
  1 = Item, 2 = Tool, 3 = Supporter, 4 = Stadium -> TRAINER_CATS
  5 = Basic Energy, 6 = Special Energy -> OUT OF SCOPE (no trainer cats)

Run:
  PYTHONPATH=. python scripts/gen_effect_data.py
"""
import os
import re
import sys

# --- repo root on path so `rl` + `sdk_cg` import ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sdk_cg.api import all_card_data, all_attack  # noqa: E402
from rl.decks import DECKS  # noqa: E402
from rl.decks_generated import GENERATED  # noqa: E402


# ----------------------------------------------------------------------------
# Text cleaning: repair cp1252 mojibake (the SDK mangles non-ascii bytes).
# "Pokémon" comes back as "Pok� mon" / "Pok�mon"; curly quotes appear
# as smart-quote codepoints. Normalize to plain ascii.
# ----------------------------------------------------------------------------
def clean(t):
    if not t:
        return ""
    t = re.sub(r"Pok.{0,2}mon", "Pokemon", t)            # mojibake / accented e
    t = t.replace("’", "'").replace("‘", "'")    # smart apostrophes
    t = t.replace("“", '"').replace("”", '"')    # smart quotes
    t = t.replace("–", "-").replace("—", "-")    # dashes
    t = "".join(ch if ord(ch) < 128 else " " for ch in t)  # drop stray bytes
    return re.sub(r"\s+", " ", t).strip()


def lc(t):
    return clean(t).lower()


def has(t, *subs):
    return any(s in t for s in subs)


def rx(t, pat):
    return re.search(pat, t) is not None


# ============================================================================
# ATTACK classifiers  (per attackId)
# ============================================================================
ATTACK_CATS = [
    "inflict_status", "ignore_weak_resist", "snipe_bench", "self_lock",
    "discard_own_energy", "energy_accel", "protect_next_turn", "lock_opp",
    "recoil_self", "spread", "disrupt_opp_hand", "heal_self",
    "discard_opp_energy", "mill_opp_deck", "self_switch_or_move_energy",
    "force_switch_gust", "draw", "place_damage_counters", "scale_dmg",
    "coin", "misc",
]

_STATUS_WORDS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def atk_inflict_status(t):
    # Active/Defending Pokemon (opponent's, or self) BECOMES a Special Condition.
    # Exclude conditional checks ("If ... is Burned, this attack does +N") which
    # only read an existing status (those are scale_dmg, not inflict).
    if rx(t, r"now (poisoned|burned|asleep|paralyzed|confused)"):
        return True
    # "is Burned/Poisoned/..." but NOT a conditional "if ... is <status>,"
    for st in _STATUS_WORDS:
        for mm in re.finditer(r"\bis (?:now )?" + st, t):
            # look back to the start of this sentence/clause; skip conditional
            # reads ("If ... is Burned, this attack does +N").
            seg = t[max(0, mm.start() - 60):mm.start()]
            seg = re.split(r"[.] ", seg)[-1]  # current sentence only
            if "if " not in seg:
                return True
    if has(t, "now burned and confused", "now asleep and poisoned"):
        return True
    if rx(t, r"affected by that special condition"):  # Cradily-style choose
        return True
    return False


def atk_ignore_weak_resist(t):
    # Damage ignores W/R or "isn't affected by any effects" on opp.
    # NOTE: the standard benched parenthetical "(Don't apply Weakness and
    # Resistance for Benched Pokemon.)" is the spread clause, not a real
    # ignore-W/R effect -> excluded.
    if "isn't affected by weakness or resistance" in t:
        return True
    if "damage isn't affected by weakness or resistance" in t:
        return True
    if rx(t, r"isn'?t affected by any effects on your opponent"):
        return True
    if rx(t, r"not affected by weakness( or resistance)?") and "benched" not in t:
        return True
    if rx(t, r"this attack'?s damage isn'?t affected by any effects"):
        return True
    if rx(t, r"damage isn'?t affected by resistance"):  # Resistance-only ignore
        return True
    if rx(t, r"this pokemon has no weakness"):
        return True
    return False


def atk_snipe_bench(t):
    # Hit a chosen single benched Pokemon (sniping), or a chosen 1-of-opp that
    # may be benched. NOT the all-benched spread case.
    if rx(t, r"to 1 of your opponent'?s benched pokemon"):
        return True
    if rx(t, r"to \d+ of your opponent'?s benched pokemon"):  # "to 2 of opp benched"
        return True
    if rx(t, r"to 1 of your opponent'?s pokemon\b") and "benched" in t:
        return True
    if rx(t, r"damage counters? on .{0,30}your opponent'?s benched"):
        return True
    if rx(t, r"to \d+ of your opponent'?s pokemon"):  # multi-target incl. bench
        return True
    return False


def atk_self_lock(t):
    # This Pokemon can't attack / can't use this attack next turn (cooldown).
    if rx(t, r"this pokemon can'?t use"):
        return True
    if rx(t, r"this pokemon can'?t attack"):
        return True
    if rx(t, r"can'?t use this attack"):
        return True
    if "until it leaves the active spot" in t and "this pokemon can" in t:
        return True
    return False


def atk_discard_own_energy(t):
    # Discard / shuffle-away Energy attached to THIS / your own Pokemon (cost).
    if rx(t, r"discard.{0,40}energy (from|attached to) this pokemon"):
        return True
    if rx(t, r"discard all energy from this pokemon"):
        return True
    if rx(t, r"discard \d+ .{0,20}energy from this"):
        return True
    if rx(t, r"discard.{0,30}energy from your (benched )?pokemon"):
        return True
    if rx(t, r"shuffle.{0,40}energy (from|attached to) this pokemon into your deck"):
        return True
    if rx(t, r"shuffle (\d+|all) .{0,30}energy.{0,20}(from|attached to) this pokemon"):
        return True
    return False


def atk_energy_accel(t):
    # Attach energy (from hand/deck/discard) to your Pokemon as part of attack.
    if rx(t, r"attach .{0,40}energy.{0,40}to (1 of your|your|this pokemon|each)"):
        return True
    if rx(t, r"search your deck for .{0,40}energy.{0,30}attach"):
        return True
    if rx(t, r"attach (it|them|that energy) to"):
        return True
    if rx(t, r"attach a basic energy.{0,30}to"):
        return True
    return False


def atk_protect_next_turn(t):
    # Reduce/prevent damage to THIS Pokemon during opponent's next turn.
    if rx(t, r"during your opponent'?s next turn.{0,80}(prevent all damage|takes? \d+ less damage|takes? \d+ fewer)"):
        return True
    if rx(t, r"prevent all damage done to this pokemon"):
        return True
    if rx(t, r"this pokemon takes \d+ less damage from attacks"):
        return True
    if "prevent all effects of attacks" in t and "next turn" in t:
        return True
    if rx(t, r"prevent all damage.{0,40}next turn"):
        return True
    return False


def atk_lock_opp(t):
    # Disrupt opponent's options next turn: can't retreat / attack / play cards /
    # use abilities; defending pokemon restrictions.
    if rx(t, r"defending pokemon can'?t retreat"):
        return True
    if rx(t, r"opponent'?s active pokemon can'?t (retreat|attack|use)"):
        return True
    if rx(t, r"that pokemon can'?t retreat"):
        return True
    if rx(t, r"they can'?t play any (item|supporter|pokemon)"):
        return True
    if rx(t, r"opponent can'?t play"):
        return True
    if rx(t, r"can'?t use any abilities"):
        return True
    if rx(t, r"defending pokemon can'?t (attack|use)"):
        return True
    if rx(t, r"opponent'?s active pokemon'?s attacks? (do|cost)"):
        return True
    # Defending Pokemon's next-turn attacks weakened/taxed (offensive disruption)
    if rx(t, r"attacks used by the defending pokemon (do \d+ less|cost)"):
        return True
    if rx(t, r"defending pokemon.{0,40}(do \d+ less damage|cost \{?c\}? more)"):
        return True
    if rx(t, r"defending pokemon tries to use an attack"):
        return True
    return False


def atk_recoil_self(t):
    # This Pokemon damages / KOs itself.
    if rx(t, r"this pokemon does \d+ damage to itself"):
        return True
    if rx(t, r"does \d+ damage to itself"):
        return True
    if rx(t, r"this pokemon is knocked out"):
        return True
    if rx(t, r"discard this pokemon and all"):  # self-destruct
        return True
    if "both active pokemon are knocked out" in t:
        return True
    return False


def atk_spread(t):
    # Damage to EACH of opponent's (benched/all) Pokemon, or each Pokemon.
    if rx(t, r"to each of your opponent'?s (benched )?pokemon"):
        return True
    if rx(t, r"to each of your opponent'?s pokemon"):
        return True
    if rx(t, r"damage to each of your (benched )?pokemon"):
        return True
    if rx(t, r"to all of your opponent'?s"):
        return True
    if rx(t, r"to 2 of your opponent'?s (benched )?pokemon"):
        return True
    if rx(t, r"to each pokemon"):
        return True
    if rx(t, r"counters? on each of your opponent'?s"):
        return True
    return False


def atk_disrupt_opp_hand(t):
    # Affect opponent's hand: discard/shuffle/bottom/reveal-and-pick.
    if "opponent's hand" in t:
        return True
    if rx(t, r"opponent (reveals their hand|chooses.{0,30}from their hand)"):
        return True
    if rx(t, r"from your opponent'?s hand"):
        return True
    if rx(t, r"opponent.{0,30}shuffles? (those )?cards? into their deck"):
        return True
    if rx(t, r"discard a random card from your opponent"):
        return True
    if rx(t, r"on the bottom of their deck") and "opponent" in t:
        return True
    return False


def atk_heal_self(t):
    # Heal / remove damage counters from YOUR (incl. this) Pokemon.
    if rx(t, r"heal \d+ damage from"):
        return True
    if rx(t, r"heal all damage from"):
        return True
    if rx(t, r"remove \d+ damage counters? from"):
        return True
    if rx(t, r"remove all damage counters? from"):
        return True
    if rx(t, r"recovers? from all special conditions") and "opponent" not in t:
        return True
    return False


def atk_discard_opp_energy(t):
    if rx(t, r"discard.{0,40}energy from (your opponent|1 of your opponent|the attacking)"):
        return True
    if rx(t, r"discard an energy from your opponent"):
        return True
    if rx(t, r"put an energy attached to your opponent'?s.{0,20}into their hand"):
        return True
    if rx(t, r"discard a special energy from.{0,20}opponent"):
        return True
    return False


def atk_mill_opp_deck(t):
    if rx(t, r"discard the top \d* ?cards? of your opponent'?s deck"):
        return True
    if rx(t, r"top.{0,20}cards? of your opponent'?s deck"):
        return True
    if rx(t, r"discard.{0,30}from your opponent'?s deck"):
        return True
    return False


def atk_self_switch_or_move_energy(t):
    # Switch/return THIS Pokemon, or move your own energy between your Pokemon.
    if rx(t, r"switch this pokemon"):
        return True
    if rx(t, r"return this pokemon"):
        return True
    if rx(t, r"put this pokemon.{0,30}(bench|back|into your hand)"):
        return True
    if rx(t, r"shuffle this pokemon and all attached cards into your deck"):
        return True
    if rx(t, r"switch your active pokemon with 1 of your benched"):
        return True
    if rx(t, r"move (an?|any|up to|\d+).{0,30}energy from this pokemon to"):
        return True
    if rx(t, r"move .{0,30}energy from.{0,30}to (1 of your benched|your benched|your active)"):
        return True
    if rx(t, r"move an energy from this"):
        return True
    return False


def atk_force_switch_gust(t):
    # Force opponent's active to switch (gust the bench up, or push active out).
    if rx(t, r"switch in 1 of your opponent'?s benched pokemon"):
        return True
    if rx(t, r"switch out your opponent'?s active"):
        return True
    if rx(t, r"your opponent switches their active"):
        return True
    if rx(t, r"opponent.{0,20}switch.{0,20}active pokemon with"):
        return True
    return False


def atk_draw(t):
    # Card advantage: draw / search a card to hand.
    if rx(t, r"draw \d+ cards?"):
        return True
    if rx(t, r"draw a card"):
        return True
    if rx(t, r"draw cards until"):
        return True
    if rx(t, r"into your hand"):  # tutor a card to hand
        return True
    if rx(t, r"put.{0,40}from your discard pile into your hand"):
        return True
    return False


def atk_place_damage_counters(t):
    # Place/put damage counters directly (not HP damage).
    if rx(t, r"put \d+ damage counters?"):
        return True
    if rx(t, r"place \d+ damage counters?"):
        return True
    if rx(t, r"put damage counters? on"):
        return True
    if rx(t, r"move .{0,20}damage counters?"):
        return True
    return False


def atk_scale_dmg(t):
    # Variable / conditional damage modulation of this same attack.
    if rx(t, r"\d+ (more|less) damage for each"):
        return True
    if rx(t, r"does \d+ damage for each"):
        return True
    if rx(t, r"\d+ more damage"):
        return True
    if rx(t, r"this attack does \d+ more damage"):
        return True
    if "for each" in t and "damage" in t:
        return True
    if rx(t, r"times the number"):
        return True
    if rx(t, r"if .{0,80}this attack does \d+ more"):
        return True
    if rx(t, r"plus \d+ more"):
        return True
    return False


def atk_coin(t):
    return has(t, "flip a coin", "flip 2 coins", "flip 3 coins", "flip 4 coins",
               "flip a number of coins", "flip coins", "flip that many coins")


ATTACK_FNS = [
    ("inflict_status", atk_inflict_status),
    ("ignore_weak_resist", atk_ignore_weak_resist),
    ("snipe_bench", atk_snipe_bench),
    ("self_lock", atk_self_lock),
    ("discard_own_energy", atk_discard_own_energy),
    ("energy_accel", atk_energy_accel),
    ("protect_next_turn", atk_protect_next_turn),
    ("lock_opp", atk_lock_opp),
    ("recoil_self", atk_recoil_self),
    ("spread", atk_spread),
    ("disrupt_opp_hand", atk_disrupt_opp_hand),
    ("heal_self", atk_heal_self),
    ("discard_opp_energy", atk_discard_opp_energy),
    ("mill_opp_deck", atk_mill_opp_deck),
    ("self_switch_or_move_energy", atk_self_switch_or_move_energy),
    ("force_switch_gust", atk_force_switch_gust),
    ("draw", atk_draw),
    ("place_damage_counters", atk_place_damage_counters),
    ("scale_dmg", atk_scale_dmg),
    ("coin", atk_coin),
]


def classify_attack(text):
    t = lc(text)
    if not t:
        return []
    cats = [name for name, fn in ATTACK_FNS if fn(t)]
    if not cats:
        cats = ["misc"]   # has effect text but no specific category
    return cats


# ============================================================================
# ABILITY classifiers  (per cardId, Pokemon w/ ability)
# ============================================================================
ABILITY_CATS = [
    "energy_accel_move", "put_into_play", "switch_gust", "search_deck",
    "draw_dig", "dmg_reduction_wall", "ability_lock", "direct_damage",
    "status", "immunity_protection", "heal_remove_dmg", "prize_tempo",
    "attack_cost_reduce", "dmg_boost", "disrupt_opp", "endure_survive_ko",
    "misc_buff", "uncategorized", "is_passive",
]

# is_passive computed DETERMINISTICALLY (last bit).
_ACTIVATED_MARKERS = (
    "once during your turn",
    "once during each",
    "you may use this ability",
    "as often as you like during your turn",
    "once during your first turn",
)


def ability_is_passive(t):
    return not any(m in t for m in _ACTIVATED_MARKERS)


def ab_energy_accel_move(t):
    if rx(t, r"attach .{0,50}energy.{0,30}(card )?(from your (hand|discard pile|deck)|to)"):
        return True
    if rx(t, r"move (an?|any|up to|\d+).{0,30}energy from"):
        return True
    if rx(t, r"each basic.{0,20}energy.{0,20}provides"):
        return True
    if rx(t, r"energy attached.{0,30}provides"):
        return True
    if rx(t, r"attach .{0,40}energy"):
        return True
    if rx(t, r"put all basic.{0,10}energy attached.{0,40}into your hand"):
        return True
    return False


def ab_put_into_play(t):
    # Put a Pokemon into play / onto bench (from deck/discard/hand).
    if rx(t, r"put .{0,60}pokemon.{0,20}onto (your|their|the) bench"):
        return True
    if rx(t, r"put it onto your bench"):
        return True
    if rx(t, r"put this pokemon onto your bench"):
        return True
    if rx(t, r"can evolve during your first turn"):
        return True
    if rx(t, r"may put it face down in the active spot"):
        return True
    if rx(t, r"put .{0,40}pokemon.{0,40}from your discard pile onto your bench"):
        return True
    return False


def ab_switch_gust(t):
    if rx(t, r"switch in 1 of your opponent'?s benched pokemon"):
        return True
    if rx(t, r"switch (this pokemon|it) with your active"):
        return True
    if rx(t, r"switch 1 of your benched.{0,30}with your active"):
        return True
    if rx(t, r"switch your active pokemon with 1 of your benched"):
        return True
    if rx(t, r"switch out your opponent'?s active"):
        return True
    if rx(t, r"moves? from (your )?bench to the active"):
        return True
    return False


def ab_search_deck(t):
    if rx(t, r"search your deck for"):
        return True
    if rx(t, r"search their deck for"):
        return True
    return False


def ab_draw_dig(t):
    if rx(t, r"draw \d+ cards?"):
        return True
    if rx(t, r"draw a card"):
        return True
    if rx(t, r"draw cards until"):
        return True
    if rx(t, r"each player draws? a card"):
        return True
    if rx(t, r"look at the top \d+ cards?"):
        return True
    if rx(t, r"look at the top card"):
        return True
    if rx(t, r"reveal.{0,30}put .{0,20}into your hand"):
        return True
    return False


def ab_dmg_reduction_wall(t):
    if rx(t, r"takes? \d+ less damage from attacks"):
        return True
    if rx(t, r"take \d+ less damage from attacks"):
        return True
    if rx(t, r"do \d+ less damage"):
        return True
    if rx(t, r"attacks used by your opponent'?s active pokemon do \d+ less"):
        return True
    return False


def ab_ability_lock(t):
    if rx(t, r"have no abilities"):
        return True
    if rx(t, r"can'?t play any (item|supporter|pokemon tool|pokemon)"):
        return True
    if rx(t, r"can'?t play any.{0,20}cards"):
        return True
    if rx(t, r"lose any ability"):
        return True
    if rx(t, r"can'?t play any.{0,30}from their hand"):
        return True
    if rx(t, r"no abilities, except"):
        return True
    return False


def ab_direct_damage(t):
    # Put damage counters / KO directly via the ability.
    if rx(t, r"put \d+ (more )?damage counters?"):
        return True
    if rx(t, r"place \d+ damage counters?"):
        return True
    if rx(t, r"this pokemon is knocked out") and "if you use this ability" in t:
        return True
    if rx(t, r"on the attacking pokemon"):
        return True
    if rx(t, r"knock out 1 of your opponent"):
        return True
    if rx(t, r"move .{0,30}damage counters?.{0,40}to 1 of your opponent"):
        return True
    return False


def ab_status(t):
    if rx(t, r"now (poisoned|burned|asleep|paralyzed|confused)"):
        return True
    if rx(t, r"is (poisoned|burned|asleep|paralyzed|confused)"):
        return True
    if rx(t, r"make your opponent'?s active pokemon (burned|poisoned|confused|asleep|paralyzed)"):
        return True
    if rx(t, r"affected by that special condition"):
        return True
    return False


def ab_immunity_protection(t):
    if rx(t, r"prevent all damage (from and effects of attacks|done to)"):
        return True
    if rx(t, r"prevent all effects of"):
        return True
    if rx(t, r"this pokemon can'?t be asleep"):
        return True
    if rx(t, r"can'?t be (affected by|put into)"):
        return True
    if rx(t, r"prevent all damage from attacks"):
        return True
    if rx(t, r"if heads, prevent that damage"):
        return True
    return False


def ab_heal_remove_dmg(t):
    if rx(t, r"heal \d+ damage"):
        return True
    if rx(t, r"heal all damage"):
        return True
    if rx(t, r"remove .{0,20}damage counters?"):
        return True
    if rx(t, r"move .{0,30}damage counters?.{0,40}to another of your"):
        return True
    return False


def ab_prize_tempo(t):
    if rx(t, r"take \d+ (more|fewer) prize"):
        return True
    if rx(t, r"takes \d+ (more|fewer) prize"):
        return True
    if rx(t, r"can'?t take any prize cards"):
        return True
    if rx(t, r"\d+ fewer prize card"):
        return True
    return False


def ab_attack_cost_reduce(t):
    if rx(t, r"cost.{0,20}less"):
        return True
    if rx(t, r"costs? \{?c\}? less"):
        return True
    if rx(t, r"ignore all \{?c\}? energy in the costs"):
        return True
    if rx(t, r"can use the.{0,30}attack for \{"):
        return True
    return False


def ab_dmg_boost(t):
    if rx(t, r"do \d+ more damage"):
        return True
    if rx(t, r"\d+ more damage to your opponent"):
        return True
    if rx(t, r"gets? \+\d+ hp"):  # HP buff treated under misc_buff too; keep boost narrow
        return False
    return False


def ab_disrupt_opp(t):
    if rx(t, r"discard.{0,40}from your opponent"):
        return True
    if rx(t, r"discard a random card from your opponent"):
        return True
    if rx(t, r"opponent reveals their hand"):
        return True
    if rx(t, r"opponent shuffle.{0,30}hand"):
        return True
    if rx(t, r"discard the top.{0,20}of your opponent'?s deck"):
        return True
    if rx(t, r"discard an energy from your opponent"):
        return True
    if rx(t, r"can'?t be put into your opponent'?s hand"):
        return True
    if rx(t, r"devolve 1 of your opponent"):
        return True
    if rx(t, r"put .{0,20}onto their bench") and "opponent" in t:
        return True
    if rx(t, r"put an energy attached to your opponent'?s.{0,20}into their hand"):
        return True
    if rx(t, r"discard an energy from the attacking pokemon"):
        return True
    if rx(t, r"discard a stadium"):
        return True
    return False


def ab_endure_survive_ko(t):
    # Would-be-KO survive (endure) effects.
    if rx(t, r"would be knocked out by damage.{0,40}it is not knocked out"):
        return True
    if rx(t, r"would be knocked out.{0,60}not knocked out"):
        return True
    if rx(t, r"is not knocked out, and its remaining hp becomes"):
        return True
    if rx(t, r"if this pokemon would be knocked out"):
        return True
    return False


def ab_misc_buff(t):
    # Passive stat buffs / type changes / retreat changes / evolve-twice etc.
    if rx(t, r"gets? \+\d+ hp"):
        return True
    if rx(t, r"have no retreat cost"):
        return True
    if rx(t, r"has no retreat cost"):
        return True
    if rx(t, r"\+\d+ hp"):
        return True
    if rx(t, r"it is \{[a-z]\} and \{[a-z]\} type"):
        return True
    if rx(t, r"may use an attack it has twice"):
        return True
    if rx(t, r"can use any attack from its previous evolutions"):
        return True
    if rx(t, r"this pokemon can use.{0,30}attack"):
        return True
    if rx(t, r"can evolve into any pokemon"):
        return True
    if rx(t, r"gets \+\d+ hp for each"):
        return True
    return False


ABILITY_FNS = [
    ("energy_accel_move", ab_energy_accel_move),
    ("put_into_play", ab_put_into_play),
    ("switch_gust", ab_switch_gust),
    ("search_deck", ab_search_deck),
    ("draw_dig", ab_draw_dig),
    ("dmg_reduction_wall", ab_dmg_reduction_wall),
    ("ability_lock", ab_ability_lock),
    ("direct_damage", ab_direct_damage),
    ("status", ab_status),
    ("immunity_protection", ab_immunity_protection),
    ("heal_remove_dmg", ab_heal_remove_dmg),
    ("prize_tempo", ab_prize_tempo),
    ("attack_cost_reduce", ab_attack_cost_reduce),
    ("dmg_boost", ab_dmg_boost),
    ("disrupt_opp", ab_disrupt_opp),
    ("endure_survive_ko", ab_endure_survive_ko),
    ("misc_buff", ab_misc_buff),
]


def classify_ability(text):
    t = lc(text)
    if not t:
        # ability-bearing card with empty text (alt print) -> all zeros except
        # deterministic is_passive (no activation markers -> passive).
        return ["uncategorized", "is_passive"]
    cats = [name for name, fn in ABILITY_FNS if fn(t)]
    if not cats:
        cats.append("uncategorized")
    if ability_is_passive(t):
        cats.append("is_passive")
    return cats


# ============================================================================
# TRAINER classifiers  (per cardId; Item/Supporter/Stadium/Tool)
# ============================================================================
TRAINER_CATS = [
    "DRAW_REFRESH", "SEARCH_POKEMON", "SEARCH_ENERGY", "SEARCH_ANY_TRAINER",
    "SWITCH", "DISRUPT_OPP", "ENERGY_ACCEL_RECUR", "HEAL_PROTECT", "DMG_MOD",
    "DIG_DECKMANIP", "MISC",
]


def tr_draw_refresh(t):
    if rx(t, r"draws? \d+ cards?"):       # "draw 4" or "draws 4"
        return True
    if rx(t, r"draws? a card"):
        return True
    if rx(t, r"draw cards until"):
        return True
    if rx(t, r"draw that many cards"):
        return True
    if rx(t, r"draw \d+ more cards"):
        return True
    if rx(t, r"shuffle your hand into your deck.{0,30}draw"):
        return True
    if rx(t, r"discard your hand and draw"):
        return True
    if rx(t, r"a card for each"):
        return True
    return False


def tr_search_pokemon(t):
    if rx(t, r"search (your|their) deck for .{0,60}pokemon"):
        return True
    if rx(t, r"search (your|their) deck for a pokemon"):
        return True
    return False


def tr_search_energy(t):
    if rx(t, r"search your deck for .{0,40}energy"):
        return True
    return False


def tr_search_any_trainer(t):
    if rx(t, r"search your deck for .{0,60}(trainer|item|supporter|stadium|tool) card"):
        return True
    if rx(t, r"search your deck for (a card|up to \d+ cards|\d+ cards)"):
        return True
    if rx(t, r"search your deck for up to \d+ cards"):
        return True
    return False


def tr_switch(t):
    if rx(t, r"switch in 1 of your opponent'?s benched pokemon"):
        return True
    if rx(t, r"switch your active pokemon with 1 of your benched"):
        return True
    if rx(t, r"switch out your opponent'?s active"):
        return True
    if rx(t, r"switch (their|your) active"):
        return True
    if "retreat cost" in t and "less" in t:
        return True
    if "no retreat cost" in t:
        return True
    return False


def tr_disrupt_opp(t):
    if rx(t, r"discard.{0,40}from (1 of )?your opponent"):
        return True
    if rx(t, r"discard a special energy from"):
        return True
    if rx(t, r"discard all pokemon tools and special energy"):
        return True
    if rx(t, r"opponent (discards|reveals their hand)"):
        return True
    if rx(t, r"opponent.{0,30}(shuffles their hand|on the bottom of their deck)"):
        return True
    if rx(t, r"discard a stadium"):
        return True
    if rx(t, r"devolve"):
        return True
    if rx(t, r"put \d+ damage counters? on 1 of your opponent"):
        return True
    if rx(t, r"on the bottom of their deck") and "opponent" in t:
        return True
    if rx(t, r"discard a pokemon tool"):
        return True
    if rx(t, r"have no effect") and "tool" in t:
        return True
    if rx(t, r"now burned and confused"):
        return True
    # Tool removal (incl. both-sides Tool Scrapper), hand-size attacks, prize peek
    if rx(t, r"pokemon tools? attached to pokemon.{0,40}discard them"):
        return True
    if rx(t, r"choose up to \d+ pokemon tools?.{0,40}discard them"):
        return True
    if rx(t, r"each player discards cards from their hand"):
        return True
    if rx(t, r"turn 1 of your opponent'?s face-down prize"):
        return True
    # Stadium/tool taxes & lockers that hit the opponent
    if rx(t, r"attacks used by your opponent'?s.{0,20}pokemon cost \{?c\}? more"):
        return True
    if rx(t, r"attacks used by .{0,30}pokemon.{0,20}cost \{?c\}? more"):
        return True
    if rx(t, r"pokemon.{0,30}have no abilities"):
        return True
    if rx(t, r"put an energy attached to 1 of your opponent'?s.{0,20}into their hand"):
        return True
    return False


def tr_energy_accel_recur(t):
    # Attach energy from discard pile / move energy on board; recover energy.
    if rx(t, r"attach .{0,40}energy.{0,30}from your discard pile"):
        return True
    if rx(t, r"move .{0,20}energy from"):
        return True
    if rx(t, r"attach a basic energy.{0,30}to"):
        return True
    if rx(t, r"basic energy cards? from your discard pile into your hand"):
        return True
    if rx(t, r"shuffle.{0,30}energy.{0,30}from your discard pile into your deck"):
        return True
    if rx(t, r"energy retrieval"):
        return True
    if rx(t, r"put up to \d+ basic energy cards? from your discard pile"):
        return True
    if rx(t, r"attach.{0,40}energy.{0,20}to each of your"):
        return True
    if rx(t, r"basic.{0,10}energy cards? from their discard pile into their hand"):
        return True
    return False


def tr_heal_protect(t):
    if rx(t, r"heal \d+ damage"):
        return True
    if rx(t, r"heal all damage"):
        return True
    if rx(t, r"remove a special condition"):
        return True
    if rx(t, r"prevent all damage"):
        return True
    if rx(t, r"prevent all effects of attacks"):
        return True
    if rx(t, r"recovers? from all special conditions"):
        return True
    if rx(t, r"is not knocked out, and its remaining hp"):
        return True
    if rx(t, r"\d+ less damage from attacks"):
        return True
    if rx(t, r"takes? \d+ less damage"):
        return True
    if rx(t, r"do \d+ less damage"):                 # opp attacks weakened
        return True
    if rx(t, r"prevent all effects of that card"):   # effect immunity (fossils)
        return True
    if rx(t, r"\+\d+ hp"):
        return True
    return False


def tr_dmg_mod(t):
    if rx(t, r"\d+ more damage to your opponent"):
        return True
    if rx(t, r"do \d+ more damage"):
        return True
    if rx(t, r"put \d+ damage counters?"):
        return True
    if rx(t, r"place \d+ damage counters?"):
        return True
    if rx(t, r"is now (burned|poisoned|confused|asleep|paralyzed)"):
        return True
    if rx(t, r"take \d+ more prize"):
        return True
    if rx(t, r"costs? \d+ energy less"):
        return True
    if rx(t, r"attack costs.{0,20}less"):
        return True
    if rx(t, r"put \d+ more damage counters?"):       # Perilous Jungle
        return True
    if rx(t, r"gets? -\d+ hp"):                        # Gravity Mountain
        return True
    return False


def tr_dig_deckmanip(t):
    if rx(t, r"look at the top \d+ cards?"):
        return True
    if rx(t, r"look at the bottom \d+ cards?"):
        return True
    if rx(t, r"reveal the top \d+ cards?"):
        return True
    if rx(t, r"discard the top \d+ cards? of your deck"):
        return True
    if rx(t, r"put them (back )?in any order"):
        return True
    if rx(t, r"on top of (it|your deck) in any order"):
        return True
    if rx(t, r"on the bottom of your deck"):
        return True
    if rx(t, r"prize card"):
        return True
    if rx(t, r"shuffle.{0,30}from your discard pile into your deck"):
        return True
    if rx(t, r"discard the top \d+ cards?"):
        return True
    # recover cards (Pokemon/Supporter/Trainer/combo) from discard pile to hand
    if rx(t, r"from your discard pile into your hand"):
        return True
    if rx(t, r"shuffle up to \d+ pokemon from your discard pile"):
        return True
    if rx(t, r"put a card from their hand on top of their deck"):
        return True
    if rx(t, r"top \d+ cards? of your opponent'?s deck"):
        return True
    if rx(t, r"discard the bottom card of your deck"):
        return True
    return False


TRAINER_FNS = [
    ("DRAW_REFRESH", tr_draw_refresh),
    ("SEARCH_POKEMON", tr_search_pokemon),
    ("SEARCH_ENERGY", tr_search_energy),
    ("SEARCH_ANY_TRAINER", tr_search_any_trainer),
    ("SWITCH", tr_switch),
    ("DISRUPT_OPP", tr_disrupt_opp),
    ("ENERGY_ACCEL_RECUR", tr_energy_accel_recur),
    ("HEAL_PROTECT", tr_heal_protect),
    ("DMG_MOD", tr_dmg_mod),
    ("DIG_DECKMANIP", tr_dig_deckmanip),
]


def classify_trainer(text):
    t = lc(text)
    if not t:
        return ["MISC"]
    cats = [name for name, fn in TRAINER_FNS if fn(t)]
    if not cats:
        cats = ["MISC"]
    return cats


# ============================================================================
# MAIN: build the lookups and emit rl/effect_data.py
# ============================================================================
def main():
    cards = all_card_data()
    attacks = {a.attackId: a for a in all_attack()}

    # ---- pool (for reporting only) ----
    pool = set()
    for v in DECKS.values():
        pool.update(v)
    for v in GENERATED.values():
        pool.update(v)

    # ---- ATTACKS (every attackId in all_attack) ----
    ATTACK_EFFECTS = {}
    for aid, a in attacks.items():
        ATTACK_EFFECTS[aid] = classify_attack(a.text)

    # ---- ABILITIES (Pokemon, cardType 0, with skills) ----
    ABILITY_EFFECTS = {}
    ability_card_ids = set()
    for c in cards:
        if c.cardType == 0 and c.skills:
            ABILITY_EFFECTS[c.cardId] = classify_ability(c.skills[0].text)
            ability_card_ids.add(c.cardId)

    # ---- TRAINERS (Item/Supporter/Stadium/Tool) ----
    TRAINER_EFFECTS = {}
    trainer_card_ids = set()
    for c in cards:
        if c.cardType in (1, 2, 3, 4):
            txt = c.skills[0].text if c.skills else ""
            TRAINER_EFFECTS[c.cardId] = classify_trainer(txt)
            trainer_card_ids.add(c.cardId)

    # ------------------------------------------------------------------
    # Reporting (printed to stdout; not embedded in module)
    # ------------------------------------------------------------------
    def avg_bits(eff, drop=None):
        n = len(eff)
        if not n:
            return 0.0
        tot = 0
        for cats in eff.values():
            cc = [x for x in cats if x != drop]
            tot += len(cc)
        return tot / n

    def fam_report(label, eff, ids_pool, cats, drop=None):
        from collections import Counter
        cnt = Counter()
        for cats_list in eff.values():
            for c in cats_list:
                if c != drop:
                    cnt[c] += 1
        npool = sum(1 for k in eff if k in ids_pool)
        print(f"\n=== {label} ===")
        print(f"  total ids labeled: {len(eff)}  (in pool: {npool})")
        print(f"  avg bits/card: {avg_bits(eff, drop):.2f}")
        for c in cats:
            if c == drop:
                continue
            print(f"    {c:24s} {cnt.get(c,0):5d}")

    print("=" * 70)
    print(f"cards={len(cards)} attacks={len(attacks)} pool_ids={len(pool)}")
    fam_report("ATTACK", ATTACK_EFFECTS, pool, ATTACK_CATS)
    fam_report("ABILITY", ABILITY_EFFECTS, pool, ABILITY_CATS, drop="is_passive")
    # passive split
    npass = sum(1 for v in ABILITY_EFFECTS.values() if "is_passive" in v)
    print(f"    [is_passive]            {npass:5d}  (activated: {len(ABILITY_EFFECTS)-npass})")
    fam_report("TRAINER", TRAINER_EFFECTS, pool, TRAINER_CATS)

    # ------------------------------------------------------------------
    # Emit rl/effect_data.py
    # ------------------------------------------------------------------
    out_path = os.path.join(ROOT, "rl", "effect_data.py")
    write_module(out_path, ATTACK_EFFECTS, ABILITY_EFFECTS, TRAINER_EFFECTS)
    print(f"\nWROTE {out_path}")

    # disputed-card resolution dump
    disputed = {
        1097: "Night Stretcher", 1110: "Max Rod", 1182: "Boss's Orders",
        1124: "Pokemon Catcher", 210: "Pikachu ex (Resolute Heart)",
        886: "Mega Hawlucha ex", 75: "Iron Leaves ex",
    }
    print("\n=== DISPUTED CARD RESOLUTION ===")
    for cid, nm in disputed.items():
        if cid in ABILITY_EFFECTS:
            print(f"  [{cid}] {nm} (ABILITY): {ABILITY_EFFECTS[cid]}")
        if cid in TRAINER_EFFECTS:
            print(f"  [{cid}] {nm} (TRAINER): {TRAINER_EFFECTS[cid]}")


def write_module(path, attack_eff, ability_eff, trainer_eff):
    def fmt(d):
        lines = []
        for k in sorted(d):
            v = d[k]
            inner = ", ".join(repr(x) for x in v)
            lines.append(f"    {k}: [{inner}],")
        return "\n".join(lines)

    body = f'''# -*- coding: utf-8 -*-
"""FROZEN effect-category multi-hot lookup. AUTO-GENERATED by
scripts/gen_effect_data.py -- DO NOT EDIT BY HAND.

Stdlib-only at import time (no SDK). Maps engine ids -> multi-hot category
vectors over fixed category orderings.

  ATTACK_CATS / ABILITY_CATS / TRAINER_CATS : ordered category name lists
  ATTACK_EFFECTS / ABILITY_EFFECTS / TRAINER_EFFECTS : {{id: [cat_name, ...]}}
  attack_multihot(attackId) / ability_multihot(cardId) / trainer_multihot(cardId)
      -> fixed-width list[float] (0/1) over the matching CATS; unknown id -> zeros
  N_ATTACK_FX / N_ABILITY_FX / N_TRAINER_FX : vector widths

ABILITY_CATS last bit `is_passive` is computed DETERMINISTICALLY by the
generator (activated iff text has "once during your turn" / "you may use this
ability" / "as often as you like during your turn" / "once during each" /
"once during your first turn"; else passive).
"""

ATTACK_CATS = {ATTACK_CATS!r}
ABILITY_CATS = {ABILITY_CATS!r}
TRAINER_CATS = {TRAINER_CATS!r}

N_ATTACK_FX = len(ATTACK_CATS)
N_ABILITY_FX = len(ABILITY_CATS)
N_TRAINER_FX = len(TRAINER_CATS)

_ATTACK_IDX = {{c: i for i, c in enumerate(ATTACK_CATS)}}
_ABILITY_IDX = {{c: i for i, c in enumerate(ABILITY_CATS)}}
_TRAINER_IDX = {{c: i for i, c in enumerate(TRAINER_CATS)}}

ATTACK_EFFECTS = {{
{fmt(attack_eff)}
}}

ABILITY_EFFECTS = {{
{fmt(ability_eff)}
}}

TRAINER_EFFECTS = {{
{fmt(trainer_eff)}
}}


def _multihot(cats, idx, n):
    v = [0.0] * n
    for c in cats:
        i = idx.get(c)
        if i is not None:
            v[i] = 1.0
    return v


def attack_multihot(attack_id):
    """Fixed-width 0/1 vector over ATTACK_CATS; unknown id -> all zeros."""
    return _multihot(ATTACK_EFFECTS.get(attack_id, ()), _ATTACK_IDX, N_ATTACK_FX)


def ability_multihot(card_id):
    """Fixed-width 0/1 vector over ABILITY_CATS; unknown id -> all zeros."""
    return _multihot(ABILITY_EFFECTS.get(card_id, ()), _ABILITY_IDX, N_ABILITY_FX)


def trainer_multihot(card_id):
    """Fixed-width 0/1 vector over TRAINER_CATS; unknown id -> all zeros."""
    return _multihot(TRAINER_EFFECTS.get(card_id, ()), _TRAINER_IDX, N_TRAINER_FX)
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


if __name__ == "__main__":
    main()
