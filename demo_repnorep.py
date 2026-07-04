"""Numerical demo: rep vs norep concreteness scores and denominators.

Runs the real library so every number is what the code actually produces.
We isolate one POS (nouns, "NN") to keep the arithmetic readable.
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                                 "..", "GitHub", "lingprops_repackaged", "src"))
# Fallback: use the known absolute src path
sys.path.insert(0, r"C:\GitHub\lingprops_repackaged\src")

import numpy as np
from scipy.special import comb
from lingprops import compute_concreteness
from lingprops import concreteness as C

legacy = C._init_legacy()


def depth_of(word, POS="NN"):
    """Depth the library assigns to a noun form under the 'first' strategy."""
    nouns, _ = legacy.noun_lemmas({(word, POS): 1})
    lemma = nouns[(word, POS)]
    if isinstance(lemma, list):
        return [(l, legacy.hyp_num(l, POS)) for l in lemma], lemma
    return legacy.hyp_num(lemma, POS), lemma


def multiset_term(depth, f):
    """The per-wordform contribution log C(depth + f, f)."""
    return math.log(comb(depth + f, f))


def show(title, text, wsd="first"):
    print("=" * 70)
    print(title)
    print("text:", repr(text))
    r = compute_concreteness(text, wsd=wsd, ner=False)["NN"]
    print(f"  REP   : score={r['score']:.4f}  denom(count)={r['count']}"
          f"  normalized={r['normalized_score']:.4f}")
    print(f"  NOREP : score={r['score_norep']:.4f}  denom(count_norep)={r['count_norep']}"
          f"  normalized={r['normalized_score_norep']:.4f}")
    return r


print("\n### Depths used below (strategy='first') ###")
for w in ["cat", "dog", "bank"]:
    d, lemma = depth_of(w)
    print(f"  {w:6s} -> lemma={lemma!r:12s} depth={d}")

d_cat = depth_of("cat")[0]
d_dog = depth_of("dog")[0]

# ----------------------------------------------------------------------
# Case 1: a single noun, one occurrence.  rep == norep here.
# ----------------------------------------------------------------------
show("CASE 1  single 'cat' (f=1)", "The cat slept.")
print(f"  hand-check: log C({d_cat}+1,1)=log({d_cat+1}) = {multiset_term(d_cat,1):.4f}")

# ----------------------------------------------------------------------
# Case 2: same noun repeated in ONE sentence (f=3).
#   rep  : one term log C(depth+3,3), denom += 3
#   norep: deduped to one lemma, log(depth+1), denom = 1
# ----------------------------------------------------------------------
r2 = show("CASE 2  'cat' x3 in one sentence (f=3)",
          "The cat saw the cat near the cat.")
print(f"  hand-check REP  : log C({d_cat}+3,3)=log C({d_cat+3},3) = {multiset_term(d_cat,3):.4f}, denom=3")
print(f"  hand-check NOREP: log({d_cat}+1) = {multiset_term(d_cat,1):.4f}, denom=1")

# ----------------------------------------------------------------------
# Case 3: same noun spread across 3 DIFFERENT sentences (f=3 total).
#   Whole-text aggregation => identical to Case 2.
#   This is the key point: per-sentence contribution does NOT double count.
# ----------------------------------------------------------------------
r3 = show("CASE 3  'cat' once in each of 3 sentences (f=3 total)",
          "The cat slept. A cat ran. My cat purred.")
print("  -> IDENTICAL to Case 2: aggregation is whole-text, not per-sentence.")
assert abs(r2["score"] - r3["score"]) < 1e-9
assert r2["count"] == r3["count"] == 3
assert r2["count_norep"] == r3["count_norep"] == 1

# ----------------------------------------------------------------------
# Case 4: two DIFFERENT nouns, one each (cat, dog).
#   rep  : sum of two f=1 terms, denom = 2
#   norep: same two lemmas, denom = 2  (rep==norep, no repetition)
# ----------------------------------------------------------------------
show("CASE 4  'cat' + 'dog' (distinct, f=1 each)", "The cat and the dog played.")
print(f"  hand-check: log({d_cat}+1)+log({d_dog}+1) = "
      f"{multiset_term(d_cat,1)+multiset_term(d_dog,1):.4f}, denom=2")

# ----------------------------------------------------------------------
# Case 5: mix -- 'cat' x3 and 'dog' x1.
#   rep  : log C(dcat+3,3) + log C(ddog+1,1), denom = 3+1 = 4
#   norep: log(dcat+1) + log(ddog+1),         denom = 1+1 = 2
# ----------------------------------------------------------------------
r5 = show("CASE 5  'cat' x3 + 'dog' x1",
          "The cat chased a cat while another cat watched the dog.")
rep_hand = multiset_term(d_cat, 3) + multiset_term(d_dog, 1)
norep_hand = multiset_term(d_cat, 1) + multiset_term(d_dog, 1)
print(f"  hand-check REP  : {rep_hand:.4f}, denom=4, norm={rep_hand/4:.4f}")
print(f"  hand-check NOREP: {norep_hand:.4f}, denom=2, norm={norep_hand/2:.4f}")

# ----------------------------------------------------------------------
# Case 6: limiting case -- how repetition inflates the REP term.
#   Show log C(depth+f, f) growth vs f for a fixed depth, and that
#   norep is flat (always log(depth+1)).
# ----------------------------------------------------------------------
print("=" * 70)
print(f"CASE 6  repetition growth of the REP term for depth(cat)={d_cat}")
print(f"  {'f':>3} {'REP term=logC(d+f,f)':>22} {'REP/f (normalized)':>20} {'NOREP term':>12}")
for f in [1, 2, 3, 5, 10, 50]:
    rep = multiset_term(d_cat, f)
    print(f"  {f:>3} {rep:>22.4f} {rep/f:>20.4f} {multiset_term(d_cat,1):>12.4f}")
print("  Note: NOREP term is constant (log(depth+1)); REP grows sublinearly,")
print("  so REP-normalized (score/denom) DECREASES as a word is repeated.")

# ----------------------------------------------------------------------
# Case 7: WSD -- senses are NOT separate; only the (word,POS) form matters.
#   'bank' x2; the picker assigns ONE depth for the form regardless of
#   how many senses appear. Counts are identical across strategies;
#   only the score can differ because a different single depth is chosen.
# ----------------------------------------------------------------------
print("=" * 70)
print("CASE 7  WSD: 'bank' twice, two plausible senses in context")
txt = "I sat by the river bank. Then I went to the bank to deposit cash."
for strat in ["first", "lesk"]:
    r = compute_concreteness(txt, wsd=strat, ner=False)["NN"]
    print(f"  wsd={strat:5s}: REP score={r['score']:.4f} denom={r['count']} | "
          f"NOREP score={r['score_norep']:.4f} denom={r['count_norep']}")
print("  -> denominators do NOT change with WSD: 'bank' is ONE wordform")
print("     (rep denom counts its frequency; norep counts it once).")
print("     WSD only changes WHICH single depth the form gets.")
