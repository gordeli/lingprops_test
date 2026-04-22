"""Tangibility (BWK concreteness rating) computation.

Uses the Brysbaert, Warriner & Kuperman (2014) human-rated concreteness
norms (~40K English words, 1–5 scale) to compute a tangibility score for
text.  The score is the average BWK rating across content words that have
a non-zero (i.e., present in the BWK table) rating.

Design mirrors ``compute_concreteness``:

* Each POS category is scored independently.
* Both with-repetitions and without-repetitions (unique lemmas) variants
  are computed in a single call.
* Normalized by the count of words with a non-zero BWK rating.

Reference
---------
Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness
ratings for 40 thousand generally known English word lemmas. *Behavior
Research Methods*, 46(3), 904–911.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Iterable, Tuple

DEFAULT_POS_GROUPS: Tuple[str, ...] = ("NN", "VB", "JJ", "RB")

_DATA_DIR = Path(__file__).resolve().parent / "data"
_BWK_FILE = _DATA_DIR / "Concreteness_ratings_Brysbaert_et_al_BRM.xls"


@functools.lru_cache(maxsize=1)
def _load_bwk() -> Dict[str, float]:
    """Load the BWK ratings into a {word: Conc.M} dict (cached)."""
    import pandas as pd
    df = pd.read_excel(_BWK_FILE)
    return dict(zip(df["Word"].str.lower(), df["Conc.M"]))


def _tag_to_wn_pos(tag: str):
    """Penn Treebank tag → WordNet POS character for lemmatisation."""
    if tag.startswith("NN") or tag.startswith("CD"):
        return "n"
    if tag.startswith("VB"):
        return "v"
    if tag.startswith("JJ"):
        return "a"
    if tag.startswith("RB"):
        return "r"
    return None


def _compute_tang_pos(word_forms, postag_prefixes, wnl, bwk,
                      exclusion_list):
    """Score a single POS partition — WITH repetitions.

    For each content word matching *postag_prefixes*:
    1. Lemmatise (WordNet lemmatiser).
    2. Look up the lemma in the BWK table.
    3. If found, accumulate ``rating × frequency``.

    Returns (score_sum, norm_count) where
    ``tangibility = score_sum / norm_count``.
    """
    score = 0.0
    norm_count = 0

    for (word, tag), freq in word_forms.items():
        if tag[:2] not in postag_prefixes:
            continue
        wn_pos = _tag_to_wn_pos(tag)
        if wn_pos is None:
            continue
        lemma = wnl.lemmatize(word, wn_pos)
        if lemma in exclusion_list:
            continue
        rating = bwk.get(lemma)
        if rating is None:
            continue
        score += rating * freq
        norm_count += freq

    return score, norm_count


def _compute_tang_pos_norep(word_forms, postag_prefixes, wnl, bwk,
                            exclusion_list):
    """Score a single POS partition — WITHOUT repetitions.

    Deduplication by lemma (before any further processing), strictly
    within the POS partition.  Each unique lemma contributes once.

    Returns (score_sum, norm_count).
    """
    score = 0.0
    norm_count = 0
    seen_lemmas: set[str] = set()

    for (word, tag), freq in word_forms.items():
        if tag[:2] not in postag_prefixes:
            continue
        wn_pos = _tag_to_wn_pos(tag)
        if wn_pos is None:
            continue
        lemma = wnl.lemmatize(word, wn_pos)
        if lemma in seen_lemmas:
            continue
        seen_lemmas.add(lemma)
        if lemma in exclusion_list:
            continue
        rating = bwk.get(lemma)
        if rating is None:
            continue
        score += rating
        norm_count += 1

    return score, norm_count


def compute_tangibility(
    text: str,
    pos_groups: Iterable[str] = DEFAULT_POS_GROUPS,
    exclude: Iterable[str] = (),
) -> Dict[str, Dict[str, float]]:
    """Compute BWK tangibility (concreteness-rating) metrics for a text.

    Each POS category is treated as a fully independent partition.
    Two variants are computed simultaneously:

    - **With repetitions** (``score``, ``count``, ``normalized_score``):
      every word token counts with its actual frequency.
    - **Without repetitions** (``score_norep``, ``count_norep``,
      ``normalized_score_norep``): unique lemmas only (f = 1).
      Uniqueness is checked at the lemma level, within each POS.

    The normalised score is the **average BWK rating** (1–5 scale) across
    words that have a BWK entry.

    Parameters
    ----------
    text : str
        Input text.
    pos_groups : iterable of str, default ("NN","VB","JJ","RB")
        POS prefixes to include.  CD is omitted by default (numbers
        rarely have BWK ratings).
    exclude : iterable of str, default ()
        Lemmas to exclude from calculation.

    Returns
    -------
    dict
        Per-POS metrics and overall totals.  Each POS entry contains:

        - ``score`` / ``count`` / ``normalized_score`` – with repetitions.
        - ``score_norep`` / ``count_norep`` / ``normalized_score_norep``
          – without repetitions.

        The ``"total"`` entry is the weighted average across all POS.
    """
    from .concreteness import _init_legacy

    legacy = _init_legacy()
    wnl = legacy.wnl
    bwk = _load_bwk()

    word_forms = legacy.wordformtion(text)
    exclusion_list = set(exclude)

    results: Dict[str, Dict[str, float]] = {}
    total_score = 0.0
    total_count = 0
    total_score_nr = 0.0
    total_count_nr = 0

    for pos in pos_groups:
        prefixes = ["NN", "CD"] if pos == "NN" else [pos]

        s, c = _compute_tang_pos(
            word_forms, prefixes, wnl, bwk, exclusion_list,
        )
        s_nr, c_nr = _compute_tang_pos_norep(
            word_forms, prefixes, wnl, bwk, exclusion_list,
        )

        results[pos] = {
            "score": s,
            "count": c,
            "normalized_score": s / c if c > 0 else 0.0,
            "score_norep": s_nr,
            "count_norep": c_nr,
            "normalized_score_norep": s_nr / c_nr if c_nr > 0 else 0.0,
        }
        total_score += s
        total_count += c
        total_score_nr += s_nr
        total_count_nr += c_nr

    results["total"] = {
        "score": total_score,
        "count": total_count,
        "normalized_score": total_score / total_count if total_count > 0 else 0.0,
        "score_norep": total_score_nr,
        "count_norep": total_count_nr,
        "normalized_score_norep": (
            total_score_nr / total_count_nr if total_count_nr > 0 else 0.0
        ),
    }
    return results
