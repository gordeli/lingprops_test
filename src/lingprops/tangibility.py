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
        # A word used only as an auxiliary has been moved to the 'AU' tag by
        # ``wordformtion``, leaving this entry at zero.  The with-repetitions
        # path multiplies by ``freq`` and so ignores it; this path does not
        # look at ``freq`` at all and would otherwise score it once.  Tested
        # before ``seen_lemmas`` is updated so a genuine later wordform with
        # the same lemma still counts.
        if freq <= 0:
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
    word_forms = legacy.wordformtion(text)
    return _score_tangibility(
        word_forms, legacy, pos_groups=pos_groups, exclude=exclude,
    )


def _score_tangibility(
    word_forms,
    legacy,
    *,
    pos_groups: Iterable[str] = DEFAULT_POS_GROUPS,
    exclude: Iterable[str] = (),
) -> Dict[str, Dict[str, float]]:
    """Score tangibility from a pre-computed ``word_forms`` dict.

    Internal: shared between :func:`compute_tangibility` and
    :func:`compute_all`.
    """
    wnl = legacy.wnl
    bwk = _load_bwk()
    exclusion_list = set(exclude)

    results: Dict[str, Dict[str, float]] = {}
    total_score = 0.0
    total_count = 0
    total_score_nr = 0.0
    total_count_nr = 0

    pos_groups = list(pos_groups)
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

        # Cardinals live in the noun partition (see the matching comment in
        # concreteness._score_concreteness).  DEFAULT_POS_GROUPS here does not
        # include "CD", so this guard only bites when a caller asks for it
        # explicitly — but then it must not be added to the totals twice.
        if pos == "CD" and "NN" in pos_groups:
            continue

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


# ---------------------------------------------------------------------------
# Classic BWK text score — the measure as the literature computes it
# ---------------------------------------------------------------------------

def compute_bwk_classic(text: str, *, lemmatize: bool = False) -> Dict[str, float]:
    """Brysbaert-style text concreteness, as published pipelines compute it.

    Every token of the text is looked up in the BWK table and the ratings of
    those found are averaged, **with repetitions and with no POS filtering**:
    function words, auxiliaries and determiners all count if they are rated
    (and most frequent words are — ``the`` 1.43, ``of`` 1.67, ``was`` 1.69).

    This is deliberately *not* :func:`compute_tangibility`, which restricts
    itself to content words so that it is computed on the same word set as the
    specificity score.  Reproducing the published measure needs this looser
    definition: against Le et al.'s own ``bryscore`` the all-token variant
    correlates r = .92, where our content-word tangibility reaches r = .81 and
    sits 0.43 higher in level (see ``data/bwk_comparison.md`` in the
    Time_construal_review project).

    Parameters
    ----------
    text : str
        Input text.
    lemmatize : bool, default False
        ``False`` reproduces the common implementation: lowercase the token and
        look it up verbatim.  ``True`` tries the WordNet lemma of the token
        under each open-class part of speech and takes the first hit, which
        raises coverage by a few words per text but moves further from the
        published pipelines.

    Returns
    -------
    dict
        ``score`` – mean BWK rating of the matched tokens (``nan`` if none).
        ``count`` – number of matched tokens (the denominator).
        ``tokens`` – number of word tokens considered (punctuation excluded).
        ``coverage`` – ``count / tokens``.

    Notes
    -----
    The BWK release also rates ~2,900 two-word expressions; like the published
    implementations, this function matches single tokens only.
    """
    import math
    import re

    bwk = _load_bwk()
    if not text:
        return {"score": math.nan, "count": 0, "tokens": 0, "coverage": 0.0}

    import nltk
    from .concreteness import ensure_nltk_data
    ensure_nltk_data()

    toks = [t.lower() for t in nltk.word_tokenize(text)]
    toks = [t for t in toks if re.search(r"\w", t)]

    wnl = None
    if lemmatize:
        from . import _wordnet
        wnl = _wordnet.get_lemmatizer()

    ratings = []
    for t in toks:
        r = bwk.get(t)
        if r is None and wnl is not None:
            for pos in ("n", "v", "a", "r"):
                r = bwk.get(wnl.lemmatize(t, pos))
                if r is not None:
                    break
        if r is not None:
            ratings.append(r)

    n = len(ratings)
    return {
        "score": (sum(ratings) / n) if n else math.nan,
        "count": n,
        "tokens": len(toks),
        "coverage": (n / len(toks)) if toks else 0.0,
    }
