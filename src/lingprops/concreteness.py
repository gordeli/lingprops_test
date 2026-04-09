"""Public API for concreteness computation built on top of the original code.

We keep the original implementation in `_concreteness_legacy.py` untouched and
provide a clean, documented function here.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

DEFAULT_POS_GROUPS: Tuple[str, ...] = ("NN", "VB", "JJ", "RB", "CD")

# Mapping from POS group labels to the Penn Treebank tag prefixes they cover
_POS_TAG_PREFIXES: Dict[str, List[str]] = {
    "NN": ["NN"],        # NN, NNS, NNP, NNPS
    "VB": ["VB"],        # VB, VBD, VBG, VBN, VBP, VBZ
    "JJ": ["JJ"],        # JJ, JJR, JJS
    "RB": ["RB"],        # RB, RBR, RBS
    "CD": ["CD"],        # Cardinal numbers
}

def ensure_nltk_data() -> None:
    """Download NLTK resources required by the concreteness code.

    We attempt both legacy and new resource names where applicable, quietly.
    """
    import nltk
    needed = [
        ("tokenizers", "punkt"),
        ("tokenizers", "punkt_tab"),  # best-effort; may not exist on all installs
        ("taggers", "averaged_perceptron_tagger"),
        ("taggers", "averaged_perceptron_tagger_eng"),  # newer naming on some builds
        ("corpora", "wordnet"),
        ("corpora", "omw-1.4"),
    ]

    def _have(kind: str, res: str) -> bool:
        try:
            nltk.data.find(f"{kind}/{res}")
            return True
        except LookupError:
            return False

    for kind, res in needed:
        if not _have(kind, res):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                # Non-fatal; some resources (e.g., punkt_tab) may not exist
                pass


def _init_legacy():
    """Ensure NLTK data is present and return a ready-to-use legacy module."""
    ensure_nltk_data()
    from importlib import reload
    from . import _concreteness_legacy as legacy
    try:
        legacy = reload(legacy)
    except Exception:
        pass
    from nltk.corpus import wordnet as wn
    _ = wn.synsets("dog", pos="n")
    legacy.wn = wn
    return legacy


def count_words(text: str) -> Dict[str, int]:
    """Count total words and content-word counts by POS category.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    dict
        ``{"total": int, "NN": int, "VB": int, "JJ": int, "RB": int, "CD": int}``
        where each content-word count includes all Penn Treebank sub-tags
        (e.g. ``"NN"`` covers NN, NNS, NNP, NNPS).
    """
    legacy = _init_legacy()
    word_forms = legacy.wordformtion(text)

    total = 0
    counts: Dict[str, int] = {pos: 0 for pos in _POS_TAG_PREFIXES}

    for (word, tag), freq in word_forms.items():
        total += freq
        for pos_label, prefixes in _POS_TAG_PREFIXES.items():
            if any(tag.startswith(p) for p in prefixes):
                counts[pos_label] += freq
                break

    counts["total"] = total
    return counts


def _tag_to_wn_pos(tag: str):
    """Convert a Penn Treebank POS tag to the WordNet POS character
    used by ``WordNetLemmatizer.lemmatize``."""
    if tag.startswith("NN") or tag.startswith("CD"):
        return "n"
    if tag.startswith("VB"):
        return "v"
    if tag.startswith("JJ"):
        return "a"
    if tag.startswith("RB"):
        return "r"
    return None


def _score_wordform(wordform, word_forms, nouns, frequency, exclusion_list,
                    legacy):
    """Score a single wordform.  Returns (conc_delta, valid)."""
    import numpy as np
    from scipy.special import comb

    try:
        noun = nouns[wordform]
    except KeyError:
        return 0.0, False
    if noun in exclusion_list:
        return 0.0, False

    conc = 0.0
    valid = False

    if isinstance(noun, list):
        for n in noun:
            try:
                depth = legacy.hyp_num(n, wordform[1])
            except Exception:
                continue
            if len(n) < 2 or (depth == 0 and n != "entity"):
                continue
            conc += np.log(comb(depth + 1 + frequency - 1, frequency))
            valid = True
    else:
        try:
            depth = legacy.hyp_num(noun, wordform[1])
            if wordform[0] in ("gameplay", "ios", "pt") or \
               "smartophone" in wordform[0]:
                depth += 1
        except Exception:
            return 0.0, False
        if len(noun) < 2 or (depth == 0 and noun != "entity"):
            return 0.0, False
        conc = np.log(comb(depth + 1 + frequency - 1, frequency))
        valid = True

    return conc, valid


def _compute_pos_score(word_forms, nouns, postag_prefixes,
                       exclusion_list, legacy):
    """Compute concreteness for a single POS partition — WITH repetitions.

    Returns (score, normalization_count).
    """
    conc = 0.0
    norm_count = 0

    for wordform in word_forms:
        if wordform[1][:2] not in postag_prefixes:
            continue
        frequency = word_forms[wordform]
        delta, valid = _score_wordform(
            wordform, word_forms, nouns, frequency, exclusion_list, legacy,
        )
        if valid:
            conc += delta
            norm_count += frequency

    return conc, norm_count


def _compute_pos_score_norep(word_forms, nouns, postag_prefixes,
                             exclusion_list, legacy):
    """Compute concreteness for a single POS partition — WITHOUT repetitions.

    Deduplication is by **lemma** (before nounification), strictly within
    the POS partition.  Each unique lemma contributes with ``f = 1``.

    Returns (score, normalization_count).
    """
    conc = 0.0
    norm_count = 0
    seen_lemmas: set[str] = set()
    wnl = legacy.wnl  # WordNetLemmatizer

    for wordform in word_forms:
        if wordform[1][:2] not in postag_prefixes:
            continue

        # Lemmatise (before nounification) for deduplication
        wn_pos = _tag_to_wn_pos(wordform[1])
        lemma = wnl.lemmatize(wordform[0], wn_pos) if wn_pos else wordform[0]
        if lemma in seen_lemmas:
            continue
        seen_lemmas.add(lemma)

        frequency = 1  # unique-word mode
        delta, valid = _score_wordform(
            wordform, word_forms, nouns, frequency, exclusion_list, legacy,
        )
        if valid:
            conc += delta
            norm_count += 1

    return conc, norm_count


def compute_concreteness(
    text: str,
    pos_groups: Iterable[str] = DEFAULT_POS_GROUPS,
    exclude: Iterable[str] = (),
) -> Dict[str, Dict[str, float]]:
    """Compute concreteness metrics for a text.

    Each POS category (nouns, verbs, adjectives, adverbs) is treated as
    a fully independent partition.  Tokenization and noun-lemmatisation
    are performed once, then each POS partition is scored in isolation.

    Two variants are computed simultaneously:

    - **With repetitions** (``score``, ``count``, ``normalized_score``):
      every word token counts with its actual frequency ``f``.
    - **Without repetitions** (``score_norep``, ``count_norep``,
      ``normalized_score_norep``): the text is treated as a bag of
      unique lemmas (``f = 1``).  Uniqueness is checked at the lemma
      level (before nounification), strictly within each POS.

    Parameters
    ----------
    text : str
        Input text.
    pos_groups : iterable of str, default ("NN","VB","JJ","RB","CD")
        POS prefixes to include.
    exclude : iterable of str, default ()
        Words to exclude from calculation.

    Returns
    -------
    dict
        Per-POS metrics and overall totals.  Each POS entry contains:

        - ``score`` / ``count`` / ``normalized_score`` – with repetitions.
        - ``score_norep`` / ``count_norep`` / ``normalized_score_norep``
          – without repetitions (unique lemmas, ``f = 1``).

        The ``"total"`` entry additionally contains:

        - ``word_count`` – total number of word tokens in the text.
        - ``content_word_counts`` – word counts per content POS category.
    """
    legacy = _init_legacy()

    # --- Preprocess ONCE for the entire text ---
    word_forms = legacy.wordformtion(text)
    nouns, _ = legacy.noun_lemmas(word_forms)

    exclude_list = list(exclude)
    results: Dict[str, Dict[str, float]] = {}
    total_score = 0.0
    total_count = 0
    total_score_nr = 0.0
    total_count_nr = 0

    # --- Compute each POS partition independently ---
    for pos in pos_groups:
        prefixes = ["NN", "CD"] if pos == "NN" else [pos]

        # With repetitions
        score, count = _compute_pos_score(
            word_forms, nouns, prefixes, exclude_list, legacy,
        )
        s = float(score or 0.0)
        c = int(count or 0)

        # Without repetitions (unique lemmas, f=1)
        score_nr, count_nr = _compute_pos_score_norep(
            word_forms, nouns, prefixes, exclude_list, legacy,
        )
        s_nr = float(score_nr or 0.0)
        c_nr = int(count_nr or 0)

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

    # Word counts from the already-tokenised word_forms (no re-tokenisation)
    wc_total = 0
    wc_pos: Dict[str, int] = {pos: 0 for pos in _POS_TAG_PREFIXES}
    for (word, tag), freq in word_forms.items():
        wc_total += freq
        for pos_label, prefixes in _POS_TAG_PREFIXES.items():
            if any(tag.startswith(p) for p in prefixes):
                wc_pos[pos_label] += freq
                break

    results["total"] = {
        "score": float(total_score),
        "count": int(total_count),
        "normalized_score": total_score / total_count if total_count > 0 else 0.0,
        "score_norep": float(total_score_nr),
        "count_norep": int(total_count_nr),
        "normalized_score_norep": (
            total_score_nr / total_count_nr if total_count_nr > 0 else 0.0
        ),
        "word_count": wc_total,
        "content_word_counts": wc_pos,
    }
    return results
