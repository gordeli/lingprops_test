"""Public API for concreteness computation built on top of the original code.

We keep the original implementation in `_concreteness_legacy.py` untouched and
provide a clean, documented function here.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

DEFAULT_POS_GROUPS: Tuple[str, ...] = ("NN", "VB", "JJ", "RB", "CD")

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


def compute_concreteness(
    text: str,
    pos_groups: Iterable[str] = DEFAULT_POS_GROUPS,
    repetitions: bool = True,
    exclude: Iterable[str] = (),
) -> Dict[str, Dict[str, float]]:
    """Compute concreteness-like metrics for a text.

    Parameters
    ----------
    text : str
        Input text.
    pos_groups : iterable of str, default ("NN","VB","JJ","RB","CD")
        POS prefixes to include (as used in the legacy code).
    repetitions : bool, default True
        Whether to count repeated wordforms (True) or unique (False).
    exclude : iterable of str, default ()
        Words to exclude from calculation.

    Returns
    -------
    dict
        Per-POS metrics and overall totals.
    """
    # 1) Ensure resources are present
    ensure_nltk_data()

    # 2) Import/reload legacy AFTER data is present; then force-bind WordNet
    from importlib import reload
    from . import _concreteness_legacy as legacy
    try:
        legacy = reload(legacy)
    except Exception:
        pass

    from nltk.corpus import wordnet as wn
    # Touch WordNet to ensure it's loadable; raises if truly missing
    _ = wn.synsets("dog", pos="n")
    legacy.wn = wn  # critical: ensure legacy module uses a valid handle

    exclude_list = list(exclude)
    results: Dict[str, Dict[str, float]] = {}
    total_score = 0.0
    total_count = 0

    for pos in pos_groups:
        score, count = legacy.text_depth(
            text,
            [pos] if pos != "NN" else ["NN", "CD"],
            repetitions,
            exclude_list,
        )
        results[pos] = {"score": float(score or 0.0), "count": int(count or 0)}
        total_score += float(score or 0.0)
        total_count += int(count or 0)

    results["total"] = {"score": float(total_score), "count": int(total_count)}
    return results
