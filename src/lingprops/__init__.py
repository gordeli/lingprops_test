"""lingprops: Linguistic property utilities (incl. concreteness).

High-level functions:
    from lingprops import compute_concreteness, ensure_nltk_data

Word-sense disambiguation strategies (for ``compute_concreteness(..., wsd=...)``)
are listed in :data:`WSD_CHOICES` and implemented in :mod:`lingprops.wsd`.
"""
from .concreteness import (
    compute_concreteness,
    count_words,
    ensure_nltk_data,
    DEFAULT_WSD,
    WSD_CHOICES,
)
from .exact_count import compute_exact_text_count, compute_exact_text_count_optimized
from .tangibility import compute_tangibility
from . import wsd

__all__ = [
    "compute_concreteness",
    "compute_exact_text_count",
    "compute_exact_text_count_optimized",
    "compute_tangibility",
    "count_words",
    "ensure_nltk_data",
    "DEFAULT_WSD",
    "WSD_CHOICES",
    "wsd",
]
__version__ = "0.1.0"
