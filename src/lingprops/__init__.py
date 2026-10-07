"""lingprops: Linguistic property utilities (incl. concreteness).

High-level functions:
    from lingprops import compute_concreteness, ensure_nltk_data

Word-sense disambiguation strategies (for ``compute_concreteness(..., wsd=...)``)
are listed in :data:`WSD_CHOICES` and implemented in :mod:`lingprops.wsd`.
"""
from .concreteness import (
    compute_all,
    compute_concreteness,
    count_words,
    ensure_nltk_data,
    DEFAULT_WSD,
    WSD_CHOICES,
    DEFAULT_NER,
    DEFAULT_NER_BACKEND,
    NER_BACKENDS,
)
from ._wordnet import (
    DEFAULT_WORDNET_VERSION,
    WORDNET_VERSIONS,
    installed_wordnet_version,
    set_wordnet_version,
    wordnet_version,
)
from .exact_count import compute_exact_text_count, compute_exact_text_count_optimized
from .tangibility import compute_bwk_classic, compute_tangibility
from . import wsd
from . import ner
from .ner import ensure_spacy_model

__all__ = [
    "compute_all",
    "compute_concreteness",
    "compute_exact_text_count",
    "compute_exact_text_count_optimized",
    "compute_bwk_classic",
    "compute_tangibility",
    "count_words",
    "ensure_nltk_data",
    "ensure_spacy_model",
    "DEFAULT_WSD",
    "WSD_CHOICES",
    "DEFAULT_NER",
    "DEFAULT_NER_BACKEND",
    "NER_BACKENDS",
    "DEFAULT_WORDNET_VERSION",
    "WORDNET_VERSIONS",
    "installed_wordnet_version",
    "set_wordnet_version",
    "wordnet_version",
    "wsd",
    "ner",
]
__version__ = "1.4.0"
