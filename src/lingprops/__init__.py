"""lingprops: Linguistic property utilities (incl. concreteness).

High-level functions:
    from lingprops import compute_concreteness, ensure_nltk_data
"""
from .concreteness import compute_concreteness, count_words, ensure_nltk_data
from .exact_count import compute_exact_text_count, compute_exact_text_count_optimized

__all__ = [
    "compute_concreteness",
    "compute_exact_text_count",
    "compute_exact_text_count_optimized",
    "count_words",
    "ensure_nltk_data",
]
__version__ = "0.1.0"
