"""lingprops: Linguistic property utilities (incl. concreteness).

High-level functions:
    from lingprops import compute_concreteness, compute_tangibility, ensure_nltk_data
"""
from .concreteness import compute_concreteness, count_words, ensure_nltk_data
from .exact_count import compute_exact_text_count, compute_exact_text_count_optimized
from .tangibility import compute_tangibility

__all__ = [
    "compute_concreteness",
    "compute_exact_text_count",
    "compute_exact_text_count_optimized",
    "compute_tangibility",
    "count_words",
    "ensure_nltk_data",
]
__version__ = "0.1.0"
