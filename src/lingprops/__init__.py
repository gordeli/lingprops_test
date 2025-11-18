"""lingprops: Linguistic property utilities (incl. concreteness).

High-level functions:
    from lingprops import compute_concreteness, ensure_nltk_data
"""
from .concreteness import compute_concreteness, ensure_nltk_data

__all__ = ["compute_concreteness", "ensure_nltk_data"]
__version__ = "0.1.0"
