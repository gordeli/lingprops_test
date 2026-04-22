"""Word-sense disambiguation (WSD) strategies for concreteness depth.

The concreteness depth of a noun is computed from the transitive hypernyms
of a chosen WordNet synset.  Historically the library always used the first
candidate synset; this module exposes two context-aware alternatives.

Strategies
----------
``"first"``   (default)
    Reproduces the library's original behaviour: the first synset returned
    by ``wn.synsets(word, 'n')`` that is not an instance hypernym.  Fastest,
    context-free.

``"lesk"``
    :func:`nltk.wsd.lesk` chooses the synset whose gloss overlaps most with
    the surrounding sentence.  On ties or misses, falls back to
    Most-Frequent-Sense (MFS) by the sum of ``lemma.count()`` over each
    synset's lemmas (SemCor-derived tagged counts that ship with WordNet).
    ~2x slower than ``"first"``; pure-Python, no extra dependencies.

``"neural"``
    A sentence-transformer (``all-MiniLM-L6-v2``) embeds the context and each
    candidate gloss; the synset with the highest cosine similarity wins.
    ~100x slower on CPU but much more accurate.  Requires the optional
    ``sentence-transformers`` dependency (``pip install lingprops[neural]``).

Proper nouns (POS ``"NNP"``) are always disambiguated with the legacy
behaviour regardless of strategy, since the depth calculation follows an
``instance_hypernyms`` promotion path that is incompatible with sense picks.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

# Module-level caches for the neural strategy; cleared via `reset_caches`.
_ST_MODEL = None
_GLOSS_CACHE: dict = {}      # word -> (synsets, gloss_embeddings_np)
_TEXT_EMB_CACHE: dict = {}   # id(text) -> context_embedding_np


def _candidate_synsets(word: str, POS: str):
    """Return the synsets considered by ``hyp_num`` for a given POS tag.

    For common nouns this drops synsets that have ``instance_hypernyms``
    (proper-noun-like senses); if that leaves nothing, the full list is kept.
    """
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(word, 'n')
    if not synsets:
        return []
    if POS == 'NNP':
        return synsets
    filtered = [s for s in synsets if not s.instance_hypernyms()]
    return filtered or synsets


def pick_first(word: str, POS: str, context=None, text=None):
    """Original library behaviour - first candidate synset."""
    cands = _candidate_synsets(word, POS)
    return cands[0] if cands else None


def pick_lesk_mfs(word: str, POS: str,
                  context: Optional[Sequence[str]] = None,
                  text: Optional[str] = None):
    """Lesk gloss-overlap with MFS-by-SemCor fallback."""
    from nltk.wsd import lesk

    cands = _candidate_synsets(word, POS)
    if not cands:
        return None
    if POS == 'NNP' or len(cands) == 1:
        return cands[0]

    chosen = None
    if context:
        chosen = lesk(context, word, 'n', synsets=cands)
    if chosen is not None:
        return chosen

    # Most-frequent-sense fallback using WordNet's tagged counts
    def sense_count(s):
        return sum(l.count() for l in s.lemmas())
    return max(cands, key=sense_count)


def _neural_model():
    """Lazy-load the sentence-transformer; raises ImportError with a hint."""
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'neural' WSD strategy requires sentence-transformers. "
                "Install with:  pip install lingprops[neural]\n"
                "or:             pip install sentence-transformers"
            ) from e
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def pick_neural(word: str, POS: str,
                context: Optional[Sequence[str]] = None,
                text: Optional[str] = None):
    """Embed the text and each candidate gloss; return the best match.

    The context embedding is cached by ``id(text)`` so that repeated calls
    on the same text do not re-encode it.  Gloss embeddings are cached by
    word globally.
    """
    cands = _candidate_synsets(word, POS)
    if not cands:
        return None
    if POS == 'NNP' or len(cands) == 1:
        return cands[0]

    import numpy as np
    model = _neural_model()

    # Context embedding (once per text)
    ctx_source = text if text else " ".join(context or [word])
    key = id(text) if text is not None else None
    if key is not None and key in _TEXT_EMB_CACHE:
        ctx_emb = _TEXT_EMB_CACHE[key]
    else:
        ctx_emb = model.encode([ctx_source], convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)[0]
        if key is not None:
            _TEXT_EMB_CACHE[key] = ctx_emb

    # Gloss embeddings (once per word)
    if word not in _GLOSS_CACHE:
        glosses = [s.definition() for s in cands]
        gloss_emb = model.encode(glosses, convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
        _GLOSS_CACHE[word] = (cands, gloss_emb)
    synset_list, gloss_emb = _GLOSS_CACHE[word]

    # The legacy module sets `np.seterr(all='raise')`, which turns harmless
    # underflow on tiny cosine similarities into a FloatingPointError.
    with np.errstate(under='ignore'):
        sims = gloss_emb @ ctx_emb
    return synset_list[int(np.argmax(sims))]


PICKERS: dict = {
    "first": pick_first,
    "lesk": pick_lesk_mfs,
    "neural": pick_neural,
}


def get_picker(name: str) -> Callable:
    """Return the picker callable for ``name`` (``first``/``lesk``/``neural``)."""
    if name not in PICKERS:
        raise ValueError(
            f"Unknown WSD strategy '{name}'. "
            f"Choose from: {sorted(PICKERS)}."
        )
    return PICKERS[name]


def depth_from_synset(synset, POS: str) -> int:
    """Compute the hypernym depth for a chosen synset.

    Replicates the counting rule of ``_concreteness_legacy.hyp_num``:
    transitive unique hypernyms, with a ``+1`` for proper-noun instance
    hypernyms.  ``synset`` of ``None`` returns depth 0.
    """
    if synset is None:
        return 0
    from ._concreteness_legacy import get_hypernyms
    if POS == 'NNP':
        instance_parents = synset.instance_hypernyms()
        base = instance_parents[0] if instance_parents else synset
        return 1 + len(get_hypernyms(base))
    return len(get_hypernyms(synset))


def reset_caches() -> None:
    """Clear neural per-text/per-word caches.

    Useful between large batches to keep memory bounded.  The loaded model
    itself is kept; to release it, set :data:`_ST_MODEL` to ``None``.
    """
    _TEXT_EMB_CACHE.clear()
    _GLOSS_CACHE.clear()
