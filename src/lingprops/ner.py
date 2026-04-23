"""Named-entity recognition for proper-noun concreteness.

Historically the library handled a small, hand-curated list of brand names
and person names (``_concreteness_legacy.noun_lemmas``), mapping each to a
WordNet-addressable category lemma (e.g. ``kevin → person``).  This module
generalises that rule: any proper noun the NER recognises is substituted
with the lemma of its category, and the existing NNP depth formula
(``1 + |transitive hypernyms of the category synset|``) provides the depth.

Design
------
- Override is applied **only when the token is not in WordNet**.  This
  protects genuine common nouns that happen to be capitalised (``apple``
  the fruit) and proper nouns WordNet already knows as instances
  (``einstein``, ``paris``), both of which the existing path handles.
- Two backends: NLTK's ``ne_chunk`` (default, no extra deps) and spaCy
  (optional, more accurate, more entity types).
- The detection runs once per text and yields a ``{surface_lower: label}``
  map that is then applied to each wordform during scoring.

Entity label → WordNet lemma mapping
------------------------------------
Labels come from both NLTK (``PERSON``, ``ORGANIZATION``, ``GPE``,
``LOCATION``, ``FACILITY``) and spaCy (``PERSON``, ``ORG``, ``GPE``,
``LOC``, ``FAC``, ``NORP``, ``PRODUCT``, ``EVENT``, ``WORK_OF_ART``,
``LAW``, ``LANGUAGE``, ``DATE``, ``TIME``, ``MONEY``, ``QUANTITY``,
``CARDINAL``, ``ORDINAL``, ``PERCENT``).  Each is mapped to a single
common-noun lemma present in WordNet so that depth is well-defined.
"""
from __future__ import annotations

from typing import Dict, Optional


ENTITY_TO_LEMMA: Dict[str, str] = {
    # NLTK labels
    "PERSON":       "person",
    "ORGANIZATION": "organization",
    "GPE":          "country",       # geo-political entity
    "LOCATION":     "location",
    "FACILITY":     "facility",
    # spaCy labels (extends / overlaps NLTK)
    "ORG":          "organization",
    "LOC":          "location",
    "FAC":          "facility",
    "NORP":         "group",         # nationalities / religious / political
    "PRODUCT":      "product",
    "EVENT":        "event",
    "WORK_OF_ART":  "creation",
    "LAW":          "law",
    "LANGUAGE":     "language",
    "DATE":         "time",
    "TIME":         "time",
    "MONEY":        "money",
    "QUANTITY":     "measure",
    "CARDINAL":     "number",
    "ORDINAL":      "order",
    "PERCENT":      "ratio",
}


def _ensure_nltk_ner_data() -> None:
    import nltk
    for res in ("maxent_ne_chunker", "maxent_ne_chunker_tab", "words"):
        try:
            nltk.data.find(res)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass


def detect_entities_nltk(text: str) -> Dict[str, str]:
    """Run NLTK's ``ne_chunk`` and return ``{surface_lower: label}``.

    Multi-word entities are recorded twice: once per individual token and
    once for the whole surface form (e.g. ``"john"``, ``"smith"``, and
    ``"john smith"`` all map to ``"PERSON"``).
    """
    if not text:
        return {}
    _ensure_nltk_ner_data()
    from nltk import ne_chunk, pos_tag, word_tokenize

    ents: Dict[str, str] = {}
    try:
        tree = ne_chunk(pos_tag(word_tokenize(text)))
    except LookupError:
        return ents
    for subtree in tree:
        if not hasattr(subtree, "label"):
            continue
        label = subtree.label()
        tokens = [t for t, _ in subtree.leaves()]
        for tok in tokens:
            ents[tok.lower()] = label
        ents[" ".join(tokens).lower()] = label
    return ents


_SPACY_NLP = None


def detect_entities_spacy(text: str) -> Dict[str, str]:
    """Run spaCy's NER on ``text``.  Requires ``spacy`` + ``en_core_web_sm``."""
    global _SPACY_NLP
    if not text:
        return {}
    if _SPACY_NLP is None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy backend requires: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            ) from e
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Install with:\n"
                "  python -m spacy download en_core_web_sm"
            ) from e

    doc = _SPACY_NLP(text)
    ents: Dict[str, str] = {}
    for ent in doc.ents:
        label = ent.label_
        for tok in ent:
            ents[tok.text.lower()] = label
        ents[ent.text.lower()] = label
    return ents


def detect_entities(text: str, backend: str = "auto") -> Dict[str, str]:
    """Dispatch to a backend.

    ``backend="auto"`` prefers spaCy if importable and its model is
    available, else falls back to NLTK.  Use ``"nltk"`` or ``"spacy"``
    to pin a specific backend.
    """
    if backend == "nltk":
        return detect_entities_nltk(text)
    if backend == "spacy":
        return detect_entities_spacy(text)
    if backend == "auto":
        try:
            import spacy  # noqa: F401
            try:
                return detect_entities_spacy(text)
            except Exception:
                return detect_entities_nltk(text)
        except ImportError:
            return detect_entities_nltk(text)
    raise ValueError(f"Unknown NER backend '{backend}'. "
                     "Use 'auto', 'nltk', or 'spacy'.")


def category_lemma(word: str, ner_map: Dict[str, str]) -> Optional[str]:
    """Return the WordNet category lemma for ``word`` (e.g. ``'person'``)
    or ``None`` if the word was not tagged as an entity."""
    label = ner_map.get(word.lower())
    if label is None:
        return None
    return ENTITY_TO_LEMMA.get(label)


def augment_nouns_with_ner(word_forms, nouns, ner_map,
                           already_wordnet_known) -> Dict:
    """Add NER-derived category lemmas to the ``nouns`` dict in place.

    A wordform is augmented only when:
      - its POS starts with ``NN`` (noun or proper noun),
      - the surface form was tagged as an entity by NER,
      - the entity label maps to a known WordNet category lemma, and
      - the word has no WordNet synsets (``already_wordnet_known(word)``
        returns ``False``).  The last rule protects capitalised common
        nouns (``apple``) and WordNet-known instances (``einstein``).

    Returns the (possibly modified) ``nouns`` dict.  Wordforms that
    already had an entry in ``nouns`` are left untouched.
    """
    if not ner_map:
        return nouns
    for wordform in list(word_forms.keys()):
        word, tag = wordform
        if not tag.startswith("NN"):
            continue
        if wordform in nouns:
            continue
        cat = category_lemma(word, ner_map)
        if cat is None:
            continue
        if already_wordnet_known(word):
            continue
        nouns[wordform] = cat
    return nouns
