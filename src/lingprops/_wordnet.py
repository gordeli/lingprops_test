"""Central, switchable WordNet access.

Every synset lookup and every lemmatisation in this package goes through here,
so that the release actually used is explicit, recorded in outputs, and can be
changed in one place.

NLTK's default ``wordnet`` corpus is WordNet **3.0**; **3.1** ships separately
as ``wordnet31``.  The package default is 3.1, which is what the CAS paper
describes.  Switch back with::

    import lingprops
    lingprops.set_wordnet_version("3.0")

before scoring anything (the reader is cached on first use).

Measured difference between the two releases, on a random sample of 4,000 noun
lemmas present in both: the first-sense ancestor count is identical for 99.53 %
of them; the 0.47 % that differ move by a mean of -0.21 levels (range -3 to
+7).  ``entity.n.01`` is the single root of the noun hierarchy in both.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_WORDNET_VERSION: str = "3.1"
WORDNET_VERSIONS: tuple[str, ...] = ("3.0", "3.1")

_NLTK_CORPUS = {"3.0": "wordnet", "3.1": "wordnet31"}

_version: str = DEFAULT_WORDNET_VERSION
_readers: dict[str, object] = {}
_lemmatizers: dict[str, object] = {}


def set_wordnet_version(version: str) -> None:
    """Choose the WordNet release used for all subsequent scoring."""
    if version not in WORDNET_VERSIONS:
        raise ValueError(
            f"Unknown WordNet version {version!r}; choose one of {WORDNET_VERSIONS}"
        )
    global _version
    _version = version


def wordnet_version() -> str:
    """The WordNet release currently selected (as configured)."""
    return _version


def installed_wordnet_version(version: Optional[str] = None) -> str:
    """The version string the loaded corpus reports about itself."""
    return get_wordnet(version).get_version()


def ensure_corpus(version: Optional[str] = None) -> None:
    """Download the corpus for *version* if NLTK does not already have it."""
    import nltk

    res = _NLTK_CORPUS[version or _version]
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)


def get_wordnet(version: Optional[str] = None):
    """Return the WordNet corpus reader for *version* (cached, eagerly loaded)."""
    v = version or _version
    if v not in _readers:
        ensure_corpus(v)
        from nltk.corpus import wordnet, wordnet31

        reader = {"3.0": wordnet, "3.1": wordnet31}[v]
        reader.synsets("dog", pos="n")  # force the LazyCorpusLoader
        _readers[v] = reader
    return _readers[v]


class _VersionedLemmatizer:
    """``WordNetLemmatizer`` bound to a specific WordNet release.

    NLTK's lemmatiser imports ``nltk.corpus.wordnet`` inside ``_morphy``, so it
    would silently keep using 3.0 even when synset lookups moved to 3.1.  The
    exception lists differ very little between releases, but the measure must
    not lemmatise against one release and take depths from another.
    """

    def __init__(self, wn):
        from nltk.stem import WordNetLemmatizer

        self._wn = wn
        self._base = WordNetLemmatizer()

    def _morphy(self, form, pos, check_exceptions=True):
        return self._wn._morphy(form, pos, check_exceptions)

    def morphy(self, form, pos=None, check_exceptions=True):
        lemmas = self._morphy(form, pos, check_exceptions) if pos else []
        return lemmas[0] if lemmas else None

    def lemmatize(self, word: str, pos: str = "n") -> str:
        lemmas = self._morphy(word, pos)
        return min(lemmas, key=len) if lemmas else word

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_VersionedLemmatizer(wordnet={self._wn.get_version()})"


def get_lemmatizer(version: Optional[str] = None):
    """Return a lemmatiser bound to the same WordNet release as scoring."""
    v = version or _version
    if v not in _lemmatizers:
        _lemmatizers[v] = _VersionedLemmatizer(get_wordnet(v))
    return _lemmatizers[v]
