import pytest

from lingprops import compute_all, compute_concreteness, compute_tangibility, count_words


def test_smoke():
    out = compute_concreteness("Cats chase mice. Dogs sleep.")
    assert "total" in out


def test_normalized_score_present():
    out = compute_concreteness("Cats chase mice. Dogs sleep.")
    for pos in ("NN", "VB", "JJ", "RB", "CD", "total"):
        assert "normalized_score" in out[pos], f"missing normalized_score for {pos}"
    assert "word_count" in out["total"]
    assert "content_word_counts" in out["total"]


def test_normalized_score_value():
    out = compute_concreteness("The cat sat on the mat.")
    for pos in ("NN", "VB", "JJ", "RB", "CD"):
        entry = out[pos]
        if entry["count"] > 0:
            assert abs(entry["normalized_score"] - entry["score"] / entry["count"]) < 1e-12
        else:
            assert entry["normalized_score"] == 0.0


def test_count_words():
    wc = count_words("The big cat quickly chased two mice.")
    assert wc["total"] > 0
    assert isinstance(wc["NN"], int)
    assert isinstance(wc["VB"], int)
    assert isinstance(wc["JJ"], int)
    assert isinstance(wc["RB"], int)


def test_empty_text():
    out = compute_concreteness("")
    assert out["total"]["score"] == 0.0
    assert out["total"]["normalized_score"] == 0.0
    assert out["total"]["count"] == 0


# --- Tangibility (BWK) tests ---

def test_tangibility_smoke():
    out = compute_tangibility("The cat sat on the mat.")
    assert "total" in out
    assert out["total"]["count"] > 0
    # BWK ratings are on a 1-5 scale
    assert 1.0 <= out["total"]["normalized_score"] <= 5.0


def test_tangibility_norep_fields():
    out = compute_tangibility("Cats chase mice. Dogs sleep.")
    for pos in ("NN", "VB", "total"):
        assert "score_norep" in out[pos]
        assert "count_norep" in out[pos]
        assert "normalized_score_norep" in out[pos]


def test_tangibility_concrete_vs_abstract():
    concrete = compute_tangibility("The big red truck drove past the wooden fence.")
    abstract = compute_tangibility("Freedom and justice require constant vigilance.")
    assert concrete["total"]["normalized_score"] > abstract["total"]["normalized_score"]


def test_tangibility_repetitions():
    out = compute_tangibility("The dog ran. The dog ran. The dog ran.")
    # With rep: 3 dog + 3 ran = 6 tokens
    # Without rep: 1 dog + 1 ran = 2 unique lemmas
    assert out["total"]["count"] == 6
    assert out["total"]["count_norep"] == 2
    # Average should be the same (same words repeated)
    assert abs(out["total"]["normalized_score"] -
               out["total"]["normalized_score_norep"]) < 1e-10


def test_tangibility_empty():
    out = compute_tangibility("")
    assert out["total"]["score"] == 0.0
    assert out["total"]["count"] == 0


# --- WSD strategy tests ---

TEXT = ("The pitcher threw the ball across the field. "
        "Players on the bench watched the game closely.")


def test_wsd_default_is_lesk():
    """Default wsd='lesk' matches calling with no wsd argument."""
    a = compute_concreteness(TEXT)
    b = compute_concreteness(TEXT, wsd="lesk")
    assert a["NN"]["score"] == b["NN"]["score"]
    assert a["total"]["normalized_score"] == b["total"]["normalized_score"]


def test_wsd_first_still_available():
    """wsd='first' remains available for reproducing pre-WSD-flip results."""
    base = compute_concreteness(TEXT, wsd="first")
    out  = compute_concreteness(TEXT, wsd="lesk")
    # Same text -> same token/noun partitioning; counts are identical
    assert out["NN"]["count"] == base["NN"]["count"]
    assert out["total"]["word_count"] == base["total"]["word_count"]
    # Scores are allowed to differ (different synsets may be picked)
    assert isinstance(out["NN"]["normalized_score"], float)


def test_wsd_lesk_is_not_first():
    """Regression test for the dispatch bug fixed 2026-09-18.

    After the default flipped to "lesk" (82cbdc4), ``_score_concreteness``
    compared ``wsd == DEFAULT_WSD`` to select the legacy first-synset path,
    so ``wsd="lesk"`` silently produced first-sense scores.  On this sentence
    the Lesk picker chooses ``savings_bank.n.02`` (depth 7) for *bank* while
    first-sense gives ``bank.n.01`` (depth 5), so the noun scores must differ,
    and the lesk score must match the picker's own choice.
    """
    import math
    import nltk
    from lingprops import wsd as W

    sentence = "I deposited the cheque at the bank before the loan meeting."
    first = compute_concreteness(sentence, wsd="first", ner=False)
    lesk = compute_concreteness(sentence, wsd="lesk", ner=False)
    assert first["NN"]["count_norep"] == lesk["NN"]["count_norep"]
    assert first["NN"]["score_norep"] != lesk["NN"]["score_norep"],         "wsd='lesk' produced the first-sense score: the picker was not used"

    # Rebuild the expected difference from the pickers themselves: for every
    # noun lemma in the sentence, sum log(d+1) under each strategy (f = 1).
    from lingprops.concreteness import _init_legacy
    legacy = _init_legacy()
    word_forms = legacy.wordformtion(sentence)
    nouns, _ = legacy.noun_lemmas(word_forms)
    ctx = nltk.word_tokenize(sentence)
    expected = 0.0
    changed = 0
    for (word, tag), lemma in nouns.items():
        if not tag.startswith("NN") or isinstance(lemma, list):
            continue
        d_first = W.depth_from_synset(W.pick_first(lemma, tag), tag)
        d_lesk = W.depth_from_synset(
            W.pick_lesk_mfs(lemma, tag, context=ctx, text=sentence), tag)
        if d_first and d_lesk:
            expected += math.log(d_lesk + 1) - math.log(d_first + 1)
            changed += d_first != d_lesk
    assert changed >= 1  # "bank" (5 -> 7) at least
    assert abs((lesk["NN"]["score_norep"] - first["NN"]["score_norep"]) - expected) < 1e-9


def test_instance_only_noun_is_not_dropped():
    """Regression test for v1.2.1.

    Words whose every WordNet sense is an instance (proper names such as
    "Hawaii") used to be dropped when the tagger labelled them a common noun,
    which happens whenever the writer does not capitalise them: hyp_num raised
    UnboundLocalError on the legacy path, and depth_from_synset returned 0 on
    the lesk/neural path, so the word left both the score and its denominator.
    Depth must not depend on capitalisation.
    """
    from lingprops.concreteness import _init_legacy
    from lingprops.wsd import depth_from_synset, pick_first
    legacy = _init_legacy()

    for word in ("hawaii", "boise", "nile"):
        assert legacy.hyp_num(word, "NN") == legacy.hyp_num(word, "NNP") > 0
        assert depth_from_synset(pick_first(word, "NN"), "NN") ==                depth_from_synset(pick_first(word, "NNP"), "NNP") > 0

    lower = compute_concreteness("we flew to hawaii last summer", wsd="first", ner=False)
    upper = compute_concreteness("We flew to Hawaii last summer", wsd="first", ner=False)
    assert lower["total"]["count_norep"] == upper["total"]["count_norep"]
    assert abs(lower["total"]["normalized_score_norep"]
               - upper["total"]["normalized_score_norep"]) < 1e-12


def test_entity_counts_in_the_denominator():
    """`entity` is the only WordNet noun lemma with depth 0 (it is the root of
    the noun hierarchy). It must contribute 0 to the score but still count as a
    scored word, so the denominator is not silently reduced."""
    out = compute_concreteness("The entity moved.", wsd="first", ner=False)
    assert out["total"]["count_norep"] == 2        # entity + moved
    from lingprops.concreteness import _init_legacy
    assert _init_legacy().hyp_num("entity", "NN") == 0


def test_wsd_invalid_raises():
    with pytest.raises(ValueError):
        compute_concreteness(TEXT, wsd="not-a-strategy")


def test_wsd_neural_optional():
    """Neural strategy: skip gracefully if the optional dep is missing."""
    pytest.importorskip("sentence_transformers")
    base = compute_concreteness(TEXT, wsd="first")
    out = compute_concreteness(TEXT, wsd="neural")
    assert out["NN"]["count"] == base["NN"]["count"]
    assert out["total"]["word_count"] == base["total"]["word_count"]


# --- NER tests ---

def test_ner_default_on():
    """NER is enabled by default; passing ner=True is a no-op."""
    t = "Alice and Bob walked through Central Park."
    a = compute_concreteness(t)
    b = compute_concreteness(t, ner=True)
    assert a["NN"]["score"] == b["NN"]["score"]
    assert a["NN"]["count"] == b["NN"]["count"]


def test_ner_can_be_disabled():
    """ner=False restores the pre-NER behaviour for reproducibility."""
    t = "Alice and Bob walked through Central Park."
    default = compute_concreteness(t)            # ner=True (default)
    no_ner  = compute_concreteness(t, ner=False)
    # Alice is OOV: default counts her, no_ner does not.
    assert default["NN"]["count"] > no_ner["NN"]["count"]


def test_ner_picks_up_oov_proper_nouns():
    """A name not in WordNet and not in the manual list should contribute
    when NER is on (the default)."""
    # 'Alice' has 0 WordNet synsets and isn't in the legacy manual list,
    # so she drops out when ner=False.  With NER (default) she is tagged
    # as PERSON and substituted with the lemma 'person'.
    t = "Alice and Bob walked through Central Park."
    without = compute_concreteness(t, ner=False)
    with_ner = compute_concreteness(t)           # default: ner=True
    assert with_ner["NN"]["count"] > without["NN"]["count"]
    assert with_ner["NN"]["score"] > without["NN"]["score"]


def test_ner_does_not_override_wordnet_known_words():
    """Capitalised common nouns and WordNet-known instances keep the
    existing depth calculation rather than being re-classified."""
    # 'apple' is in WordNet (fruit); 'einstein' is a WordNet instance.
    # NER must not clobber either.
    t_apple    = "I bought an apple today."
    t_einstein = "Einstein studied physics."
    a_off = compute_concreteness(t_apple, ner=False)
    a_on  = compute_concreteness(t_apple, ner=True)
    e_off = compute_concreteness(t_einstein, ner=False)
    e_on  = compute_concreteness(t_einstein, ner=True)
    assert a_off["NN"]["score"] == a_on["NN"]["score"]
    assert e_off["NN"]["score"] == e_on["NN"]["score"]


def test_ner_person_depth_is_person_plus_one():
    """For detected PERSON tokens not in WordNet, the score delta is
    exactly ``n * log(1 + depth(person))`` where ``depth(person)`` is the
    NNP depth of the category lemma."""
    import math
    from lingprops._concreteness_legacy import hyp_num

    # 'Barack' and 'Obama' are both OOV (0 WordNet synsets) and are
    # reliably tagged PERSON by NLTK's ne_chunk when used together.
    t = "Barack Obama visited London yesterday."
    base = compute_concreteness(t, ner=False)
    out  = compute_concreteness(t, ner=True)

    expected_delta = 2 * math.log(hyp_num("person", "NNP") + 1)
    actual_delta = out["NN"]["score"] - base["NN"]["score"]
    assert abs(actual_delta - expected_delta) < 1e-9
    assert out["NN"]["count"] == base["NN"]["count"] + 2


def test_ner_invalid_backend_raises():
    with pytest.raises(ValueError):
        compute_concreteness("Alice is here.", ner_backend="bogus")


# --- compute_all: shared-tokenisation parity ---

@pytest.mark.parametrize("text", [
    "Cats chase mice. Dogs sleep.",
    "The big red truck drove past the wooden fence.",
    "Alice and Bob walked through Central Park yesterday.",
    "Freedom and justice require constant vigilance from every citizen.",
    "",  # empty input: both calls must agree on the zero-state
])
def test_compute_all_matches_separate_calls(text):
    """compute_all(text) must be bit-exactly equal to
    {compute_concreteness(text), compute_tangibility(text)}."""
    bundled = compute_all(text)
    conc_solo = compute_concreteness(text)
    tang_solo = compute_tangibility(text)
    assert bundled["concreteness"] == conc_solo
    assert bundled["tangibility"]  == tang_solo
