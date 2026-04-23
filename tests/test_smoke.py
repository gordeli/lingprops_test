import pytest

from lingprops import compute_concreteness, compute_tangibility, count_words


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


def test_wsd_default_is_first():
    """Default wsd='first' matches calling with no wsd argument."""
    a = compute_concreteness(TEXT)
    b = compute_concreteness(TEXT, wsd="first")
    assert a["NN"]["score"] == b["NN"]["score"]
    assert a["total"]["normalized_score"] == b["total"]["normalized_score"]


def test_wsd_lesk_runs():
    """Lesk+MFS strategy runs and yields the same normalization count."""
    base = compute_concreteness(TEXT, wsd="first")
    out = compute_concreteness(TEXT, wsd="lesk")
    # Sanity: same text -> same token/noun partitioning; counts are identical
    assert out["NN"]["count"] == base["NN"]["count"]
    assert out["total"]["word_count"] == base["total"]["word_count"]
    # Scores are allowed to differ (different synsets may be picked)
    assert isinstance(out["NN"]["normalized_score"], float)


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

def test_ner_default_off_is_noop():
    """Default ner=False matches ner unset."""
    t = "Alice loves ios."
    a = compute_concreteness(t)
    b = compute_concreteness(t, ner=False)
    assert a["NN"]["score"] == b["NN"]["score"]
    assert a["NN"]["count"] == b["NN"]["count"]


def test_ner_picks_up_oov_proper_nouns():
    """A name not in WordNet and not in the manual list should contribute
    when ner=True."""
    # 'Alice' has 0 WordNet synsets and isn't in the legacy manual list,
    # so it drops out of the baseline score entirely.  With NER it is
    # tagged as an entity and substituted with a category lemma.
    t = "Alice loves ios."
    base = compute_concreteness(t)
    ner  = compute_concreteness(t, ner=True)
    assert ner["NN"]["count"] > base["NN"]["count"]
    assert ner["NN"]["score"] > base["NN"]["score"]


def test_ner_does_not_override_wordnet_known_words():
    """Capitalised common nouns and WordNet-known instances keep the
    existing depth calculation rather than being re-classified."""
    # 'apple' is in WordNet (fruit); 'einstein' is a WordNet instance.
    # NER must not clobber either.
    t_apple    = "I bought an apple today."
    t_einstein = "Einstein studied physics."
    a1 = compute_concreteness(t_apple)
    a2 = compute_concreteness(t_apple, ner=True)
    e1 = compute_concreteness(t_einstein)
    e2 = compute_concreteness(t_einstein, ner=True)
    assert a1["NN"]["score"] == a2["NN"]["score"]
    assert e1["NN"]["score"] == e2["NN"]["score"]


def test_ner_person_depth_is_person_plus_one():
    """For detected PERSON tokens not in WordNet, the score delta is
    exactly ``n * log(1 + depth(person))`` where ``depth(person)`` is the
    NNP depth of the category lemma."""
    import math
    from lingprops._concreteness_legacy import hyp_num

    # 'Barack' and 'Obama' are both OOV (0 WordNet synsets) and are
    # reliably tagged PERSON by NLTK's ne_chunk when used together.
    t = "Barack Obama visited London yesterday."
    base = compute_concreteness(t)
    out  = compute_concreteness(t, ner=True)

    expected_delta = 2 * math.log(hyp_num("person", "NNP") + 1)
    actual_delta = out["NN"]["score"] - base["NN"]["score"]
    assert abs(actual_delta - expected_delta) < 1e-9
    assert out["NN"]["count"] == base["NN"]["count"] + 2


def test_ner_invalid_backend_raises():
    with pytest.raises(ValueError):
        compute_concreteness("Alice.", ner=True, ner_backend="bogus")
