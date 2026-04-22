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
