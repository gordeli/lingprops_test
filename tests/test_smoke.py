from lingprops import compute_concreteness, count_words


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
