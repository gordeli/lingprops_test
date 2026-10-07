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

    # 'Barack' and 'Obama' are reliably tagged PERSON by NLTK's ne_chunk when
    # used together.  How many of them are OOV is version-dependent: WordNet
    # 3.1 added obama.n.01, so under 3.1 only 'Barack' needs substituting.
    from lingprops._wordnet import get_wordnet
    wn = get_wordnet()
    t = "Barack Obama visited London yesterday."
    oov = [w for w in ("barack", "obama") if not wn.synsets(w, "n")]
    assert oov, "test needs at least one out-of-vocabulary PERSON token"

    base = compute_concreteness(t, ner=False)
    out  = compute_concreteness(t, ner=True)

    expected_delta = len(oov) * math.log(hyp_num("person", "NNP") + 1)
    actual_delta = out["NN"]["score"] - base["NN"]["score"]
    assert abs(actual_delta - expected_delta) < 1e-9
    assert out["NN"]["count"] == base["NN"]["count"] + len(oov)


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


# --- cardinal numbers must not be counted twice (fixed after v1.2.1) ---

NUMERAL_TEXT = "I bought two shirts and three hats. Two dogs barked at a hundred birds."


def test_cardinals_are_scored_inside_the_noun_partition():
    """The noun partition keeps numerals, as in the original pipeline, which
    called text_depth(text, ['NN', 'CD'], ...) and had no CD partition."""
    out = compute_concreteness(NUMERAL_TEXT, wsd="first", ner=False)
    nn_only = compute_concreteness(NUMERAL_TEXT, pos_groups=("NN",),
                                   wsd="first", ner=False)
    assert out["NN"] == nn_only["NN"]
    assert out["CD"]["count"] > 0, "test text must contain scoreable numerals"
    assert out["NN"]["count"] > out["CD"]["count"]


def test_cardinals_not_double_counted_in_total():
    """total must equal NN + VB + JJ + RB; the reported CD partition is a
    subset of NN and must not be added again."""
    out = compute_concreteness(NUMERAL_TEXT, wsd="first", ner=False)
    for key in ("count", "count_norep"):
        assert out["total"][key] == sum(out[p][key] for p in ("NN", "VB", "JJ", "RB")), key
        assert out["total"][key] != sum(
            out[p][key] for p in ("NN", "VB", "JJ", "RB", "CD")
        ), f"{key} still includes the CD partition twice"
    for key in ("score", "score_norep"):
        assert out["total"][key] == pytest.approx(
            sum(out[p][key] for p in ("NN", "VB", "JJ", "RB"))
        ), key


def test_cd_alone_still_contributes_to_the_total():
    """Asking for CD without NN must score it — the guard only suppresses the
    double count, it does not make the partition unusable on its own."""
    out = compute_concreteness(NUMERAL_TEXT, pos_groups=("CD",),
                               wsd="first", ner=False)
    assert out["total"]["count"] == out["CD"]["count"] > 0


def test_tangibility_cardinals_not_double_counted():
    out = compute_tangibility(NUMERAL_TEXT, pos_groups=("NN", "VB", "JJ", "RB", "CD"))
    assert out["total"]["count"] == sum(
        out[p]["count"] for p in ("NN", "VB", "JJ", "RB")
    )


# --- auxiliary verbs must stay stripped in the f = 1 path (fixed after v1.2.1) ---

AUX_TEXT = ("I was planning a trip to Italy. It has been 3 years since I was there. "
            "We will have 2 weeks and we have booked 4 hotels. It was amazing!")

# Here "was" appears ONLY as an auxiliary, so nothing else carries the lemma
# "be" and the bug changes the verb count rather than just which wordform wins.
AUX_BITES = "The parcel was delivered yesterday and the box arrived broken."

AUXILIARIES = {
    "am", "is", "are", "was", "were", "being", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could",
}


def _zero_count_wordforms(text):
    """Wordforms left at frequency <= 0 because every occurrence was auxiliary."""
    from lingprops.concreteness import _init_legacy
    legacy = _init_legacy()
    wf = legacy.wordformtion(text)
    return {k for k, v in wf.items() if v <= 0}


def test_wordformtion_leaves_zero_count_auxiliaries():
    """Guard for the premise of the next tests: the legacy tokeniser really
    does leave (word, VB*) entries behind at zero when it moves a word to AU."""
    zeros = _zero_count_wordforms(AUX_TEXT)
    assert zeros, "test text no longer produces a zero-count auxiliary"
    assert all(w in AUXILIARIES for w, _tag in zeros)


def test_norep_does_not_resurrect_zero_count_auxiliaries():
    """The f = 1 path must not score a wordform that no longer occurs.

    In AUX_BITES "was" occurs only as an auxiliary, so ``wordformtion`` leaves
    ('was','VBD') at zero.  Before the fix the f = 1 path scored it anyway and
    the verb partition came out as 3 unique verbs (was, delivered, arrived)
    instead of 2 — a 50 % inflation of its denominator.
    """
    out = compute_concreteness(AUX_BITES, wsd="first", ner=False)

    # Recompute the verb partition by hand, with and without the guard.
    from lingprops.concreteness import (_init_legacy, _score_wordform,
                                        _tag_to_wn_pos)
    legacy = _init_legacy()
    wf = legacy.wordformtion(AUX_BITES)
    nouns, _ = legacy.noun_lemmas(wf)

    def verb_partition(skip_zero):
        seen, score, count = set(), 0.0, 0
        for w in wf:
            if w[1][:2] != "VB":
                continue
            if skip_zero and wf[w] <= 0:
                continue
            p = _tag_to_wn_pos(w[1])
            lemma = legacy.wnl.lemmatize(w[0], p) if p else w[0]
            if lemma in seen:
                continue
            seen.add(lemma)
            delta, ok = _score_wordform(w, wf, nouns, 1, [], legacy)
            if ok:
                score += delta
                count += 1
        return score, count

    fixed, buggy = verb_partition(True), verb_partition(False)
    assert buggy[1] == fixed[1] + 1, "test text must exercise the bug"
    assert fixed[1] == 2
    assert out["VB"]["count_norep"] == fixed[1]
    assert out["VB"]["score_norep"] == pytest.approx(fixed[0])


def test_norep_still_counts_a_verb_used_as_both_main_and_auxiliary():
    """'was' occurs twice as an auxiliary and once as a copula here, so it
    survives with a positive count and must still be scored."""
    wf_zeros = {w for w, _t in _zero_count_wordforms(AUX_TEXT)}
    assert "was" not in wf_zeros
    out = compute_concreteness(AUX_TEXT, wsd="first", ner=False)
    assert out["VB"]["count_norep"] > 0


def test_tangibility_norep_skips_zero_count_auxiliaries():
    out = compute_tangibility(AUX_TEXT)
    with_aux = compute_tangibility("I was planning a trip to Italy.")
    assert out["VB"]["count_norep"] > 0
    assert with_aux["total"]["count_norep"] > 0


# --- WordNet release selection (new in 1.3.0) ---

def test_default_wordnet_is_31():
    import lingprops
    assert lingprops.wordnet_version() == "3.1"
    assert lingprops.installed_wordnet_version() == "3.1"


def test_entity_is_the_only_noun_root_in_both_releases():
    """The measure takes a depth from the root, which presupposes one root."""
    from lingprops._wordnet import get_wordnet
    for version in ("3.0", "3.1"):
        wn = get_wordnet(version)
        roots = [s.name() for s in wn.all_synsets("n")
                 if not s.hypernyms() and not s.instance_hypernyms()]
        assert roots == ["entity.n.01"], (version, roots)


def test_lemmatiser_is_bound_to_the_same_release():
    """Depths and lemmas must not come from different WordNet releases."""
    from lingprops.concreteness import _init_legacy
    from lingprops._wordnet import wordnet_version
    legacy = _init_legacy()
    assert legacy.wn.get_version() == wordnet_version()
    assert legacy.wnl._wn is legacy.wn
    assert legacy.wnl.lemmatize("hotels", "n") == "hotel"
    assert legacy.wnl.lemmatize("was", "v") == "be"


def test_wordnet_version_switch_is_validated():
    import pytest as _pytest
    from lingprops import set_wordnet_version, wordnet_version
    with _pytest.raises(ValueError):
        set_wordnet_version("2.9")
    assert wordnet_version() == "3.1"


# --- word_count excludes punctuation (fixed after v1.2.1) ---

def test_word_count_excludes_punctuation():
    """Reproduces the original rule: the POS tag must start with a letter."""
    t = "I was planning a trip to Italy. It has been 3 years since I was there."
    assert count_words(t)["total"] == len(t.split())
    out = compute_concreteness(t, wsd="first", ner=False)
    assert out["total"]["word_count"] == len(t.split())


def test_word_count_keeps_auxiliaries_as_words():
    """Auxiliaries are re-tagged AU and are not scored, but they are words."""
    out = compute_concreteness("The parcel was delivered yesterday.",
                               wsd="first", ner=False)
    assert out["total"]["word_count"] == 5
    assert out["VB"]["count"] == 1  # 'delivered' only; 'was' is auxiliary
