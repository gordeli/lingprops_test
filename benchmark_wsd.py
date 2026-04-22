"""Benchmark three WSD strategies for the `hyp_num` depth lookup.

Strategies:
    1. CURRENT   - wn.synsets(word, 'n')[0]    (library's current behaviour)
    2. LESK+MFS  - nltk.wsd.lesk on the sentence, MFS-by-SemCor fallback
    3. NEURAL    - sentence-transformer gloss matching against the context

Interface: each strategy is `disambiguate_text(text, tokens, targets) -> [Synset]`
so that per-text overhead (e.g. neural context embedding) is charged once per
text, not once per noun. Warmup excludes first-call / model-load cost.
"""
from __future__ import annotations

import os
import random
import time
from typing import List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer

for res in ("wordnet", "omw-1.4", "punkt", "punkt_tab",
            "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(res)
    except LookupError:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

wnl = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Strategy 1: CURRENT
# ---------------------------------------------------------------------------
def disambiguate_current(text, tokens, targets):
    out = []
    for word, _ctx in targets:
        ss = wn.synsets(word, 'n')
        if not ss:
            out.append(None); continue
        pick = next((s for s in ss if not s.instance_hypernyms()), ss[0])
        out.append(pick)
    return out


# ---------------------------------------------------------------------------
# Strategy 2: LESK + MFS fallback
# ---------------------------------------------------------------------------
def _sense_count(s):
    return sum(l.count() for l in s.lemmas())


def disambiguate_lesk_mfs(text, tokens, targets):
    out = []
    for word, ctx in targets:
        candidates = wn.synsets(word, 'n')
        if not candidates:
            out.append(None); continue
        common = [s for s in candidates if not s.instance_hypernyms()] or candidates
        chosen = lesk(ctx, word, 'n', synsets=common)
        if chosen is None:
            chosen = max(common, key=_sense_count)
        out.append(chosen)
    return out


# ---------------------------------------------------------------------------
# Strategy 3: NEURAL gloss matching (context embedded ONCE per text)
# ---------------------------------------------------------------------------
_st_model = None
_gloss_cache = {}   # word -> (synsets, gloss_embeddings_np)


def _get_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def disambiguate_neural(text, tokens, targets):
    import numpy as np
    model = _get_model()

    if not targets:
        return []

    # Encode context ONCE per text
    ctx_emb = model.encode([text], convert_to_numpy=True,
                           normalize_embeddings=True,
                           show_progress_bar=False)[0]

    out = []
    for word, _ctx in targets:
        candidates = wn.synsets(word, 'n')
        if not candidates:
            out.append(None); continue
        common = [s for s in candidates if not s.instance_hypernyms()] or candidates
        if len(common) == 1:
            out.append(common[0]); continue

        if word not in _gloss_cache:
            glosses = [s.definition() for s in common]
            gloss_emb = model.encode(glosses, convert_to_numpy=True,
                                     normalize_embeddings=True,
                                     show_progress_bar=False)
            _gloss_cache[word] = (common, gloss_emb)
        cands, gloss_emb = _gloss_cache[word]
        sims = gloss_emb @ ctx_emb
        out.append(cands[int(np.argmax(sims))])
    return out


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------
AMBIGUOUS_NOUNS = [
    "ball", "bank", "bark", "bass", "bat", "bow", "can", "club",
    "court", "crane", "fair", "fan", "file", "flat", "foot", "game",
    "hand", "head", "jam", "key", "light", "mine", "note", "organ",
    "palm", "park", "pen", "pitch", "plant", "point", "pool", "port",
    "post", "race", "ring", "rock", "row", "seal", "set", "spring",
    "star", "stick", "table", "tie", "trunk", "wave", "well", "yard",
    "arm", "bear", "book", "case", "cell", "chair", "check",
    "company", "court", "date", "deck", "face", "field", "fire", "floor",
    "force", "form", "front", "ground", "house", "kind", "law",
    "level", "line", "list", "look", "market", "matter", "mind", "money",
    "name", "night", "number", "office", "order", "party", "place", "plan",
    "play", "power", "present", "problem", "question", "reason", "record",
    "room", "sense", "service", "side", "state", "story", "system", "term",
]
FILLER = ("the a an of to in for with and or but on at by from as is was "
          "were are be been being this that these those i you he she it we "
          "they what which who how when where why very really quite just "
          "also about over under through ").split()


def make_text(target_words: int, seed: int) -> str:
    rng = random.Random(seed)
    n_amb = target_words // 4
    n_fill = target_words - n_amb
    words = [rng.choice(AMBIGUOUS_NOUNS) for _ in range(n_amb)] + \
            [rng.choice(FILLER) for _ in range(n_fill)]
    rng.shuffle(words)
    return " ".join(words)


def extract_noun_targets(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    targets = []
    for w, tag in tagged:
        if tag.startswith("NN"):
            lemma = wnl.lemmatize(w, 'n')
            targets.append((lemma, tokens))
    return tokens, targets


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_strategy(name, fn, payload, warmup_texts: int = 5):
    # Warmup — pay model-load cost, prime caches, then discard
    for t, tokens, targets in payload[:warmup_texts]:
        fn(t, tokens, targets)

    t0 = time.perf_counter()
    total_calls = 0
    for t, tokens, targets in payload:
        fn(t, tokens, targets)
        total_calls += len(targets)
    dt = time.perf_counter() - t0
    return dt, total_calls


def fmt_time(s: float) -> str:
    if s < 1e-3:   return f"{s*1e6:.1f} us"
    if s < 1:      return f"{s*1e3:.1f} ms"
    if s < 60:     return f"{s:.2f} s"
    m = s / 60
    if m < 60:     return f"{m:.2f} min"
    h = m / 60
    if h < 48:     return f"{h:.2f} h"
    d = h / 24
    return f"{d:.2f} days"


def main():
    N_TEXTS = 200
    WORDS_PER_TEXT = 100
    THREADS = 28

    print("=" * 78)
    print(f"Test corpus: {N_TEXTS} texts x {WORDS_PER_TEXT} words")
    print("=" * 78)
    texts = [make_text(WORDS_PER_TEXT, seed=i) for i in range(N_TEXTS)]

    t_pre = time.perf_counter()
    payload = []
    for t in texts:
        tokens, targets = extract_noun_targets(t)
        payload.append((t, tokens, targets))
    pre_dt = time.perf_counter() - t_pre
    total_targets = sum(len(p[2]) for p in payload)
    print(f"  preprocessing: {pre_dt:.2f} s | "
          f"{total_targets} noun-targets | "
          f"{total_targets/N_TEXTS:.1f} nouns/text avg")

    wn.synsets("dog")    # prime wn loader

    print()
    print("Running strategies (warmup = 5 texts, discarded)...")
    print()
    results = {}
    for name, fn in [("CURRENT  (first synset)  ", disambiguate_current),
                     ("LESK+MFS fallback        ", disambiguate_lesk_mfs),
                     ("NEURAL   (MiniLM glosses)", disambiguate_neural)]:
        print(f"  {name} ... ", end="", flush=True)
        dt, n = run_strategy(name, fn, payload, warmup_texts=5)
        per_call_us = dt / n * 1e6
        per_text_s = dt / N_TEXTS
        print(f"{dt:8.3f} s   "
              f"({per_call_us:8.1f} us/noun, "
              f"{per_text_s*1e3:8.2f} ms/text)")
        results[name] = (dt, n, per_call_us, per_text_s)

    print()
    print("=" * 78)
    print(f"Extrapolation to 100k and 100M texts "
          f"(100 words each, {THREADS} threads)")
    print("=" * 78)
    hdr = f"{'strategy':<28}{'1 thread 100k':>18}{'28 thr 100k':>16}{'1 thread 100M':>18}{'28 thr 100M':>16}"
    print(hdr)
    print("-" * len(hdr))
    for name, (dt, n, _, per_text_s) in results.items():
        t_100k = per_text_s * 100_000
        t_100m = per_text_s * 100_000_000
        print(f"{name:<28}"
              f"{fmt_time(t_100k):>18}"
              f"{fmt_time(t_100k/THREADS):>16}"
              f"{fmt_time(t_100m):>18}"
              f"{fmt_time(t_100m/THREADS):>16}")

    print()
    print("Notes:")
    print("  - 28-thread extrapolation assumes perfect parallel scaling "
          "(no I/O or GIL contention).")
    print("  - Neural times are CPU-only (MiniLM-L6-v2); a GPU would be "
          "~10-40x faster for the encoder path.")


if __name__ == "__main__":
    main()
