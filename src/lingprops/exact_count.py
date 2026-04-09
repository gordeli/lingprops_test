"""Exact computation of distinct generalized texts.

The concreteness formula estimates the number of texts producible by
replacing each word with any of its hypernyms as prod C(d_k + f_k, f_k).
This treats each word's hypernym chain as independent, overcounting when
branches intersect in WordNet.

This module provides two implementations:

- ``compute_exact_text_count`` — brute-force Cartesian-product enumeration.
  Correct but feasible only for ~5 content words.
- ``compute_exact_text_count_optimized`` — incremental set multiplication
  with deduplication, integer-encoded state, group merging, and greedy
  ordering.  Handles 10+ content words when overlap is significant.
"""
from __future__ import annotations

import time
from itertools import combinations_with_replacement, product as cartesian_product
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import comb as scipy_comb


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_synset(noun: str, POS: str, wn):
    """Replicate the synset-selection logic used by hyp_num in the legacy code."""
    synsets = wn.synsets(noun, "n")
    if not synsets:
        return None
    if POS != "NNP":
        for s in synsets:
            if not s.instance_hypernyms():
                return s
        return synsets[0]
    else:
        for s in synsets:
            if s.instance_hypernyms():
                return s.instance_hypernyms()[0]
        return synsets[0]


def _replacement_set(synset, get_hypernyms_fn) -> frozenset:
    """Return the frozenset of synset-name strings this word can map to."""
    hyps = get_hypernyms_fn(synset)
    return frozenset(s.name() for s in ({synset} | hyps))


def _extract_entries(text, repetitions, pos_groups, legacy, wn):
    """Run the legacy pipeline and collect (word, freq, replacement_set) entries."""
    word_forms = legacy.wordformtion(text)
    nouns_dict, _ = legacy.noun_lemmas(word_forms)

    postag_prefixes: list[str] = []
    for pg in pos_groups:
        postag_prefixes.extend(["NN", "CD"] if pg == "NN" else [pg])

    entries: list[dict] = []
    seen_nouns: set[str] = set()

    for wf_key in word_forms:
        if wf_key[1][:2] not in postag_prefixes:
            continue
        noun = nouns_dict.get(wf_key)
        if noun is None:
            continue
        if not repetitions:
            nk = str(noun)
            if nk in seen_nouns:
                continue
            seen_nouns.add(nk)

        noun_list = noun if isinstance(noun, list) else [noun]
        rep_set: set[str] = set()
        for n in noun_list:
            syn = _get_synset(n, wf_key[1], wn)
            if syn is None:
                continue
            hyps = legacy.get_hypernyms(syn)
            if len(n) < 2 or (len(hyps) == 0 and n != "entity"):
                continue
            rep_set |= _replacement_set(syn, legacy.get_hypernyms)
        if not rep_set:
            continue
        entries.append({
            "word": wf_key[0],
            "pos": wf_key[1],
            "noun": noun,
            "frequency": word_forms[wf_key] if repetitions else 1,
            "rep_set": frozenset(rep_set),
        })
    return entries


# ---------------------------------------------------------------------------
# Brute-force exact count  (feasible for ~5 content words)
# ---------------------------------------------------------------------------

def compute_exact_text_count(
    text: str,
    repetitions: bool = True,
    pos_groups: Tuple[str, ...] = ("NN", "VB", "JJ", "RB", "CD"),
    max_combinations: int = 10_000_000,
) -> Dict:
    """Compute the exact number of distinct generalized texts.

    A 'generalized text' is formed by replacing each content word with any
    synset on its hypernym path (including itself).  The exact count accounts
    for overlapping hypernym branches; the formula estimate does not.

    Parameters
    ----------
    text : str
        Input text.
    repetitions : bool
        If True, each word token is counted separately (default).
    pos_groups : tuple of str
        POS categories to include.
    max_combinations : int
        Safety limit on the Cartesian-product size before aborting enumeration.

    Returns
    -------
    dict with keys:
        formula_estimate, formula_log, exact_count, overcounting_ratio,
        word_details, overlapping_pairs.
        If enumeration exceeds *max_combinations*, exact_count is None.
    """
    from .concreteness import _init_legacy
    legacy = _init_legacy()
    wn = legacy.wn

    entries = _extract_entries(text, repetitions, pos_groups, legacy, wn)

    if not entries:
        return {
            "formula_estimate": 1, "formula_log": 0.0,
            "exact_count": 1, "overcounting_ratio": 1.0,
            "word_details": [], "overlapping_pairs": [],
        }

    # Formula estimate
    formula_log = 0.0
    formula_estimate_float = 1.0
    for e in entries:
        d = len(e["rep_set"]) - 1
        f = e["frequency"]
        c = float(scipy_comb(d + f, f, exact=True))
        formula_log += np.log(c)
        formula_estimate_float *= c
        e["depth"] = d

    # Feasibility check
    total_combos = 1
    for e in entries:
        n_ms = int(scipy_comb(len(e["rep_set"]) + e["frequency"] - 1,
                              e["frequency"], exact=True))
        e["n_multisets"] = n_ms
        total_combos *= n_ms
    feasible = total_combos <= max_combinations

    # Overlap analysis
    overlapping_pairs = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            shared = entries[i]["rep_set"] & entries[j]["rep_set"]
            if shared:
                overlapping_pairs.append({
                    "word_a": entries[i]["word"],
                    "word_b": entries[j]["word"],
                    "shared_count": len(shared),
                    "set_a_size": len(entries[i]["rep_set"]),
                    "set_b_size": len(entries[j]["rep_set"]),
                    "shared_fraction_a": len(shared) / len(entries[i]["rep_set"]),
                    "shared_fraction_b": len(shared) / len(entries[j]["rep_set"]),
                    "shared_examples": sorted(shared)[:5],
                })

    # Exact enumeration
    exact_count: Optional[int] = None
    if feasible:
        per_entry_ms = [
            list(combinations_with_replacement(sorted(e["rep_set"]), e["frequency"]))
            for e in entries
        ]
        distinct: set[tuple] = set()
        for combo in cartesian_product(*per_entry_ms):
            flat = tuple(sorted(item for ms in combo for item in ms))
            distinct.add(flat)
        exact_count = len(distinct)

    word_details = [
        {"word": e["word"], "pos": e["pos"], "noun": e["noun"],
         "frequency": e["frequency"], "depth": e["depth"],
         "replacement_set_size": len(e["rep_set"]),
         "multisets": e["n_multisets"]}
        for e in entries
    ]

    result = {
        "formula_estimate": int(formula_estimate_float),
        "formula_log": formula_log,
        "exact_count": exact_count,
        "overcounting_ratio": (
            formula_estimate_float / exact_count
            if exact_count is not None and exact_count > 0 else None
        ),
        "total_combinations_checked": total_combos if feasible else None,
        "word_details": word_details,
        "overlapping_pairs": overlapping_pairs,
    }
    if not feasible:
        result["note"] = (
            f"Enumeration skipped: {total_combos:,} combinations "
            f"exceed limit of {max_combinations:,}."
        )
    return result


# ---------------------------------------------------------------------------
# Optimized exact count  (incremental set multiplication)
# ---------------------------------------------------------------------------

def _solve_partition(entries_partition, max_states):
    """Solve the exact count for a single POS partition via incremental set mult.

    Returns (exact_count_or_None, formula_est, formula_log, peak_states, detail).
    """
    # Group by identical replacement set
    group_map: dict[frozenset, dict] = {}
    for e in entries_partition:
        key = e["rep_set"]
        if key not in group_map:
            group_map[key] = {"rep_set": key, "total_freq": 0, "words": []}
        group_map[key]["total_freq"] += e["frequency"]
        group_map[key]["words"].append(e["word"])
    groups = list(group_map.values())

    # Universe and integer encoding
    all_syns: set[str] = set()
    for g in groups:
        all_syns |= g["rep_set"]
    universe = sorted(all_syns)
    n_syns = len(universe)
    syn_idx = {s: i for i, s in enumerate(universe)}

    total_freq = sum(g["total_freq"] for g in groups)
    BASE = total_freq + 1
    base_pows = [BASE ** i for i in range(n_syns)]

    # Precompute encoded choices
    per_group_choices: list[list[int]] = []
    formula_est = 1.0
    formula_log = 0.0
    for g in groups:
        rs_indices = sorted(syn_idx[s] for s in g["rep_set"])
        seen: set[int] = set()
        choices: list[int] = []
        for ms in combinations_with_replacement(rs_indices, g["total_freq"]):
            val = 0
            for idx in ms:
                val += base_pows[idx]
            if val not in seen:
                seen.add(val)
                choices.append(val)
        per_group_choices.append(choices)
        d = len(g["rep_set"]) - 1
        f = g["total_freq"]
        c = float(scipy_comb(d + f, f, exact=True))
        formula_log += np.log(c)
        formula_est *= c
        g["depth"] = d
        g["n_choices"] = len(choices)

    # Greedy ordering
    processed_union: set[str] = set()
    remaining = list(range(len(groups)))
    order: list[int] = []
    while remaining:
        if not processed_union:
            best = min(remaining, key=lambda i: len(per_group_choices[i]))
        else:
            best = max(
                remaining,
                key=lambda i: (
                    len(groups[i]["rep_set"] & processed_union),
                    -len(per_group_choices[i]),
                ),
            )
        order.append(best)
        remaining.remove(best)
        processed_union |= groups[best]["rep_set"]

    # Incremental multiplication
    state: set[int] = {0}
    peak = 1
    step_log: list[dict] = []

    for step_num, gi in enumerate(order):
        choices = per_group_choices[gi]
        state = {e + c for e in state for c in choices}
        sz = len(state)
        if sz > peak:
            peak = sz
        step_log.append({
            "step": step_num + 1,
            "group_words": groups[gi]["words"],
            "choices": len(choices),
            "state_size": sz,
        })
        if sz > max_states:
            detail = {
                "groups": [
                    {"words": g["words"], "freq": g["total_freq"],
                     "depth": g["depth"], "choices": g["n_choices"]}
                    for g in groups],
                "step_log": step_log,
                "aborted_at": f"{step_num + 1}/{len(groups)}",
            }
            return None, formula_est, formula_log, peak, detail

    detail = {
        "groups": [
            {"words": g["words"], "freq": g["total_freq"],
             "depth": g["depth"], "choices": g["n_choices"]}
            for g in groups],
        "step_log": step_log,
    }
    return len(state), formula_est, formula_log, peak, detail


def compute_exact_text_count_optimized(
    text: str,
    repetitions: bool = True,
    pos_groups: Tuple[str, ...] = ("NN", "VB", "JJ", "RB", "CD"),
    max_states: int = 5_000_000,
) -> Dict:
    """Compute the exact number of distinct generalized texts (optimized).

    Key optimizations:

    1. **POS independence** — words from different POS categories (nouns,
       verbs, adjectives, adverbs) are solved separately and multiplied,
       because they map to disjoint WordNet subgraphs.
    2. **Incremental deduplication** — within each POS partition, word-type
       groups are processed one at a time with set deduplication after each
       step, so overlapping hypernym branches compress the state.
    3. **Group merging** — word types with identical replacement sets are
       combined: their joint multiset count is smaller than the product.
    4. **Integer encoding** — each multiset is packed into a single Python
       ``int``; merging = ``int + int`` (no per-element loop).
    5. **Set comprehension** — tighter C-level loop in CPython.
    6. **Greedy ordering** — most-overlapping groups processed first.

    Parameters
    ----------
    text : str
        Input text.
    repetitions : bool
        Count repeated word tokens separately (default True).
    pos_groups : tuple of str
        POS categories to include.
    max_states : int
        Abort a POS partition if its state set exceeds this size.

    Returns
    -------
    dict
        ``exact_count``, ``formula_estimate``, ``overcounting_ratio``,
        ``elapsed_ms``, ``peak_states``, and per-POS ``partitions`` detail.
    """
    t_start = time.perf_counter()

    from .concreteness import _init_legacy
    legacy = _init_legacy()
    wn = legacy.wn

    entries = _extract_entries(text, repetitions, pos_groups, legacy, wn)

    if not entries:
        return {
            "exact_count": 1, "formula_estimate": 1,
            "overcounting_ratio": 1.0, "elapsed_ms": 0.0,
            "peak_states": 1, "partitions": {},
        }

    # --- Partition entries by POS category ---
    # Map POS tag prefix to the pos_group label
    tag_to_group: dict[str, str] = {}
    for pg in pos_groups:
        if pg == "NN":
            for p in ("NN", "CD"):
                tag_to_group[p] = "NN"
        else:
            tag_to_group[pg] = pg

    partitions: dict[str, list[dict]] = {}
    for e in entries:
        prefix = e["pos"][:2]
        pg = tag_to_group.get(prefix, prefix)
        partitions.setdefault(pg, []).append(e)

    # --- Solve each partition independently, multiply results ---
    total_exact = 1
    total_formula = 1.0
    total_formula_log = 0.0
    global_peak = 1
    aborted = False
    partition_details: dict[str, dict] = {}

    for pg in sorted(partitions):
        part_entries = partitions[pg]
        if not part_entries:
            continue

        exact, fest, flog, peak, detail = _solve_partition(
            part_entries, max_states,
        )

        partition_details[pg] = {
            "n_words": len(part_entries),
            "words": [e["word"] for e in part_entries],
            "formula_estimate": int(fest),
            "exact_count": exact,
            "overcounting_ratio": fest / exact if exact and exact > 0 else None,
            "peak_states": peak,
            **detail,
        }

        total_formula *= fest
        total_formula_log += flog
        if peak > global_peak:
            global_peak = peak

        if exact is None:
            aborted = True
            total_exact = None
        elif total_exact is not None:
            total_exact *= exact

    elapsed = (time.perf_counter() - t_start) * 1000

    result = {
        "exact_count": total_exact,
        "formula_estimate": int(total_formula),
        "formula_log": total_formula_log,
        "overcounting_ratio": (
            total_formula / total_exact
            if total_exact is not None and total_exact > 0 else None
        ),
        "elapsed_ms": round(elapsed, 1),
        "peak_states": global_peak,
        "partitions": partition_details,
    }
    if aborted:
        result["note"] = "One or more POS partitions exceeded the state limit."
    return result
