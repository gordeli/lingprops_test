# lingprops

Utilities for computing linguistic properties of text in Python — starting with **concreteness** (noun/verb/adj/adv variants) built on NLTK + WordNet.

> **Please cite** if you use this code (APA below; more styles further down):  
> Kronrod, A., Gordeliy, I., & Lee, J. K. (2023). Been There, Done That: How Episodic and Semantic Memory Affects the Language of Authentic and Fictitious Reviews. *Journal of Consumer Research, 50*(2), 405–425. https://doi.org/10.1093/jcr/ucac056  
> Kronrod, A., Lee, J. K., & Gordeliy, I. (2017). Detecting fictitious consumer reviews: A theory-driven approach combining automated text analysis and experimental design. *Marketing Science Institute Working Papers Series*, 17–124.

---

## ✨ Features
- `compute_concreteness(text)` — WordNet hypernym-depth concreteness, per-POS and total, with and without repetitions
- `compute_tangibility(text)` — BWK (Brysbaert et al. 2014) human-rated concreteness norms (1–5 scale), per-POS and total, with and without repetitions
- `count_words(text)` — standalone word counts by POS category
- POS independence: nouns, verbs, adjectives, and adverbs are scored as fully separate partitions (frequencies and deduplication are within-POS only)
- Normalised scores: divided by the count of words with non-zero contribution
- Pluggable word-sense disambiguation: `wsd="lesk"` (default), `"first"`, or `"neural"`
- Automatic named-entity recognition (on by default): unknown proper nouns (people, organisations, places) are folded into the score via their WordNet category lemma; pass `ner=False` to disable
- Robust NLTK resource bootstrap via `ensure_nltk_data()`
- Command-line interface: `python -m lingprops.scripts.concreteness_cli --text "..."`
- Tests included

---

## 🔧 Requirements
- Python **3.8+**
- OS: Windows, macOS, Linux
- Will download NLTK data on first use (WordNet, taggers, tokenizers)
- **spaCy + `en_core_web_sm`** (installed automatically; see below for the
  one-time model download) — used as the default NER backend, which is on
  by default. ~13× faster and ~40 F1 points more accurate than the NLTK
  fallback.

---

## 📥 Installation

### Option A — pip (editable install for development)
```bash
# in your shell
git clone https://github.com/yourname/lingprops.git
cd lingprops
python -m venv .venv         # or: conda create -n lp314 python=3.14
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

### Option B — Conda (recommended on Windows)
```bash
conda create -n lp314 python=3.14 -y
conda activate lp314
git clone https://github.com/yourname/lingprops.git
cd lingprops
python -m pip install -U pip
python -m pip install -e .
```

> **Tip:** Avoid mixing Conda and `venv` simultaneously. If using Conda, skip creating `.venv`.

---

## 🧠 First run (download models)

Run once in Python:
```python
from lingprops import ensure_nltk_data, ensure_spacy_model
ensure_nltk_data()       # WordNet, taggers, tokenizers (~30 MB)
ensure_spacy_model()     # spaCy en_core_web_sm (~12 MB) for the default NER
```

…or from the shell:
```bash
python -m lingprops.scripts.concreteness_cli --text "The quick brown fox."  # NLTK data
python -m spacy download en_core_web_sm                                      # spaCy model
```

NLTK fetches: `wordnet31` (the default hierarchy), `wordnet`, `omw-1.4`,
`punkt` (and `punkt_tab` if available), `averaged_perceptron_tagger` (and
`_eng`), `maxent_ne_chunker`, `words`.

### Three concreteness scores, not two

`compute_concreteness` (specificity) and `compute_tangibility` (BWK ratings) are
computed on the **same word set** — content words, nounified, auxiliaries
excluded — so that the two constructs can be compared word for word. That makes
`compute_tangibility` deliberately *not* the Brysbaert score as published
pipelines compute it.

`compute_bwk_classic` is that published measure: every token found in the BWK
table, averaged, with repetitions and no POS filtering, so function words and
auxiliaries count.

```python
from lingprops import compute_bwk_classic
compute_bwk_classic("I booked four hotels.")
# {"score": ..., "count": 4, "tokens": 4, "coverage": 1.0}
compute_bwk_classic(text, lemmatize=True)   # higher coverage, further from the published pipelines
```

Against the original authors' own scores, the classic variant correlates r = .92
with Le et al.'s `bryscore` where the content-word tangibility reaches r = .81
and sits about 0.43 higher in level. Report whichever answers the question:
`compute_tangibility` for "does specificity differ from tangibility on identical
words", `compute_bwk_classic` for "does this replicate the measure the field
uses".

### WordNet release

Depths come from **WordNet 3.1** by default (NLTK's `wordnet31` corpus); NLTK's
plain `wordnet` corpus is 3.0. Both the hierarchy and the lemmatiser are bound
to the same release, and the choice is reported so an output file can record it:

```python
import lingprops
lingprops.wordnet_version()            # "3.1"  — what is selected
lingprops.installed_wordnet_version()  # "3.1"  — what the corpus reports
lingprops.set_wordnet_version("3.0")   # before scoring, to reproduce older runs
```

`entity.n.01` is the single root of the noun hierarchy in both releases. On a
random sample of 4,000 noun lemmas present in both, the first-sense ancestor
count is identical for 99.53 % of them; the 0.47 % that differ move by a mean
of −0.21 levels. 3.1 has slightly better coverage (117,953 noun lemmas against
117,798), which also means fewer proper nouns need the NER fallback —
`obama`, for instance, is in 3.1 but not in 3.0.

> **Skipping spaCy?** Pass `ner_backend="nltk"` (slower, lower accuracy) or
> `ner=False` (no NER at all). The library will still work, but the default
> `ner_backend="spacy"` will fail with a clear download instruction the
> first time it tries to load the model.

---

## 🚀 Quick start

### Python
```python
from lingprops import compute_concreteness, compute_tangibility, count_words

r = compute_concreteness("The cat chased the cat quickly.")

# --- With repetitions (each token counts with its frequency) ---
r["NN"]["score"]              # raw log-combinatorial sum for nouns
r["NN"]["count"]              # normalization count (tokens with non-zero concreteness)
r["NN"]["normalized_score"]   # score / count

# --- Without repetitions (unique lemmas only, f=1) ---
r["NN"]["score_norep"]
r["NN"]["count_norep"]
r["NN"]["normalized_score_norep"]

# --- Total across all POS ---
r["total"]["normalized_score"]        # with repetitions
r["total"]["normalized_score_norep"]  # without repetitions
r["total"]["word_count"]              # words in the text (punctuation excluded)
r["total"]["content_word_counts"]     # {"NN": .., "VB": .., "JJ": .., "RB": .., "CD": ..}

# --- Standalone word counts ---
count_words("The cat chased the cat quickly.")
# {"NN": 2, "VB": 1, "JJ": 0, "RB": 1, "CD": 0, "total": 6}
```

Same fields (`score`, `count`, `normalized_score`, `score_norep`, `count_norep`,
`normalized_score_norep`) are available for each POS key: `"NN"`, `"VB"`, `"JJ"`,
`"RB"`, `"CD"`, and `"total"`.

**Design:** Each POS category is computed independently — word frequencies and
lemma deduplication are strictly within-POS. The no-repetitions mode deduplicates
by lemma (before nounification): "cats" and "cat" are one lemma; "big" and "large"
are two.

### Word-sense disambiguation (WSD)

Concreteness depth depends on which WordNet synset is selected for each noun.
`compute_concreteness` exposes a `wsd=` option to control that selection:

```python
compute_concreteness(text)                    # default: wsd="first"
compute_concreteness(text, wsd="lesk")        # Lesk gloss-overlap + MFS fallback
compute_concreteness(text, wsd="neural")      # sentence-transformer matching
```

| Strategy | Uses context? | Extra deps | Relative CPU cost | When to use |
|---|---|---|---|---|
| `"first"` (default) | no  | none                    | 1×    | Maximum speed, reproduces the original paper's numbers |
| `"lesk"`            | yes | none (stdlib NLTK)      | ~2×   | Context-aware at negligible extra cost — recommended for new analyses |
| `"neural"`          | yes | `sentence-transformers` | ~100× (CPU) | Highest accuracy; install with `pip install lingprops[neural]` |

Rough throughput on a single CPU thread for 100-word texts (~19 nouns each):
`first` ≈ 0.2 ms/text, `lesk` ≈ 0.4 ms/text, `neural` ≈ 20 ms/text. With 28
threads (e.g. `ProcessPoolExecutor`), 100 k texts take ≲1 s for `first`/`lesk`
and ~1 min for `neural`. See `benchmark_wsd.py` for the full comparison.

> **Reproducibility:** `wsd="first"` preserves the library's original
> (context-free) behaviour exactly — so results from prior publications
> using this package remain reproducible by passing `wsd="first"` (and
> `ner=False`) explicitly. The library default is now `wsd="lesk"`,
> which is context-aware and recommended for new analyses.

### Choosing parameters by dataset size

The default (`wsd="lesk"`, `ner=True`, `ner_backend="spacy"`) is calibrated
for medium datasets (10 k – 1 M texts). For other regimes, here is what
to switch:

| Dataset size | Recommended `wsd` | NER | Why |
|---|---|---|---|
| **< 10 k texts** (small)            | `"neural"`              | on  | Highest accuracy; the ~100× CPU cost is acceptable here (~3 min for 10 k texts on a single thread). Best for case studies, paper experiments, or any analysis where each text matters. |
| **10 k – 1 M** (medium, **default**)| `"lesk"`                | on  | Context-aware sense selection at ~2× the cost of `"first"`. Sub-minute for 100 k on 28 threads. |
| **1 M – 100 M** (large)             | `"lesk"`                | on  | Same defaults; spaCy NER scales fine (~30 h for 100 M on 28 threads). Switch `ner_backend` to `"nltk"` only if you cannot install the spaCy model. |
| **> 100 M** (very large)            | `"first"`               | optional | Speed dominates; the synset pick is less consequential at scale than throughput. Disabling NER halves end-to-end time. |
| **Reproducing prior publications**  | `"first"`               | off | Restores the library's original behaviour exactly. |

**Switching from defaults — examples:**

```python
# Small dataset, max accuracy
compute_concreteness(text, wsd="neural")

# Very large dataset, throughput-first
compute_concreteness(text, wsd="first", ner=False)

# Reproducing the JCR 2023 paper numbers
compute_concreteness(text, wsd="first", ner=False)
```

> **Why `lesk` and not `first` as the new default?** Lesk is context-aware
> (the original always picked the first WordNet synset, abstract or
> concrete, without looking at the sentence) at only ~2× the cost. For
> 99% of analyses outside of strict reproductions of prior papers, this
> is the correct trade-off.

### Named-entity recognition (NER)

Proper nouns not in WordNet (personal names like *Alice*, brand names
like *Microsoft*, or arbitrary place names) would otherwise drop
silently out of the concreteness score. The library historically
compensated with a hand-curated list (e.g. `kevin → person`). NER is
now **on by default**: any entity the NER tagger recognises is
substituted with the lemma of its category, and depth is computed by
the NNP rule as `1 + depth(category)`.

```python
compute_concreteness(text)                              # ner=True, spaCy (default)
compute_concreteness(text, ner_backend="nltk")          # NLTK ne_chunk (no extra deps)
compute_concreteness(text, ner_backend="auto")          # spaCy if installed, else NLTK
compute_concreteness(text, ner=False)                   # reproduce pre-NER numbers
```

**Backend comparison** (30 hand-labelled sentences, 200×100-word texts on CPU):

| Backend | Speed (ms/text) | F1 | Notes |
|---|---|---|---|
| spaCy `en_core_web_sm` (default) | **30**   | **0.84** | Recommended; needs `python -m spacy download en_core_web_sm` |
| NLTK `ne_chunk`                  | 388      | 0.60     | Bundled with NLTK; ~13× slower, ~40 F1 points worse  |

Entity-label → WordNet-lemma mapping (see `lingprops/ner.py`):

| Label | Lemma | Label | Lemma |
|---|---|---|---|
| PERSON      | person       | PRODUCT | product  |
| ORGANIZATION / ORG | organization | EVENT   | event    |
| GPE         | country      | WORK_OF_ART | creation |
| LOCATION / LOC | location  | LAW     | law      |
| FACILITY / FAC | facility  | LANGUAGE | language |
| NORP        | group        | DATE / TIME | time |

**Guard rail:** the override fires **only when the token has no WordNet
synsets at all**, so capitalised common nouns (`apple`) and WordNet-known
instances (`einstein`, `paris`) keep their existing depth. Tokens the
legacy hand-curated list already handles (`kevin → person`, `pt →
therapist`, ...) also keep precedence over NER.

**Backends:** `ner_backend="spacy"` is the default. `ner_backend="auto"`
prefers spaCy and falls back to NLTK if the spaCy model isn't installed —
useful in environments where you can't run the model download.

> **Reproducibility with prior work:** pass `ner=False` **and** `wsd="first"`
> to reproduce numbers from the original library exactly. The current
> defaults (`ner=True`, `wsd="lesk"`) instead use context-aware sense
> selection and resolve OOV proper nouns through NER — both typically
> nudge scores upward where ambiguous nouns or names appear.

### Tangibility (BWK ratings)
```python
t = compute_tangibility("The cat sat on the wooden mat.")

t["total"]["normalized_score"]        # avg BWK rating (1-5), with repetitions
t["total"]["normalized_score_norep"]  # avg BWK rating, unique lemmas only
t["NN"]["normalized_score"]           # nouns only
```

Uses the Brysbaert, Warriner & Kuperman (2014) concreteness norms (~40K words).
Same per-POS independence and with/without-repetitions design as `compute_concreteness`.

### CLI
```bash
python -m lingprops.scripts.concreteness_cli --text "Cats chase mice. Dogs sleep."
python -m lingprops.scripts.concreteness_cli --file review.txt
```

---

## 🧪 Testing
```bash
python -m pip install -U pytest
pytest -q
```

---

## 🛠 Troubleshooting

- **`ModuleNotFoundError: lingprops`**  
  Ensure you ran `python -m pip install -e .` from the project root and that you’re using the same interpreter (`python -c "import sys; print(sys.executable)"`).

- **NLTK tagging/WordNet errors** (e.g., `wn is None` or tagger missing)  
  Call `ensure_nltk_data()` once. On Windows/Conda, you can direct downloads into the env:
  ```python
  import nltk, os
  nltk_dir = os.path.join(os.getenv("CONDA_PREFIX", ""), "nltk_data")
  if nltk_dir:
      os.makedirs(nltk_dir, exist_ok=True)
      for r in ["wordnet","omw-1.4","punkt","averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
          nltk.download(r, download_dir=nltk_dir)
  ```

- **Mixing Conda + venv**  
  Prefer one. If using Conda, open a **terminal from that env** and avoid `.venv`.

---

## 📝 Please Cite

If this package helps your research, please cite the following works.

### APA
- Kronrod, A., Gordeliy, I., & Lee, J. K. (2023). *Been There, Done That: How Episodic and Semantic Memory Affects the Language of Authentic and Fictitious Reviews*. Journal of Consumer Research, 50(2), 405–425. https://doi.org/10.1093/jcr/ucac056
- Kronrod, A., Lee, J. K., & Gordeliy, I. (2017). *Detecting fictitious consumer reviews: A theory-driven approach combining automated text analysis and experimental design*. Marketing Science Institute Working Papers Series, 17–124.

### BibTeX
```bibtex
@article{KronrodGordeliyLee2023JCR,
  author  = {Kronrod, Ann and Gordeliy, Ivan and Lee, Jeffrey K},
  title   = {Been There, Done That: How Episodic and Semantic Memory Affects the Language of Authentic and Fictitious Reviews},
  journal = {Journal of Consumer Research},
  year    = {2023},
  volume  = {50},
  number  = {2},
  pages   = {405--425},
  doi     = {10.1093/jcr/ucac056}
}

@techreport{KronrodLeeGordeliy2017MSI,
  author      = {Kronrod, Ann and Lee, Jeffrey K. and Gordeliy, Ivan},
  title       = {Detecting fictitious consumer reviews: A theory-driven approach combining automated text analysis and experimental design},
  institution = {Marketing Science Institute},
  type        = {Working Paper},
  number      = {17-124},
  year        = {2017}
}
```

### Chicago
- Kronrod, Ann, Ivan Gordeliy, and Jeffrey K. Lee. 2023. “Been There, Done That: How Episodic and Semantic Memory Affects the Language of Authentic and Fictitious Reviews.” *Journal of Consumer Research* 50 (2): 405–25. https://doi.org/10.1093/jcr/ucac056.  
- Kronrod, Ann, Jeffrey K. Lee, and Ivan Gordeliy. 2017. “Detecting Fictitious Consumer Reviews: A Theory-Driven Approach Combining Automated Text Analysis and Experimental Design.” *Marketing Science Institute Working Papers Series*, 17–124.

### MLA
- Kronrod, Ann, et al. “Been There, Done That: How Episodic and Semantic Memory Affects the Language of Authentic and Fictitious Reviews.” *Journal of Consumer Research*, vol. 50, no. 2, 2023, pp. 405–425.  
- Kronrod, Ann, Jeffrey K. Lee, and Ivan Gordeliy. “Detecting Fictitious Consumer Reviews: A Theory-Driven Approach Combining Automated Text Analysis and Experimental Design.” *Marketing Science Institute Working Papers Series*, 2017, pp. 17–124.

---

## 📄 License
MIT — see [LICENSE](LICENSE).

## 🤝 Contributing
Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
