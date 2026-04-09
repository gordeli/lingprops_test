# lingprops

Utilities for computing linguistic properties of text in Python — starting with **concreteness** (noun/verb/adj/adv variants) built on NLTK + WordNet.

> **Please cite** if you use this code (APA below; more styles further down):  
> Kronrod, A., Gordeliy, I., & Lee, J. K. (2023). Been There, Done That: How Episodic and Semantic Memory Affects the Language of Authentic and Fictitious Reviews. *Journal of Consumer Research, 50*(2), 405–425. https://doi.org/10.1093/jcr/ucac056  
> Kronrod, A., Lee, J. K., & Gordeliy, I. (2017). Detecting fictitious consumer reviews: A theory-driven approach combining automated text analysis and experimental design. *Marketing Science Institute Working Papers Series*, 17–124.

---

## ✨ Features
- Clean API: `compute_concreteness(text)` — per-POS and total scores, with and without repetitions, computed in a single call
- `count_words(text)` — standalone word counts by POS category
- POS independence: nouns, verbs, adjectives, and adverbs are scored as fully separate partitions (frequencies and deduplication are within-POS only)
- Normalized concreteness: raw score divided by the count of words with non-zero contribution
- Robust NLTK resource bootstrap via `ensure_nltk_data()`
- Command-line interface: `python -m lingprops.scripts.concreteness_cli --text "..."`
- Tests included

---

## 🔧 Requirements
- Python **3.8+**
- OS: Windows, macOS, Linux
- Will download NLTK data on first use (WordNet, taggers, tokenizers)

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

## 🧠 First run (download NLTK models)

Either call the helper in Python:
```python
from lingprops import ensure_nltk_data
ensure_nltk_data()
```

…or run the CLI once (it calls the helper internally):
```bash
python -m lingprops.scripts.concreteness_cli --text "The quick brown fox."
```

This fetches: `wordnet`, `omw-1.4`, `punkt` (and `punkt_tab` if available), `averaged_perceptron_tagger` (and `_eng`).

---

## 🚀 Quick start

### Python
```python
from lingprops import compute_concreteness, count_words

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
r["total"]["word_count"]              # all tokens in text
r["total"]["content_word_counts"]     # {"NN": .., "VB": .., "JJ": .., "RB": .., "CD": ..}

# --- Standalone word counts ---
count_words("The cat chased the cat quickly.")
# {"NN": 2, "VB": 1, "JJ": 0, "RB": 1, "CD": 0, "total": 7}
```

Same fields (`score`, `count`, `normalized_score`, `score_norep`, `count_norep`,
`normalized_score_norep`) are available for each POS key: `"NN"`, `"VB"`, `"JJ"`,
`"RB"`, `"CD"`, and `"total"`.

**Design:** Each POS category is computed independently — word frequencies and
lemma deduplication are strictly within-POS. The no-repetitions mode deduplicates
by lemma (before nounification): "cats" and "cat" are one lemma; "big" and "large"
are two.

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
