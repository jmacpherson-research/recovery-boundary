# Static Mimicry, Dynamic Failure: Recovery Boundaries in AI-Generated Text

This repository contains data and analysis code for a computational linguistics study examining structural differences between human-written and AI-generated text through the lens of error recovery dynamics.

## Abstract

We present an 8-dimensional compact structural vector that captures the dynamics of text production across multiple dimensions: excursion rate, stress rate, recovery rate, and content-independent statistical measures (compression, vocabulary novelty, and lexical reuse).

Our key finding: **single-pass AI-generated text shows 0% recovery from structural excursions (0/98 excursions across 6 model families and 4 content domains)**, while human-written text recovers an average of 68% of the time. Importantly, multi-turn AI generation shows 34–79% recovery, indicating that generation structure (not content or model family) determines the boundary. This structural asymmetry produces perfect leave-one-out cross-validation accuracy (43/43 corpora) without using lexical, semantic, or prompt-based features.

The 8 dimensions of the compact vector are:
- **d1**: Excursion rate (proportion of sessions with ≥1 structural failure)
- **d2**: Stress rate (mean per-session frequency of sub-threshold segments)
- **d3**: Recovery rate (proportion of excursions followed by return-to-baseline within session)
- **d4**: Compression mean (zlib compressibility, 1200-token chunks)
- **d5**: Compression SD (within-session compression variance)
- **d6**: Novelty CV (one-off vocabulary coefficient of variation across chunks)
- **d7**: Novelty Gini (one-off vocabulary inequality index)
- **d8**: TF-IDF one-off rate (proportion of type-1-occurrence terms)

## Repository Structure

```
.
├── README.md
├── LICENSE
├── data/
│   └── compact_vector.csv              # 52 corpora × 8-dimensional feature vectors
├── scripts/
│   ├── feature_extraction.py           # Core pipeline: Phases A-C + Phase D (compact vector)
│   ├── novelty_budget.py               # Budgeted novelty subsampling (25K words × 50 iter)
│   ├── audit_fix.py                    # Adaptive novelty budgeting + PCA rebuild
│   ├── sensitivity_sweep.py            # 75-combo parameter sweep + diagnostics + held-out
│   ├── shuffle_null.py                 # Shuffle null model (200 permutations)
│   ├── permanova.py                    # PERMANOVA + PERMDISP analyses
│   ├── test_runner.py                  # Validation suite (bootstrap CIs, permutation tests)
│   └── figures/
│       ├── fig_recovery_pca.py         # Recovery bar chart + PCA scatter
│       ├── fig_trace.py                # Time-series schematic
│       ├── fig_lag.py                  # Recovery lag distribution
│       └── fig_coherence.py            # Coherence correlation plot
├── generation/
│   ├── prompts.md                      # Complete generation prompts (all AI corpora)
│   ├── build_standard_ai.py            # G0-G4 constraint gradient (Claude)
│   ├── build_multimodel.py             # GPT-4o, Gemini, Llama
│   ├── build_multiturn.py              # Multi-turn stress test
│   ├── build_crossdomain.py            # Cross-domain single-shot
│   ├── build_matched_ai.py             # Domain-matched AI variants
│   ├── build_reasoning.py              # DeepSeek-R1, o3-mini
│   ├── build_temperature.py            # Temperature sweep
│   ├── build_twopass.py                # Two-pass self-revision
│   ├── build_adversarial.py            # Adversarial null models
│   └── build_multiturn_crossdomain.py  # Multi-turn cross-domain
├── results/                            # Pre-computed analysis outputs (JSON)
│   ├── audit_fix_results.json          # Adaptive novelty budget + PCA rebuild
│   ├── phase3a_sensitivity.json        # 75-combo parameter sweep
│   ├── phase3b_diagnostics.json        # Session-level diagnostics
│   ├── phase3c_statistics.json         # Binomial CIs, ANOVA, PERMANOVA
│   ├── phase3d_qualitative.json        # 55 annotated text examples
│   ├── phase3e_heldout.json            # Held-out validation (3/3)
│   ├── shuffle_null_results.json       # Shuffle null model (18 corpora)
│   ├── permdisp_sensitivity_results.json
│   ├── bootstrap_subsample_results.json
│   ├── novelty_budget_results.json
│   ├── topic_coherence_results.json
│   ├── round2_analyses.json            # Entity-only filter, segment correlation
│   ├── phase2_normalization_results.json
│   ├── phase2_session_vectors_results.json  # ICC + session-level vectors
│   ├── phase2_lmm_results.json         # Mixed-effects models
│   ├── drift_subsampling_results.json   # dim10 kill evidence
│   └── final_prewriting_results.json    # PERMANOVA, per-corpus ablation
└── paper/
    ├── paper3_scrp_cl.tex              # Manuscript (CL journal format, submitted version)
    ├── paper3_scrp.tex                 # Manuscript (ACL format, original)
    ├── paper3_macros.tex               # Data macros (120+ verified values)
    ├── paper3_refs.bib                 # Bibliography
    └── figures/
        ├── fig_recovery.pdf
        ├── fig_pca.pdf
        ├── fig_trace.pdf
        ├── fig_lag.pdf
        └── fig_coherence.pdf
```

## Requirements

**Core requirements** (Python 3.10+):
```
numpy >= 1.20
scipy >= 1.7
scikit-learn >= 1.0
matplotlib >= 3.5
pandas >= 1.3
```

**Optional** (for embedding-based extended analyses only):
```
sentence-transformers >= 2.2
```

Install via pip:
```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

For embedding-based work:
```bash
pip install sentence-transformers
```

## Quick Start

### Loading the Compact Vector

```python
import pandas as pd

# Load the compact vector data
cv = pd.read_csv('data/compact_vector.csv', index_col='corpus_id')

# View all corpora and their 8 dimensions
print(cv.head())
print(cv.describe())

# Identify single-pass AI (all with d3 = NULL or NaN)
single_pass_ai = cv[cv['dim3'].isna()]
print(f"Single-pass AI corpora (n={len(single_pass_ai)})")
print(f"Recovery rate (d3): {single_pass_ai['dim3'].min():.1f}% (all missing)")

# Identify human corpora (d3 > 0)
human_corpora = cv[cv['dim3'] > 0]
print(f"\nHuman/natural corpora (n={len(human_corpora)})")
print(f"Recovery rate (d3): {human_corpora['dim3'].mean():.1f}% (mean)")
```

### Reproducing the PCA

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
cv = pd.read_csv('data/compact_vector.csv', index_col='corpus_id')

# Remove calibration corpora (optional - paper reports 52 corpora including them)
# calibration_ids = ['ecc_hamming74', 'sat_3sat', 'gencode_splice']
# cv_filtered = cv.drop(calibration_ids, errors='ignore')

# Prepare data: fill NaN with 0 (recoveryless AI has implicit d3=0)
X = cv.fillna(0).values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=2)
pc_scores = pca.fit_transform(X_scaled)

# Explained variance
print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.1%}")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Color by corpus type
colors = {'natural': 'blue', 'ai': 'red', 'calibration': 'gray'}
corpus_types = {}  # Map corpus_id -> type (populate as needed)

for i, corpus_id in enumerate(cv.index):
    if corpus_id in ['ecc_hamming74', 'sat_3sat', 'gencode_splice']:
        color = colors['calibration']
        label = 'Calibration'
    elif corpus_id.startswith('synth') or corpus_id.startswith('xdom_') or 'reasoning' in corpus_id or corpus_id in ['adv_constraint', 'adv_style', 'edit_e1', 'edit_e2', 'edit_e3']:
        color = colors['ai']
        label = 'AI'
    else:
        color = colors['natural']
        label = 'Natural'

    ax.scatter(pc_scores[i, 0], pc_scores[i, 1], c=color, s=100, alpha=0.6, label=label)
    ax.annotate(corpus_id, (pc_scores[i, 0], pc_scores[i, 1]), fontsize=8, alpha=0.7)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("52-Corpus PCA: Natural vs Single-Pass AI")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_reproduction.pdf', dpi=300)
plt.show()
```

## Data Description

### `compact_vector.csv`

A 52-row × 8-column matrix where each row is a corpus and each column is a compact vector dimension.

**Corpus types:**
- **Natural (15)**: law_of_one, seth, bashar, acim, cwg, ll_research, carla_prose, vatican, fomc, scotus, annomi, academic_arxiv, news_reuters, gutenberg_fiction, dailydialog
- **Calibration (3)**: ecc_hamming74 (74-bit Hamming code), sat_3sat (SAT solver instance), gencode_splice (genomic data)
- **Single-Pass AI (26)**: synth_g0–g4 (Claude gradient), synth_gpt4o_g0, synth_llama70b_g0, synth_gemini_g0 (multi-model), synth_claude_t02/t15, synth_gpt4o_t02/t15 (temperature sweep), synth_twopass_claude, synth_fiction/news/academic/dialogue_g0 (domain-matched), adv_constraint, adv_style (adversarial), edit_e1/e2/e3 (human-edited), reasoning_deepseek_r1, reasoning_o3mini (single-shot reasoning)
- **Multi-Turn AI (8)**: multiturn_claude, multiturn_gpt4o (same-domain), reasoning_deepseek_r1_mt, reasoning_o3mini_mt (reasoning), xdom_claude/gpt4o_scotus/fomc/therapy (cross-domain)

**Dimension definitions:**

| Dimension | Type | Scale | Definition |
|-----------|------|-------|-----------|
| d1 | Excursion rate | [0, 1] | Proportion of sessions with ≥1 excursion |
| d2 | Stress | [0, ∞) | Stress rate: mean per-session segments below recovery threshold |
| d3 | Recovery | [0, 1] | Recovery rate: proportion of excursions followed by return within session |
| d4 | Compression | [0, 1] | Mean zlib compression ratio (1200-token chunks); lower = more random |
| d5 | Compression SD | [0, 1] | Within-session SD of compression ratio; AI ≈ 0, natural > 0.04 |
| d6 | Novelty CV | [0, ∞) | Coefficient of variation in one-off word rate across chunks |
| d7 | Novelty Gini | [0, 1] | Gini index of one-off vocabulary distribution across chunks |
| d8 | TF-IDF | [0, 1] | Proportion of type-1-occurrence (hapax) terms in vocabulary |

**Measurement notes:**
- Compression computed on non-overlapping 1200-token chunks using zlib (level 6)
- Recovery threshold = 50th percentile of chunk-level compression within corpus
- Excursion = segment with compression ratio below threshold
- Recovery = excursion followed by segment above threshold within same session (lag ≤ 5 segments typical)
- d6/d7 require ≥150K words; smaller corpora may show NaN
- d3 is structurally NULL for single-pass generation (only 1 segment per session); empty string in CSV represents this
- AI synthetics show d3 = NULL across all 6 model families (Claude, GPT-4o, Gemini, Llama, DeepSeek-R1, o3-mini)

## Reproducing Results

### Key Findings

1. **Recovery boundary is generation-structure dependent, not content-dependent.**
   - Single-pass AI: 0/98 excursions recover (0%)
   - Multi-turn AI: 72/110 excursions recover (65.5%)
   - Human-written: mean 68% recovery across domains (range 33–96%)
   - Conclusion: Recovery is determined by whether the model can observe and revise its own output, not what it is writing about.

2. **Leave-one-out cross-validation (LOO-CV) on single-pass corpora: 43/43 = 100% accuracy**
   - Natural (n=15): 15/15 correct
   - Single-pass AI (n=26): 26/26 correct
   - Reasoning single-shot (n=2): 2/2 correct
   - Excluded by design: 3 calibration (random data), 2 adversarial prompt-adapted (confounded with prompt), 2 reasoning multi-turn (excluded to avoid false negatives from role-alternation artifacts)

3. **Content-independence verified across 5 domains (cross-domain AI):**
   - FOMC (economics): 43.8% (GPT), 0% (Claude single-pass)
   - Therapy (dialogue): 50.0% (GPT), 30.0% (Claude) — multi-turn
   - SCOTUS (law): 57.1% (GPT), 0% (Claude single-pass)
   - All single-pass show 0% regardless of domain

4. **Multi-model generalization (R2.1):**
   - Claude Sonnet 4.5: d3 = NULL (single-pass), dim4/dim5 consistent with other AI
   - GPT-4o: d3 = NULL, dim4/dim5 identical to Claude (both ~0.009)
   - Gemini 2.5 Flash: d3 = NULL, dim4/dim5 = 0.009
   - Llama 3.1 70B: d3 = NULL, dim4/dim5 = 0.027 (slightly higher variance)
   - DeepSeek-R1 (single-shot): d3 = NULL; (multi-turn) d3 = 63.0% (entity-only recovery)
   - o3-mini (single-shot): d3 = NULL; (multi-turn) d3 = 89.5% (reasoning chain uses internal refinement)

### Sensitivity Analysis

Parameter sweeps confirm robustness:
- **Recovery threshold**: Tested percentiles 40–60 (paper uses 50th). All yield d3 = 0 for single-pass AI.
- **Chunk size**: Tested 800–2400 tokens (paper uses 1200). Rank correlations ρ > 0.89 across all corpora.
- **Compressor**: Tested gzip, bzip2, lzma. Inter-compressor ρ > 0.92 for both d4 and d5.
- **Session granularity**: All results use session-level recovery (within same session). Cross-session recovery: 0/98 for single-pass AI.

### Interpretation & Limitations

**What the boundary does measure:**
- Structural asymmetry in production: token-by-token generation vs. ability to observe and revise
- A necessary condition for natural language: online constraint management during production

**What it does not measure:**
- Content quality, semantic coherence, factuality, or alignment
- Detectability under adversarial prompting (15/16 adversarial AI show recovery via explicit constraints)
- Multi-turn standard AI recovery (34.1%–78.9% depending on model)
- Reasoning-augmented generation (DeepSeek-R1 multi-turn: 63.0% entity-only recovery; o3-mini: 89.5%)

**Known limitations:**
1. **Temporal span.** Natural corpora span decades or centuries; AI synthetics are single-session. Long-term stylistic drift unmeasurable.
2. **Session length.** Single-pass AI constrained to 1 segment per session (max_tokens ~1024). Recovery is structurally undefined. Natural corpora average 20–100 segments per session.
3. **Content diversity.** AI training data domain mismatch for some corpora (e.g., SCOTUS oral arguments). Cross-domain AI mitigates but doesn't eliminate.
4. **Held-out test set.** LOO-CV on 45 corpora from authors' own data. Three external corpora (news_reuters, reasoning_deepseek_r1, reasoning_o3mini) confirm pattern but are small (N=1–50 sessions).
5. **Compression as proxy.** Assumes zlib entropy correlates with linguistic structure. Validation: compression rank-stable across gzip/bzip2/lzma (ρ > 0.92).
6. **Topic coherence non-significant.** N=8 multi-turn corpora insufficient to rule out coherence as moderator. Revision: "multi-turn AI recovery is high regardless of topic coherence" (r = 0.10, p = 0.811).
7. **No human annotation.** All segmentation and recovery detection automated via compression threshold. Ground-truth validation would strengthen claims.
8. **No comparison to existing detectors.** GPTZero, TurnItIn, DetectGPT: not implemented. Cross-detector correlation unknown.

## Citation

If you use this data or methodology, please cite:

```bibtex
@article{macpherson2026recovery,
  title = {Static Mimicry, Dynamic Failure: Recovery Boundaries in {AI}-Generated Text},
  author = {MacPherson, James},
  journal = {Computational Linguistics (submitted)},
  year = {2026},
  note = {Available at \url{https://github.com/jmacpherson-research/recovery-boundary}}
}
```

For specific corpus data, cite:
- Law of One: Elkins, D., & Elkins, P. (1989–present). *The Law of One: A Channeling Text*. L/L Research.
- FOMC: Federal Reserve Board (1993–present). *Federal Open Market Committee statements and meeting minutes*. Data via [Federal Reserve Board](https://www.federalreserve.gov/monetarypolicy/fomc.htm).
- SCOTUS: Justia Oyez (2000–present). *Supreme Court oral arguments*. Data via [Justia Oyez](https://www.oyez.org/).

## License

This repository is released under the MIT License. See LICENSE file for details.

---

## Questions & Collaboration

For questions, suggestions, or interested in collaboration:
- Open an issue on this repository
- Contact: jamesmacpherson130 at gmail.com

**Status (March 2026):** Paper submitted for peer review. Code and detailed analysis in progress.

**Known TODOs for full release:**
- Raw segment-level data for 52 corpora (under review for privacy/licensing)
- Supplementary analysis results (ICC tables, normalization ablations, LMM outputs)
- Natural corpus retrieval scripts (Oyez API, HuggingFace, Project Gutenberg, FRED)
