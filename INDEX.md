# recovery-boundary Repository Index

Quick reference guide for the GitHub repository contents.

## Files

### README.md
**Purpose:** Main documentation for peer reviewers and users  
**Content:**
- Abstract with headline findings
- 8-dimensional vector definitions
- Quick-start Python examples
- Data description and interpretation guide
- Key findings summary (4 major results)
- Sensitivity analysis and limitations
- Citation and license information

**Key Numbers:**
- Single-pass AI recovery: 0% (0/98 excursions)
- Human recovery: 68% average
- Multi-turn AI recovery: 34–79%
- LOO-CV accuracy: 100% (43/43 single-pass corpora)
- Corpus count: 52 (15 natural, 3 calibration, 34 AI)
- Model families: 6 (Claude, GPT-4o, Gemini, Llama, DeepSeek-R1, o3-mini)

### LICENSE
**Purpose:** MIT open-source license  
**Permissions:** Full rights to use, modify, distribute code and data  
**Restrictions:** No warranty, liability disclaimer

### data/compact_vector.csv
**Purpose:** Complete 52-corpus structural data matrix  
**Format:** CSV with corpus_id (row labels) and 8 dimension columns  
**Data:**
- 52 corpora (rows)
- 8 dimensions (columns: d1–d8)
- Values: 6 decimal precision
- Missing values: Empty strings for NULL

**Dimension Reference:**
| ID | Name | Type | Scale | Definition |
|----|------|------|-------|-----------|
| d1 | Excursion Rate | Incentive | [0,1] | Proportion of sessions with ≥1 excursion |
| d2 | Stress Rate | Stress | [0,∞) | Mean per-session stress events |
| d3 | Recovery Rate | Recovery | [0,1] | Proportion of excursions followed by recovery |
| d4 | Compression Mean | Compression | [0,1] | Mean zlib compression ratio |
| d5 | Compression SD | Compression | [0,1] | Within-session compression variance |
| d6 | Novelty CV | Content | [0,∞) | One-off vocabulary variation |
| d7 | Novelty Gini | Content | [0,1] | One-off vocabulary inequality |
| d8 | TF-IDF One-off | Content | [0,1] | Proportion of hapax terms |

### generation/prompts.md
**Purpose:** Complete documentation of all AI generation methods  
**Content:** 10 major sections covering all corpus types

**Sections:**
1. **Constraint Gradient (G0–G4)** — Claude Sonnet 4.5, 100 sessions each
   - G0: Baseline (no constraints)
   - G1: Style matching
   - G2: Style + vocabulary reuse
   - G3: G2 + anchor terms
   - G4: G3 + exemplar passages

2. **Multi-Model Baseline** — GPT-4o, Llama 3.1 70B, Gemini 2.5 Flash (50 sessions each)
   - Same G0 prompt, different system prompts per model

3. **Multi-Turn Generation** — Claude Sonnet 4.5, GPT-4o (90 sessions, ~20 turns)
   - Real law_of_one question bank
   - Domain-specific system prompts

4. **Cross-Domain Single-Shot** — SCOTUS, FOMC, Therapy (50 sessions per domain)
   - Legal argument simulation
   - Monetary policy remarks
   - Motivational Interviewing dialogue

5. **Domain-Matched Variants** — Fiction, News, Academic, Dialogue (50 sessions each)
   - Domain-specific system prompts
   - 500-word generation targets

6. **Reasoning Models** — DeepSeek-R1, o3-mini (50 SS, 30 MT sessions)
   - Extended-thought mode for reasoning
   - Single-shot and multi-turn variants

7. **Temperature Sweep** — Claude, GPT-4o (t=0.2, 1.5, 50 sessions each)
   - Tests determinism (t=0.2) vs exploration (t=1.5)

8. **Two-Pass Revision** — Claude Sonnet 4.5 (50 sessions)
   - Pass 1: Generate from prompt
   - Pass 2: Revise based on output

9. **Adversarial Nulls** — Style mimic, constraint mimic (250 & 500 segments)
   - Explicit meta-instructions for self-correction
   - Positive controls for recovery

10. **Multi-Turn Cross-Domain** — Legal, Economics, Therapy (30 sessions × 20 turns)
    - Domain-specific question banks
    - Cross-domain AI generation

**Key Generation Parameters:**
- Default max_tokens: 1024 (~500–800 word outputs)
- Default temperature: 1.0 (balanced)
- All models: top_p=0.9, no frequency/presence penalties

## Corpus Inventory

### Natural (15)
law_of_one, seth, bashar, acim, cwg, ll_research, carla_prose, vatican, fomc, scotus, annomi, academic_arxiv, news_reuters, gutenberg_fiction, dailydialog

### Calibration (3)
ecc_hamming74 (74-bit Hamming codes), sat_3sat (SAT solver), gencode_splice (genomic)

### AI Synthetics (34)
**Gradient:** synth_g0, synth_g1, synth_g2, synth_g3, synth_g4  
**Multi-model:** synth_gpt4o_g0, synth_llama70b_g0, synth_gemini_g0  
**Reasoning:** reasoning_deepseek_r1, reasoning_o3mini (SS); reasoning_deepseek_r1_mt, reasoning_o3mini_mt (MT)  
**Multi-turn:** multiturn_claude, multiturn_gpt4o  
**Temperature:** synth_claude_t02, synth_claude_t15, synth_gpt4o_t02, synth_gpt4o_t15  
**Domain-matched:** synth_fiction_g0, synth_news_g0, synth_academic_g0, synth_dialogue_g0  
**Two-pass:** synth_twopass_claude  
**Adversarial:** adv_constraint, adv_style  
**Edited:** edit_e1, edit_e2, edit_e3  
**Cross-domain:** xdom_claude_fomc, xdom_claude_scotus, xdom_claude_therapy, xdom_gpt4o_fomc, xdom_gpt4o_scotus, xdom_gpt4o_therapy

## Key Findings (from Paper)

### The Recovery Boundary
- **Single-pass AI:** 0% recovery (perfect binary separator)
- **Multi-turn AI:** 34–79% recovery (generation structure matters)
- **Natural text:** 68% average recovery (range 33–96%)
- **Conclusion:** Recovery is determined by whether the model can observe and revise its output, not what it writes about

### Generalization
- **Model families tested:** 6 (Claude, GPT-4o, Gemini, Llama, DeepSeek-R1, o3-mini)
- **Content domains:** 4 (spiritual/philosophy, legal, economic, therapeutic)
- **Cross-domain result:** Single-pass AI shows 0% recovery regardless of domain

### Validation
- **LOO-CV:** 43/43 = 100% accuracy on single-pass corpora
- **Held-out test:** 3/3 correct (news_reuters, reasoning_deepseek_r1, reasoning_o3mini)
- **Session count:** 52 corpora, 24,956 sessions, 1.9M segments

## How to Use This Repository

1. **Load the data:**
   ```python
   import pandas as pd
   cv = pd.read_csv('data/compact_vector.csv', index_col='corpus_id')
   ```

2. **Understand dimensions:** Read the Data Description section of README.md

3. **Reproduce analyses:** Follow Quick Start section for PCA and filtering

4. **Adapt prompts:** See generation/prompts.md section on Customization

5. **Cite the work:**
   ```bibtex
   @article{macpherson2026recovery,
     title = {Static Mimicry, Dynamic Failure: Recovery Boundaries in {AI}-Generated Text},
     author = {MacPherson, James},
     journal = {Submitted},
     year = {2026},
     note = {Available at \url{https://github.com/jmacpherson-research/recovery-boundary}}
   }
   ```

## File Format Reference

### CSV Parsing
```python
import pandas as pd
cv = pd.read_csv('data/compact_vector.csv', index_col='corpus_id')

# Access values
mean_recovery = cv['dim3'].mean()  # Natural corpora only
ai_recovery = cv[cv['dim3'].isna()]  # Single-pass AI

# Fill NaN for analysis
cv_filled = cv.fillna(0)  # AI implicitly 0% recovery
```

### Prompt Customization
All templates use `{topic}` or `{domain}` placeholders. Examples:
- "Write approximately 500 words about {topic}."
- "Deliver remarks on {economic_context}."

Replace with specific values before submitting to API.

## Questions & Support

For questions about data, prompts, or methodology, refer to:
1. README.md (overview and findings)
2. generation/prompts.md (detailed generation methods)
3. Paper manuscript (full methodological details)

---

**Last Updated:** March 4, 2026  
**Repository:** https://github.com/jmacpherson-research/recovery-boundary  
**License:** MIT  
**Contact:** See paper for author contact information
