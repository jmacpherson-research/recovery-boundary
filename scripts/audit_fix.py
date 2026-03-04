#!/usr/bin/env python3
r"""
Audit Fix: Recompute novelty budgeted + Phase D for 52 paper corpora.
Session 116.

Fixes:
1. Adaptive budget: starts at 25K, doubles if iterations fail (< 4 sessions).
   This fixes SCOTUS which only got 7/50 valid iterations at 25K.
2. Fixed seed (1337) for reproducibility.
3. Recomputes Phase D (compact_vector d6/d7, PCA, distances) for exactly
   the 52 paper corpora.

Outputs:
- Updates novelty_budgeted table in corpus.db
- Updates compact_vector d6/d7 columns
- Updates compact_pca table (52-corpus PCA only)
- Updates compact_distances table (52-corpus distances only)
- Writes audit_fix_results.json with all values for verification

Usage:
    cd C:\NDE_Research_Project\pipeline
    python .\run_audit_fix.py
"""

import sqlite3
import json
import math
import re
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get('CORPUS_DB_PATH', r'C:\NDE_Research_Project\corpus.db')
WORD_RE = re.compile(r"[A-Za-z']+")
NGRAM_N = 5
TOPK_TERMS = 80
SEED = 1337
DEFAULT_BUDGET = 25000
MAX_BUDGET = 200000  # never exceed this
N_ITER = 50
MIN_SESSIONS = 4  # minimum sessions for valid novelty computation

STOPWORDS = {
    'the', 'of', 'to', 'and', 'in', 'a', 'is', 'that', 'it', 'as',
    'for', 'this', 'with', 'be', 'are', 'we', 'you', 'not', 'or', 'by',
    'from', 'an', 'at', 'which', 'on', 'i', 'but', 'have', 'has', 'was',
    'were', 'been', 'do', 'does', 'did', 'will',
}

# The 52 corpora in the paper (exact set from Appendix F)
PAPER_CORPORA = [
    # Natural (15)
    'law_of_one', 'seth', 'bashar', 'acim', 'cwg', 'scotus', 'fomc',
    'vatican', 'carla_prose', 'll_research', 'annomi', 'dailydialog',
    'gutenberg_fiction', 'news_reuters', 'academic_arxiv',
    # Calibration (3)
    'ecc_hamming74', 'sat_3sat', 'gencode_splice',
    # AI Single-Shot (12)
    'synth_g0', 'synth_g1', 'synth_g2', 'synth_g3', 'synth_g4',
    'synth_gpt4o_g0', 'synth_gemini_g0', 'synth_llama70b_g0',
    'synth_fiction_g0', 'synth_news_g0', 'synth_academic_g0', 'synth_dialogue_g0',
    # AI Multi-Turn (2)
    'multiturn_claude', 'multiturn_gpt4o',
    # AI Reasoning (4)
    'reasoning_deepseek_r1', 'reasoning_o3mini',
    'reasoning_deepseek_r1_mt', 'reasoning_o3mini_mt',
    # AI Temperature (4)
    'synth_claude_t02', 'synth_claude_t15', 'synth_gpt4o_t02', 'synth_gpt4o_t15',
    # AI Cross-Domain (6)
    'xdom_claude_fomc', 'xdom_claude_scotus', 'xdom_claude_therapy',
    'xdom_gpt4o_fomc', 'xdom_gpt4o_scotus', 'xdom_gpt4o_therapy',
    # AI Edited/Adversarial (6)
    'edit_e1', 'edit_e2', 'edit_e3', 'adv_style', 'adv_constraint',
    'synth_twopass_claude',
]

# Skip these for novelty (too small/non-text)
SKIP_NOVELTY = {'ecc_hamming74', 'gencode_splice', 'sat_3sat'}
MIN_WORDS = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def tokenize(text):
    if not text:
        return []
    return [w.lower() for w in WORD_RE.findall(text)]


def gini(arr):
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_arr) / (n * np.sum(sorted_arr))) - (n + 1) / n


def compute_novelty(session_tokens_list, n_bins=None):
    """Compute 5-gram novelty CV and Gini from a list of (session_id, tokens) pairs."""
    n_sess = len(session_tokens_list)
    if n_sess < MIN_SESSIONS:
        return None

    if n_bins is None:
        n_bins = min(96, n_sess)

    bin_size = n_sess / n_bins
    bins = []
    for b in range(n_bins):
        start = int(b * bin_size)
        end = int((b + 1) * bin_size)
        combined = []
        for sid, tokens in session_tokens_list[start:end]:
            combined.extend(tokens)
        bins.append(combined)

    # Channel 1: 5-gram novelty
    seen_5grams = set()
    new_5gram_counts = []
    for tokens in bins:
        if len(tokens) < NGRAM_N:
            new_5gram_counts.append(0)
            continue
        bin_5grams = set()
        for i in range(len(tokens) - NGRAM_N + 1):
            bin_5grams.add(tuple(tokens[i:i + NGRAM_N]))
        new_5gram_counts.append(len(bin_5grams - seen_5grams))
        seen_5grams |= bin_5grams

    # Channel 2: TF-IDF novelty
    bin_term_freqs = []
    doc_freq = Counter()
    for tokens in bins:
        content = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
        tf = Counter(content)
        bin_term_freqs.append(tf)
        for term in set(content):
            doc_freq[term] += 1

    seen_top = set()
    new_term_counts = []
    for bin_idx, tokens in enumerate(bins):
        tf = bin_term_freqs[bin_idx]
        total = sum(tf.values())
        if total == 0:
            new_term_counts.append(0)
            continue
        tfidf = {}
        for term, count in tf.items():
            idf = math.log(n_bins / (1 + doc_freq.get(term, 0)))
            tfidf[term] = (count / total) * idf
        top_terms = set(sorted(tfidf, key=tfidf.get, reverse=True)[:TOPK_TERMS])
        new_term_counts.append(len(top_terms - seen_top))
        seen_top |= top_terms

    arr_5g = np.array(new_5gram_counts, dtype=float)
    arr_tf = np.array(new_term_counts, dtype=float)

    mean_5g = np.mean(arr_5g)
    cv_5g = float(np.std(arr_5g) / mean_5g) if mean_5g > 0 else 0.0
    gini_5g = float(gini(arr_5g))

    mean_tf = np.mean(arr_tf)
    cv_tf = float(np.std(arr_tf) / mean_tf) if mean_tf > 0 else 0.0
    gini_tf = float(gini(arr_tf))

    return (cv_5g, gini_5g, cv_tf, gini_tf, n_bins)


def subsample_sessions(session_tokens, budget, rng):
    """Randomly sample sessions until word budget is reached."""
    session_ids = list(session_tokens.keys())
    word_counts = {sid: len(toks) for sid, toks in session_tokens.items()}
    total_words = sum(word_counts.values())

    if total_words <= budget:
        return [(sid, session_tokens[sid]) for sid in sorted(session_ids)]

    shuffled = session_ids.copy()
    rng.shuffle(shuffled)

    selected = []
    running_words = 0
    for sid in shuffled:
        wc = word_counts[sid]
        if running_words + wc > budget * 1.1:
            if running_words >= budget * 0.8:
                break
        selected.append(sid)
        running_words += wc
        if running_words >= budget:
            break

    selected_set = set(selected)
    ordered = [(sid, session_tokens[sid]) for sid in sorted(session_ids) if sid in selected_set]
    return ordered


# ---------------------------------------------------------------------------
# Part 1: Novelty Budgeting with Adaptive Budget
# ---------------------------------------------------------------------------

def run_novelty_budgeting():
    """Rerun novelty budgeting with adaptive budget for all corpora."""
    print("=" * 60)
    print("PART 1: NOVELTY BUDGETING (adaptive budget)")
    print("=" * 60)

    conn = get_db()
    all_corpora = [r['corpus_id'] for r in conn.execute(
        "SELECT DISTINCT corpus_id FROM text_entropy_session ORDER BY corpus_id"
    ).fetchall()]
    conn.close()

    target = [c for c in all_corpora if c not in SKIP_NOVELTY]
    print(f"Processing {len(target)} corpora (skipping {len(all_corpora) - len(target)} calibration)\n")

    results = {}

    for idx, corpus_id in enumerate(target, 1):
        corpus_start = datetime.now()

        # Load all session tokens
        conn = get_db()
        rows = conn.execute("""
            SELECT s.session_id, seg.text
            FROM sessions s
            JOIN segments seg ON s.session_id = seg.session_id
            WHERE s.corpus_id = ?
            ORDER BY s.session_id, seg.sequence_order
        """, (corpus_id,)).fetchall()
        conn.close()

        session_tokens = defaultdict(list)
        for row in rows:
            tokens = tokenize(row['text'])
            session_tokens[row['session_id']].extend(tokens)

        total_words = sum(len(t) for t in session_tokens.values())
        n_sessions = len(session_tokens)

        if total_words < MIN_WORDS:
            print(f"[{idx}/{len(target)}] {corpus_id}: {total_words:,} words — SKIPPED")
            continue

        # Adaptive budget: try 25K, double if too many failures
        budget = DEFAULT_BUDGET
        best_result = None

        while budget <= MAX_BUDGET:
            rng = np.random.RandomState(SEED)  # Reset seed for each budget attempt

            iter_cv_5g = []
            iter_gini_5g = []
            iter_cv_tf = []
            iter_gini_tf = []

            for it in range(N_ITER):
                sampled = subsample_sessions(session_tokens, budget, rng)
                result = compute_novelty(sampled)
                if result is None:
                    continue
                cv_5g, gini_5g, cv_tf, gini_tf, n_bins = result
                iter_cv_5g.append(cv_5g)
                iter_gini_5g.append(gini_5g)
                iter_cv_tf.append(cv_tf)
                iter_gini_tf.append(gini_tf)

            n_valid = len(iter_cv_5g)

            if n_valid >= N_ITER * 0.8:  # 80% success rate = good enough
                best_result = (iter_cv_5g, iter_gini_5g, iter_cv_tf, iter_gini_tf, budget, n_valid)
                break
            elif n_valid > 0 and best_result is None:
                best_result = (iter_cv_5g, iter_gini_5g, iter_cv_tf, iter_gini_tf, budget, n_valid)

            # Double budget and retry
            old_budget = budget
            budget *= 2
            if budget <= MAX_BUDGET:
                print(f"    {corpus_id}: {n_valid}/{N_ITER} valid at {old_budget:,}w — bumping to {budget:,}w")

        if best_result is None:
            print(f"[{idx}/{len(target)}] {corpus_id}: no valid iterations at any budget — SKIPPED")
            continue

        iter_cv_5g, iter_gini_5g, iter_cv_tf, iter_gini_tf, used_budget, n_valid = best_result

        med_cv_5g = float(np.median(iter_cv_5g))
        med_gini_5g = float(np.median(iter_gini_5g))
        med_cv_tf = float(np.median(iter_cv_tf))
        med_gini_tf = float(np.median(iter_gini_tf))
        sd_cv_5g = float(np.std(iter_cv_5g))
        sd_gini_5g = float(np.std(iter_gini_5g))

        elapsed_c = (datetime.now() - corpus_start).total_seconds()

        results[corpus_id] = {
            'corpus_id': corpus_id,
            'budget': used_budget,
            'n_iter': n_valid,
            'total_words': total_words,
            'n_sessions': n_sessions,
            'subsampled': 1 if total_words > used_budget else 0,
            'median_cv_5gram': round(med_cv_5g, 6),
            'median_gini_5gram': round(med_gini_5g, 6),
            'sd_cv_5gram': round(sd_cv_5g, 6),
            'sd_gini_5gram': round(sd_gini_5g, 6),
            'median_cv_tfidf': round(med_cv_tf, 6),
            'median_gini_tfidf': round(med_gini_tf, 6),
        }

        tag = f"budget={used_budget:,}" if used_budget != DEFAULT_BUDGET else "25K"
        print(f"[{idx}/{len(target)}] {corpus_id}: cv={med_cv_5g:.4f} gini={med_gini_5g:.4f} "
              f"({tag}, {n_valid}/{N_ITER} valid, {elapsed_c:.1f}s)")

    # Write to database
    print(f"\nWriting {len(results)} results to novelty_budgeted table...")
    conn = get_db()
    conn.execute("DROP TABLE IF EXISTS novelty_budgeted")
    conn.execute("""CREATE TABLE novelty_budgeted (
        corpus_id TEXT PRIMARY KEY,
        budget INTEGER NOT NULL,
        n_iter INTEGER NOT NULL,
        total_words INTEGER NOT NULL,
        n_sessions INTEGER NOT NULL,
        subsampled INTEGER NOT NULL,
        median_cv_5gram REAL,
        median_gini_5gram REAL,
        sd_cv_5gram REAL,
        sd_gini_5gram REAL,
        median_cv_tfidf REAL,
        median_gini_tfidf REAL
    )""")

    for r in results.values():
        cols = list(r.keys())
        placeholders = ', '.join(['?' for _ in cols])
        col_names = ', '.join(cols)
        conn.execute(f"INSERT INTO novelty_budgeted ({col_names}) VALUES ({placeholders})",
                     [r[c] for c in cols])
    conn.commit()
    conn.close()

    return results


# ---------------------------------------------------------------------------
# Part 2: Update compact_vector d6/d7 from novelty_budgeted
# ---------------------------------------------------------------------------

def update_compact_vector_d6d7(novelty_results):
    """Update d6 (dim6) and d7 (dim7) in compact_vector from novelty_budgeted."""
    print("\n" + "=" * 60)
    print("PART 2: UPDATE compact_vector d6/d7")
    print("=" * 60)

    conn = get_db()

    # Only update corpora that are in both PAPER_CORPORA and novelty_results
    updated = 0
    for corpus_id in PAPER_CORPORA:
        if corpus_id in novelty_results:
            r = novelty_results[corpus_id]
            conn.execute("""
                UPDATE compact_vector SET dim6 = ?, dim7 = ?
                WHERE corpus_id = ?
            """, (r['median_cv_5gram'], r['median_gini_5gram'], corpus_id))
            updated += 1
        elif corpus_id in SKIP_NOVELTY:
            # Calibration corpora — keep NULL
            pass
        else:
            # Check if corpus has novelty from full-text computation
            # (small corpora below budget get full computation, not subsampled)
            pass

    conn.commit()
    conn.close()
    print(f"Updated d6/d7 for {updated} corpora in compact_vector")


# ---------------------------------------------------------------------------
# Part 3: Recompute PCA for 52 paper corpora
# ---------------------------------------------------------------------------

def recompute_pca_52():
    """Recompute PCA on z-scored 8D vectors for the 52 paper corpora only."""
    print("\n" + "=" * 60)
    print("PART 3: RECOMPUTE PCA (52 paper corpora)")
    print("=" * 60)

    conn = get_db()

    # Get 8D vectors for all 52 paper corpora
    placeholders = ','.join(['?' for _ in PAPER_CORPORA])
    rows = conn.execute(f"""
        SELECT corpus_id, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8
        FROM compact_vector
        WHERE corpus_id IN ({placeholders})
        ORDER BY corpus_id
    """, PAPER_CORPORA).fetchall()

    corpus_ids = [r['corpus_id'] for r in rows]
    n = len(corpus_ids)
    print(f"Loaded {n} corpora from compact_vector")

    # Build matrix (handle NULLs: replace with column mean for PCA)
    dim_names = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8']
    raw_matrix = np.zeros((n, 8))
    null_mask = np.zeros((n, 8), dtype=bool)

    for i, r in enumerate(rows):
        for j, d in enumerate(dim_names):
            val = r[d]
            if val is None:
                null_mask[i, j] = True
                raw_matrix[i, j] = np.nan
            else:
                raw_matrix[i, j] = val

    # Z-score each dimension (using only non-null values)
    z_matrix = np.copy(raw_matrix)
    for j in range(8):
        col = raw_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            m, s = np.mean(valid), np.std(valid)
            if s > 0:
                z_matrix[:, j] = (col - m) / s
            else:
                z_matrix[:, j] = 0.0
        # Replace NaN with 0 for PCA (mean-imputation in z-space)
        z_matrix[np.isnan(z_matrix[:, j]), j] = 0.0

    # PCA via SVD
    centered = z_matrix - np.mean(z_matrix, axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Variance explained
    var_explained = (S ** 2) / np.sum(S ** 2)
    pc1_var = round(var_explained[0] * 100, 1)
    pc2_var = round(var_explained[1] * 100, 1)
    print(f"PC1: {pc1_var}% variance, PC2: {pc2_var}% variance")

    # Project onto PC1 and PC2
    scores = centered @ Vt.T
    pc1_raw = scores[:, 0]
    pc2_raw = scores[:, 1]

    # Orient: naturals should load positively on PC1
    natural_ids = {'law_of_one', 'seth', 'bashar', 'acim', 'cwg', 'scotus', 'fomc',
                   'vatican', 'carla_prose', 'll_research', 'annomi', 'dailydialog',
                   'gutenberg_fiction', 'news_reuters', 'academic_arxiv'}
    nat_mask = np.array([cid in natural_ids for cid in corpus_ids])
    if np.mean(pc1_raw[nat_mask]) < 0:
        pc1_raw = -pc1_raw
        print("  Flipped PC1 sign (naturals now positive)")
    if np.mean(pc2_raw[nat_mask]) < 0:
        # Don't flip PC2 — keep whatever sign comes out
        pass

    # Round for display
    pca_results = {}
    for i, cid in enumerate(corpus_ids):
        pca_results[cid] = {
            'pc1': round(float(pc1_raw[i]), 2),
            'pc2': round(float(pc2_raw[i]), 2),
        }

    # Write to compact_pca table
    conn.execute("DROP TABLE IF EXISTS compact_pca")
    conn.execute("""CREATE TABLE compact_pca (
        corpus_id TEXT PRIMARY KEY,
        pc1 REAL,
        pc2 REAL
    )""")
    for cid, pca in pca_results.items():
        conn.execute("INSERT INTO compact_pca (corpus_id, pc1, pc2) VALUES (?, ?, ?)",
                     (cid, pca['pc1'], pca['pc2']))
    conn.commit()

    # Print natural vs AI separation
    nat_pc1 = [pca_results[c]['pc1'] for c in corpus_ids if c in natural_ids]
    ai_pc1 = [pca_results[c]['pc1'] for c in corpus_ids if c not in natural_ids and c not in SKIP_NOVELTY]
    print(f"  Natural PC1 range: {min(nat_pc1):.2f} to {max(nat_pc1):.2f}")
    print(f"  AI PC1 range: {min(ai_pc1):.2f} to {max(ai_pc1):.2f}")

    conn.close()
    return pca_results, pc1_var, pc2_var


# ---------------------------------------------------------------------------
# Part 4: Recompute distances for 52 paper corpora
# ---------------------------------------------------------------------------

def recompute_distances():
    """Recompute pairwise L2 distances in z-space for 52 paper corpora."""
    print("\n" + "=" * 60)
    print("PART 4: RECOMPUTE DISTANCES (52 paper corpora)")
    print("=" * 60)

    conn = get_db()

    placeholders = ','.join(['?' for _ in PAPER_CORPORA])
    rows = conn.execute(f"""
        SELECT corpus_id, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8
        FROM compact_vector
        WHERE corpus_id IN ({placeholders})
        ORDER BY corpus_id
    """, PAPER_CORPORA).fetchall()

    corpus_ids = [r['corpus_id'] for r in rows]
    n = len(corpus_ids)

    # Build and z-score matrix (same as PCA)
    dim_names = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8']
    raw_matrix = np.zeros((n, 8))
    for i, r in enumerate(rows):
        for j, d in enumerate(dim_names):
            val = r[d]
            raw_matrix[i, j] = val if val is not None else np.nan

    z_matrix = np.copy(raw_matrix)
    for j in range(8):
        col = raw_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            m, s = np.mean(valid), np.std(valid)
            if s > 0:
                z_matrix[:, j] = (col - m) / s
            else:
                z_matrix[:, j] = 0.0
        z_matrix[np.isnan(z_matrix[:, j]), j] = 0.0

    # Pairwise L2 distances
    distances = {}
    conn.execute("DROP TABLE IF EXISTS compact_distances")
    conn.execute("""CREATE TABLE compact_distances (
        corpus_a TEXT NOT NULL,
        corpus_b TEXT NOT NULL,
        l2_distance REAL,
        PRIMARY KEY (corpus_a, corpus_b)
    )""")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(z_matrix[i] - z_matrix[j]))
            conn.execute("INSERT INTO compact_distances (corpus_a, corpus_b, l2_distance) VALUES (?, ?, ?)",
                         (corpus_ids[i], corpus_ids[j], d))
            distances[(corpus_ids[i], corpus_ids[j])] = round(d, 4)

    conn.commit()
    conn.close()

    # Print key distances
    key_pairs = [
        ('law_of_one', 'reasoning_deepseek_r1_mt'),
        ('law_of_one', 'multiturn_claude'),
        ('law_of_one', 'seth'),
        ('law_of_one', 'reasoning_o3mini_mt'),
        ('synth_gpt4o_g0', 'synth_gemini_g0'),
    ]
    print(f"Computed {len(distances)} pairwise distances")
    print("\nKey distances:")
    for a, b in key_pairs:
        if (a, b) in distances:
            print(f"  {a} <-> {b}: {distances[(a, b)]:.4f}")

    return distances


# ---------------------------------------------------------------------------
# Part 5: Generate complete output for Appendix F update
# ---------------------------------------------------------------------------

def generate_appendix_f_output(novelty_results, pca_results, distances, pc1_var, pc2_var):
    """Generate complete Appendix F data and key macro values."""
    print("\n" + "=" * 60)
    print("PART 5: GENERATE OUTPUT")
    print("=" * 60)

    conn = get_db()

    # Get full compact_vector for all 52 corpora
    placeholders = ','.join(['?' for _ in PAPER_CORPORA])
    rows = conn.execute(f"""
        SELECT corpus_id, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8
        FROM compact_vector
        WHERE corpus_id IN ({placeholders})
        ORDER BY corpus_id
    """, PAPER_CORPORA).fetchall()

    appendix_f = {}
    for r in rows:
        cid = r['corpus_id']
        pca = pca_results.get(cid, {'pc1': None, 'pc2': None})
        appendix_f[cid] = {
            'd1': round(r['dim1'], 3) if r['dim1'] is not None else None,
            'd2': round(r['dim2'], 3) if r['dim2'] is not None else None,
            'd3': r['dim3'],  # keep encoded (null/1/1.5/2)
            'd4': round(r['dim4'], 3) if r['dim4'] is not None else None,
            'd5': round(r['dim5'], 3) if r['dim5'] is not None else None,
            'd6': round(r['dim6'], 3) if r['dim6'] is not None else None,
            'd7': round(r['dim7'], 3) if r['dim7'] is not None else None,
            'd8': round(r['dim8'], 3) if r['dim8'] is not None else None,
            'pc1': pca['pc1'],
            'pc2': pca['pc2'],
        }

    conn.close()

    # Key macro values
    macros = {
        'pcaVarOne': pc1_var,
        'pcaVarTwo': pc2_var,
    }

    # Extract specific PCA macros
    macro_pca_map = {
        'pcaACIM': ('acim', 'pc1'),
        'pcaBashar': ('bashar', 'pc1'),
        'pcaCWG': ('cwg', 'pc1'),
        'pcaVatican': ('vatican', 'pc1'),
        'pcaSynthTwopass': ('synth_twopass_claude', 'pc1'),
        'pcaSynthTwopassPCtwo': ('synth_twopass_claude', 'pc2'),
        'pcaMTClaude': ('multiturn_claude', 'pc1'),
        'pcaMTGPT': ('multiturn_gpt4o', 'pc1'),
        'pcaReasoningDSMT': ('reasoning_deepseek_r1_mt', 'pc1'),
        'pcaReasoningOiiiMT': ('reasoning_o3mini_mt', 'pc1'),
    }
    for macro_name, (cid, dim) in macro_pca_map.items():
        val = pca_results.get(cid, {}).get(dim)
        if val is not None:
            macros[macro_name] = f"+{val:.2f}" if val >= 0 else f"{val:.2f}"

    # PCA ranges
    natural_ids = {'law_of_one', 'seth', 'bashar', 'acim', 'cwg', 'scotus', 'fomc',
                   'vatican', 'carla_prose', 'll_research', 'annomi', 'dailydialog',
                   'gutenberg_fiction', 'news_reuters', 'academic_arxiv'}
    nat_pc1 = [pca_results[c]['pc1'] for c in PAPER_CORPORA if c in natural_ids and c in pca_results]
    ai_pc1 = [pca_results[c]['pc1'] for c in PAPER_CORPORA
              if c not in natural_ids and c not in SKIP_NOVELTY and c in pca_results]
    macros['pcaNatPConeLo'] = f"{min(nat_pc1):.2f}"
    macros['pcaNatPConeHi'] = f"+{max(nat_pc1):.2f}"
    macros['pcaAIPConeLo'] = f"{min(ai_pc1):.2f}"
    macros['pcaAIPConeHi'] = f"+{max(ai_pc1):.2f}"

    # Key distances
    dist_macros = {
        'distReasoningDSMTToLoO': ('law_of_one', 'reasoning_deepseek_r1_mt'),
        'distMTClaudeToLoO': ('law_of_one', 'multiturn_claude'),
        'distSethToLoO': ('law_of_one', 'seth'),
        'distReasoningOiiiMTToLoO': ('law_of_one', 'reasoning_o3mini_mt'),
        'distGPTToGemini': ('synth_gpt4o_g0', 'synth_gemini_g0'),
    }
    for macro_name, (a, b) in dist_macros.items():
        if (a, b) in distances:
            macros[macro_name] = round(distances[(a, b)], 2)

    # Print summary
    print("\nKey PCA macros:")
    for k, v in sorted(macros.items()):
        print(f"  \\{k}{{{v}}}")

    print(f"\nAppendix F data for {len(appendix_f)} corpora ready")

    # Print the scotus d6/d7 specifically
    sc = appendix_f.get('scotus', {})
    print(f"\nSCOTUS d6={sc.get('d6')}, d7={sc.get('d7')}, PC1={sc.get('pc1')}, PC2={sc.get('pc2')}")

    return appendix_f, macros


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = datetime.now()
    print(f"AUDIT FIX SCRIPT — Session 116")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print(f"Seed: {SEED}, Default budget: {DEFAULT_BUDGET:,}, Iterations: {N_ITER}")
    print(f"Paper corpora: {len(PAPER_CORPORA)}")
    print()

    # Part 1: Novelty budgeting
    novelty_results = run_novelty_budgeting()

    # Part 2: Update compact_vector
    update_compact_vector_d6d7(novelty_results)

    # Part 3: PCA
    pca_results, pc1_var, pc2_var = recompute_pca_52()

    # Part 4: Distances
    distances = recompute_distances()

    # Part 5: Generate output
    appendix_f, macros = generate_appendix_f_output(novelty_results, pca_results, distances, pc1_var, pc2_var)

    # Save everything to JSON
    elapsed = (datetime.now() - start).total_seconds()

    output = {
        'script': 'run_audit_fix.py',
        'session': 116,
        'timestamp': start.isoformat(),
        'elapsed_seconds': elapsed,
        'seed': SEED,
        'default_budget': DEFAULT_BUDGET,
        'n_iter': N_ITER,
        'n_corpora': len(PAPER_CORPORA),
        'pca_variance': {'pc1': pc1_var, 'pc2': pc2_var},
        'novelty_results': {k: v for k, v in novelty_results.items()},
        'appendix_f': appendix_f,
        'macros': macros,
        'key_distances': {f"{a}__{b}": d for (a, b), d in distances.items()
                         if 'law_of_one' in (a, b) or (a, b) == ('synth_gpt4o_g0', 'synth_gemini_g0')},
    }

    output_path = os.path.join(os.path.dirname(DB_PATH), 'audit_fix_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"AUDIT FIX COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Results: {output_path}")
    print(f"  Tables updated: novelty_budgeted, compact_vector (d6/d7), compact_pca, compact_distances")
    print(f"  Next: Read audit_fix_results.json to update paper3_scrp.tex Appendix F")


if __name__ == '__main__':
    main()
