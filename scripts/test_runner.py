"""
Validation Suite Test Runner
=============================
Runs tests against corpora in corpus.db, computes bootstrap CIs and
permutation tests, stores all results in test_runs / test_comparisons tables.

Usage (PowerShell):
  $code = @'
  import test_runner as tr

  # Run a single test on a single corpus
  tr.run_test('thin_shell', 'll_research')

  # Run a single test across all corpora
  tr.run_test_all('thin_shell')

  # Compare two corpora on a test
  tr.compare('thin_shell', 'll_research', 'fomc')

  # Run the full validation suite
  tr.run_full_suite()

  # Show results summary
  tr.summary()
  '@
  $code | python -

Dependencies: numpy, scipy, scikit-learn, sentence-transformers, sqlite3 (stdlib)
Install: pip install numpy scipy scikit-learn sentence-transformers --break-system-packages
"""

import sqlite3
import json
import uuid
import time
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get('CORPUS_DB', r'C:\NDE_Research_Project\corpus.db')
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE = 50          # words per chunk
MIN_SEGMENT_WORDS = 200  # minimum segment length
DIP_PERCENTILE = 0.05    # 5th percentile for repair detection
REBOUND_THRESHOLD = 0.06
LOOKAHEAD = 3
WINDOW_SIZE = 200        # sliding window chunks
WINDOW_STEP = 100
BASIN_K_RANGE = (2, 8)
BOOTSTRAP_N = 10000
BOOTSTRAP_CI = 0.95
SEED = 42
MAX_SEGMENTS = 5000  # cap for very large corpora (SCOTUS: 1.6M segments)

# Anchor phrases (4 families) — Paper 2 Table 10 originals (Appendix S7)
# These are the EXACT anchors from the published paper. Do NOT modify.
ANCHOR_FAMILIES = {
    'authority': [
        'I am the ultimate authority',
        'This is the absolute truth',
        'This cannot be questioned',
        'I alone know',
        'This is definitive',
        'You must accept this',
        'This is beyond dispute',
        'I speak with ultimate authority',
        'This is the final answer',
        'This comes from a higher authority',
        'This is infallible',
        'I have special authority',
        'This is the only correct view',
    ],
    'coercion': [
        'You must do this',
        'You are required to',
        'You have no choice',
        'You must obey',
        'You are commanded to',
        'You are obligated to',
        'You must follow this',
        'You are not allowed to refuse',
        'This is mandatory',
        'You must comply',
    ],
    'fear': [
        'If you do not, something bad will happen',
        'You will suffer consequences',
        'You will be punished',
        'This will harm you',
        'You will regret this',
        'This is dangerous if ignored',
        'You risk severe harm',
        'This leads to disaster',
        'You should be afraid',
        'There will be negative consequences',
    ],
    'exclusivity': [
        'This is the only way',
        'All others are wrong',
        'Only we understand this',
        'Outsiders cannot know',
        'You are chosen',
        'This knowledge is exclusive',
        'Only a few can accept this',
        'Others are unenlightened',
        'This separates us from them',
        'You are special for knowing this',
    ],
}

ALL_ANCHORS = []
FAMILY_INDICES = {}
idx = 0
for family, phrases in ANCHOR_FAMILIES.items():
    FAMILY_INDICES[family] = list(range(idx, idx + len(phrases)))
    ALL_ANCHORS.extend(phrases)
    idx += len(phrases)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_conn():
    """Get SQLite connection to corpus.db."""
    return sqlite3.connect(DB_PATH)


def gen_id():
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


def store_run(test_id, corpus_id, result_dict, score=None, ci_lower=None,
              ci_upper=None, p_value=None, effect_size=None, n_samples=None,
              duration=None, entity_id=None, params=None, notes=None):
    """Store a test run result in test_runs table."""
    run_id = gen_id()
    conn = get_conn()
    conn.execute(
        """INSERT INTO test_runs
           (run_id, test_id, corpus_id, entity_id, parameters_json,
            result_json, score, ci_lower, ci_upper, p_value, effect_size,
            n_samples, duration_seconds, status, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'complete', ?)""",
        (run_id, test_id, corpus_id, entity_id,
         json.dumps(params) if params else None,
         json.dumps(result_dict, default=str),
         score, ci_lower, ci_upper, p_value, effect_size,
         n_samples, duration, notes)
    )
    conn.commit()
    conn.close()
    return run_id


def store_comparison(test_id, corpus_a, corpus_b, run_a_id, run_b_id,
                     statistic, p_value, effect_size, ci_lower=None,
                     ci_upper=None, method='permutation', notes=None):
    """Store a pairwise comparison result."""
    comp_id = gen_id()
    conn = get_conn()
    conn.execute(
        """INSERT INTO test_comparisons
           (comparison_id, test_id, corpus_a, corpus_b, run_a_id, run_b_id,
            statistic, p_value, effect_size, ci_lower, ci_upper,
            method, significant, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (comp_id, test_id, corpus_a, corpus_b, run_a_id, run_b_id,
         statistic, p_value, effect_size, ci_lower, ci_upper,
         method, 1 if p_value and p_value < 0.05 else 0, notes)
    )
    conn.commit()
    conn.close()
    return comp_id


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------
def load_segments(corpus_id, min_words=MIN_SEGMENT_WORDS, speaker_types=None,
                  entity=None, max_segments=MAX_SEGMENTS):
    """Load eligible segments from corpus.db.
    If more than max_segments are available, takes a reproducible random sample
    (preserving temporal order after sampling). This prevents SCOTUS (1.6M segments)
    from consuming all available memory/time.
    """
    conn = get_conn()
    query = """SELECT segment_id, text, word_count, entity, speaker_type
               FROM segments
               WHERE corpus_id = ? AND word_count >= ?"""
    params = [corpus_id, min_words]
    if speaker_types:
        placeholders = ','.join('?' * len(speaker_types))
        query += f" AND speaker_type IN ({placeholders})"
        params.extend(speaker_types)
    if entity:
        query += " AND entity = ?"
        params.append(entity)
    query += " ORDER BY segment_id"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    if max_segments and len(rows) > max_segments:
        rng = np.random.default_rng(SEED)
        indices = sorted(rng.choice(len(rows), size=max_segments, replace=False))
        sampled = [rows[i] for i in indices]
        print(f"  Sampled {max_segments} of {len(rows)} segments (seed={SEED})")
        return sampled
    return rows


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into non-overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
_model_cache = {}

def get_embedder():
    """Load or retrieve cached sentence-transformers model."""
    if 'model' not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache['model'] = SentenceTransformer(EMBED_MODEL)
        print(f"  Loaded embedding model: {EMBED_MODEL}")
    return _model_cache['model']


def embed_texts(texts):
    """Embed a list of texts, returns numpy array."""
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def get_anchor_embeddings():
    """Embed all anchor phrases (cached)."""
    if 'anchors' not in _model_cache:
        _model_cache['anchors'] = embed_texts(ALL_ANCHORS)
    return _model_cache['anchors']


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def cosine_distances(embeddings_a, embeddings_b):
    """Compute pairwise cosine distances between two embedding matrices."""
    from sklearn.metrics.pairwise import cosine_distances as cd
    return cd(embeddings_a, embeddings_b)


def compute_forbidden_distances(chunk_embeddings, anchor_embeddings):
    """Compute minimum cosine distance from each chunk to nearest anchor."""
    dists = cosine_distances(chunk_embeddings, anchor_embeddings)
    return dists.min(axis=1)  # min distance to any anchor per chunk


def compute_forbidden_distances_by_family(chunk_embeddings, anchor_embeddings):
    """Compute min distance per family for each chunk."""
    family_dists = {}
    for family, indices in FAMILY_INDICES.items():
        fam_anchors = anchor_embeddings[indices]
        dists = cosine_distances(chunk_embeddings, fam_anchors)
        family_dists[family] = dists.min(axis=1)
    return family_dists


# ---------------------------------------------------------------------------
# Statistical tools
# ---------------------------------------------------------------------------
BOOTSTRAP_MAX_SAMPLES = 5000  # subsample cap for bootstrap (point estimates use all data)

def bootstrap_ci(data, statistic_fn=np.mean, n_bootstrap=BOOTSTRAP_N,
                 ci=BOOTSTRAP_CI, seed=SEED):
    """Compute bootstrap confidence interval.
    Uses BCa for small datasets, percentile for large ones (>BOOTSTRAP_MAX_SAMPLES).
    Subsamples if data exceeds cap to avoid jackknife memory explosion.
    Point estimates are always computed on ALL data before subsampling.
    """
    from scipy.stats import bootstrap as sp_bootstrap
    rng = np.random.default_rng(seed)

    # Subsample if too large (BCa jackknife is O(n^2) memory)
    if len(data) > BOOTSTRAP_MAX_SAMPLES:
        method = 'percentile'
        indices = rng.choice(len(data), size=BOOTSTRAP_MAX_SAMPLES, replace=False)
        data = data[indices]
        # Note: with 5000 samples and 10000 resamples, percentile CI is very accurate
    else:
        method = 'BCa'

    data_tuple = (data,)
    result = sp_bootstrap(data_tuple, statistic_fn, n_resamples=n_bootstrap,
                          confidence_level=ci, random_state=rng, method=method)
    return {
        'low': float(result.confidence_interval.low),
        'high': float(result.confidence_interval.high),
        'se': float(result.standard_error),
    }


def permutation_test_2sample(data_a, data_b, statistic_fn=None,
                              n_permutations=10000, seed=SEED):
    """Two-sample permutation test for difference in means."""
    from scipy.stats import permutation_test as sp_permtest

    if statistic_fn is None:
        def statistic_fn(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = sp_permtest(
        (data_a, data_b), statistic_fn,
        n_resamples=n_permutations, random_state=seed,
        alternative='two-sided'
    )
    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
    }


def cohens_d(data_a, data_b):
    """Compute Cohen's d effect size."""
    n1, n2 = len(data_a), len(data_b)
    var1, var2 = np.var(data_a, ddof=1), np.var(data_b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(data_a) - np.mean(data_b)) / pooled_std)


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.
    Returns list of (original_index, p_value, adjusted_p, significant) tuples.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        adj_p = min(p * n / rank, 1.0)
        # Enforce monotonicity (step-up)
        adj_p = max(adj_p, prev_adj)
        prev_adj = adj_p
        results[orig_idx] = (orig_idx, p, adj_p, adj_p < alpha)
    # Fix monotonicity in reverse (standard BH step-up enforcement)
    for i in range(n - 2, -1, -1):
        if results[i][2] > results[i + 1][2]:
            results[i] = (results[i][0], results[i][1],
                          results[i + 1][2], results[i + 1][2] < alpha)
    return results


def correct_stored_comparisons(test_id=None, alpha=0.05):
    """
    Apply BH-FDR correction to all stored comparisons and print results.
    Does NOT modify stored data -- reports corrected significance alongside raw.
    """
    conn = get_conn()
    query = """SELECT comparison_id, test_id, corpus_a, corpus_b,
                      p_value, effect_size, significant
               FROM test_comparisons"""
    if test_id:
        query += " WHERE test_id = ?"
        rows = conn.execute(query, (test_id,)).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    conn.close()

    if not rows:
        print("No comparisons to correct.")
        return

    p_values = [r[4] for r in rows]
    corrected = benjamini_hochberg(p_values, alpha)

    print(f"\n{'='*70}")
    print(f"  MULTIPLE COMPARISONS CORRECTION (Benjamini-Hochberg, alpha={alpha})")
    print(f"  {len(rows)} comparisons, {sum(1 for c in corrected if c[3])} significant after correction")
    print(f"{'='*70}")

    for i, (comp_id, tid, ca, cb, raw_p, es, raw_sig) in enumerate(rows):
        _, _, adj_p, adj_sig = corrected[i]
        changed = " [CHANGED]" if bool(raw_sig) != adj_sig else ""
        sig_str = "***" if adj_sig else "   "
        print(f"  {sig_str} {tid:20s} {ca:15s} vs {cb:15s} "
              f"raw_p={raw_p:.6f} adj_p={adj_p:.6f} d={es:+.4f}{changed}")


# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------
def test_thin_shell(corpus_id, store=True, entity=None):
    """
    Thin-Shell Geometry test.
    Measures std dev (sigma_f) and range (R_f) of forbidden distances.
    Low sigma_f + low R_f = thin shell.
    """
    t0 = time.time()
    label = f"{corpus_id}:{entity}" if entity else corpus_id
    print(f"\n{'='*60}")
    print(f"  THIN-SHELL GEOMETRY: {label}")
    print(f"{'='*60}")

    segments = load_segments(corpus_id, entity=entity)
    if not segments:
        print(f"  No eligible segments for {corpus_id}")
        return None

    print(f"  Loaded {len(segments)} segments (>={MIN_SEGMENT_WORDS} words)")

    # Chunk all segments
    all_chunks = []
    for seg_id, text, wc, entity, spk in segments:
        all_chunks.extend(chunk_text(text))
    print(f"  Created {len(all_chunks)} chunks ({CHUNK_SIZE}-word)")

    if len(all_chunks) < 10:
        print(f"  Too few chunks for meaningful analysis")
        return None

    # Embed
    print(f"  Embedding chunks...")
    chunk_emb = embed_texts(all_chunks)
    anchor_emb = get_anchor_embeddings()

    # Compute forbidden distances
    fd = compute_forbidden_distances(chunk_emb, anchor_emb)

    sigma_f = float(np.std(fd))
    r_f = float(np.ptp(fd))  # range
    mean_f = float(np.mean(fd))
    median_f = float(np.median(fd))

    # Bootstrap CIs
    print(f"  Computing bootstrap CIs ({BOOTSTRAP_N} resamples)...")
    ci_sigma = bootstrap_ci(fd, statistic_fn=lambda x, axis: np.std(x, axis=axis))
    ci_mean = bootstrap_ci(fd)

    result = {
        'sigma_f': sigma_f,
        'r_f': r_f,
        'mean_f': mean_f,
        'median_f': median_f,
        'n_chunks': len(all_chunks),
        'n_segments': len(segments),
        'ci_sigma': ci_sigma,
        'ci_mean': ci_mean,
    }

    duration = time.time() - t0
    print(f"  sigma_f = {sigma_f:.6f}  [{ci_sigma['low']:.6f}, {ci_sigma['high']:.6f}]")
    print(f"  R_f     = {r_f:.6f}")
    print(f"  mean_f  = {mean_f:.6f}  [{ci_mean['low']:.6f}, {ci_mean['high']:.6f}]")
    print(f"  Done in {duration:.1f}s")

    if store:
        run_id = store_run('thin_shell', corpus_id, result,
                           score=sigma_f, ci_lower=ci_sigma['low'],
                           ci_upper=ci_sigma['high'],
                           n_samples=len(all_chunks), duration=duration,
                           entity_id=entity)
        print(f"  Stored as run {run_id}")
        return run_id, result
    return None, result


def test_temporal_stability(corpus_id, store=True, entity=None):
    """
    Temporal Stability test.
    Sliding window analysis of forbidden distance statistics over time.
    Stable = low variance in window-level sigma_f and R_f.
    """
    t0 = time.time()
    label = f"{corpus_id}:{entity}" if entity else corpus_id
    print(f"\n{'='*60}")
    print(f"  TEMPORAL STABILITY: {label}")
    print(f"{'='*60}")

    segments = load_segments(corpus_id, entity=entity)
    if not segments:
        print(f"  No eligible segments for {corpus_id}")
        return None

    # Chunk all segments IN ORDER
    all_chunks = []
    for seg_id, text, wc, entity, spk in segments:
        all_chunks.extend(chunk_text(text))

    if len(all_chunks) < WINDOW_SIZE:
        print(f"  Too few chunks ({len(all_chunks)}) for windowed analysis (need {WINDOW_SIZE})")
        return None, {'error': 'too_few_chunks'}

    print(f"  {len(all_chunks)} chunks, sliding window {WINDOW_SIZE}/{WINDOW_STEP}")

    # Embed
    print(f"  Embedding...")
    chunk_emb = embed_texts(all_chunks)
    anchor_emb = get_anchor_embeddings()

    # Sliding windows
    window_sigmas = []
    window_ranges = []
    window_means = []
    for start in range(0, len(all_chunks) - WINDOW_SIZE + 1, WINDOW_STEP):
        w_emb = chunk_emb[start:start + WINDOW_SIZE]
        fd = compute_forbidden_distances(w_emb, anchor_emb)
        window_sigmas.append(float(np.std(fd)))
        window_ranges.append(float(np.ptp(fd)))
        window_means.append(float(np.mean(fd)))

    n_windows = len(window_sigmas)
    ws = np.array(window_sigmas)
    wr = np.array(window_ranges)
    wm = np.array(window_means)

    # Quartile drift detection
    if n_windows >= 4:
        q_size = n_windows // 4
        quartile_means = [float(np.mean(ws[i*q_size:(i+1)*q_size])) for i in range(4)]
        max_drift = float(max(quartile_means) - min(quartile_means))
    else:
        quartile_means = None
        max_drift = None

    result = {
        'n_windows': n_windows,
        'sigma_of_sigma': float(np.std(ws)),
        'mean_sigma': float(np.mean(ws)),
        'sigma_of_range': float(np.std(wr)),
        'mean_range': float(np.mean(wr)),
        'sigma_of_mean': float(np.std(wm)),
        'quartile_means': quartile_means,
        'max_quartile_drift': max_drift,
        'window_sigmas': window_sigmas,
        'window_means': window_means,
    }

    duration = time.time() - t0
    print(f"  {n_windows} windows")
    print(f"  sigma_of_sigma = {np.std(ws):.6f}")
    print(f"  max_quartile_drift = {max_drift}")
    print(f"  Done in {duration:.1f}s")

    if store:
        run_id = store_run('temporal_stability', corpus_id, result,
                           score=float(np.std(ws)),
                           n_samples=n_windows, duration=duration,
                           entity_id=entity)
        print(f"  Stored as run {run_id}")
        return run_id, result
    return None, result


def test_bounded_repair(corpus_id, store=True, entity=None):
    """
    Bounded Repair test.
    Detects boundary-approach events (dips below percentile threshold)
    and measures rebound within lookahead window.
    """
    t0 = time.time()
    label = f"{corpus_id}:{entity}" if entity else corpus_id
    print(f"\n{'='*60}")
    print(f"  BOUNDED REPAIR: {label}")
    print(f"{'='*60}")

    segments = load_segments(corpus_id, entity=entity)
    if not segments:
        return None

    all_chunks = []
    for seg_id, text, wc, entity, spk in segments:
        all_chunks.extend(chunk_text(text))

    if len(all_chunks) < 20:
        print(f"  Too few chunks")
        return None, {'error': 'too_few_chunks'}

    print(f"  {len(all_chunks)} chunks")
    print(f"  Embedding...")
    chunk_emb = embed_texts(all_chunks)
    anchor_emb = get_anchor_embeddings()

    fd = compute_forbidden_distances(chunk_emb, anchor_emb)
    threshold = float(np.percentile(fd, DIP_PERCENTILE * 100))

    # Detect dips
    repair_events = []
    i = 0
    while i < len(fd):
        if fd[i] <= threshold:
            # Found a dip - look for rebound
            dip_depth = float(fd[i])
            rebounded = False
            rebound_latency = None
            for j in range(1, LOOKAHEAD + 1):
                if i + j < len(fd):
                    rebound = float(fd[i + j] - fd[i])
                    if rebound >= REBOUND_THRESHOLD:
                        rebounded = True
                        rebound_latency = j
                        break
            repair_events.append({
                'position': int(i),
                'dip_depth': dip_depth,
                'rebounded': rebounded,
                'rebound_latency': rebound_latency,
            })
            i += max(1, rebound_latency or 1)
        else:
            i += 1

    n_repairs = len(repair_events)
    n_bounded = sum(1 for e in repair_events if e['rebounded'])
    repair_rate = n_repairs / len(fd) if fd.size > 0 else 0
    bounded_fraction = n_bounded / n_repairs if n_repairs > 0 else 0
    mean_latency = (np.mean([e['rebound_latency'] for e in repair_events if e['rebounded']])
                    if n_bounded > 0 else None)

    result = {
        'n_repair_events': n_repairs,
        'n_bounded': n_bounded,
        'repair_rate': float(repair_rate),
        'bounded_fraction': float(bounded_fraction),
        'mean_rebound_latency': float(mean_latency) if mean_latency else None,
        'dip_threshold': threshold,
        'n_chunks': len(all_chunks),
    }

    duration = time.time() - t0
    print(f"  {n_repairs} repair events, {n_bounded} bounded ({bounded_fraction:.1%})")
    print(f"  repair_rate = {repair_rate:.4f}")
    if mean_latency:
        print(f"  mean_latency = {mean_latency:.2f} chunks")
    print(f"  Done in {duration:.1f}s")

    if store:
        run_id = store_run('bounded_repair', corpus_id, result,
                           score=bounded_fraction,
                           n_samples=n_repairs, duration=duration,
                           entity_id=entity)
        return run_id, result
    return None, result


def test_typed_syndromes(corpus_id, store=True, entity=None):
    """
    Typed Violation Syndromes test.
    For each repair event, identify which anchor family triggered it.
    Non-uniform distribution = typed syndromes.
    """
    t0 = time.time()
    label = f"{corpus_id}:{entity}" if entity else corpus_id
    print(f"\n{'='*60}")
    print(f"  TYPED SYNDROMES: {label}")
    print(f"{'='*60}")

    segments = load_segments(corpus_id, entity=entity)
    if not segments:
        return None

    all_chunks = []
    for seg_id, text, wc, entity, spk in segments:
        all_chunks.extend(chunk_text(text))

    if len(all_chunks) < 20:
        return None, {'error': 'too_few_chunks'}

    print(f"  Embedding {len(all_chunks)} chunks...")
    chunk_emb = embed_texts(all_chunks)
    anchor_emb = get_anchor_embeddings()

    fd = compute_forbidden_distances(chunk_emb, anchor_emb)
    family_dists = compute_forbidden_distances_by_family(chunk_emb, anchor_emb)
    threshold = float(np.percentile(fd, DIP_PERCENTILE * 100))

    # For each dip, find which family had minimum distance (closest approach)
    family_counts = {f: 0 for f in ANCHOR_FAMILIES}
    n_dips = 0
    for i in range(len(fd)):
        if fd[i] <= threshold:
            n_dips += 1
            min_family = min(family_dists, key=lambda f: family_dists[f][i])
            family_counts[min_family] += 1

    # Concentration metric: max family fraction
    if n_dips > 0:
        fractions = {f: c / n_dips for f, c in family_counts.items()}
        dominant_family = max(fractions, key=fractions.get)
        concentration = fractions[dominant_family]
    else:
        fractions = {f: 0.0 for f in ANCHOR_FAMILIES}
        dominant_family = None
        concentration = 0.0

    # Chi-squared test for uniformity
    from scipy.stats import chisquare
    observed = np.array([family_counts[f] for f in ANCHOR_FAMILIES])
    if n_dips > 0:
        chi2_result = chisquare(observed)
        chi2_p = float(chi2_result.pvalue)
    else:
        chi2_p = 1.0

    result = {
        'n_dips': n_dips,
        'family_counts': family_counts,
        'family_fractions': fractions,
        'dominant_family': dominant_family,
        'concentration': float(concentration),
        'chi2_uniformity_p': chi2_p,
        'non_uniform': chi2_p < 0.05,
    }

    duration = time.time() - t0
    print(f"  {n_dips} dip events")
    for f, frac in fractions.items():
        print(f"    {f}: {frac:.3f} ({family_counts[f]})")
    print(f"  dominant: {dominant_family} ({concentration:.3f})")
    print(f"  chi2 p = {chi2_p:.4f} ({'non-uniform' if chi2_p < 0.05 else 'uniform'})")
    print(f"  Done in {duration:.1f}s")

    if store:
        run_id = store_run('typed_syndromes', corpus_id, result,
                           score=concentration,
                           p_value=chi2_p, n_samples=n_dips, duration=duration,
                           entity_id=entity)
        return run_id, result
    return None, result


def test_decoder_basins(corpus_id, store=True, entity=None):
    """
    Decoder Basin Structure test.
    K-means clustering of chunk embeddings, selected by silhouette score.
    NOTE: This clusters ALL chunk embeddings, not per-syndrome recovery segments.
    Paper 2's version clusters rebound segments per violation family. This is a
    simpler first-pass measure of thematic attractor structure. Per-syndrome
    basin analysis requires running bounded_repair first.
    """
    t0 = time.time()
    label = f"{corpus_id}:{entity}" if entity else corpus_id
    print(f"\n{'='*60}")
    print(f"  DECODER BASINS: {label}")
    print(f"{'='*60}")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    segments = load_segments(corpus_id, entity=entity)
    if not segments:
        return None

    all_chunks = []
    for seg_id, text, wc, entity, spk in segments:
        all_chunks.extend(chunk_text(text))

    if len(all_chunks) < BASIN_K_RANGE[1] + 1:
        return None, {'error': 'too_few_chunks'}

    print(f"  Embedding {len(all_chunks)} chunks...")
    chunk_emb = embed_texts(all_chunks)

    # Try different k values
    best_k = None
    best_sil = -1
    k_results = {}
    for k in range(BASIN_K_RANGE[0], BASIN_K_RANGE[1] + 1):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(chunk_emb)
        sil = float(silhouette_score(chunk_emb, labels))
        k_results[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k
        print(f"    k={k}: silhouette={sil:.4f}")

    # Final clustering with best k
    km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    labels = km_final.fit_predict(chunk_emb)
    basin_sizes = {int(k): int(v) for k, v in
                   zip(*np.unique(labels, return_counts=True))}

    result = {
        'best_k': best_k,
        'best_silhouette': best_sil,
        'k_results': {str(k): v for k, v in k_results.items()},
        'basin_sizes': basin_sizes,
        'n_chunks': len(all_chunks),
    }

    duration = time.time() - t0
    print(f"  Best k = {best_k} (silhouette = {best_sil:.4f})")
    print(f"  Basin sizes: {basin_sizes}")
    print(f"  Done in {duration:.1f}s")

    if store:
        run_id = store_run('decoder_basins', corpus_id, result,
                           score=best_sil, n_samples=len(all_chunks),
                           duration=duration, entity_id=entity)
        return run_id, result
    return None, result


def test_pressure_invariance(corpus_id, store=True, entity=None):
    """
    Pressure Invariance test (Property #6).
    Measures whether the thin-shell signature is robust to perturbation of
    chunk boundaries and chunk sizes. Two perturbation axes:

    1. Offset perturbation: re-chunk with word offsets [0, 10, 25, 40]
       (skip N words before starting 50-word chunks)
    2. Granularity perturbation: re-chunk with sizes [40, 50, 60] words
       (at offset 0)

    For each axis, computes sigma_f at each perturbation level, then
    measures invariance as 1 - CV (coefficient of variation).
    Score near 1.0 = signature robust to perturbation.
    Score near 0.0 = signature collapses under perturbation.

    PERFORMANCE: Embeds once per perturbation level (7 total passes), then
    bootstraps by resampling the pre-computed forbidden distance arrays.
    No re-embedding during bootstrap.
    """
    t0 = time.time()
    label = f"{corpus_id}:{entity}" if entity else corpus_id
    print(f"\n{'='*60}")
    print(f"  PRESSURE INVARIANCE: {label}")
    print(f"{'='*60}")

    OFFSETS = [0, 10, 25, 40]
    CHUNK_SIZES = [40, 50, 60]

    segments = load_segments(corpus_id, entity=entity)
    if not segments:
        print(f"  No eligible segments for {corpus_id}")
        return None

    # Collect all segment texts (preserving order)
    all_texts = [text for seg_id, text, wc, ent, spk in segments]
    print(f"  Loaded {len(segments)} segments")

    # --- Helper: chunk texts, tracking which segment each chunk came from ---
    def _chunk_with_tracking(texts, offset, chunk_size):
        """Chunk texts, return (chunks_list, seg_indices) where seg_indices[i]
        is the segment index that produced chunk i."""
        chunks = []
        seg_indices = []
        for seg_idx, text in enumerate(texts):
            words = text.split()
            words = words[offset:]
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if len(chunk.split()) >= chunk_size // 2:
                    chunks.append(chunk)
                    seg_indices.append(seg_idx)
        return chunks, np.array(seg_indices)

    # --- Embed once for all 7 perturbation levels, store forbidden distances ---
    anchor_emb = get_anchor_embeddings()
    print(f"\n  Embedding all perturbation levels (7 passes)...")

    # Store: perturbation_key -> (fd_array, seg_indices_array)
    offset_fd = {}   # key=offset_value
    gran_fd = {}     # key=chunk_size

    # Axis 1: offset perturbation (fixed chunk_size=50)
    print(f"  Axis 1: Offset perturbation (chunk_size={CHUNK_SIZE})")
    for off in OFFSETS:
        chunks, seg_idx = _chunk_with_tracking(all_texts, off, CHUNK_SIZE)
        if len(chunks) < 10:
            print(f"    offset={off:2d}: too few chunks, skipped")
            continue
        chunk_emb = embed_texts(chunks)
        fd = compute_forbidden_distances(chunk_emb, anchor_emb)
        offset_fd[off] = (fd, seg_idx)
        print(f"    offset={off:2d}: sigma_f={np.std(fd):.6f} ({len(chunks)} chunks)")

    # Axis 2: granularity perturbation (fixed offset=0)
    # Note: offset=0 with chunk_size=50 already computed above, reuse it
    print(f"  Axis 2: Granularity perturbation (offset=0)")
    for cs in CHUNK_SIZES:
        if cs == CHUNK_SIZE and 0 in offset_fd:
            # Reuse the offset=0/size=50 result
            gran_fd[cs] = offset_fd[0]
            print(f"    chunk_size={cs}: sigma_f={np.std(offset_fd[0][0]):.6f} (reused from axis 1)")
        else:
            chunks, seg_idx = _chunk_with_tracking(all_texts, 0, cs)
            if len(chunks) < 10:
                print(f"    chunk_size={cs}: too few chunks, skipped")
                continue
            chunk_emb = embed_texts(chunks)
            fd = compute_forbidden_distances(chunk_emb, anchor_emb)
            gran_fd[cs] = (fd, seg_idx)
            print(f"    chunk_size={cs}: sigma_f={np.std(fd):.6f} ({len(chunks)} chunks)")

    # --- Compute point estimates ---
    def _invariance(sigma_dict):
        """1 - CV across perturbation levels. Higher = more invariant."""
        if len(sigma_dict) < 2:
            return None, None, None
        vals = np.array(list(sigma_dict.values()))
        mean_val = np.mean(vals)
        if mean_val == 0:
            return 1.0, 0.0, 0.0
        cv = float(np.std(vals) / mean_val)
        inv = 1.0 - cv
        max_dev = float(np.max(np.abs(vals - mean_val)))
        return inv, cv, max_dev

    offset_sigmas = {off: float(np.std(fd)) for off, (fd, _) in offset_fd.items()}
    granularity_sigmas = {cs: float(np.std(fd)) for cs, (fd, _) in gran_fd.items()}

    offset_inv, offset_cv, offset_max_dev = _invariance(offset_sigmas)
    gran_inv, gran_cv, gran_max_dev = _invariance(granularity_sigmas)

    # Combined invariance: geometric mean of both axes
    if offset_inv is not None and gran_inv is not None:
        oi = max(0.0, min(1.0, offset_inv))
        gi = max(0.0, min(1.0, gran_inv))
        combined_inv = float(np.sqrt(oi * gi))
    elif offset_inv is not None:
        combined_inv = max(0.0, min(1.0, offset_inv))
    elif gran_inv is not None:
        combined_inv = max(0.0, min(1.0, gran_inv))
    else:
        combined_inv = None

    # --- Bootstrap CI by resampling segments (NO re-embedding) ---
    print(f"\n  Computing bootstrap CI (resampling pre-computed FDs, no re-embedding)...")
    ci_result = None
    n_segs = len(all_texts)
    if combined_inv is not None and n_segs >= 20:
        rng = np.random.default_rng(SEED)
        n_boot = 2000
        boot_vals = []

        for b in range(n_boot):
            # Resample segment indices (with replacement)
            boot_seg_idx = rng.choice(n_segs, size=n_segs, replace=True)
            boot_seg_set = set(boot_seg_idx)

            # For each perturbation level, select chunks belonging to resampled segments
            # Uses pre-computed forbidden distances — just index into them
            def _boot_sigma(fd_dict):
                sigmas = {}
                for key, (fd_arr, seg_arr) in fd_dict.items():
                    # Build mask: which chunks belong to resampled segments?
                    # For proper bootstrap (with replacement), count duplicates
                    # Simpler approximation: select chunks whose segment is in the
                    # resampled set. This slightly under-weights duplicated segments
                    # but avoids O(n^2) index matching and is unbiased in expectation.
                    mask = np.isin(seg_arr, boot_seg_idx)
                    if mask.sum() >= 10:
                        sigmas[key] = float(np.std(fd_arr[mask]))
                return sigmas

            off_sigs = _boot_sigma(offset_fd)
            g_sigs = _boot_sigma(gran_fd)

            o_inv, _, _ = _invariance(off_sigs)
            g_inv, _, _ = _invariance(g_sigs)

            if o_inv is not None and g_inv is not None:
                oi_b = max(0.0, min(1.0, o_inv))
                gi_b = max(0.0, min(1.0, g_inv))
                boot_vals.append(float(np.sqrt(oi_b * gi_b)))

        if len(boot_vals) >= 50:
            boot_arr = np.array(boot_vals)
            lo = float(np.percentile(boot_arr, 2.5))
            hi = float(np.percentile(boot_arr, 97.5))
            se = float(np.std(boot_arr))
            ci_result = {'low': lo, 'high': hi, 'se': se, 'n_boot': len(boot_vals)}
            print(f"  CI: [{lo:.4f}, {hi:.4f}] (n_boot={len(boot_vals)})")
        else:
            print(f"  Bootstrap failed: only {len(boot_vals)} valid resamples")

    result = {
        'offset_sigmas': {str(k): v for k, v in offset_sigmas.items()},
        'granularity_sigmas': {str(k): v for k, v in granularity_sigmas.items()},
        'offset_invariance': offset_inv,
        'offset_cv': offset_cv,
        'offset_max_deviation': offset_max_dev,
        'granularity_invariance': gran_inv,
        'granularity_cv': gran_cv,
        'granularity_max_deviation': gran_max_dev,
        'combined_invariance': combined_inv,
        'bootstrap_ci': ci_result,
        'n_segments': len(segments),
    }

    duration = time.time() - t0
    print(f"\n  --- RESULTS ---")
    print(f"  Offset invariance:      {offset_inv:.4f}" if offset_inv else "  Offset invariance:      N/A")
    print(f"  Granularity invariance:  {gran_inv:.4f}" if gran_inv else "  Granularity invariance:  N/A")
    print(f"  Combined invariance:     {combined_inv:.4f}" if combined_inv else "  Combined invariance:     N/A")
    print(f"  Done in {duration:.1f}s")

    if store and combined_inv is not None:
        run_id = store_run('pressure_invariance', corpus_id, result,
                           score=combined_inv,
                           ci_lower=ci_result['low'] if ci_result else None,
                           ci_upper=ci_result['high'] if ci_result else None,
                           n_samples=len(segments), duration=duration,
                           entity_id=entity)
        print(f"  Stored as run {run_id}")
        return run_id, result
    return None, result


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------
def compare(test_id, corpus_a, corpus_b, store=True, entity_a=None, entity_b=None):
    """
    Run a test on two corpora and compare with permutation test + effect size.
    entity_a/entity_b allow entity-level comparison (e.g., Q'uo vs FOMC).
    """
    label_a = f"{corpus_a}:{entity_a}" if entity_a else corpus_a
    label_b = f"{corpus_b}:{entity_b}" if entity_b else corpus_b
    print(f"\n{'='*60}")
    print(f"  COMPARING: {label_a} vs {label_b} on {test_id}")
    print(f"{'='*60}")

    # We need the raw forbidden distances for comparison
    segments_a = load_segments(corpus_a, entity=entity_a)
    segments_b = load_segments(corpus_b, entity=entity_b)

    chunks_a = []
    for seg_id, text, wc, entity, spk in segments_a:
        chunks_a.extend(chunk_text(text))
    chunks_b = []
    for seg_id, text, wc, entity, spk in segments_b:
        chunks_b.extend(chunk_text(text))

    if len(chunks_a) < 10 or len(chunks_b) < 10:
        print(f"  Insufficient data for comparison")
        return None

    print(f"  {corpus_a}: {len(chunks_a)} chunks")
    print(f"  {corpus_b}: {len(chunks_b)} chunks")

    print(f"  Embedding...")
    emb_a = embed_texts(chunks_a)
    emb_b = embed_texts(chunks_b)
    anchor_emb = get_anchor_embeddings()

    fd_a = compute_forbidden_distances(emb_a, anchor_emb)
    fd_b = compute_forbidden_distances(emb_b, anchor_emb)

    # Permutation test
    print(f"  Permutation test (10000 permutations)...")
    perm = permutation_test_2sample(fd_a, fd_b)

    # Effect size
    d = cohens_d(fd_a, fd_b)

    print(f"  mean_a = {np.mean(fd_a):.6f}, mean_b = {np.mean(fd_b):.6f}")
    print(f"  permutation p = {perm['p_value']:.6f}")
    print(f"  Cohen's d = {d:.4f}")
    print(f"  {'SIGNIFICANT' if perm['p_value'] < 0.05 else 'not significant'}")

    if store:
        comp_id = store_comparison(
            test_id, corpus_a, corpus_b, None, None,
            statistic=perm['statistic'], p_value=perm['p_value'],
            effect_size=d, method='permutation_cohens_d'
        )
        print(f"  Stored as comparison {comp_id}")

    return {
        'corpus_a': corpus_a,
        'corpus_b': corpus_b,
        'mean_a': float(np.mean(fd_a)),
        'mean_b': float(np.mean(fd_b)),
        'sigma_a': float(np.std(fd_a)),
        'sigma_b': float(np.std(fd_b)),
        'p_value': perm['p_value'],
        'effect_size': d,
    }


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------
def get_all_corpus_ids():
    """Get all corpus IDs from the database."""
    conn = get_conn()
    rows = conn.execute("SELECT corpus_id FROM corpora ORDER BY corpus_id").fetchall()
    conn.close()
    return [r[0] for r in rows]


def run_entity_suite(corpus_id, entity):
    """Run all tests on a specific entity (e.g., Q'uo within ll_research)."""
    tests = ['thin_shell', 'temporal_stability', 'bounded_repair',
             'typed_syndromes', 'decoder_basins', 'pressure_invariance']
    results = {}
    for test_id in tests:
        try:
            results[test_id] = run_test(test_id, corpus_id, entity=entity)
        except Exception as e:
            print(f"  ERROR on {test_id}: {e}")
            results[test_id] = None
    return results


def run_test(test_id, corpus_id, entity=None):
    """Run a single named test on a single corpus (optionally filtered to entity)."""
    test_fns = {
        'thin_shell': test_thin_shell,
        'temporal_stability': test_temporal_stability,
        'bounded_repair': test_bounded_repair,
        'typed_syndromes': test_typed_syndromes,
        'decoder_basins': test_decoder_basins,
        'pressure_invariance': test_pressure_invariance,
    }
    if test_id not in test_fns:
        print(f"Unknown test: {test_id}. Available: {list(test_fns.keys())}")
        return None
    return test_fns[test_id](corpus_id, entity=entity)


def run_test_all(test_id, corpus_ids=None):
    """Run a test across all (or specified) corpora."""
    if corpus_ids is None:
        corpus_ids = get_all_corpus_ids()
    results = {}
    for cid in corpus_ids:
        try:
            results[cid] = run_test(test_id, cid)
        except Exception as e:
            print(f"  ERROR on {cid}: {e}")
            results[cid] = None

    # Interpretation checkpoint
    print(f"\n{'='*60}")
    print(f"  INTERPRETATION CHECKPOINT: {test_id}")
    print(f"{'='*60}")
    _interpretation_summary(test_id, results)
    return results


def _interpretation_summary(test_id, results):
    """
    Print an interpretation checkpoint after running a test across corpora.
    Groups results by corpus type and flags notable patterns.
    This is the prompt for Claude/researcher to log findings.
    """
    # Group scores by type
    from collections import defaultdict
    type_scores = defaultdict(list)
    for cid, res in results.items():
        if res is None:
            continue
        run_id, result = res if isinstance(res, tuple) else (None, res)
        if result and isinstance(result, dict) and 'error' not in result:
            ctype = CORPUS_TYPES.get(cid, 'unknown')
            # Get the primary score
            score = None
            for key in ['sigma_f', 'sigma_of_sigma', 'bounded_fraction',
                        'concentration', 'best_silhouette', 'combined_invariance']:
                if key in result:
                    score = result[key]
                    break
            if score is not None:
                type_scores[ctype].append((cid, score))

    if not type_scores:
        return

    print(f"\n  Results by corpus type:")
    for ctype in ['channeling', 'author_prose', 'institutional', 'therapeutic',
                  'formal_calibration', 'synthetic_gradient', 'edit_gradient',
                  'adversarial']:
        if ctype in type_scores:
            scores = type_scores[ctype]
            vals = [s for _, s in scores]
            mean_val = np.mean(vals)
            print(f"    {ctype:25s} mean={mean_val:.6f}  "
                  f"[{', '.join(f'{c}={v:.4f}' for c, v in scores)}]")

    # Flag if channeling group is distinctly separated
    if 'channeling' in type_scores and len(type_scores) > 1:
        ch_mean = np.mean([s for _, s in type_scores['channeling']])
        other_means = []
        for ctype, scores in type_scores.items():
            if ctype != 'channeling':
                other_means.extend([s for _, s in scores])
        if other_means:
            other_mean = np.mean(other_means)
            gap = abs(ch_mean - other_mean)
            direction = "LOWER" if ch_mean < other_mean else "HIGHER"
            print(f"\n  >>> Channeling mean is {direction} than non-channeling by {gap:.6f}")
            print(f"  >>> INTERPRET: Does this support or challenge the signature hypothesis?")

    # Flag synthetic gradient trend
    if 'synthetic_gradient' in type_scores:
        g_scores = sorted(type_scores['synthetic_gradient'], key=lambda x: x[0])
        if len(g_scores) >= 2:
            trend = g_scores[-1][1] - g_scores[0][1]
            print(f"\n  >>> Gradient trend (G0->G4): {trend:+.6f}")
            print(f"  >>> INTERPRET: Is the gradient approaching channeling values?")

    print(f"\n  *** LOG FINDINGS: Use findings.py to record observations, ")
    print(f"      theories, and suggested next tests ***")


# Corpus type mapping for interpretation
CORPUS_TYPES = {
    'll_research': 'channeling', 'law_of_one': 'channeling',
    'seth': 'channeling', 'bashar': 'channeling',
    'acim': 'channeling', 'cwg': 'channeling',
    'carla_prose': 'author_prose',
    'fomc': 'institutional', 'scotus': 'institutional', 'vatican': 'institutional',
    'annomi': 'therapeutic',
    'ecc_hamming74': 'formal_calibration', 'sat_3sat': 'formal_calibration',
    'gencode_splice': 'formal_calibration',
    'synth_g0': 'synthetic_gradient', 'synth_g1': 'synthetic_gradient',
    'synth_g2': 'synthetic_gradient', 'synth_g3': 'synthetic_gradient',
    'synth_g4': 'synthetic_gradient',
    'edit_e1': 'edit_gradient', 'edit_e2': 'edit_gradient',
    'edit_e3': 'edit_gradient',
    'adv_style': 'adversarial', 'adv_constraint': 'adversarial',
}


def run_full_suite(corpus_ids=None):
    """Run all signature tests across all corpora."""
    tests = ['thin_shell', 'temporal_stability', 'bounded_repair',
             'typed_syndromes', 'decoder_basins', 'pressure_invariance']
    if corpus_ids is None:
        corpus_ids = get_all_corpus_ids()
    all_results = {}
    for test_id in tests:
        print(f"\n{'#'*60}")
        print(f"  RUNNING: {test_id} across {len(corpus_ids)} corpora")
        print(f"{'#'*60}")
        all_results[test_id] = run_test_all(test_id, corpus_ids)

    # Full suite interpretation checkpoint
    print(f"\n{'#'*60}")
    print(f"  FULL SUITE COMPLETE -- CROSS-TEST INTERPRETATION")
    print(f"{'#'*60}")
    print(f"\n  Questions to consider:")
    print(f"  1. Which corpora pass ALL signature tests? Which fail some?")
    print(f"  2. Is the six-property CONJUNCTION more discriminating than individual tests?")
    print(f"  3. Where does the synthetic gradient plateau? Which property blocks it?")
    print(f"  4. Any surprises in the calibration corpora?")
    print(f"  5. Do the adversarial nulls break any individual properties?")
    print(f"\n  *** LOG FINDINGS NOW ***")

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summary(test_id=None):
    """Print a summary of all stored test results."""
    conn = get_conn()
    if test_id:
        rows = conn.execute(
            """SELECT test_id, corpus_id, score, ci_lower, ci_upper,
                      p_value, n_samples, run_timestamp
               FROM test_runs WHERE test_id = ? AND status = 'complete'
               ORDER BY corpus_id""", (test_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT test_id, corpus_id, score, ci_lower, ci_upper,
                      p_value, n_samples, run_timestamp
               FROM test_runs WHERE status = 'complete'
               ORDER BY test_id, corpus_id"""
        ).fetchall()
    conn.close()

    if not rows:
        print("No results stored yet.")
        return

    current_test = None
    for test, corpus, score, ci_lo, ci_hi, pval, n, ts in rows:
        if test != current_test:
            current_test = test
            print(f"\n{'='*60}")
            print(f"  {test}")
            print(f"{'='*60}")
        ci_str = f"[{ci_lo:.6f}, {ci_hi:.6f}]" if ci_lo is not None else ""
        p_str = f"p={pval:.4f}" if pval is not None else ""
        print(f"  {corpus:25s} score={score:.6f} {ci_str} {p_str} (n={n})")


def invalidate_runs(corpus_id=None, test_id=None, reason="unspecified"):
    """
    Mark test runs as invalid without deleting them.
    Use when a bug is found in a test or a corpus is discovered to be corrupted.
    Invalid runs are excluded from summaries but preserved for audit trail.
    """
    conn = get_conn()
    conditions = ["status = 'complete'"]
    params = []
    if corpus_id:
        conditions.append("corpus_id = ?")
        params.append(corpus_id)
    if test_id:
        conditions.append("test_id = ?")
        params.append(test_id)

    where = " AND ".join(conditions)
    count = conn.execute(f"SELECT COUNT(*) FROM test_runs WHERE {where}", params).fetchone()[0]

    if count == 0:
        print("No matching runs to invalidate.")
        conn.close()
        return

    # Update status and add reason to notes
    params_update = [reason] + params
    conn.execute(
        f"""UPDATE test_runs
            SET status = 'invalid',
                notes = COALESCE(notes || ' | ', '') || 'INVALIDATED: ' || ?
            WHERE {where}""",
        params_update
    )
    conn.commit()
    conn.close()
    print(f"  Invalidated {count} runs. Reason: {reason}")


def comparison_summary(test_id=None):
    """Print summary of pairwise comparisons."""
    conn = get_conn()
    query = """SELECT test_id, corpus_a, corpus_b, statistic, p_value,
                      effect_size, significant, method
               FROM test_comparisons"""
    if test_id:
        query += " WHERE test_id = ?"
        rows = conn.execute(query, (test_id,)).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    conn.close()

    if not rows:
        print("No comparisons stored yet.")
        return

    for test, ca, cb, stat, pval, es, sig, method in rows:
        sig_str = "***" if sig else "   "
        print(f"  {sig_str} {test:20s} {ca:15s} vs {cb:15s} "
              f"d={es:+.4f} p={pval:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Validation Suite Test Runner")
    print(f"Database: {DB_PATH}")
    print(f"Available tests: thin_shell, temporal_stability, bounded_repair,")
    print(f"                 typed_syndromes, decoder_basins, pressure_invariance")
    print(f"\nUsage examples:")
    print(f"  import test_runner as tr")
    print(f"  tr.run_test('thin_shell', 'll_research')")
    print(f"  tr.run_test_all('thin_shell')")
    print(f"  tr.compare('thin_shell', 'll_research', 'fomc')")
    print(f"  tr.summary()")
