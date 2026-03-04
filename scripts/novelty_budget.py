#!/usr/bin/env python3
r"""
Budgeted Novelty Subsampling: Size-independent d6/d7 for all corpora.
Session 94.

For every corpus, subsamples to a fixed word budget (25K words), computes
novelty CV (d6) and novelty Gini (d7), repeats N_ITER times, takes median.
This removes the >=150K word threshold for d6/d7 entirely.

Stores results in a new table `novelty_budgeted` in corpus.db.
Also writes JSON backup to same folder as corpus.db.

Usage:
    cd C:\NDE_Research_Project\pipeline
    python .\run_novelty_budget.py

    # Custom budget:
    python .\run_novelty_budget.py --budget 50000

    # Fewer iterations (faster):
    python .\run_novelty_budget.py --iters 20
"""

import sqlite3
import json
import math
import re
import os
import sys
import argparse
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
DEFAULT_ITERS = 50

STOPWORDS = {
    'the', 'of', 'to', 'and', 'in', 'a', 'is', 'that', 'it', 'as',
    'for', 'this', 'with', 'be', 'are', 'we', 'you', 'not', 'or', 'by',
    'from', 'an', 'at', 'which', 'on', 'i', 'but', 'have', 'has', 'was',
    'were', 'been', 'do', 'does', 'did', 'will',
}

# Skip these — too small for novelty to be meaningful
SKIP_CORPORA = {'ecc_hamming74', 'gencode_splice', 'sat_3sat'}
MIN_WORDS = 5000  # need at least this many words for novelty computation


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
    """
    Compute 5-gram novelty CV and Gini from a list of (session_id, tokens) pairs.
    Returns (cv_5gram, gini_5gram, cv_tfidf, gini_tfidf, n_bins_used).
    """
    n_sess = len(session_tokens_list)
    if n_sess < 4:
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

    # Compute stats
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
    """
    Randomly sample sessions (without replacement) until word budget is reached.
    Returns list of (session_id, tokens) in original session order.
    """
    session_ids = list(session_tokens.keys())
    word_counts = {sid: len(toks) for sid, toks in session_tokens.items()}
    total_words = sum(word_counts.values())

    if total_words <= budget:
        # Corpus is at or below budget — use all sessions in order
        return [(sid, session_tokens[sid]) for sid in sorted(session_ids)]

    # Shuffle session order for random sampling
    shuffled = session_ids.copy()
    rng.shuffle(shuffled)

    selected = []
    running_words = 0
    for sid in shuffled:
        wc = word_counts[sid]
        if running_words + wc > budget * 1.1:  # allow 10% overshoot
            if running_words >= budget * 0.8:  # already have 80%+ of budget
                break
        selected.append(sid)
        running_words += wc
        if running_words >= budget:
            break

    # Sort selected sessions back into original order
    selected_set = set(selected)
    ordered = [(sid, session_tokens[sid]) for sid in sorted(session_ids) if sid in selected_set]
    return ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Budgeted Novelty Subsampling (Session 94)')
    parser.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                       help=f'Word budget per iteration (default: {DEFAULT_BUDGET})')
    parser.add_argument('--iters', type=int, default=DEFAULT_ITERS,
                       help=f'Number of iterations (default: {DEFAULT_ITERS})')
    args = parser.parse_args()

    BUDGET = args.budget
    N_ITER = args.iters

    start = datetime.now()
    print(f"Budgeted Novelty Subsampling — Session 94")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print(f"Budget: {BUDGET:,} words, {N_ITER} iterations")
    print()

    rng = np.random.RandomState(SEED)
    conn = get_db()

    # Get all corpus IDs
    all_corpora = [r['corpus_id'] for r in conn.execute(
        "SELECT DISTINCT corpus_id FROM text_entropy_session ORDER BY corpus_id"
    ).fetchall()]
    conn.close()

    print(f"Found {len(all_corpora)} corpora")
    skip = [c for c in all_corpora if c in SKIP_CORPORA]
    target = [c for c in all_corpora if c not in SKIP_CORPORA]
    if skip:
        print(f"Skipping {len(skip)} calibration corpora: {', '.join(skip)}")
    print(f"Processing {len(target)} corpora\n")

    # Process each corpus
    results = []
    for idx, corpus_id in enumerate(target, 1):
        corpus_start = datetime.now()

        # Load all session tokens for this corpus
        conn = get_db()
        rows = conn.execute("""
            SELECT s.session_id, seg.text
            FROM sessions s
            JOIN segments seg ON s.session_id = seg.session_id
            WHERE s.corpus_id = ?
            ORDER BY s.session_id, seg.sequence_order
        """, (corpus_id,)).fetchall()
        conn.close()

        # Group by session, tokenize
        session_tokens = defaultdict(list)
        for row in rows:
            tokens = tokenize(row['text'])
            session_tokens[row['session_id']].extend(tokens)

        total_words = sum(len(t) for t in session_tokens.values())
        n_sessions = len(session_tokens)

        if total_words < MIN_WORDS:
            print(f"[{idx}/{len(target)}] {corpus_id}: {total_words:,} words — SKIPPED (< {MIN_WORDS})")
            continue

        # Run N_ITER iterations
        iter_cv_5g = []
        iter_gini_5g = []
        iter_cv_tf = []
        iter_gini_tf = []

        for it in range(N_ITER):
            sampled = subsample_sessions(session_tokens, BUDGET, rng)
            result = compute_novelty(sampled)
            if result is None:
                continue
            cv_5g, gini_5g, cv_tf, gini_tf, n_bins = result
            iter_cv_5g.append(cv_5g)
            iter_gini_5g.append(gini_5g)
            iter_cv_tf.append(cv_tf)
            iter_gini_tf.append(gini_tf)

        if not iter_cv_5g:
            print(f"[{idx}/{len(target)}] {corpus_id}: no valid iterations — SKIPPED")
            continue

        # Take medians
        med_cv_5g = float(np.median(iter_cv_5g))
        med_gini_5g = float(np.median(iter_gini_5g))
        med_cv_tf = float(np.median(iter_cv_tf))
        med_gini_tf = float(np.median(iter_gini_tf))
        sd_cv_5g = float(np.std(iter_cv_5g))
        sd_gini_5g = float(np.std(iter_gini_5g))

        elapsed_c = (datetime.now() - corpus_start).total_seconds()

        # Determine if corpus was subsampled or used in full
        was_subsampled = total_words > BUDGET

        results.append({
            'corpus_id': corpus_id,
            'budget': BUDGET,
            'n_iter': len(iter_cv_5g),
            'total_words': total_words,
            'n_sessions': n_sessions,
            'subsampled': 1 if was_subsampled else 0,
            'median_cv_5gram': round(med_cv_5g, 6),
            'median_gini_5gram': round(med_gini_5g, 6),
            'sd_cv_5gram': round(sd_cv_5g, 6),
            'sd_gini_5gram': round(sd_gini_5g, 6),
            'median_cv_tfidf': round(med_cv_tf, 6),
            'median_gini_tfidf': round(med_gini_tf, 6),
        })

        tag = "subsampled" if was_subsampled else "full"
        print(f"[{idx}/{len(target)}] {corpus_id}: cv={med_cv_5g:.4f} gini={med_gini_5g:.4f} "
              f"({tag}, {total_words:,}w, {elapsed_c:.1f}s)")

    # Write to database
    print(f"\nWriting {len(results)} results to database...")
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

    cols = list(results[0].keys())
    placeholders = ', '.join(['?' for _ in cols])
    col_names = ', '.join(cols)
    values = [[r[c] for c in cols] for r in results]
    conn.executemany(f"INSERT INTO novelty_budgeted ({col_names}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()

    elapsed = (datetime.now() - start).total_seconds()

    # JSON backup
    backup = {
        'script': 'run_novelty_budget.py',
        'session': 94,
        'timestamp': start.isoformat(),
        'elapsed_seconds': elapsed,
        'budget': BUDGET,
        'n_iter': N_ITER,
        'n_corpora': len(results),
        'results': results,
    }
    backup_path = os.path.join(os.path.dirname(DB_PATH), 'novelty_budget_results.json')
    with open(backup_path, 'w') as f:
        json.dump(backup, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"BUDGETED NOVELTY COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  {len(results)} corpora processed")
    print(f"  Budget: {BUDGET:,} words × {N_ITER} iterations")
    print(f"  Table: novelty_budgeted")
    print(f"  Backup: {backup_path}")

    # Quick comparison: subsampled vs original
    print(f"\n  Subsampled: {sum(1 for r in results if r['subsampled'])}")
    print(f"  Full (below budget): {sum(1 for r in results if not r['subsampled'])}")


if __name__ == '__main__':
    main()
