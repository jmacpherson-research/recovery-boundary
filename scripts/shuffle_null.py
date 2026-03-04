"""
Shuffle Null Model for Recovery Events
=======================================
Tests whether recovery is an artifact of regression-to-the-mean
by shuffling chunk order within sessions and recomputing recovery.

If observed recovery >> shuffled recovery for naturals, recovery reflects
structured temporal dynamics (not just statistical fluctuation).
If AI shows 0% under both observed and shuffled, it confirms genuine absence.

Output: shuffle_null_results.json in same folder as corpus.db
Also stores results in corpus.db table: shuffle_null_results

Usage: cd C:\\NDE_Research_Project\\pipeline && python .\\run_shuffle_null.py
"""

import os
import sys
import json
import gzip
import sqlite3
import random
import numpy as np
from collections import defaultdict

# ── paths ──
DB_PATH = os.path.join(os.path.dirname(os.getcwd()), "corpus.db")
OUT_DIR = os.path.dirname(DB_PATH)
CHUNK_TOKENS = 1200
N_SHUFFLES = 200  # number of shuffle iterations per session
EXCURSION_THRESHOLD_SD = 0.5  # excursion = chunk > mean + 0.5*SD
RECOVERY_WINDOW = 7  # chunks after excursion to check for recovery

# Target corpora: multi-turn AI + representative naturals
# Focus on corpora with enough chunks per session for meaningful analysis
TARGET_CORPORA = [
    # Multi-turn AI (the key test)
    "multiturn_claude", "multiturn_gpt4o",
    # Reasoning MT
    "reasoning_deepseek_r1_mt", "reasoning_o3mini_mt",
    # Natural corpora (representative sample across domains)
    "law_of_one", "seth", "bashar", "acim", "cwg",
    "scotus", "fomc", "vatican", "ll_research", "annomi",
    "carla_prose", "dailydialog",
    "gutenberg_fiction", "news_reuters", "academic_arxiv",
    # Cross-domain AI
    "xdom_claude_fomc", "xdom_claude_scotus", "xdom_claude_therapy",
    "xdom_gpt4o_fomc", "xdom_gpt4o_scotus", "xdom_gpt4o_therapy",
    # Adversarial
    "adv_style", "adv_constraint",
]


def tokenize_simple(text):
    """Simple whitespace tokenizer for chunking."""
    return text.split()


def chunk_text(text, chunk_size=CHUNK_TOKENS):
    """Split text into non-overlapping chunks of chunk_size tokens."""
    tokens = tokenize_simple(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) >= chunk_size * 0.5:  # at least half-size
            chunks.append(" ".join(chunk_tokens))
    return chunks


def compress_ratio(text):
    """Compute gzip compression ratio."""
    raw = text.encode("utf-8")
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)


def compute_recovery(ratios, threshold_sd=EXCURSION_THRESHOLD_SD, window=RECOVERY_WINDOW):
    """
    Compute excursion count and recovery rate from a sequence of compression ratios.

    Returns: (n_excursions, n_recovered, recovery_rate)
    """
    if len(ratios) < 3:
        return 0, 0, None  # too few chunks

    mean_c = np.mean(ratios)
    sd_c = np.std(ratios)

    if sd_c < 1e-10:
        return 0, 0, None  # zero variance

    threshold = mean_c + threshold_sd * sd_c

    n_excursions = 0
    n_recovered = 0

    for i, r in enumerate(ratios):
        if r > threshold:
            n_excursions += 1
            # Check if any chunk within window returns below mean
            for j in range(i + 1, min(i + 1 + window, len(ratios))):
                if ratios[j] <= mean_c:
                    n_recovered += 1
                    break

    if n_excursions == 0:
        return 0, 0, None

    return n_excursions, n_recovered, n_recovered / n_excursions


def process_corpus(conn, corpus_id):
    """Process all sessions for a corpus: observed + shuffled recovery."""
    print(f"  Processing {corpus_id}...")

    # Get all sessions with their concatenated text
    cur = conn.cursor()
    cur.execute("""
        SELECT s.session_id, GROUP_CONCAT(seg.text, ' ')
        FROM sessions s
        JOIN segments seg ON seg.session_id = s.session_id
        WHERE s.corpus_id = ?
        GROUP BY s.session_id
        ORDER BY s.session_id
    """, (corpus_id,))

    sessions = cur.fetchall()

    observed_excursions = 0
    observed_recovered = 0
    shuffled_excursions_total = 0
    shuffled_recovered_total = 0
    n_sessions_with_chunks = 0
    session_results = []

    for session_id, text in sessions:
        if not text:
            continue

        chunks = chunk_text(text)
        if len(chunks) < 3:
            continue

        n_sessions_with_chunks += 1

        # Compute compression ratios
        ratios = [compress_ratio(c) for c in chunks]

        # Observed recovery
        n_exc, n_rec, obs_rate = compute_recovery(ratios)
        observed_excursions += n_exc
        observed_recovered += n_rec

        # Shuffled recovery (N_SHUFFLES iterations)
        shuffle_exc_sum = 0
        shuffle_rec_sum = 0

        for _ in range(N_SHUFFLES):
            shuffled = ratios.copy()
            random.shuffle(shuffled)
            s_exc, s_rec, _ = compute_recovery(shuffled)
            shuffle_exc_sum += s_exc
            shuffle_rec_sum += s_rec

        avg_shuffle_exc = shuffle_exc_sum / N_SHUFFLES
        avg_shuffle_rec = shuffle_rec_sum / N_SHUFFLES

        session_results.append({
            "session_id": session_id,
            "n_chunks": len(chunks),
            "observed_excursions": n_exc,
            "observed_recovered": n_rec,
            "observed_rate": obs_rate,
            "shuffle_avg_excursions": round(avg_shuffle_exc, 2),
            "shuffle_avg_recovered": round(avg_shuffle_rec, 2),
            "shuffle_avg_rate": round(avg_shuffle_rec / avg_shuffle_exc, 4) if avg_shuffle_exc > 0 else None,
        })

    # Aggregate
    obs_rate = observed_recovered / observed_excursions if observed_excursions > 0 else None

    # Aggregate shuffled across all sessions
    total_shuffle_exc = sum(s["shuffle_avg_excursions"] for s in session_results)
    total_shuffle_rec = sum(s["shuffle_avg_recovered"] for s in session_results)
    shuffle_rate = total_shuffle_rec / total_shuffle_exc if total_shuffle_exc > 0 else None

    result = {
        "corpus_id": corpus_id,
        "n_sessions": len(sessions),
        "n_sessions_with_chunks": n_sessions_with_chunks,
        "observed_excursions": observed_excursions,
        "observed_recovered": observed_recovered,
        "observed_rate": round(obs_rate * 100, 1) if obs_rate is not None else None,
        "shuffle_avg_excursions": round(total_shuffle_exc, 1),
        "shuffle_avg_recovered": round(total_shuffle_rec, 1),
        "shuffle_rate": round(shuffle_rate * 100, 1) if shuffle_rate is not None else None,
        "n_shuffle_iterations": N_SHUFFLES,
    }

    if obs_rate is not None and shuffle_rate is not None:
        result["observed_minus_shuffle"] = round((obs_rate - shuffle_rate) * 100, 1)
    else:
        result["observed_minus_shuffle"] = None

    print(f"    {corpus_id}: observed={result['observed_rate']}%, "
          f"shuffled={result['shuffle_rate']}%, "
          f"diff={result['observed_minus_shuffle']}pp, "
          f"sessions={n_sessions_with_chunks}")

    return result


def main():
    random.seed(42)
    np.random.seed(42)

    print(f"Shuffle Null Model for Recovery Events")
    print(f"DB: {DB_PATH}")
    print(f"Chunk size: {CHUNK_TOKENS} tokens")
    print(f"Shuffle iterations: {N_SHUFFLES}")
    print(f"Excursion threshold: mean + {EXCURSION_THRESHOLD_SD}*SD")
    print(f"Recovery window: {RECOVERY_WINDOW} chunks")
    print()

    conn = sqlite3.connect(DB_PATH)

    # Check which target corpora actually exist
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT corpus_id FROM sessions")
    available = {row[0] for row in cur.fetchall()}

    corpora = [c for c in TARGET_CORPORA if c in available]
    missing = [c for c in TARGET_CORPORA if c not in available]
    if missing:
        print(f"Warning: {len(missing)} corpora not found: {missing}")

    print(f"Processing {len(corpora)} corpora...")
    print()

    results = []
    for corpus_id in corpora:
        result = process_corpus(conn, corpus_id)
        results.append(result)

    conn.close()

    # ── Save JSON ──
    out_path = os.path.join(OUT_DIR, "shuffle_null_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "parameters": {
                "chunk_tokens": CHUNK_TOKENS,
                "n_shuffles": N_SHUFFLES,
                "excursion_threshold_sd": EXCURSION_THRESHOLD_SD,
                "recovery_window": RECOVERY_WINDOW,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Save to DB ──
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS shuffle_null_results")
    conn.execute("""
        CREATE TABLE shuffle_null_results (
            corpus_id TEXT PRIMARY KEY,
            n_sessions INTEGER,
            n_sessions_with_chunks INTEGER,
            observed_excursions INTEGER,
            observed_recovered INTEGER,
            observed_rate REAL,
            shuffle_avg_excursions REAL,
            shuffle_avg_recovered REAL,
            shuffle_rate REAL,
            observed_minus_shuffle REAL,
            n_shuffle_iterations INTEGER
        )
    """)
    for r in results:
        conn.execute("""
            INSERT INTO shuffle_null_results VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["corpus_id"], r["n_sessions"], r["n_sessions_with_chunks"],
            r["observed_excursions"], r["observed_recovered"], r["observed_rate"],
            r["shuffle_avg_excursions"], r["shuffle_avg_recovered"], r["shuffle_rate"],
            r["observed_minus_shuffle"], r["n_shuffle_iterations"],
        ))
    conn.commit()
    conn.close()
    print(f"Results stored in corpus.db table: shuffle_null_results")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SHUFFLE NULL MODEL SUMMARY")
    print("=" * 80)
    print(f"{'Corpus':<30} {'Observed%':>10} {'Shuffled%':>10} {'Diff(pp)':>10} {'N_exc':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["observed_rate"] or -1, reverse=True):
        obs = f"{r['observed_rate']:.1f}" if r["observed_rate"] is not None else "---"
        shf = f"{r['shuffle_rate']:.1f}" if r["shuffle_rate"] is not None else "---"
        diff = f"{r['observed_minus_shuffle']:+.1f}" if r["observed_minus_shuffle"] is not None else "---"
        print(f"{r['corpus_id']:<30} {obs:>10} {shf:>10} {diff:>10} {r['observed_excursions']:>8}")

    print("\nKey question: Is observed natural recovery >> shuffled natural recovery?")
    print("If yes → recovery reflects structured temporal dynamics, not regression-to-mean.")
    print("If AI shows 0% under both observed AND shuffled → genuine structural absence.")


if __name__ == "__main__":
    main()
