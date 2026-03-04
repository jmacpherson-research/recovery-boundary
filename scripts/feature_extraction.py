#!/usr/bin/env python3
r"""
Round 1 Feature Extraction: Phases A-C for 17 new corpora + Phase D rebuild (52 total).
Session 94.

Auto-detects which corpora lack Phase A data (text_compression as proxy).
Processes one corpus at a time to handle large corpora (news_reuters = 117K segments).
Phase D rebuilds from scratch for ALL corpora (DROP + recreate).

Usage:
    cd C:\NDE_Research_Project\pipeline
    python ..\run_round1_extraction.py

    # Specific corpora only:
    python ..\run_round1_extraction.py --corpora gutenberg_fiction dailydialog

    # Skip Phase D:
    python ..\run_round1_extraction.py --skip-phase-d

    # Phase D only (skip A-C):
    python ..\run_round1_extraction.py --phase-d-only
"""

import sqlite3
import json
import math
import re
import gzip
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
CHUNK_TOKENS = 1200
SEED = 1337
NGRAM_N = 5
TOPK_TERMS = 80
N_BINS = 96
REFERENCE_CORPUS = 'law_of_one'
MIN_SEGS_FOR_REPAIR = 3

# Function words (104, from QUOCON S2.3)
FUNCTION_WORDS = {
    'the', 'of', 'to', 'and', 'in', 'a', 'is', 'that', 'it', 'as',
    'for', 'this', 'with', 'be', 'are', 'we', 'you', 'not', 'or', 'by',
    'from', 'an', 'at', 'which', 'will', 'may', 'shall', 'can', 'would',
    'there', 'been', 'has', 'have', 'was', 'were', 'if', 'then', 'so',
    'do', 'does', 'did', 'but', 'into', 'their', 'its', 'our', 'your',
    'these', 'those', 'such', 'than', 'what', 'when', 'where', 'how',
    'why', 'who', 'whom', 'because', 'therefore', 'thus', 'however',
    'only', 'also', 'most', 'any', 'each', 'all', 'one', 'no', 'nor',
    'upon', 'within', 'without', 'toward', 'towards', 'between', 'among',
    'again',
    'he', 'she', 'his', 'her', 'they', 'them', 'me', 'him', 'my',
    'us', 'up', 'out', 'had', 'more', 'about', 'on', 'i', 'some',
    'could', 'should', 'other', 'too', 'under',
}

STOPWORDS = {
    'the', 'of', 'to', 'and', 'in', 'a', 'is', 'that', 'it', 'as',
    'for', 'this', 'with', 'be', 'are', 'we', 'you', 'not', 'or', 'by',
    'from', 'an', 'at', 'which', 'on', 'i', 'but', 'have', 'has', 'was',
    'were', 'been', 'do', 'does', 'did', 'will',
}

# Incentive patterns (from RA12 S2.2)
INCENTIVE_COARSE = {
    "STATUS": [r"\bstatus\b", r"\bprestige\b", r"\bhonor\b", r"\besteem\b", r"\belite\b", r"\bsuperior\b", r"\bchosen\b", r"\bspecial\s+one\b"],
    "AUTHORITY_ENFORCE": [r"\byou\s+must\b", r"\byou\s+shall\b", r"\bobey\b", r"\bcommand\b", r"\bwithout\s+question\b", r"\bthe\s+truth\s+is\b"],
    "RECRUITMENT": [r"\bjoin\b", r"\bfollow\s+me\b", r"\bfollowers\b", r"\bconvert\b", r"\bspread\s+the\s+word\b", r"\bbelieve\s+in\s+me\b"],
    "URGENCY": [r"\burgent\b", r"\bimmediately\b", r"\bact\s+now\b", r"\btime\s+is\s+running\s+out\b", r"\bdeadline\b"],
    "FLATTERY": [r"\byou\s+are\s+special\b", r"\byou\s+are\s+chosen\b", r"\byou\s+are\s+advanced\b"],
    "THREAT_FEAR": [r"\bfear\b", r"\bterrify\b", r"\bdoom\b", r"\bpunish\b", r"\bwrath\b", r"\bhell\b", r"\bdamnation\b"],
    "GUILT_SHAME": [r"\bguilt\b", r"\bshame\b", r"\bashamed\b", r"\bsin\b", r"\bsinful\b", r"\bblame\b"],
    "SELF_JUSTIFICATION": [r"\bI\s+assure\s+you\b", r"\bI\s+promise\b", r"\btrust\s+me\b", r"\bbelieve\s+me\b", r"\bI\s+swear\b"],
}
INCENTIVE_STRICT = {
    "STATUS_LEVERAGE": [r"\byou\s+are\s+(special|chosen|elite|superior|advanced)\b", r"\bas\s+one\s+of\s+the\s+chosen\b", r"\bmost\s+excellent\b"],
    "AUTHORITY_ENFORCE": [r"\byou\s+must\b", r"\byou\s+shall\b", r"\bobey\b", r"\bwithout\s+question\b", r"\bbecause\s+I\s+say\s+so\b", r"\bthe\s+truth\s+is\s+that\s+you\s+must\b"],
    "RECRUITMENT": [r"\b(join|convert|recruit)\b.*\b(us|me|this)\b", r"\bfollow\s+(me|us)\b", r"\bspread\s+the\s+word\b", r"\baccept\s+this\b.*\bnow\b"],
    "URGENCY_PRESSURE": [r"\bact\s+now\b", r"\bwithout\s+delay\b", r"\bfinal\s+chance\b", r"\bnow\s+or\s+else\b"],
    "THREAT_FEAR": [r"\byou\s+will\s+(suffer|be\s+punished|pay)\b", r"\bor\s+else\b.*\byou\b", r"\bpunishment\b.*\byou\b"],
    "GUILT_SHAME": [r"\byou\s+should\s+be\s+ashamed\b", r"\byour\s+shame\b", r"\byour\s+guilt\b", r"\bsinful\b.*\byou\b"],
    "SELF_JUSTIFICATION": [r"\btrust\s+me\b", r"\bbelieve\s+me\b", r"\bI\s+assure\s+you\b", r"\bwe\s+assure\s+you\b"],
}
INCENTIVE_COARSE_COMPILED = {
    axis: [re.compile(p, re.IGNORECASE) for p in patterns]
    for axis, patterns in INCENTIVE_COARSE.items()
}
INCENTIVE_STRICT_COMPILED = {
    axis: [re.compile(p, re.IGNORECASE) for p in patterns]
    for axis, patterns in INCENTIVE_STRICT.items()
}

# Error patterns (Phase C)
ERROR_PATTERNS = {
    'CORRECTION': re.compile(r'\b(correction|correcting|correct this|let me correct|i misspoke)\b', re.IGNORECASE),
    'CLARIFICATION': re.compile(r'\b(clarif|to clarify|what i mean|in other words|to be more precise)\b', re.IGNORECASE),
    'REFUSAL': re.compile(r'\b(cannot|will not|unable to|not able to|we cannot|i cannot)\b', re.IGNORECASE),
    'DEPRIORITIZE': re.compile(r'\b(not important|unimportant|trivial|beside the point|irrelevant)\b', re.IGNORECASE),
}
INSTRUMENT_PATTERN = re.compile(r'\b(instrument|contact|session|transfer|the one known as)\b', re.IGNORECASE)
INSTRUMENT_CORPORA = {'ll_research', 'law_of_one'}

# Phase D
CALIBRATION = {'sat_3sat', 'ecc_hamming74', 'gencode_splice'}
INCENTIVE_THRESHOLD = 0.5
ORIGINAL_DISTANCES = {
    'synth_g0': 8.24, 'synth_g1': 6.87, 'synth_g2': 6.42,
    'synth_g3': 6.15, 'synth_g4': 6.08,
}

# All synthetic corpora (for construction resistance)
SYNTH_TARGETS = [
    'synth_g0', 'synth_g1', 'synth_g2', 'synth_g3', 'synth_g4',
    'edit_e1', 'edit_e2', 'edit_e3',
    'synth_gpt4o_g0', 'synth_llama70b_g0', 'synth_gemini_g0',
    'multiturn_claude', 'multiturn_gpt4o',
    'xdom_claude_scotus', 'xdom_claude_fomc', 'xdom_claude_therapy',
    'xdom_gpt4o_scotus', 'xdom_gpt4o_fomc', 'xdom_gpt4o_therapy',
    # New Wave 2 synthetics
    'synth_fiction_g0', 'synth_news_g0', 'synth_academic_g0', 'synth_dialogue_g0',
    'reasoning_deepseek_r1', 'reasoning_deepseek_r1_mt',
    'reasoning_o3mini', 'reasoning_o3mini_mt',
    'synth_claude_t02', 'synth_claude_t15',
    'synth_gpt4o_t02', 'synth_gpt4o_t15',
    'synth_twopass_claude',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def tokenize(text):
    if not text:
        return []
    return [w.lower() for w in WORD_RE.findall(text)]


def shannon_entropy(token_list):
    if not token_list:
        return 0.0
    counts = Counter(token_list)
    total = len(token_list)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def gini(arr):
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_arr) / (n * np.sum(sorted_arr))) - (n + 1) / n


def insert_batch(conn, table, rows):
    """Batch insert list of dicts into table."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ', '.join(['?' for _ in cols])
    col_names = ', '.join(cols)
    values = [[r[c] for c in cols] for r in rows]
    conn.executemany(f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})", values)
    conn.commit()
    return len(values)


def insert_tuples(conn, table, tuples):
    """Batch insert list of tuples into table."""
    if not tuples:
        return 0
    placeholders = ', '.join(['?' for _ in range(len(tuples[0]))])
    conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", tuples)
    conn.commit()
    return len(tuples)


def compute_percentiles(struct_features, corpus_id):
    """Compute word_count and punct_density percentiles for a corpus."""
    wcs = []
    pds = []
    for (cid, _), feat in struct_features.items():
        if cid == corpus_id:
            wcs.append(feat['word_count'])
            pds.append(feat['punct_density'])
    if not wcs:
        return None
    wcs = np.array(wcs)
    pds = np.array(pds)
    return {
        'wc_p75': float(np.percentile(wcs, 75)),
        'wc_p95': float(np.percentile(wcs, 95)),
        'wc_p99': float(np.percentile(wcs, 99)),
        'pd_p75': float(np.percentile(pds, 75)),
        'pd_p95': float(np.percentile(pds, 95)),
        'pd_p99': float(np.percentile(pds, 99)),
    }


# ---------------------------------------------------------------------------
# Data loading — one corpus at a time
# ---------------------------------------------------------------------------

def load_corpus_segments(corpus_id):
    """Fetch all segments for a single corpus, organized by session."""
    conn = get_db()
    rows = conn.execute("""
        SELECT corpus_id, session_id, segment_id, text, word_count, speaker_type, sequence_order
        FROM segments
        WHERE corpus_id = ?
        ORDER BY session_id, sequence_order
    """, (corpus_id,)).fetchall()
    conn.close()

    sessions = defaultdict(list)
    for row in rows:
        sessions[row['session_id']].append(dict(row))
    return sessions


def detect_unextracted_corpora():
    """Find corpora in segments table but not in text_compression."""
    conn = get_db()
    rows = conn.execute("""
        SELECT DISTINCT s.corpus_id
        FROM segments s
        WHERE s.corpus_id NOT IN (SELECT DISTINCT corpus_id FROM text_compression)
        ORDER BY s.corpus_id
    """).fetchall()
    conn.close()
    return [r['corpus_id'] for r in rows]


# ===========================================================================
# PHASE A: Core Text Analysis (per corpus)
# ===========================================================================

def run_phase_a_corpus(corpus_id, sessions):
    """Phase A for a single corpus. Returns row counts."""
    conn = get_db()
    counts = {}

    # --- A.1: Entropy ---
    entropy_rows = []
    for session_id, segments in sessions.items():
        combined = ' '.join([s['text'] for s in segments if s['text']])
        tokens = tokenize(combined)
        entropy_rows.append({
            'corpus_id': corpus_id, 'session_id': session_id,
            'entropy': shannon_entropy(tokens),
            'token_count': len(tokens), 'segment_count': len(segments),
        })
    counts['entropy'] = insert_batch(conn, 'text_entropy_session', entropy_rows)

    # --- A.2: Vocab growth ---
    vocab_rows = []
    sorted_sids = sorted(sessions.keys())
    prev_vocab = set()
    for session_id in sorted_sids:
        segs = sessions[session_id]
        combined = ' '.join([s['text'] for s in segs if s['text']])
        tokens = tokenize(combined)
        vocab = set(tokens)
        union = vocab | prev_vocab
        overlap = len(vocab & prev_vocab) / len(union) if union else 0.0
        vocab_rows.append({
            'corpus_id': corpus_id, 'session_id': session_id,
            'vocab_size': len(vocab), 'new_words': len(vocab - prev_vocab),
            'overlap_ratio': overlap,
        })
        prev_vocab = vocab
    counts['vocab'] = insert_batch(conn, 'text_vocab_growth', vocab_rows)

    # --- A.3: Structural features ---
    struct_rows = []
    for session_id, segments in sessions.items():
        for seg in segments:
            text = seg['text'] if seg['text'] else ''
            tokens = tokenize(text)
            wc = len(tokens)
            sc = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
            if sc == 0 and text.strip():
                sc = 1
            unique = len(set(tokens))
            ttr = unique / wc if wc > 0 else 0.0
            pd = sum(1 for c in text if c in '.!?,;:') / wc if wc > 0 else 0.0
            awl = sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
            struct_rows.append({
                'corpus_id': corpus_id, 'session_id': session_id,
                'segment_id': seg['segment_id'],
                'word_count': wc, 'sentence_count': sc, 'unique_tokens': unique,
                'ttr': ttr, 'punct_density': pd, 'avg_word_length': awl,
            })
    counts['struct'] = insert_batch(conn, 'text_struct_features', struct_rows)

    # --- A.4: Compression ---
    comp_rows = []
    for session_id, segments in sessions.items():
        combined = ' '.join([s['text'] for s in segments if s['text']])
        tokens = tokenize(combined)
        if not tokens:
            comp_rows.append({
                'corpus_id': corpus_id, 'session_id': session_id,
                'original_bytes': 0, 'compressed_bytes': 0,
                'compression_ratio': 0.0, 'chunk_count': 0,
            })
            continue
        if len(tokens) < CHUNK_TOKENS:
            text_bytes = combined.encode('utf-8')
            compressed = gzip.compress(text_bytes, compresslevel=9)
            comp_rows.append({
                'corpus_id': corpus_id, 'session_id': session_id,
                'original_bytes': len(text_bytes),
                'compressed_bytes': len(compressed),
                'compression_ratio': len(compressed) / len(text_bytes) if text_bytes else 0.0,
                'chunk_count': 1,
            })
        else:
            ratios = []
            total_orig = total_comp = 0
            for i in range(0, len(tokens), CHUNK_TOKENS):
                chunk = ' '.join(tokens[i:i + CHUNK_TOKENS])
                chunk_bytes = chunk.encode('utf-8')
                compressed = gzip.compress(chunk_bytes, compresslevel=9)
                total_orig += len(chunk_bytes)
                total_comp += len(compressed)
                ratios.append(len(compressed) / len(chunk_bytes))
            comp_rows.append({
                'corpus_id': corpus_id, 'session_id': session_id,
                'original_bytes': total_orig, 'compressed_bytes': total_comp,
                'compression_ratio': sum(ratios) / len(ratios),
                'chunk_count': len(ratios),
            })
    counts['compression'] = insert_batch(conn, 'text_compression', comp_rows)

    # --- A.5: Function word profiles ---
    fw_rows = []
    for session_id, segments in sessions.items():
        combined = ' '.join([s['text'] for s in segments if s['text']])
        tokens = tokenize(combined)
        total = len(tokens)
        fw_counts = {}
        for fw in sorted(FUNCTION_WORDS):
            c = tokens.count(fw)
            fw_counts[fw] = (c / total * 1000) if total > 0 else 0.0
        fw_rows.append({
            'corpus_id': corpus_id, 'session_id': session_id,
            'funcword_counts_json': json.dumps(fw_counts),
            'total_tokens': total,
        })
    counts['funcword'] = insert_batch(conn, 'text_funcword_profile', fw_rows)

    # --- A.6: Incentive scan ---
    coarse_rows = []
    strict_rows = []
    for session_id, segments in sessions.items():
        combined = ' '.join([s['text'] for s in segments if s['text']])
        tokens = tokenize(combined)
        tc = len(tokens)
        for axis, patterns in INCENTIVE_COARSE_COMPILED.items():
            hits = sum(len(p.findall(combined)) for p in patterns)
            coarse_rows.append({
                'corpus_id': corpus_id, 'session_id': session_id,
                'axis': axis, 'hit_count': hits, 'token_count': tc,
                'rate_per_1k': (hits / tc * 1000) if tc > 0 else 0.0,
            })
        for axis, patterns in INCENTIVE_STRICT_COMPILED.items():
            hits = sum(len(p.findall(combined)) for p in patterns)
            strict_rows.append({
                'corpus_id': corpus_id, 'session_id': session_id,
                'axis': axis, 'hit_count': hits, 'token_count': tc,
                'rate_per_10k': (hits / tc * 10000) if tc > 0 else 0.0,
            })
    counts['incentive_coarse'] = insert_batch(conn, 'text_incentive_coarse', coarse_rows)
    counts['incentive_strict'] = insert_batch(conn, 'text_incentive_strict', strict_rows)

    conn.close()
    return counts


# ===========================================================================
# PHASE B: Temporal & Sequential Metrics (per corpus)
# ===========================================================================

def run_phase_b_corpus(corpus_id, sessions):
    """Phase B for a single corpus. Returns row counts."""
    conn = get_db()
    counts = {}

    # Build session-level token lists
    sorted_sids = sorted(sessions.keys())
    session_list = []
    for sid in sorted_sids:
        segs = sessions[sid]
        combined = ' '.join([s['text'] for s in segs if s['text']])
        tokens = tokenize(combined)
        session_list.append((sid, tokens))

    # --- B.1: Phrase persistence (4-gram) ---
    n_sess = len(session_list)
    phrase_rows = []
    if n_sess >= 2:
        first_seen = {}
        last_seen = {}
        for idx, (sid, tokens) in enumerate(session_list):
            if len(tokens) < 4:
                continue
            seen = set()
            for i in range(len(tokens) - 3):
                seen.add(tuple(tokens[i:i+4]))
            for gram in seen:
                if gram not in first_seen:
                    first_seen[gram] = idx
                last_seen[gram] = idx
        threshold = max(1, int(n_sess * 0.10))
        total = len(first_seen)
        persistent = 0
        max_span = 0
        span_sum = 0
        for gram in first_seen:
            span = last_seen[gram] - first_seen[gram] + 1
            span_sum += span
            if span >= threshold:
                persistent += 1
            if span > max_span:
                max_span = span
        mean_span = span_sum / total if total > 0 else 0.0
        phrase_rows.append({
            'corpus_id': corpus_id, 'n_sessions': n_sess,
            'total_4grams': total, 'persistent_count': persistent,
            'persistence_threshold': threshold, 'max_span': max_span,
            'mean_span': round(mean_span, 4),
            'persistent_fraction': round(persistent / total, 6) if total > 0 else 0.0,
        })
    counts['phrase'] = insert_batch(conn, 'text_phrase_lifespan', phrase_rows)

    # --- B.2: Zipf + Heaps ---
    zh_rows = []
    all_tokens = []
    for sid, tokens in session_list:
        all_tokens.extend(tokens)
    if len(all_tokens) >= 100:
        freq_counts = Counter(all_tokens)
        freqs = sorted(freq_counts.values(), reverse=True)
        max_rank = min(5000, len(freqs))
        if max_rank < 20:
            zipf_alpha = zipf_r2 = 0.0
        else:
            log_ranks = np.log10(np.arange(10, max_rank + 1))
            log_freqs = np.log10(np.array(freqs[9:max_rank]))
            coeffs = np.polyfit(log_ranks, log_freqs, 1)
            zipf_alpha = -coeffs[0]
            predicted = np.polyval(coeffs, log_ranks)
            ss_res = np.sum((log_freqs - predicted) ** 2)
            ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
            zipf_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n_total = len(all_tokens)
        n_samples = min(200, n_total)
        sample_points = np.linspace(100, n_total, n_samples, dtype=int)
        vocab_sizes = []
        running_vocab = set()
        token_idx = 0
        for target_n in sample_points:
            while token_idx < target_n:
                running_vocab.add(all_tokens[token_idx])
                token_idx += 1
            vocab_sizes.append(len(running_vocab))
        log_n = np.log10(sample_points.astype(float))
        log_v = np.log10(np.array(vocab_sizes, dtype=float))
        coeffs_h = np.polyfit(log_n, log_v, 1)
        heaps_beta = coeffs_h[0]
        predicted_h = np.polyval(coeffs_h, log_n)
        ss_res_h = np.sum((log_v - predicted_h) ** 2)
        ss_tot_h = np.sum((log_v - np.mean(log_v)) ** 2)
        heaps_r2 = 1 - ss_res_h / ss_tot_h if ss_tot_h > 0 else 0.0
        zh_rows.append({
            'corpus_id': corpus_id, 'total_tokens': n_total,
            'vocab_size': len(freq_counts),
            'zipf_alpha': round(zipf_alpha, 4), 'zipf_r2': round(zipf_r2, 4),
            'heaps_beta': round(heaps_beta, 4), 'heaps_r2': round(heaps_r2, 4),
        })
    counts['zipf_heaps'] = insert_batch(conn, 'text_zipf_heaps', zh_rows)

    # --- B.3 + B.4: Novelty Economics ---
    inj_rows = []
    integ_rows = []
    if n_sess >= 4:
        n_bins = min(N_BINS, n_sess)
        bin_size = n_sess / n_bins
        bins = []
        for b in range(n_bins):
            start = int(b * bin_size)
            end = int((b + 1) * bin_size)
            combined_tokens = []
            for sid, tokens in session_list[start:end]:
                combined_tokens.extend(tokens)
            bins.append((b, combined_tokens))

        # Channel 1: 5-gram novelty
        seen_5grams = set()
        new_5gram_counts = []
        fivegram_bins = defaultdict(list)
        for bin_idx, tokens in bins:
            if len(tokens) < NGRAM_N:
                new_5gram_counts.append(0)
                continue
            bin_5grams = set()
            for i in range(len(tokens) - NGRAM_N + 1):
                gram = tuple(tokens[i:i+NGRAM_N])
                bin_5grams.add(gram)
                fivegram_bins[gram].append(bin_idx)
            new_5gram_counts.append(len(bin_5grams - seen_5grams))
            seen_5grams |= bin_5grams

        # Channel 2: TF-IDF salience
        bin_term_freqs = []
        doc_freq = Counter()
        for bin_idx, tokens in bins:
            content = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
            tf = Counter(content)
            bin_term_freqs.append(tf)
            for term in set(content):
                doc_freq[term] += 1

        seen_top = set()
        new_term_counts = []
        term_bins = defaultdict(list)
        for bin_idx, (_, tokens) in enumerate(bins):
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
            for t in top_terms:
                term_bins[t].append(bin_idx)
            new_term_counts.append(len(top_terms - seen_top))
            seen_top |= top_terms

        arr_5g = np.array(new_5gram_counts, dtype=float)
        arr_tf = np.array(new_term_counts, dtype=float)
        mean_5g = np.mean(arr_5g) if len(arr_5g) > 0 else 0
        cv_5g = np.std(arr_5g) / mean_5g if mean_5g > 0 else 0
        mean_tf = np.mean(arr_tf) if len(arr_tf) > 0 else 0
        cv_tf = np.std(arr_tf) / mean_tf if mean_tf > 0 else 0

        for channel, arr, mean_v, cv_v in [('5gram', arr_5g, mean_5g, cv_5g), ('tfidf', arr_tf, mean_tf, cv_tf)]:
            inj_rows.append({
                'corpus_id': corpus_id, 'n_bins': n_bins, 'channel': channel,
                'total_new': int(np.sum(arr)), 'mean_new_per_bin': round(float(mean_v), 2),
                'cv': round(float(cv_v), 4), 'gini': round(float(gini(arr)), 4),
            })

        # Integration
        for channel, item_bins in [('5gram', fivegram_bins), ('tfidf', term_bins)]:
            oneoff = sum(1 for bins_list in item_bins.values() if len(set(bins_list)) == 1)
            total_novel = len(item_bins)
            reappear = [len(set(bl)) for bl in item_bins.values()]
            integ_rows.append({
                'corpus_id': corpus_id, 'channel': channel,
                'total_novel': total_novel,
                'oneoff_count': oneoff,
                'oneoff_fraction': round(oneoff / total_novel, 4) if total_novel > 0 else 0.0,
                'median_reappear': int(np.median(reappear)) if reappear else 0,
                'mean_reappear': round(float(np.mean(reappear)), 4) if reappear else 0.0,
            })

    counts['novelty_inj'] = insert_batch(conn, 'text_novelty_injection', inj_rows)
    counts['novelty_int'] = insert_batch(conn, 'text_novelty_integration', integ_rows)

    # --- B.5: Mutual Information ---
    mi_rows = []
    if n_sess >= 3:
        session_vocabs = []
        for sid, tokens in session_list:
            total_tok = len(tokens)
            dist = {w: c / total_tok for w, c in Counter(tokens).items()} if total_tok > 0 else {}
            session_vocabs.append(dist)
        for lag in range(1, min(11, n_sess)):
            mi_vals = []
            for i in range(n_sess - lag):
                da = session_vocabs[i]
                db = session_vocabs[i + lag]
                shared = set(da.keys()) & set(db.keys())
                if not shared:
                    mi_vals.append(0.0)
                    continue
                h_b = -sum(p * math.log2(p) for p in db.values() if p > 0)
                shared_mass = sum(db.get(w, 0) for w in shared)
                if shared_mass <= 0:
                    mi_vals.append(0.0)
                    continue
                h_b_a = -sum((db[w]/shared_mass) * math.log2(db[w]/shared_mass) for w in shared if db.get(w, 0) > 0)
                mi_vals.append(max(0.0, h_b - h_b_a))
            mi_rows.append({
                'corpus_id': corpus_id, 'lag': lag,
                'mean_mi': round(float(np.mean(mi_vals)), 4) if mi_vals else 0.0,
                'sd_mi': round(float(np.std(mi_vals)), 4) if mi_vals else 0.0,
                'n_pairs': len(mi_vals),
            })
    counts['mutual_info'] = insert_batch(conn, 'text_mutual_info', mi_rows)

    # --- B.6: Boundary effects ---
    boundary_rows = []
    tercile_metrics = {'first': [], 'middle': [], 'last': []}
    for session_id, segs in sessions.items():
        if len(segs) < 6:
            continue
        n = len(segs)
        t1 = n // 3
        t2 = 2 * n // 3
        for label, seg_slice in [('first', segs[:t1]), ('middle', segs[t1:t2]), ('last', segs[t2:])]:
            combined = ' '.join([s['text'] for s in seg_slice if s['text']])
            tokens = tokenize(combined)
            if not tokens:
                continue
            unique = len(set(tokens))
            ttr_val = unique / len(tokens)
            counts_dist = Counter(tokens)
            total_tok = len(tokens)
            entropy_val = -sum((c/total_tok) * math.log2(c/total_tok) for c in counts_dist.values())
            tercile_metrics[label].append({'entropy': entropy_val, 'ttr': ttr_val, 'tokens': total_tok})
    for label in ['first', 'middle', 'last']:
        vals = tercile_metrics[label]
        if not vals:
            continue
        boundary_rows.append({
            'corpus_id': corpus_id, 'tercile': label,
            'mean_entropy': round(float(np.mean([v['entropy'] for v in vals])), 4),
            'sd_entropy': round(float(np.std([v['entropy'] for v in vals])), 4),
            'mean_ttr': round(float(np.mean([v['ttr'] for v in vals])), 4),
            'sd_ttr': round(float(np.std([v['ttr'] for v in vals])), 4),
            'mean_tokens': round(float(np.mean([v['tokens'] for v in vals])), 1),
            'n_sessions': len(vals),
        })
    counts['boundary'] = insert_batch(conn, 'text_boundary_effects', boundary_rows)

    conn.close()
    return counts


# ===========================================================================
# PHASE C: Error/Recovery/Stress (per corpus) — PROPER recovery logic
# ===========================================================================

def run_phase_c_corpus(corpus_id, sessions):
    """Phase C for a single corpus. Uses full recovery lookahead logic."""
    conn = get_db()
    counts = {}

    # Load struct features for this corpus (from Phase A — just inserted)
    struct_features = {}
    rows = conn.execute("""
        SELECT corpus_id, session_id, segment_id, word_count, punct_density
        FROM text_struct_features WHERE corpus_id = ?
    """, (corpus_id,)).fetchall()
    for row in rows:
        struct_features[(row['corpus_id'], row['segment_id'])] = {
            'word_count': row['word_count'], 'punct_density': row['punct_density'],
        }

    # --- C.1: Error events ---
    segment_events = []
    session_events = []
    seg_has_error = defaultdict(int)

    for session_id, segs in sessions.items():
        session_total = 0
        session_tokens = 0
        for seg in segs:
            text = seg['text'] or ''
            tokens = tokenize(text)
            session_tokens += len(tokens)
            for cat, pat in ERROR_PATTERNS.items():
                hits = len(pat.findall(text))
                if hits > 0:
                    segment_events.append((corpus_id, session_id, seg['segment_id'], cat, hits))
                    session_total += hits
                    seg_has_error[(corpus_id, session_id, seg['segment_id'])] += hits
            # INSTRUMENT_SESSION for channeling corpora
            if corpus_id in INSTRUMENT_CORPORA:
                inst_hits = len(INSTRUMENT_PATTERN.findall(text))
                if inst_hits > 0:
                    segment_events.append((corpus_id, session_id, seg['segment_id'], 'INSTRUMENT_SESSION', inst_hits))
                    session_total += inst_hits
                    # Note: INSTRUMENT_SESSION excluded from seg_has_error for repair/recovery
                    # (domain vocab, not real errors — bug fix from Session 47)
        rate = (session_total / session_tokens * 1000) if session_tokens > 0 else 0.0
        session_events.append((corpus_id, session_id, session_total, rate))

    counts['error_seg'] = insert_tuples(conn, 'text_error_events', segment_events)
    counts['error_sess'] = insert_tuples(conn, 'text_error_session', session_events)

    # --- C.2: Repair latency ---
    # Check median segs/session
    seg_counts = [len(segs) for segs in sessions.values()]
    median_segs = float(np.median(seg_counts)) if seg_counts else 0

    repair_rows = []
    repair_summary = []

    if median_segs < MIN_SEGS_FOR_REPAIR:
        # Structurally undefined — insert NaN summary
        repair_summary.append((corpus_id, None, None, 0.0))
    else:
        # Find segments with errors (excluding INSTRUMENT_SESSION)
        error_segs = set()
        for (cid, sid, segid), hit_count in seg_has_error.items():
            if hit_count > 0:
                error_segs.add((cid, sid, segid))

        pctiles = compute_percentiles(struct_features, corpus_id)
        repair_lags = []
        if pctiles:
            for session_id, segs in sessions.items():
                for i, seg in enumerate(segs):
                    if (corpus_id, session_id, seg['segment_id']) not in error_segs:
                        continue
                    # Look ahead for clean segment
                    found_clean = False
                    max_lookahead = 10
                    for j in range(i + 1, min(i + 1 + max_lookahead, len(segs))):
                        future_seg = segs[j]
                        fut_has_err = seg_has_error.get((corpus_id, session_id, future_seg['segment_id']), 0) > 0
                        if fut_has_err:
                            continue
                        fut_feat = struct_features.get((corpus_id, future_seg['segment_id']))
                        if fut_feat:
                            if fut_feat['word_count'] > pctiles['wc_p75']:
                                continue
                            if fut_feat['punct_density'] > pctiles['pd_p75']:
                                continue
                        lag = j - i
                        repair_rows.append((corpus_id, session_id, seg['segment_id'], lag, 1))
                        repair_lags.append(lag)
                        found_clean = True
                        break
                    if not found_clean:
                        repair_rows.append((corpus_id, session_id, seg['segment_id'], -1, 0))
                        repair_lags.append(None)

        valid_lags = [l for l in repair_lags if l is not None]
        n_total_rep = len(repair_lags)
        if valid_lags:
            median_lag = float(np.median(valid_lags))
        else:
            median_lag = None
        p90_lag = float(np.percentile(valid_lags, 90)) if valid_lags else None
        unrepaired_frac = (n_total_rep - len(valid_lags)) / n_total_rep if n_total_rep > 0 else 0.0
        repair_summary.append((corpus_id, median_lag, p90_lag, unrepaired_frac))

    counts['repair_rows'] = insert_tuples(conn, 'text_repair_latency', repair_rows)
    counts['repair_summary'] = insert_tuples(conn, 'text_repair_summary', repair_summary)

    # --- C.3: Stress & recovery ---
    recovery_rows = []
    n_exc = 0
    n_rec = 0
    lags = []

    pctiles = compute_percentiles(struct_features, corpus_id)
    if pctiles:
        for session_id, segs in sessions.items():
            for i, seg in enumerate(segs):
                feat = struct_features.get((corpus_id, seg['segment_id']))
                if feat is None:
                    continue
                has_err = seg_has_error.get((corpus_id, session_id, seg['segment_id']), 0) > 0

                # Excursion check
                is_excursion = False
                if feat['word_count'] > pctiles['wc_p99'] or feat['punct_density'] > pctiles['pd_p99']:
                    is_excursion = True
                elif has_err and (feat['word_count'] > pctiles['wc_p95'] or feat['punct_density'] > pctiles['pd_p95']):
                    is_excursion = True
                if not is_excursion:
                    continue

                n_exc += 1

                # Look ahead for recovery (max 5 segments)
                found_clean = False
                max_lookahead = 5
                for j in range(i + 1, min(i + 1 + max_lookahead, len(segs))):
                    future_seg = segs[j]
                    fut_has_err = seg_has_error.get((corpus_id, session_id, future_seg['segment_id']), 0) > 0
                    if fut_has_err:
                        continue
                    fut_feat = struct_features.get((corpus_id, future_seg['segment_id']))
                    if fut_feat:
                        if fut_feat['word_count'] > pctiles['wc_p75']:
                            continue
                        if fut_feat['punct_density'] > pctiles['pd_p75']:
                            continue
                    lag = j - i
                    recovery_rows.append((corpus_id, session_id, seg['segment_id'], lag, 1))
                    n_rec += 1
                    lags.append(lag)
                    found_clean = True
                    break

                if not found_clean:
                    recovery_rows.append((corpus_id, session_id, seg['segment_id'], -1, 0))

    rate = n_rec / n_exc if n_exc > 0 else 0.0
    med_lag = float(np.median(lags)) if lags else -1.0
    stress_summary = [(corpus_id, n_exc, rate, med_lag)]

    counts['recovery_rows'] = insert_tuples(conn, 'text_stress_recovery', recovery_rows)
    counts['stress_summary'] = insert_tuples(conn, 'text_stress_summary', stress_summary)

    conn.close()
    return counts, {'n_excursions': n_exc, 'recovery_rate': rate, 'median_lag': med_lag}


# ===========================================================================
# PHASE D: Compact Vector + Distances + PCA + Regime (ALL corpora)
# ===========================================================================

def run_phase_d():
    """Phase D: DROP+recreate for all corpora. Reads from Phase A-C tables."""
    print(f"\n{'='*60}")
    print(f"PHASE D: Compact Vector Assembly (ALL corpora)")
    print(f"{'='*60}")

    conn = get_db()

    # D.1: Assemble vectors
    print("  D.1: Assembling compact vectors...")

    # Dim 1: strict incentive per 10k
    dim1 = {}
    for r in conn.execute("SELECT corpus_id, SUM(hit_count) as h, SUM(token_count) as t FROM text_incentive_strict GROUP BY corpus_id").fetchall():
        dim1[r['corpus_id']] = (r['h'] / r['t'] * 10000) if r['t'] > 0 else 0.0

    # Dim 2: mean stress per 1k
    dim2 = {}
    for r in conn.execute("SELECT corpus_id, AVG(events_per_1k_tokens) as v FROM text_error_session GROUP BY corpus_id").fetchall():
        dim2[r['corpus_id']] = r['v']

    # Dim 3: median clean lag (from repair summary — NOT stress summary)
    # Corpora with <3 segs/session are absent from this table → NaN → structural separator
    dim3 = {}
    for r in conn.execute("SELECT corpus_id, median_lag FROM text_repair_summary").fetchall():
        dim3[r['corpus_id']] = float(r['median_lag']) if r['median_lag'] is not None else float('nan')

    # Dim 4: mean compression ratio
    dim4 = {}
    for r in conn.execute("SELECT corpus_id, AVG(compression_ratio) as v FROM text_compression GROUP BY corpus_id").fetchall():
        dim4[r['corpus_id']] = r['v']

    # Dim 5: SD compression ratio
    dim5 = {}
    corpus_ratios = {}
    for r in conn.execute("SELECT corpus_id, compression_ratio FROM text_compression").fetchall():
        corpus_ratios.setdefault(r['corpus_id'], []).append(r['compression_ratio'])
    for cid, vals in corpus_ratios.items():
        dim5[cid] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Dim 6: novelty CV (5gram) — use budgeted table if available
    dim6 = {}
    dim7 = {}
    has_budgeted = False
    try:
        budgeted = conn.execute("SELECT corpus_id, median_cv_5gram, median_gini_5gram FROM novelty_budgeted").fetchall()
        if budgeted:
            has_budgeted = True
            for r in budgeted:
                dim6[r['corpus_id']] = r['median_cv_5gram']
                dim7[r['corpus_id']] = r['median_gini_5gram']
            print("    Dim 6+7: using BUDGETED novelty (novelty_budgeted table)")
    except Exception:
        pass

    if not has_budgeted:
        # Fallback to raw novelty injection
        for r in conn.execute("SELECT corpus_id, cv FROM text_novelty_injection WHERE channel='5gram'").fetchall():
            dim6[r['corpus_id']] = r['cv']
        for r in conn.execute("SELECT corpus_id, gini FROM text_novelty_injection WHERE channel='5gram'").fetchall():
            dim7[r['corpus_id']] = r['gini']
        print("    Dim 6+7: using RAW novelty (novelty_budgeted table not found)")

    # Dim 8: oneoff tfidf
    dim8 = {}
    for r in conn.execute("SELECT corpus_id, oneoff_fraction FROM text_novelty_integration WHERE channel='tfidf'").fetchall():
        dim8[r['corpus_id']] = r['oneoff_fraction']

    # Dim 9: DEAD (oneoff 5gram — size artifact, killed S67). Placeholder NaN.
    # Dim 10: DEAD (glossary drift — size artifact, killed S61). Placeholder NaN.
    print("    Dim 9 + Dim 10: DEAD — NaN placeholders")

    conn.close()

    # Build vectors (8 active dims + 2 NaN placeholders)
    all_dims = [dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8]
    all_corpora = set()
    for d in all_dims:
        all_corpora.update(d.keys())

    vectors = {}
    for cid in sorted(all_corpora):
        vec = [d.get(cid, float('nan')) for d in all_dims]
        vec.append(float('nan'))  # dim9 placeholder
        vec.append(float('nan'))  # dim10 placeholder
        vectors[cid] = vec

    n_active = 8
    print(f"    {len(vectors)} corpora, {n_active} active dimensions")

    # Z-score (on 8 active dims only)
    print("  Z-scoring...")
    dim_values = [[] for _ in range(n_active)]
    for cid, vec in vectors.items():
        for i in range(n_active):
            if not np.isnan(vec[i]):
                dim_values[i].append(vec[i])
    dim_stats = []
    for i in range(n_active):
        vals = dim_values[i]
        if len(vals) > 1:
            m, s = np.mean(vals), np.std(vals, ddof=1)
            if s == 0:
                s = 1.0
        else:
            m, s = 0.0, 1.0
        dim_stats.append((m, s))

    z_vectors = {}
    for cid, vec in vectors.items():
        z = []
        for i in range(n_active):
            v = vec[i]
            z.append(float('nan') if np.isnan(v) else (v - dim_stats[i][0]) / dim_stats[i][1])
        z.append(float('nan'))  # z_dim9 placeholder
        z.append(float('nan'))  # z_dim10 placeholder
        z_vectors[cid] = z

    # D.2: Distances (8 active dims)
    print("  D.2: Computing distances...")
    corpora = sorted(z_vectors.keys())
    distances = []
    for i, ca in enumerate(corpora):
        for j, cb in enumerate(corpora):
            if j <= i:
                continue
            va = z_vectors[ca][:n_active]
            vb = z_vectors[cb][:n_active]
            valid = [(va[k], vb[k]) for k in range(n_active) if not np.isnan(va[k]) and not np.isnan(vb[k])]
            if valid:
                diff = np.array([a - b for a, b in valid])
                raw_l2 = np.sqrt(np.sum(diff**2))
                scale = np.sqrt(n_active / len(valid))
                l2 = float(raw_l2 * scale)
            else:
                l2 = float('nan')
            distances.append((ca, cb, l2))
            distances.append((cb, ca, l2))
    print(f"    {len(distances)} distance pairs")

    # D.2b: PCA
    print("  D.2b: PCA projection...")
    matrix = []
    for cid in corpora:
        row = z_vectors[cid][:n_active]
        row = [0.0 if np.isnan(v) else v for v in row]
        matrix.append(row)
    X = np.array(matrix)
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    pc1 = X_c @ Vt[0]
    pc2 = X_c @ Vt[1]
    var_exp = S**2 / np.sum(S**2)
    print(f"    PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%")
    pca_rows = [(cid, float(pc1[i]), float(pc2[i])) for i, cid in enumerate(corpora)]

    # D.3: Regime classification
    print("  D.3: Regime classification...")
    non_cal = {c: v for c, v in vectors.items() if c not in CALIBRATION}
    drift_dim = 4  # using dim5 (compression SD) as drift proxy since dim10 is dead
    oneoff_vals = [v[7] for v in non_cal.values() if not np.isnan(v[7])]
    recovery_vals = [v[2] for v in non_cal.values() if not np.isnan(v[2])]
    drift_vals = [v[drift_dim] for v in non_cal.values() if not np.isnan(v[drift_dim])]
    oneoff_median = float(np.median(oneoff_vals)) if oneoff_vals else 0.7
    recovery_median = float(np.median(recovery_vals)) if recovery_vals else 2.0
    drift_median = float(np.median(drift_vals)) if drift_vals else 0.05
    print(f"    Thresholds: oneoff={oneoff_median:.3f}, recovery={recovery_median:.3f}, drift={drift_median:.4f}")

    regime_rows = []
    for cid, vec in vectors.items():
        if cid in CALIBRATION:
            regime_rows.append((cid, vec[0], vec[7], vec[2], vec[4], 'R0', 'Calibration corpus'))
            continue
        incentive = vec[0]
        oneoff = vec[7]
        recovery = vec[2]
        drift = vec[drift_dim]
        inc_o = 'high' if (not np.isnan(incentive) and incentive > INCENTIVE_THRESHOLD) else 'low'
        int_o = 'deep' if (not np.isnan(oneoff) and oneoff < oneoff_median) else ('unknown' if np.isnan(oneoff) else 'shallow')
        rec_o = 'fast' if (not np.isnan(recovery) and recovery <= recovery_median) else ('unknown' if np.isnan(recovery) else 'slow')
        dri_o = 'low' if (not np.isnan(drift) and drift < drift_median) else ('unknown' if np.isnan(drift) else 'high')

        if inc_o == 'high':
            r2c = sum(1 for x in [int_o, rec_o, dri_o] if x in ('deep', 'fast', 'low'))
            regime = 'F1' if r2c >= 2 else 'R1'
            notes = f"{'High incentive + R2-like' if r2c >= 2 else 'High incentive'}; int={int_o} rec={rec_o} drift={dri_o}"
        else:
            r2c = sum(1 for x in [int_o, rec_o, dri_o] if x in ('deep', 'fast', 'low'))
            unk = sum(1 for x in [int_o, rec_o, dri_o] if x == 'unknown')
            if r2c >= 2:
                regime = 'R2'
            elif r2c + unk >= 2:
                regime = 'R2?'
            else:
                regime = 'R1'
            notes = f"Low incentive, {r2c}/3 R2 criteria; int={int_o} rec={rec_o} drift={dri_o}"
        regime_rows.append((cid, incentive, oneoff, recovery, drift, regime, notes))

    # Print regime summary
    rc = defaultdict(list)
    for r in regime_rows:
        rc[r[5]].append(r[0])
    for regime in sorted(rc):
        print(f"    {regime}: {', '.join(sorted(rc[regime]))}")

    # D.4: Construction resistance
    print("  D.4: Construction resistance...")
    dist_lookup = {(ca, cb): l2 for ca, cb, l2 in distances}
    cr_rows = []
    for cid in SYNTH_TARGETS:
        if cid not in vectors:
            continue
        d = dist_lookup.get((cid, REFERENCE_CORPUS), dist_lookup.get((REFERENCE_CORPUS, cid), float('nan')))
        orig = ORIGINAL_DISTANCES.get(cid, float('nan'))
        delta = d - orig if not (np.isnan(d) or np.isnan(orig)) else float('nan')
        cr_rows.append((cid, d, orig, delta))
        if not np.isnan(d):
            print(f"    {cid:30s}: {d:.3f} zL2")

    # Write Phase D
    print("\n  Writing Phase D to database...")
    conn = get_db()
    conn.execute("DROP TABLE IF EXISTS compact_vector")
    conn.execute("DROP TABLE IF EXISTS compact_distances")
    conn.execute("DROP TABLE IF EXISTS compact_pca")
    conn.execute("DROP TABLE IF EXISTS regime_classification")
    conn.execute("DROP TABLE IF EXISTS construction_resistance")

    conn.execute("""CREATE TABLE compact_vector (
        corpus_id TEXT PRIMARY KEY,
        dim1 REAL, dim2 REAL, dim3 REAL, dim4 REAL, dim5 REAL,
        dim6 REAL, dim7 REAL, dim8 REAL, dim9 REAL, dim10 REAL,
        z_dim1 REAL, z_dim2 REAL, z_dim3 REAL, z_dim4 REAL, z_dim5 REAL,
        z_dim6 REAL, z_dim7 REAL, z_dim8 REAL, z_dim9 REAL, z_dim10 REAL
    )""")
    conn.execute("""CREATE TABLE compact_distances (
        corpus_a TEXT NOT NULL, corpus_b TEXT NOT NULL, l2_distance REAL,
        PRIMARY KEY (corpus_a, corpus_b)
    )""")
    conn.execute("""CREATE TABLE compact_pca (
        corpus_id TEXT PRIMARY KEY, pc1 REAL NOT NULL, pc2 REAL NOT NULL
    )""")
    conn.execute("""CREATE TABLE regime_classification (
        corpus_id TEXT PRIMARY KEY,
        incentive_order REAL, integration_order REAL,
        recovery_order REAL, drift_order REAL,
        regime TEXT NOT NULL, notes TEXT
    )""")
    conn.execute("""CREATE TABLE construction_resistance (
        corpus_id TEXT PRIMARY KEY,
        distance_to_law_of_one REAL, original_distance REAL, delta REAL
    )""")
    conn.commit()

    # Insert
    cv_rows = []
    for cid in sorted(vectors.keys()):
        raw = vectors[cid]
        z = z_vectors[cid]
        cv_rows.append((cid, *raw, *z))
    conn.executemany("INSERT INTO compact_vector VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cv_rows)
    conn.executemany("INSERT INTO compact_distances VALUES (?,?,?)", distances)
    conn.executemany("INSERT INTO compact_pca VALUES (?,?,?)", pca_rows)
    conn.executemany("INSERT INTO regime_classification VALUES (?,?,?,?,?,?,?)", regime_rows)
    conn.executemany("INSERT INTO construction_resistance VALUES (?,?,?,?)", cr_rows)
    conn.commit()
    conn.close()

    print(f"  Phase D complete: {len(vectors)} corpora in compact vector")

    return {
        'n_corpora': len(vectors),
        'n_dims': n_active,
        'pca_variance': {'pc1': float(var_exp[0]), 'pc2': float(var_exp[1])},
        'dim_stats': [{'mean': float(m), 'std': float(s)} for m, s in dim_stats],
        'tables': {
            'compact_vector': len(cv_rows),
            'compact_distances': len(distances),
            'compact_pca': len(pca_rows),
            'regime_classification': len(regime_rows),
            'construction_resistance': len(cr_rows),
        }
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Round 1 Feature Extraction (Session 94)')
    parser.add_argument('--corpora', nargs='+', default=None,
                       help='Specific corpus_ids to process (auto-detect if not given)')
    parser.add_argument('--skip-phase-d', action='store_true',
                       help='Skip Phase D rebuild')
    parser.add_argument('--phase-d-only', action='store_true',
                       help='Skip Phases A-C, only rebuild Phase D')
    args = parser.parse_args()

    start = datetime.now()
    print(f"Round 1 Feature Extraction — Session 94")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print()

    # Determine target corpora
    if args.phase_d_only:
        target = []
        print("Phase D only mode — skipping A-C")
    elif args.corpora:
        target = args.corpora
        print(f"Manual target: {', '.join(target)}")
    else:
        target = detect_unextracted_corpora()
        print(f"Auto-detected {len(target)} unextracted corpora:")
        for c in target:
            print(f"  - {c}")

    # Phases A-C: one corpus at a time
    if target:
        print(f"\n{'='*60}")
        print(f"PHASES A-C: Processing {len(target)} corpora")
        print(f"{'='*60}")

        results = {}
        for idx, corpus_id in enumerate(target, 1):
            corpus_start = datetime.now()
            print(f"\n--- [{idx}/{len(target)}] {corpus_id} ---")

            # Load segments
            sessions = load_corpus_segments(corpus_id)
            n_sess = len(sessions)
            n_segs = sum(len(s) for s in sessions.values())
            print(f"  Loaded: {n_sess} sessions, {n_segs} segments")

            # Phase A
            print("  Phase A: Core text analysis...")
            a_counts = run_phase_a_corpus(corpus_id, sessions)
            print(f"    Done: {sum(a_counts.values())} rows")

            # Phase B
            print("  Phase B: Temporal metrics...")
            b_counts = run_phase_b_corpus(corpus_id, sessions)
            print(f"    Done: {sum(b_counts.values())} rows")

            # Phase C
            print("  Phase C: Error/recovery/stress...")
            c_counts, recovery_info = run_phase_c_corpus(corpus_id, sessions)
            print(f"    Done: {sum(c_counts.values())} rows")
            print(f"    Recovery: {recovery_info['n_excursions']} excursions, "
                  f"rate={recovery_info['recovery_rate']:.3f}")

            elapsed_c = (datetime.now() - corpus_start).total_seconds()
            results[corpus_id] = {
                'sessions': n_sess, 'segments': n_segs,
                'phase_a': a_counts, 'phase_b': b_counts, 'phase_c': c_counts,
                'recovery': recovery_info, 'elapsed': elapsed_c,
            }
            print(f"  Completed in {elapsed_c:.1f}s")

            # Free memory
            del sessions

    # Phase D
    phase_d_results = None
    if not args.skip_phase_d:
        phase_d_results = run_phase_d()
    else:
        print("\nPhase D skipped (--skip-phase-d)")

    # Summary
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    if target:
        print(f"  Phases A-C: {len(target)} corpora")
        for c in target:
            if c in results:
                r = results[c]
                print(f"    {c}: {r['sessions']} sessions, recovery={r['recovery']['recovery_rate']:.3f} ({r['elapsed']:.1f}s)")
    if phase_d_results:
        print(f"  Phase D: {phase_d_results['n_corpora']} corpora, {phase_d_results['n_dims']}-dim vector")
        print(f"  PCA: PC1={phase_d_results['pca_variance']['pc1']*100:.1f}%, PC2={phase_d_results['pca_variance']['pc2']*100:.1f}%")

    # JSON backup
    backup = {
        'script': 'run_round1_extraction.py',
        'session': 94,
        'timestamp': start.isoformat(),
        'elapsed_seconds': elapsed,
        'target_corpora': target,
        'per_corpus': results if target else {},
        'phase_d': phase_d_results,
    }
    backup_path = os.path.join(os.path.dirname(DB_PATH), 'round1_extraction_results.json')
    with open(backup_path, 'w') as f:
        json.dump(backup, f, indent=2, default=str)
    print(f"\n  Results: {backup_path}")


if __name__ == '__main__':
    main()
