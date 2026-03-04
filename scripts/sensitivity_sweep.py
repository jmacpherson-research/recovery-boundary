#!/usr/bin/env python3
r"""
Phase 3 Analyses: Sensitivity, Diagnostics, Statistics, Qualitative, Held-Out.
Session 95. Addresses Round 1 reviewer feedback items 3A-3E.

All results stored to:
  1. Tables in corpus.db (queryable via MCP)
  2. JSON backup in COWORK folder (readable by Claude)

Usage:
    cd C:\NDE_Research_Project\pipeline
    python .\run_phase3_analyses.py

    # Run specific phases only:
    python .\run_phase3_analyses.py --only 3a 3b 3c

    # Skip slow phases:
    python .\run_phase3_analyses.py --skip 3d
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
# JSON output goes to COWORK folder so Claude can read it directly
# JSON output directory — defaults to parent of DB (C:\NDE_Research_Project\)
JSON_OUTPUT_DIR = os.path.dirname(os.environ.get('CORPUS_DB_PATH', r'C:\NDE_Research_Project\corpus.db'))
WORD_RE = re.compile(r"[A-Za-z']+")
SEED = 1337

# Error patterns (same as extraction script)
ERROR_PATTERNS = {
    'RESTART': re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE),
    'FILLER': re.compile(r'\b(um|uh|er|ah|hmm|mmm|huh|oh)\b', re.IGNORECASE),
    'FALSE_START': re.compile(r'--\s*\w', re.IGNORECASE),
    'CORRECTION': re.compile(r'\b(I mean|rather|that is|or rather|well actually)\b', re.IGNORECASE),
    'INCOMPLETE': re.compile(r'\.\.\.\s*$', re.IGNORECASE),
}

INSTRUMENT_CORPORA = {
    'law_of_one', 'seth', 'bashar', 'acim', 'cwg', 'll_research', 'carla_prose'
}
INSTRUMENT_PATTERN = re.compile(
    r'\b(session|instrument|contact|channeling|channeled|medium|trance|entity)\b',
    re.IGNORECASE
)

# Calibration corpora (exclude from recovery analyses)
CALIBRATION = {'ecc_hamming74', 'gencode_splice', 'sat_3sat'}

# Corpus protocol classification
NATURAL_CORPORA = {
    'law_of_one', 'seth', 'bashar', 'acim', 'cwg', 'll_research',
    'carla_prose', 'fomc', 'scotus', 'vatican', 'annomi',
    'gutenberg_fiction', 'news_reuters', 'academic_arxiv', 'dailydialog',
}
AI_STANDARD_SINGLESHOT = {
    'synth_g0', 'synth_g1', 'synth_g2', 'synth_g3', 'synth_g4',
    'synth_gpt4o_g0', 'synth_llama70b_g0', 'synth_gemini_g0',
    'synth_fiction_g0', 'synth_news_g0', 'synth_academic_g0', 'synth_dialogue_g0',
}
AI_MULTITURN = {
    'multiturn_claude', 'multiturn_gpt4o',
    'reasoning_deepseek_r1_mt', 'reasoning_o3mini_mt',
}
AI_REASONING_SS = {'reasoning_deepseek_r1', 'reasoning_o3mini'}
AI_TEMP_SWEEP = {'synth_claude_t02', 'synth_claude_t15', 'synth_gpt4o_t02', 'synth_gpt4o_t15'}
AI_TWOPASS = {'synth_twopass_claude'}
AI_ADVERSARIAL = {'adv_constraint', 'adv_style'}
AI_CROSSDOMAIN = {
    'xdom_claude_fomc', 'xdom_claude_scotus', 'xdom_claude_therapy',
    'xdom_gpt4o_fomc', 'xdom_gpt4o_scotus', 'xdom_gpt4o_therapy',
}
AI_EDITED = {'edit_e1', 'edit_e2', 'edit_e3'}

ALL_AI = (AI_STANDARD_SINGLESHOT | AI_MULTITURN | AI_REASONING_SS |
          AI_TEMP_SWEEP | AI_TWOPASS | AI_ADVERSARIAL | AI_CROSSDOMAIN | AI_EDITED)


def get_protocol(corpus_id):
    if corpus_id in NATURAL_CORPORA:
        return 'natural'
    elif corpus_id in CALIBRATION:
        return 'calibration'
    else:
        return 'ai'


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


def save_json(data, filename):
    """Save JSON to output directory."""
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(JSON_OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  JSON saved: {path}")
    return path


# ===========================================================================
# 3A: PARAMETER SENSITIVITY SWEEP
# ===========================================================================

def run_3a_sensitivity():
    """
    Grid search over recovery definition parameters.
    Excursion thresholds × recovery windows × recovery targets.
    """
    print(f"\n{'='*60}")
    print("PHASE 3A: Parameter Sensitivity Sweep")
    print(f"{'='*60}")
    start = datetime.now()

    conn = get_db()

    # Load ALL struct features + error flags for all non-calibration corpora
    print("  Loading structural features...")
    struct_rows = conn.execute("""
        SELECT sf.corpus_id, sf.session_id, sf.segment_id, sf.word_count, sf.punct_density
        FROM text_struct_features sf
        JOIN sessions s ON sf.session_id = s.session_id AND sf.corpus_id = s.corpus_id
        WHERE sf.corpus_id NOT IN ('ecc_hamming74', 'gencode_splice', 'sat_3sat')
        ORDER BY sf.corpus_id, sf.session_id, sf.segment_id
    """).fetchall()

    # Organize by corpus → session → ordered segments
    corpus_sessions = defaultdict(lambda: defaultdict(list))
    corpus_feats = defaultdict(dict)
    for row in struct_rows:
        cid, sid, segid = row['corpus_id'], row['session_id'], row['segment_id']
        corpus_sessions[cid][sid].append({
            'segment_id': segid,
            'word_count': row['word_count'],
            'punct_density': row['punct_density'],
        })
        corpus_feats[cid][segid] = {
            'word_count': row['word_count'],
            'punct_density': row['punct_density'],
        }

    # Load error events (excluding INSTRUMENT_SESSION)
    print("  Loading error events...")
    error_rows = conn.execute("""
        SELECT corpus_id, session_id, segment_id, SUM(hit_count) as hits
        FROM text_error_events
        WHERE category != 'INSTRUMENT_SESSION'
        GROUP BY corpus_id, session_id, segment_id
    """).fetchall()
    seg_has_error = defaultdict(int)
    for row in error_rows:
        seg_has_error[(row['corpus_id'], row['session_id'], row['segment_id'])] = row['hits']

    # Also need segment ordering within sessions
    print("  Loading segment ordering...")
    seg_order_rows = conn.execute("""
        SELECT corpus_id, session_id, segment_id, sequence_order
        FROM segments
        WHERE corpus_id NOT IN ('ecc_hamming74', 'gencode_splice', 'sat_3sat')
        ORDER BY corpus_id, session_id, sequence_order
    """).fetchall()

    # Rebuild ordered sessions from segments table (struct_features might have different order)
    ordered_sessions = defaultdict(lambda: defaultdict(list))
    for row in seg_order_rows:
        cid, sid, segid = row['corpus_id'], row['session_id'], row['segment_id']
        if segid in corpus_feats.get(cid, {}):
            ordered_sessions[cid][sid].append(segid)

    conn.close()

    # Compute percentiles per corpus
    print("  Computing percentiles per corpus...")
    corpus_pctiles = {}
    for cid in corpus_feats:
        wcs = np.array([f['word_count'] for f in corpus_feats[cid].values()])
        pds = np.array([f['punct_density'] for f in corpus_feats[cid].values()])
        if len(wcs) == 0:
            continue
        pcts = {}
        for p in [1, 2.5, 5, 10, 15, 75, 80, 85, 90, 92.5, 95, 97.5, 99]:
            pcts[f'wc_p{p}'] = float(np.percentile(wcs, p))
            pcts[f'pd_p{p}'] = float(np.percentile(pds, p))
        # Also store median and q25 for recovery targets
        pcts['wc_median'] = float(np.median(wcs))
        pcts['wc_q25'] = float(np.percentile(wcs, 25))
        pcts['pd_median'] = float(np.median(pds))
        pcts['pd_q25'] = float(np.percentile(pds, 25))
        pcts['wc_mean'] = float(np.mean(wcs))
        pcts['wc_sd'] = float(np.std(wcs))
        pcts['pd_mean'] = float(np.mean(pds))
        pcts['pd_sd'] = float(np.std(pds))
        corpus_pctiles[cid] = pcts

    # === SWEEP PARAMETERS ===
    excursion_thresholds = [99, 97.5, 95, 90, 85]  # percentile for wc/pd
    recovery_windows = [1, 2, 3, 5, 10]  # max lookahead segments
    recovery_targets = ['median', 'q75', 'q25']  # what counts as "clean"
    # recovery_targets: 'median' = ≤ session median (default), 'q75' = ≤ p75 (current), 'q25' = ≤ p25 (strict)

    grid_results = []
    total_combos = len(excursion_thresholds) * len(recovery_windows) * len(recovery_targets)
    combo_idx = 0

    for exc_pct in excursion_thresholds:
        for rec_win in recovery_windows:
            for rec_tgt in recovery_targets:
                combo_idx += 1
                if combo_idx % 15 == 1:
                    print(f"  Combo {combo_idx}/{total_combos}: exc_p{exc_pct}, win={rec_win}, tgt={rec_tgt}")

                # Per-corpus recovery under this parameter set
                corpus_results = {}
                for cid in ordered_sessions:
                    if cid not in corpus_pctiles:
                        continue
                    pcts = corpus_pctiles[cid]

                    # Excursion threshold
                    wc_exc = pcts.get(f'wc_p{exc_pct}', pcts.get(f'wc_p{int(exc_pct)}'))
                    pd_exc = pcts.get(f'pd_p{exc_pct}', pcts.get(f'pd_p{int(exc_pct)}'))
                    if wc_exc is None or pd_exc is None:
                        continue

                    # For the combined error+p95 rule: use p95 relative to excursion threshold
                    # If exc_pct is already 95, use 90; if 99, use 95 (one step below)
                    err_pct = max(exc_pct - 5, 75)
                    wc_err = pcts.get(f'wc_p{err_pct}', pcts.get(f'wc_p{int(err_pct)}', wc_exc))
                    pd_err = pcts.get(f'pd_p{err_pct}', pcts.get(f'pd_p{int(err_pct)}', pd_exc))

                    # Recovery target threshold
                    if rec_tgt == 'q75':
                        wc_clean = pcts['wc_p75']
                        pd_clean = pcts['pd_p75']
                    elif rec_tgt == 'median':
                        wc_clean = pcts['wc_median']
                        pd_clean = pcts['pd_median']
                    elif rec_tgt == 'q25':
                        wc_clean = pcts['wc_q25']
                        pd_clean = pcts['pd_q25']
                    else:
                        continue

                    n_exc = 0
                    n_rec = 0

                    for sid, seg_ids in ordered_sessions[cid].items():
                        for i, segid in enumerate(seg_ids):
                            feat = corpus_feats[cid].get(segid)
                            if feat is None:
                                continue
                            has_err = seg_has_error.get((cid, sid, segid), 0) > 0

                            # Excursion check
                            is_exc = False
                            if feat['word_count'] > wc_exc or feat['punct_density'] > pd_exc:
                                is_exc = True
                            elif has_err and (feat['word_count'] > wc_err or feat['punct_density'] > pd_err):
                                is_exc = True
                            if not is_exc:
                                continue

                            n_exc += 1

                            # Recovery lookahead
                            for j in range(i + 1, min(i + 1 + rec_win, len(seg_ids))):
                                fut_segid = seg_ids[j]
                                fut_err = seg_has_error.get((cid, sid, fut_segid), 0) > 0
                                if fut_err:
                                    continue
                                fut_feat = corpus_feats[cid].get(fut_segid)
                                if fut_feat:
                                    if fut_feat['word_count'] > wc_clean:
                                        continue
                                    if fut_feat['punct_density'] > pd_clean:
                                        continue
                                n_rec += 1
                                break

                    rate = n_rec / n_exc if n_exc > 0 else None
                    corpus_results[cid] = {
                        'n_excursions': n_exc,
                        'n_recovered': n_rec,
                        'recovery_rate': rate,
                    }

                # Pool by protocol
                nat_exc = sum(r['n_excursions'] for c, r in corpus_results.items() if c in NATURAL_CORPORA)
                nat_rec = sum(r['n_recovered'] for c, r in corpus_results.items() if c in NATURAL_CORPORA)
                ai_exc = sum(r['n_excursions'] for c, r in corpus_results.items()
                            if c in ALL_AI and c not in AI_ADVERSARIAL)
                ai_rec = sum(r['n_recovered'] for c, r in corpus_results.items()
                            if c in ALL_AI and c not in AI_ADVERSARIAL)
                adv_exc = sum(r['n_excursions'] for c, r in corpus_results.items() if c in AI_ADVERSARIAL)
                adv_rec = sum(r['n_recovered'] for c, r in corpus_results.items() if c in AI_ADVERSARIAL)

                nat_rate = nat_rec / nat_exc if nat_exc > 0 else None
                ai_rate = ai_rec / ai_exc if ai_exc > 0 else None
                adv_rate = adv_rec / adv_exc if adv_exc > 0 else None

                grid_results.append({
                    'excursion_pct': exc_pct,
                    'recovery_window': rec_win,
                    'recovery_target': rec_tgt,
                    'natural_excursions': nat_exc,
                    'natural_recovered': nat_rec,
                    'natural_rate': round(nat_rate, 6) if nat_rate is not None else None,
                    'ai_excursions': ai_exc,
                    'ai_recovered': ai_rec,
                    'ai_rate': round(ai_rate, 6) if ai_rate is not None else None,
                    'adversarial_excursions': adv_exc,
                    'adversarial_recovered': adv_rec,
                    'adversarial_rate': round(adv_rate, 6) if adv_rate is not None else None,
                    'gap': round(nat_rate - ai_rate, 6) if (nat_rate is not None and ai_rate is not None) else None,
                    # Per-corpus detail
                    'per_corpus': {c: {'n_exc': r['n_excursions'], 'n_rec': r['n_recovered'],
                                       'rate': round(r['recovery_rate'], 6) if r['recovery_rate'] is not None else None}
                                  for c, r in corpus_results.items()},
                })

    # Write to database
    print(f"\n  Writing {len(grid_results)} grid cells to database...")
    conn = get_db()
    conn.execute("DROP TABLE IF EXISTS sensitivity_sweep")
    conn.execute("""CREATE TABLE sensitivity_sweep (
        excursion_pct REAL,
        recovery_window INTEGER,
        recovery_target TEXT,
        natural_excursions INTEGER,
        natural_recovered INTEGER,
        natural_rate REAL,
        ai_excursions INTEGER,
        ai_recovered INTEGER,
        ai_rate REAL,
        adversarial_excursions INTEGER,
        adversarial_recovered INTEGER,
        adversarial_rate REAL,
        gap REAL,
        PRIMARY KEY (excursion_pct, recovery_window, recovery_target)
    )""")

    for r in grid_results:
        conn.execute("""INSERT INTO sensitivity_sweep VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            r['excursion_pct'], r['recovery_window'], r['recovery_target'],
            r['natural_excursions'], r['natural_recovered'], r['natural_rate'],
            r['ai_excursions'], r['ai_recovered'], r['ai_rate'],
            r['adversarial_excursions'], r['adversarial_recovered'], r['adversarial_rate'],
            r['gap'],
        ))
    conn.commit()
    conn.close()

    elapsed = (datetime.now() - start).total_seconds()

    # Print summary
    print(f"\n  --- SENSITIVITY SUMMARY ({elapsed:.1f}s) ---")
    print(f"  {len(grid_results)} parameter combinations tested")
    ai_zero = sum(1 for r in grid_results if r['ai_rate'] is not None and r['ai_rate'] == 0.0)
    print(f"  AI recovery = 0.0%: {ai_zero}/{len(grid_results)} combos")
    if grid_results:
        max_ai = max((r['ai_rate'] for r in grid_results if r['ai_rate'] is not None), default=0)
        print(f"  Max AI recovery rate across all combos: {max_ai:.4f}")
        min_gap = min((r['gap'] for r in grid_results if r['gap'] is not None), default=0)
        max_gap = max((r['gap'] for r in grid_results if r['gap'] is not None), default=0)
        print(f"  Gap range: {min_gap:.4f} to {max_gap:.4f}")

    # Save JSON (per-corpus detail included)
    save_json({
        'phase': '3A',
        'description': 'Parameter sensitivity sweep',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_combos': len(grid_results),
        'excursion_thresholds': excursion_thresholds,
        'recovery_windows': recovery_windows,
        'recovery_targets': recovery_targets,
        'ai_zero_count': ai_zero,
        'results': grid_results,
    }, 'phase3a_sensitivity.json')

    return grid_results


# ===========================================================================
# 3B: SESSION DIAGNOSTICS
# ===========================================================================

def run_3b_diagnostics():
    """
    Session-level diagnostics: chunks/session, excursion position, runway.
    """
    print(f"\n{'='*60}")
    print("PHASE 3B: Session Diagnostics")
    print(f"{'='*60}")
    start = datetime.now()

    conn = get_db()

    # 1. Chunks per session (from compression table — each row is a session)
    print("  Loading compression data (chunks per session)...")
    chunk_rows = conn.execute("""
        SELECT corpus_id, session_id, chunk_count
        FROM text_compression
        WHERE corpus_id NOT IN ('ecc_hamming74', 'gencode_splice', 'sat_3sat')
    """).fetchall()

    chunks_by_corpus = defaultdict(list)
    for row in chunk_rows:
        chunks_by_corpus[row['corpus_id']].append(row['chunk_count'])

    chunk_summary = []
    for cid in sorted(chunks_by_corpus):
        arr = np.array(chunks_by_corpus[cid])
        protocol = get_protocol(cid)
        chunk_summary.append({
            'corpus_id': cid,
            'protocol': protocol,
            'n_sessions': len(arr),
            'mean_chunks': round(float(np.mean(arr)), 2),
            'median_chunks': round(float(np.median(arr)), 2),
            'min_chunks': int(np.min(arr)),
            'max_chunks': int(np.max(arr)),
            'sd_chunks': round(float(np.std(arr)), 2),
            'q25_chunks': round(float(np.percentile(arr, 25)), 2),
            'q75_chunks': round(float(np.percentile(arr, 75)), 2),
            'pct_1chunk': round(float(np.mean(arr == 1)) * 100, 1),
        })

    # 2. Segments per session (from segments table)
    print("  Loading segments per session...")
    seg_rows = conn.execute("""
        SELECT corpus_id, session_id, COUNT(*) as n_segs
        FROM segments
        WHERE corpus_id NOT IN ('ecc_hamming74', 'gencode_splice', 'sat_3sat')
        GROUP BY corpus_id, session_id
    """).fetchall()

    segs_by_corpus = defaultdict(list)
    for row in seg_rows:
        segs_by_corpus[row['corpus_id']].append(row['n_segs'])

    seg_summary = []
    for cid in sorted(segs_by_corpus):
        arr = np.array(segs_by_corpus[cid])
        protocol = get_protocol(cid)
        seg_summary.append({
            'corpus_id': cid,
            'protocol': protocol,
            'n_sessions': len(arr),
            'mean_segs': round(float(np.mean(arr)), 2),
            'median_segs': round(float(np.median(arr)), 2),
            'min_segs': int(np.min(arr)),
            'max_segs': int(np.max(arr)),
            'sd_segs': round(float(np.std(arr)), 2),
        })

    # 3. Excursion position within session + remaining runway
    print("  Loading excursion positions and runway...")
    exc_rows = conn.execute("""
        SELECT sr.corpus_id, sr.session_id, sr.excursion_segment_id, sr.recovery_lag, sr.recovered
        FROM text_stress_recovery sr
    """).fetchall()

    # Get segment positions within sessions
    seg_positions = {}
    seg_session_sizes = {}
    pos_rows = conn.execute("""
        SELECT corpus_id, session_id, segment_id, sequence_order
        FROM segments
        WHERE corpus_id NOT IN ('ecc_hamming74', 'gencode_splice', 'sat_3sat')
        ORDER BY corpus_id, session_id, sequence_order
    """).fetchall()

    session_sizes = defaultdict(int)
    for row in pos_rows:
        cid, sid, segid = row['corpus_id'], row['session_id'], row['segment_id']
        seg_positions[(cid, sid, segid)] = row['sequence_order']
        session_sizes[(cid, sid)] += 1

    excursion_details = []
    for row in exc_rows:
        cid = row['corpus_id']
        sid = row['session_id']
        segid = row['excursion_segment_id']
        pos = seg_positions.get((cid, sid, segid))
        sess_size = session_sizes.get((cid, sid), 0)
        if pos is not None and sess_size > 0:
            relative_pos = pos / sess_size  # 0.0 = start, 1.0 = end
            remaining = sess_size - pos - 1  # segments after excursion
            excursion_details.append({
                'corpus_id': cid,
                'session_id': sid,
                'segment_id': segid,
                'protocol': get_protocol(cid),
                'absolute_pos': pos,
                'session_size': sess_size,
                'relative_pos': round(relative_pos, 4),
                'remaining_segs': remaining,
                'recovered': row['recovered'],
                'recovery_lag': row['recovery_lag'],
            })

    conn.close()

    # 4. Compute recovery rate conditioned on remaining runway
    runway_brackets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 50), (51, 9999)]
    runway_analysis = []
    for lo, hi in runway_brackets:
        for protocol in ['natural', 'ai']:
            subset = [e for e in excursion_details
                      if e['protocol'] == protocol and lo <= e['remaining_segs'] <= hi]
            n = len(subset)
            n_rec = sum(1 for e in subset if e['recovered'] == 1)
            rate = n_rec / n if n > 0 else None
            runway_analysis.append({
                'runway_min': lo,
                'runway_max': hi,
                'protocol': protocol,
                'n_excursions': n,
                'n_recovered': n_rec,
                'recovery_rate': round(rate, 6) if rate is not None else None,
            })

    # 5. Excursion position bins (early/middle/late)
    position_analysis = []
    for protocol in ['natural', 'ai']:
        subset = [e for e in excursion_details if e['protocol'] == protocol]
        for label, lo, hi in [('early', 0.0, 0.33), ('middle', 0.33, 0.66), ('late', 0.66, 1.01)]:
            bin_exc = [e for e in subset if lo <= e['relative_pos'] < hi]
            n = len(bin_exc)
            n_rec = sum(1 for e in bin_exc if e['recovered'] == 1)
            rate = n_rec / n if n > 0 else None
            position_analysis.append({
                'protocol': protocol,
                'position': label,
                'n_excursions': n,
                'n_recovered': n_rec,
                'recovery_rate': round(rate, 6) if rate is not None else None,
            })

    # Write to database
    print("  Writing to database...")
    conn = get_db()

    conn.execute("DROP TABLE IF EXISTS session_diagnostics_chunks")
    conn.execute("""CREATE TABLE session_diagnostics_chunks (
        corpus_id TEXT PRIMARY KEY,
        protocol TEXT,
        n_sessions INTEGER,
        mean_chunks REAL, median_chunks REAL,
        min_chunks INTEGER, max_chunks INTEGER,
        sd_chunks REAL, q25_chunks REAL, q75_chunks REAL,
        pct_1chunk REAL
    )""")
    for r in chunk_summary:
        conn.execute("INSERT INTO session_diagnostics_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            r['corpus_id'], r['protocol'], r['n_sessions'],
            r['mean_chunks'], r['median_chunks'], r['min_chunks'], r['max_chunks'],
            r['sd_chunks'], r['q25_chunks'], r['q75_chunks'], r['pct_1chunk'],
        ))

    conn.execute("DROP TABLE IF EXISTS session_diagnostics_segs")
    conn.execute("""CREATE TABLE session_diagnostics_segs (
        corpus_id TEXT PRIMARY KEY,
        protocol TEXT,
        n_sessions INTEGER,
        mean_segs REAL, median_segs REAL,
        min_segs INTEGER, max_segs INTEGER,
        sd_segs REAL
    )""")
    for r in seg_summary:
        conn.execute("INSERT INTO session_diagnostics_segs VALUES (?,?,?,?,?,?,?,?)", (
            r['corpus_id'], r['protocol'], r['n_sessions'],
            r['mean_segs'], r['median_segs'], r['min_segs'], r['max_segs'], r['sd_segs'],
        ))

    conn.commit()
    conn.close()

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n  --- DIAGNOSTICS SUMMARY ({elapsed:.1f}s) ---")
    print(f"  {len(chunk_summary)} corpora in chunk summary")
    print(f"  {len(excursion_details)} excursion events with position data")
    print(f"  Runway analysis: {len(runway_analysis)} cells")

    save_json({
        'phase': '3B',
        'description': 'Session diagnostics',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'chunk_summary': chunk_summary,
        'seg_summary': seg_summary,
        'runway_analysis': runway_analysis,
        'position_analysis': position_analysis,
        'n_excursion_details': len(excursion_details),
    }, 'phase3b_diagnostics.json')

    return {
        'chunk_summary': chunk_summary,
        'seg_summary': seg_summary,
        'runway_analysis': runway_analysis,
        'position_analysis': position_analysis,
    }


# ===========================================================================
# 3C: STATISTICAL EXTRAS
# ===========================================================================

def run_3c_statistics():
    """
    Binomial CI, per-corpus excursion rate, within-protocol domain tests,
    independence assessment.
    """
    print(f"\n{'='*60}")
    print("PHASE 3C: Statistical Extras")
    print(f"{'='*60}")
    start = datetime.now()

    conn = get_db()

    # Load stress summary
    stress = {}
    for r in conn.execute("SELECT * FROM text_stress_summary").fetchall():
        stress[r['corpus_id']] = dict(r)

    # Load session counts + word counts per corpus
    corpus_info = {}
    for r in conn.execute("""
        SELECT s.corpus_id,
               COUNT(DISTINCT s.session_id) as n_sessions,
               SUM(seg.word_count) as total_words
        FROM sessions s
        JOIN (SELECT corpus_id, session_id, SUM(LENGTH(text) - LENGTH(REPLACE(text, ' ', ''))) + 1 as word_count
              FROM segments GROUP BY corpus_id, session_id) seg
        ON s.corpus_id = seg.corpus_id AND s.session_id = seg.session_id
        GROUP BY s.corpus_id
    """).fetchall():
        corpus_info[r['corpus_id']] = dict(r)

    # Actually, simpler word count from struct_features
    word_counts = {}
    for r in conn.execute("""
        SELECT corpus_id, SUM(word_count) as total_words, COUNT(*) as n_segs
        FROM text_struct_features GROUP BY corpus_id
    """).fetchall():
        word_counts[r['corpus_id']] = {'total_words': r['total_words'], 'n_segs': r['n_segs']}

    # Load individual excursions for independence check
    exc_detail = conn.execute("""
        SELECT corpus_id, session_id, excursion_segment_id, recovered
        FROM text_stress_recovery
    """).fetchall()

    conn.close()

    results = {}

    # --- 1. Binomial CI (rule of three) ---
    print("  Computing binomial CIs...")
    binomial_cis = []
    for cid in sorted(stress):
        s = stress[cid]
        protocol = get_protocol(cid)
        n = s['n_excursions']
        k = round(s['recovery_rate'] * n) if n > 0 else 0

        # Rule of three: if k=0, upper 95% CI = 3/N
        if n > 0 and k == 0:
            upper_95 = 3.0 / n
            ci_method = 'rule_of_three'
        elif n > 0:
            # Wilson score interval
            z = 1.96
            p_hat = k / n
            denom = 1 + z**2 / n
            centre = (p_hat + z**2 / (2 * n)) / denom
            margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
            upper_95 = min(centre + margin, 1.0)
            ci_method = 'wilson'
        else:
            upper_95 = None
            ci_method = 'none'

        lower_95 = 0.0
        if n > 0 and k > 0:
            z = 1.96
            p_hat = k / n
            denom = 1 + z**2 / n
            centre = (p_hat + z**2 / (2 * n)) / denom
            margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
            lower_95 = max(centre - margin, 0.0)

        binomial_cis.append({
            'corpus_id': cid,
            'protocol': protocol,
            'n_excursions': n,
            'n_recovered': k,
            'recovery_rate': round(s['recovery_rate'], 6),
            'ci_lower_95': round(lower_95, 6),
            'ci_upper_95': round(upper_95, 6) if upper_95 is not None else None,
            'ci_method': ci_method,
        })

    results['binomial_cis'] = binomial_cis

    # Pooled AI binomial CI (excluding adversarial)
    ai_n = sum(s['n_excursions'] for cid, s in stress.items()
               if cid in ALL_AI and cid not in AI_ADVERSARIAL)
    ai_k = sum(round(s['recovery_rate'] * s['n_excursions'])
               for cid, s in stress.items()
               if cid in ALL_AI and cid not in AI_ADVERSARIAL)
    pooled_upper = 3.0 / ai_n if ai_n > 0 and ai_k == 0 else None
    results['pooled_ai'] = {
        'n_excursions': ai_n,
        'n_recovered': ai_k,
        'recovery_rate': ai_k / ai_n if ai_n > 0 else None,
        'upper_95_ci': round(pooled_upper, 6) if pooled_upper is not None else None,
        'rule_of_three': f'0/{ai_n} gives upper 95% CI = {pooled_upper:.4f}' if pooled_upper else f'{ai_k}/{ai_n}',
    }

    # --- 2. Per-corpus excursion rate ---
    print("  Computing per-corpus excursion rates...")
    exc_rates = []
    for cid in sorted(stress):
        s = stress[cid]
        wc = word_counts.get(cid, {})
        n_segs = wc.get('n_segs', 0)
        total_words = wc.get('total_words', 0)
        exc_per_1k_segs = (s['n_excursions'] / n_segs * 1000) if n_segs > 0 else 0
        exc_per_100k_words = (s['n_excursions'] / total_words * 100000) if total_words > 0 else 0

        exc_rates.append({
            'corpus_id': cid,
            'protocol': get_protocol(cid),
            'n_excursions': s['n_excursions'],
            'n_segments': n_segs,
            'total_words': total_words,
            'exc_per_1k_segs': round(exc_per_1k_segs, 4),
            'exc_per_100k_words': round(exc_per_100k_words, 4),
        })

    results['excursion_rates'] = exc_rates

    # --- 3. Independence assessment ---
    print("  Assessing excursion independence...")
    exc_by_session = defaultdict(lambda: defaultdict(int))
    for row in exc_detail:
        exc_by_session[row['corpus_id']][row['session_id']] += 1

    independence = []
    for cid in sorted(exc_by_session):
        session_exc = list(exc_by_session[cid].values())
        n_sessions_with_exc = len(session_exc)
        n_exc_total = sum(session_exc)
        max_from_one = max(session_exc) if session_exc else 0
        pct_from_max = (max_from_one / n_exc_total * 100) if n_exc_total > 0 else 0

        independence.append({
            'corpus_id': cid,
            'protocol': get_protocol(cid),
            'n_excursions': n_exc_total,
            'n_sessions_with_exc': n_sessions_with_exc,
            'max_from_one_session': max_from_one,
            'pct_from_max_session': round(pct_from_max, 1),
            'mean_per_session': round(n_exc_total / n_sessions_with_exc, 2) if n_sessions_with_exc > 0 else 0,
        })

    results['independence'] = independence

    # --- 4. Within-protocol domain tests ---
    print("  Computing within-protocol domain effects...")
    # Group naturals by domain
    domain_map = {
        'law_of_one': 'channeling', 'seth': 'channeling', 'bashar': 'channeling',
        'acim': 'channeling', 'cwg': 'channeling', 'll_research': 'channeling',
        'carla_prose': 'channeling',
        'fomc': 'institutional', 'scotus': 'institutional', 'vatican': 'institutional',
        'annomi': 'dialogue',
        'gutenberg_fiction': 'fiction', 'news_reuters': 'news',
        'academic_arxiv': 'academic', 'dailydialog': 'dialogue',
    }

    # Within-natural: do different domains have different recovery rates?
    domain_recovery = defaultdict(lambda: {'n_exc': 0, 'n_rec': 0})
    for cid, s in stress.items():
        if cid not in NATURAL_CORPORA:
            continue
        domain = domain_map.get(cid, 'other')
        k = round(s['recovery_rate'] * s['n_excursions'])
        domain_recovery[domain]['n_exc'] += s['n_excursions']
        domain_recovery[domain]['n_rec'] += k

    within_natural = []
    for domain in sorted(domain_recovery):
        d = domain_recovery[domain]
        rate = d['n_rec'] / d['n_exc'] if d['n_exc'] > 0 else None
        within_natural.append({
            'domain': domain,
            'n_excursions': d['n_exc'],
            'n_recovered': d['n_rec'],
            'recovery_rate': round(rate, 6) if rate is not None else None,
        })

    results['within_natural_domains'] = within_natural

    # Within-AI: do different conditions have different recovery rates?
    ai_conditions = {
        'standard_singleshot': AI_STANDARD_SINGLESHOT,
        'multiturn': AI_MULTITURN,
        'reasoning_singleshot': AI_REASONING_SS,
        'temp_sweep': AI_TEMP_SWEEP,
        'twopass': AI_TWOPASS,
        'adversarial': AI_ADVERSARIAL,
        'crossdomain': AI_CROSSDOMAIN,
        'edited': AI_EDITED,
    }
    within_ai = []
    for cond_name, cond_set in ai_conditions.items():
        n_exc = sum(stress[c]['n_excursions'] for c in cond_set if c in stress)
        n_rec = sum(round(stress[c]['recovery_rate'] * stress[c]['n_excursions'])
                    for c in cond_set if c in stress)
        rate = n_rec / n_exc if n_exc > 0 else None
        within_ai.append({
            'condition': cond_name,
            'n_corpora': len([c for c in cond_set if c in stress]),
            'n_excursions': n_exc,
            'n_recovered': n_rec,
            'recovery_rate': round(rate, 6) if rate is not None else None,
        })

    results['within_ai_conditions'] = within_ai

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n  --- STATISTICS SUMMARY ({elapsed:.1f}s) ---")
    print(f"  Pooled AI: {results['pooled_ai']['rule_of_three']}")
    if pooled_upper:
        print(f"  Upper 95% CI: {pooled_upper:.4f} ({pooled_upper*100:.2f}%)")
    print(f"  Within-natural domains: {len(within_natural)} domains")
    print(f"  Within-AI conditions: {len(within_ai)} conditions")
    for w in within_ai:
        print(f"    {w['condition']}: {w['n_recovered']}/{w['n_excursions']} = "
              f"{w['recovery_rate']:.4f}" if w['recovery_rate'] is not None else
              f"    {w['condition']}: no excursions")

    save_json({
        'phase': '3C',
        'description': 'Statistical extras',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'results': results,
    }, 'phase3c_statistics.json')

    return results


# ===========================================================================
# 3D: QUALITATIVE EXAMPLES
# ===========================================================================

def run_3d_qualitative():
    """
    Extract actual text from excursion+recovery and excursion+no-recovery events.
    Prefer non-channeling naturals for recovery examples.
    """
    print(f"\n{'='*60}")
    print("PHASE 3D: Qualitative Examples")
    print(f"{'='*60}")
    start = datetime.now()

    conn = get_db()

    # Priority naturals for recovery examples (non-channeling)
    priority_naturals = ['fomc', 'annomi', 'scotus', 'news_reuters', 'gutenberg_fiction',
                         'academic_arxiv', 'dailydialog']
    # Also include some channeling for diversity
    channeling_naturals = ['law_of_one', 'seth', 'bashar', 'cwg']

    # Get excursions with recovery from priority naturals
    print("  Finding natural recovery examples...")
    recovery_examples = []
    for cid in priority_naturals + channeling_naturals:
        rows = conn.execute("""
            SELECT corpus_id, session_id, excursion_segment_id, recovery_lag, recovered
            FROM text_stress_recovery
            WHERE corpus_id = ? AND recovered = 1
            ORDER BY recovery_lag ASC
            LIMIT 10
        """, (cid,)).fetchall()

        for row in rows:
            # Get the excursion segment text + surrounding context
            sid = row['session_id']
            exc_segid = row['excursion_segment_id']
            lag = row['recovery_lag']

            # Get ordered segments for this session
            segs = conn.execute("""
                SELECT segment_id, text, sequence_order, speaker_type
                FROM segments
                WHERE corpus_id = ? AND session_id = ?
                ORDER BY sequence_order
            """, (cid, sid)).fetchall()

            # Find excursion segment position
            for idx, seg in enumerate(segs):
                if seg['segment_id'] == exc_segid:
                    # Get context: 1 before, excursion, lag segments after
                    context_start = max(0, idx - 1)
                    context_end = min(len(segs), idx + lag + 2)  # include recovery + 1 after
                    context_segs = []
                    for k in range(context_start, context_end):
                        s = segs[k]
                        role = 'BEFORE' if k < idx else ('EXCURSION' if k == idx else
                               ('RECOVERY' if k == idx + lag else 'BETWEEN' if k < idx + lag else 'AFTER'))
                        text = s['text'] or ''
                        # Truncate long text
                        if len(text) > 500:
                            text = text[:500] + '...'
                        context_segs.append({
                            'role': role,
                            'segment_id': s['segment_id'],
                            'speaker_type': s['speaker_type'],
                            'text': text,
                            'seq_order': s['sequence_order'],
                        })

                    recovery_examples.append({
                        'corpus_id': cid,
                        'session_id': sid,
                        'excursion_segment': exc_segid,
                        'recovery_lag': lag,
                        'protocol': 'natural',
                        'context': context_segs,
                    })
                    break

            if len(recovery_examples) >= 20:
                break
        if len(recovery_examples) >= 20:
            break

    # Get AI non-recovery examples
    print("  Finding AI non-recovery examples...")
    ai_example_corpora = ['synth_g0', 'synth_gpt4o_g0', 'multiturn_claude',
                          'synth_fiction_g0', 'synth_news_g0', 'reasoning_deepseek_r1',
                          'synth_claude_t02', 'synth_twopass_claude']
    nonrecovery_examples = []

    for cid in ai_example_corpora:
        rows = conn.execute("""
            SELECT corpus_id, session_id, excursion_segment_id, recovery_lag, recovered
            FROM text_stress_recovery
            WHERE corpus_id = ? AND recovered = 0
            LIMIT 5
        """, (cid,)).fetchall()

        for row in rows:
            sid = row['session_id']
            exc_segid = row['excursion_segment_id']

            segs = conn.execute("""
                SELECT segment_id, text, sequence_order, speaker_type
                FROM segments
                WHERE corpus_id = ? AND session_id = ?
                ORDER BY sequence_order
            """, (cid, sid)).fetchall()

            for idx, seg in enumerate(segs):
                if seg['segment_id'] == exc_segid:
                    context_start = max(0, idx - 1)
                    context_end = min(len(segs), idx + 6)  # show 5 after
                    context_segs = []
                    for k in range(context_start, context_end):
                        s = segs[k]
                        role = 'BEFORE' if k < idx else ('EXCURSION' if k == idx else 'AFTER')
                        text = s['text'] or ''
                        if len(text) > 500:
                            text = text[:500] + '...'
                        context_segs.append({
                            'role': role,
                            'segment_id': s['segment_id'],
                            'speaker_type': s['speaker_type'],
                            'text': text,
                            'seq_order': s['sequence_order'],
                        })

                    nonrecovery_examples.append({
                        'corpus_id': cid,
                        'session_id': sid,
                        'excursion_segment': exc_segid,
                        'recovery_lag': -1,
                        'protocol': 'ai',
                        'context': context_segs,
                    })
                    break

            if len(nonrecovery_examples) >= 15:
                break
        if len(nonrecovery_examples) >= 15:
            break

    # Also get adversarial recovery examples (these are interesting!)
    print("  Finding adversarial recovery examples...")
    adv_examples = []
    for cid in ['adv_constraint', 'adv_style']:
        rows = conn.execute("""
            SELECT corpus_id, session_id, excursion_segment_id, recovery_lag, recovered
            FROM text_stress_recovery
            WHERE corpus_id = ? AND recovered = 1
            LIMIT 5
        """, (cid,)).fetchall()

        for row in rows:
            sid = row['session_id']
            exc_segid = row['excursion_segment_id']
            lag = row['recovery_lag']

            segs = conn.execute("""
                SELECT segment_id, text, sequence_order, speaker_type
                FROM segments
                WHERE corpus_id = ? AND session_id = ?
                ORDER BY sequence_order
            """, (cid, sid)).fetchall()

            for idx, seg in enumerate(segs):
                if seg['segment_id'] == exc_segid:
                    context_start = max(0, idx - 1)
                    context_end = min(len(segs), idx + lag + 2)
                    context_segs = []
                    for k in range(context_start, context_end):
                        s = segs[k]
                        role = 'BEFORE' if k < idx else ('EXCURSION' if k == idx else
                               ('RECOVERY' if k == idx + lag else 'BETWEEN' if k < idx + lag else 'AFTER'))
                        text = s['text'] or ''
                        if len(text) > 500:
                            text = text[:500] + '...'
                        context_segs.append({
                            'role': role,
                            'segment_id': s['segment_id'],
                            'speaker_type': s['speaker_type'],
                            'text': text,
                            'seq_order': s['sequence_order'],
                        })

                    adv_examples.append({
                        'corpus_id': cid,
                        'session_id': sid,
                        'excursion_segment': exc_segid,
                        'recovery_lag': lag,
                        'protocol': 'adversarial_ai',
                        'context': context_segs,
                    })
                    break

    # Reasoning multi-turn recovery examples
    print("  Finding reasoning multi-turn recovery examples...")
    reasoning_examples = []
    for cid in ['reasoning_deepseek_r1_mt', 'reasoning_o3mini_mt']:
        rows = conn.execute("""
            SELECT corpus_id, session_id, excursion_segment_id, recovery_lag, recovered
            FROM text_stress_recovery
            WHERE corpus_id = ? AND recovered = 1
            LIMIT 5
        """, (cid,)).fetchall()

        for row in rows:
            sid = row['session_id']
            exc_segid = row['excursion_segment_id']
            lag = row['recovery_lag']

            segs = conn.execute("""
                SELECT segment_id, text, sequence_order, speaker_type
                FROM segments
                WHERE corpus_id = ? AND session_id = ?
                ORDER BY sequence_order
            """, (cid, sid)).fetchall()

            for idx, seg in enumerate(segs):
                if seg['segment_id'] == exc_segid:
                    context_start = max(0, idx - 1)
                    context_end = min(len(segs), idx + lag + 2)
                    context_segs = []
                    for k in range(context_start, context_end):
                        s = segs[k]
                        role = 'BEFORE' if k < idx else ('EXCURSION' if k == idx else
                               ('RECOVERY' if k == idx + lag else 'BETWEEN' if k < idx + lag else 'AFTER'))
                        text = s['text'] or ''
                        if len(text) > 500:
                            text = text[:500] + '...'
                        context_segs.append({
                            'role': role,
                            'segment_id': s['segment_id'],
                            'speaker_type': s['speaker_type'],
                            'text': text,
                            'seq_order': s['sequence_order'],
                        })

                    reasoning_examples.append({
                        'corpus_id': cid,
                        'session_id': sid,
                        'excursion_segment': exc_segid,
                        'recovery_lag': lag,
                        'protocol': 'reasoning_multiturn',
                        'context': context_segs,
                    })
                    break

    conn.close()

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n  --- QUALITATIVE SUMMARY ({elapsed:.1f}s) ---")
    print(f"  Natural recovery examples: {len(recovery_examples)}")
    print(f"  AI non-recovery examples: {len(nonrecovery_examples)}")
    print(f"  Adversarial recovery examples: {len(adv_examples)}")
    print(f"  Reasoning multi-turn recovery examples: {len(reasoning_examples)}")

    save_json({
        'phase': '3D',
        'description': 'Qualitative examples',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'natural_recovery': recovery_examples,
        'ai_nonrecovery': nonrecovery_examples,
        'adversarial_recovery': adv_examples,
        'reasoning_multiturn_recovery': reasoning_examples,
    }, 'phase3d_qualitative.json')

    return {
        'n_natural_recovery': len(recovery_examples),
        'n_ai_nonrecovery': len(nonrecovery_examples),
        'n_adversarial': len(adv_examples),
        'n_reasoning': len(reasoning_examples),
    }


# ===========================================================================
# 3E: HELD-OUT VALIDATION
# ===========================================================================

def run_3e_heldout():
    """
    Designate 3 corpora as held-out. Build compact vector on remaining ~49.
    Evaluate recovery boundary + PCA placement on held-out 3.
    """
    print(f"\n{'='*60}")
    print("PHASE 3E: Held-Out Validation")
    print(f"{'='*60}")
    start = datetime.now()

    conn = get_db()

    # Held-out corpora (as specified in action plan)
    held_out = ['news_reuters', 'synth_news_g0', 'reasoning_deepseek_r1']

    # Load full compact vector
    print("  Loading compact vectors...")
    cv_rows = conn.execute("SELECT * FROM compact_vector").fetchall()
    all_vectors = {}
    for row in cv_rows:
        cid = row['corpus_id']
        dims = []
        for d in range(1, 11):
            val = row[f'dim{d}']
            dims.append(float(val) if val is not None else float('nan'))
        all_vectors[cid] = dims

    # Split into training and held-out
    train_ids = [cid for cid in all_vectors if cid not in held_out]
    test_ids = [cid for cid in all_vectors if cid in held_out]

    print(f"  Training: {len(train_ids)} corpora")
    print(f"  Held-out: {test_ids}")

    # Active dims: 1-8 (9,10 are NaN placeholders)
    active_dims = list(range(8))  # indices 0-7

    # Compute z-scores from training set only
    train_matrix = []
    for cid in train_ids:
        vec = [all_vectors[cid][d] for d in active_dims]
        train_matrix.append(vec)
    train_matrix = np.array(train_matrix)

    # Handle NaN: use nanmean/nanstd
    means = np.nanmean(train_matrix, axis=0)
    stds = np.nanstd(train_matrix, axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero

    # Z-score all vectors using training stats
    def z_score(vec):
        return [(v - means[i]) / stds[i] if not np.isnan(v) else 0.0
                for i, v in enumerate(vec)]

    train_z = {}
    for cid in train_ids:
        vec = [all_vectors[cid][d] for d in active_dims]
        train_z[cid] = z_score(vec)

    test_z = {}
    for cid in test_ids:
        vec = [all_vectors[cid][d] for d in active_dims]
        test_z[cid] = z_score(vec)

    # PCA on training set
    print("  Computing PCA on training set...")
    train_z_matrix = np.array([train_z[cid] for cid in train_ids])
    cov = np.cov(train_z_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # Project training and test
    train_pca = {}
    for i, cid in enumerate(train_ids):
        coords = train_z_matrix[i] @ eigenvectors[:, :2]
        train_pca[cid] = {'pc1': float(coords[0]), 'pc2': float(coords[1])}

    test_pca = {}
    for cid in test_ids:
        vec = np.array(test_z[cid])
        coords = vec @ eigenvectors[:, :2]
        test_pca[cid] = {'pc1': float(coords[0]), 'pc2': float(coords[1])}

    # Distance to law_of_one (using training z-scores)
    ref_z = train_z.get('law_of_one')
    test_distances = {}
    if ref_z:
        for cid in test_ids:
            z = test_z[cid]
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(z, ref_z)))
            test_distances[cid] = round(dist, 4)

    # Recovery boundary check
    stress = {}
    for r in conn.execute("SELECT * FROM text_stress_summary").fetchall():
        stress[r['corpus_id']] = dict(r)
    conn.close()

    # Training set boundary: find minimum natural recovery rate and maximum AI recovery rate
    train_nat_rates = [stress[c]['recovery_rate'] for c in train_ids
                       if c in NATURAL_CORPORA and c in stress and stress[c]['n_excursions'] > 0]
    train_ai_rates = [stress[c]['recovery_rate'] for c in train_ids
                      if c in ALL_AI and c not in AI_ADVERSARIAL and c in stress and stress[c]['n_excursions'] > 0]

    min_nat_train = min(train_nat_rates) if train_nat_rates else None
    max_ai_train = max(train_ai_rates) if train_ai_rates else None

    # Evaluate held-out
    heldout_results = []
    for cid in test_ids:
        s = stress.get(cid, {})
        protocol = get_protocol(cid)
        rate = s.get('recovery_rate')
        n_exc = s.get('n_excursions', 0)

        # Does it fall on the correct side of the boundary?
        if protocol == 'natural':
            boundary_correct = rate > 0 if rate is not None else None
        else:
            boundary_correct = rate == 0 if rate is not None else None

        heldout_results.append({
            'corpus_id': cid,
            'protocol': protocol,
            'n_excursions': n_exc,
            'recovery_rate': round(rate, 6) if rate is not None else None,
            'boundary_correct': boundary_correct,
            'pca_pc1': test_pca.get(cid, {}).get('pc1'),
            'pca_pc2': test_pca.get(cid, {}).get('pc2'),
            'distance_to_law_of_one': test_distances.get(cid),
            'z_vector': [round(v, 4) for v in test_z.get(cid, [])],
        })

    # Variance explained
    total_var = sum(eigenvalues)
    var_explained = [float(e / total_var) for e in eigenvalues[:3]]

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n  --- HELD-OUT VALIDATION RESULTS ({elapsed:.1f}s) ---")
    for r in heldout_results:
        status = "CORRECT" if r['boundary_correct'] else "WRONG" if r['boundary_correct'] is not None else "N/A"
        print(f"  {r['corpus_id']}: recovery={r['recovery_rate']}, boundary={status}, "
              f"PC1={r['pca_pc1']:.3f}, dist_lo1={r['distance_to_law_of_one']}")
    print(f"  Training PCA var explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")
    print(f"  Training boundary: min_nat={min_nat_train:.4f}, max_ai={max_ai_train:.4f}"
          if min_nat_train is not None and max_ai_train is not None else "")

    save_json({
        'phase': '3E',
        'description': 'Held-out validation',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'held_out_corpora': held_out,
        'n_training': len(train_ids),
        'training_boundary': {
            'min_natural_recovery': round(min_nat_train, 6) if min_nat_train is not None else None,
            'max_ai_recovery': round(max_ai_train, 6) if max_ai_train is not None else None,
        },
        'pca_variance_explained': var_explained,
        'heldout_results': heldout_results,
        'training_pca_sample': {cid: train_pca[cid] for cid in list(train_pca)[:5]},
    }, 'phase3e_heldout.json')

    return heldout_results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 3 Analyses (Session 95)')
    parser.add_argument('--only', nargs='+', choices=['3a', '3b', '3c', '3d', '3e'],
                       help='Run only these phases')
    parser.add_argument('--skip', nargs='+', choices=['3a', '3b', '3c', '3d', '3e'],
                       help='Skip these phases')
    args = parser.parse_args()

    phases = {'3a', '3b', '3c', '3d', '3e'}
    if args.only:
        phases = set(args.only)
    if args.skip:
        phases -= set(args.skip)

    start = datetime.now()
    print(f"Phase 3 Analyses — Session 95")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print(f"JSON output: {JSON_OUTPUT_DIR}")
    print(f"Phases: {', '.join(sorted(phases))}")
    print()

    results = {}

    if '3a' in phases:
        results['3a'] = run_3a_sensitivity()

    if '3b' in phases:
        results['3b'] = run_3b_diagnostics()

    if '3c' in phases:
        results['3c'] = run_3c_statistics()

    if '3d' in phases:
        results['3d'] = run_3d_qualitative()

    if '3e' in phases:
        results['3e'] = run_3e_heldout()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"ALL PHASE 3 ANALYSES COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Phases run: {', '.join(sorted(phases))}")
    print(f"  JSON files in: {JSON_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
