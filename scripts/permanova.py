r"""
PERMDISP + safe-vector PERMANOVA sensitivity analyses.
Addresses GPT T12 critique items:
  1. PERMDISP (betadisper equivalent) for binary/fine-grained protocol + domain
  2. Safe-vector PERMANOVA on dims 1,2,4,5 only (no size-confounded dims)
  3. PERMANOVA restricted to corpora >= 150K words (7-dim, no small-corpus issue)

Run from: C:\NDE_Research_Project\pipeline
Command:  python ..\run_permdisp_sensitivity.py
Output:   permdisp_sensitivity_results.json in same folder as corpus.db
"""

import sqlite3
import numpy as np
import json
import os
from itertools import combinations

DB_PATH = r"C:\NDE_Research_Project\corpus.db"
OUTPUT_DIR = os.path.dirname(DB_PATH)

# ─── Configuration ───────────────────────────────────────────────────────────
N_PERM = 9999

# Same 30 corpora as original PERMANOVA (exclude calibration + adversarial with incomplete vectors)
EXCLUDE = {
    'ecc_hamming74', 'gencode_splice', 'sat_3sat',  # calibration
    'adv_constraint', 'adv_style',                    # adversarial (NULL dims)
}

# Protocol binary: natural vs AI
NATURAL_CORPORA = {
    'law_of_one', 'll_research', 'seth', 'bashar', 'acim', 'cwg',
    'carla_prose', 'fomc', 'scotus', 'vatican', 'annomi'
}

# Domain groupings
DOMAIN_MAP = {
    'law_of_one': 'channeling', 'll_research': 'channeling',
    'seth': 'channeling', 'bashar': 'channeling',
    'acim': 'channeling', 'cwg': 'channeling',
    'carla_prose': 'prose', 'vatican': 'institutional',
    'fomc': 'institutional', 'scotus': 'institutional',
    'annomi': 'therapy',
    # AI corpora get domain by content
    'synth_g0': 'channeling', 'synth_g1': 'channeling',
    'synth_g2': 'channeling', 'synth_g3': 'channeling',
    'synth_g4': 'channeling',
    'edit_e1': 'channeling', 'edit_e2': 'channeling', 'edit_e3': 'channeling',
    'multiturn_claude': 'channeling', 'multiturn_gpt4o': 'channeling',
    'synth_gpt4o_g0': 'channeling', 'synth_llama70b_g0': 'channeling',
    'synth_gemini_g0': 'channeling',
    'xdom_claude_scotus': 'institutional', 'xdom_gpt4o_scotus': 'institutional',
    'xdom_claude_fomc': 'institutional', 'xdom_gpt4o_fomc': 'institutional',
    'xdom_claude_therapy': 'therapy', 'xdom_gpt4o_therapy': 'therapy',
}

# Fine-grained protocol (10 levels)
PROTOCOL_FINE_MAP = {
    'law_of_one': 'dialogue_channeling', 'll_research': 'dialogue_channeling',
    'bashar': 'dialogue_channeling',
    'seth': 'dictation_channeling', 'acim': 'dictation_channeling',
    'cwg': 'monologue_channeling',
    'fomc': 'institutional_transcript', 'scotus': 'institutional_transcript',
    'vatican': 'institutional_document',
    'carla_prose': 'authored_prose',
    'annomi': 'therapy_dialogue',
    'synth_g0': 'ai_stateless', 'synth_g1': 'ai_stateless',
    'synth_g2': 'ai_stateless', 'synth_g3': 'ai_stateless',
    'synth_g4': 'ai_stateless',
    'synth_gpt4o_g0': 'ai_stateless', 'synth_llama70b_g0': 'ai_stateless',
    'synth_gemini_g0': 'ai_stateless',
    'xdom_claude_scotus': 'ai_stateless', 'xdom_claude_fomc': 'ai_stateless',
    'xdom_claude_therapy': 'ai_stateless',
    'xdom_gpt4o_scotus': 'ai_stateless', 'xdom_gpt4o_fomc': 'ai_stateless',
    'xdom_gpt4o_therapy': 'ai_stateless',
    'multiturn_claude': 'ai_multiturn', 'multiturn_gpt4o': 'ai_multiturn',
    'edit_e1': 'ai_edited', 'edit_e2': 'ai_edited', 'edit_e3': 'ai_edited',
}

# Word counts for 150K threshold (from sessions table)
WORD_COUNTS = {
    'scotus': 76287307, 'll_research': 4427944, 'vatican': 1616356,
    'seth': 1372939, 'bashar': 1349120, 'fomc': 733961,
    'acim': 339802, 'law_of_one': 279331, 'cwg': 239580,
    'carla_prose': 196082, 'multiturn_gpt4o': 191848,
    'multiturn_claude': 163230, 'annomi': 153435,
    # Below 150K
    'xdom_gpt4o_scotus': 32171, 'xdom_gpt4o_therapy': 30985,
    'xdom_gpt4o_fomc': 30329, 'xdom_claude_therapy': 28301,
    'synth_gpt4o_g0': 27572, 'synth_gemini_g0': 27164,
    'synth_llama70b_g0': 27031, 'xdom_claude_scotus': 26994,
    'xdom_claude_fomc': 25865,
    # NULL word counts — estimate from segment counts (~50K each based on 100 sessions × ~500 words)
    'synth_g0': 50000, 'synth_g1': 50000, 'synth_g2': 50000,
    'synth_g3': 50000, 'synth_g4': 50000,
    'edit_e1': 13000, 'edit_e2': 13000, 'edit_e3': 13000,
}


# ─── Helper functions ────────────────────────────────────────────────────────

def load_vectors(conn, dims):
    """Load z-scored vectors for specified dims."""
    dim_cols = ', '.join(f'z_dim{d}' for d in dims)
    query = f"SELECT corpus_id, {dim_cols} FROM compact_vector"
    rows = conn.execute(query).fetchall()

    data = {}
    for row in rows:
        cid = row[0]
        if cid in EXCLUDE:
            continue
        vals = row[1:]
        if any(v is None for v in vals):
            continue  # skip corpora with NULL on any requested dim
        data[cid] = np.array(vals, dtype=float)

    return data


def euclidean_distance_matrix(vectors, names):
    """Compute pairwise Euclidean distance matrix."""
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((vectors[i] - vectors[j])**2))
            D[i, j] = d
            D[j, i] = d
    return D


def permanova(D, groups, n_perm=9999):
    """PERMANOVA (Anderson 2001). Returns pseudo-F, R2, p-value."""
    n = len(groups)
    unique_groups = list(set(groups))
    k = len(unique_groups)

    # Total SS
    SS_total = 0
    for i in range(n):
        for j in range(i+1, n):
            SS_total += D[i, j]**2
    SS_total /= n

    def compute_SS_within(grp_labels):
        ss_w = 0
        for g in unique_groups:
            idx = [i for i, x in enumerate(grp_labels) if x == g]
            ng = len(idx)
            if ng < 2:
                continue
            for a, b in combinations(idx, 2):
                ss_w += D[a, b]**2
            ss_w_partial = 0
            for a, b in combinations(idx, 2):
                ss_w_partial += D[a, b]**2
        # Recompute properly
        ss_w = 0
        for g in unique_groups:
            idx = [i for i, x in enumerate(grp_labels) if x == g]
            ng = len(idx)
            if ng < 1:
                continue
            for a, b in combinations(idx, 2):
                ss_w += D[a, b]**2 / ng
        return ss_w

    SS_within = compute_SS_within(groups)
    SS_between = SS_total - SS_within

    # Pseudo-F
    df_between = k - 1
    df_within = n - k
    if df_within <= 0 or df_between <= 0:
        return None, None, None
    F_obs = (SS_between / df_between) / (SS_within / df_within)
    R2 = SS_between / SS_total

    # Permutation test
    rng = np.random.default_rng(42)
    count = 1  # include observed
    for _ in range(n_perm):
        perm_groups = rng.permutation(groups).tolist()
        ss_w_perm = 0
        for g in unique_groups:
            idx = [i for i, x in enumerate(perm_groups) if x == g]
            ng = len(idx)
            if ng < 1:
                continue
            for a, b in combinations(idx, 2):
                ss_w_perm += D[a, b]**2 / ng
        ss_b_perm = SS_total - ss_w_perm
        F_perm = (ss_b_perm / df_between) / (ss_w_perm / df_within) if ss_w_perm > 0 else 0
        if F_perm >= F_obs:
            count += 1

    p_value = count / (n_perm + 1)

    return {
        'pseudo_F': round(F_obs, 4),
        'R2': round(R2, 4),
        'p_value': round(p_value, 4),
        'n_groups': k,
        'n_obs': n,
        'SS_between': round(SS_between, 4),
        'SS_within': round(SS_within, 4),
        'SS_total': round(SS_total, 4),
    }


def permdisp(D, groups, n_perm=9999):
    """
    PERMDISP (Anderson 2006) — test for homogeneity of multivariate dispersions.
    Uses distance to group centroid (via principal coordinates).
    Returns F-statistic, p-value, and per-group mean distances to centroid.
    """
    n = len(groups)
    unique_groups = sorted(set(groups))
    k = len(unique_groups)

    # Convert distance matrix to principal coordinates (PCoA)
    # Double-centering
    D2 = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort descending
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # Keep positive eigenvalues only
    pos_mask = eigenvalues > 1e-10
    eigenvalues_pos = eigenvalues[pos_mask]
    eigenvectors_pos = eigenvectors[:, pos_mask]

    # Principal coordinates
    coords = eigenvectors_pos * np.sqrt(eigenvalues_pos)

    # Compute distance to group centroid for each observation
    centroids = {}
    for g in unique_groups:
        g_idx = [i for i, x in enumerate(groups) if x == g]
        centroids[g] = np.mean(coords[g_idx], axis=0)

    distances_to_centroid = np.zeros(n)
    for i in range(n):
        g = groups[i]
        distances_to_centroid[i] = np.sqrt(np.sum((coords[i] - centroids[g])**2))

    # Per-group mean distance to centroid
    group_dispersions = {}
    for g in unique_groups:
        g_idx = [i for i, x in enumerate(groups) if x == g]
        group_dispersions[g] = {
            'mean_dist': round(float(np.mean(distances_to_centroid[g_idx])), 4),
            'median_dist': round(float(np.median(distances_to_centroid[g_idx])), 4),
            'n': len(g_idx),
        }

    # ANOVA-like F on distances to centroid
    grand_mean = np.mean(distances_to_centroid)

    SS_between = 0
    for g in unique_groups:
        g_idx = [i for i, x in enumerate(groups) if x == g]
        ng = len(g_idx)
        SS_between += ng * (np.mean(distances_to_centroid[g_idx]) - grand_mean)**2

    SS_within = 0
    for g in unique_groups:
        g_idx = [i for i, x in enumerate(groups) if x == g]
        g_mean = np.mean(distances_to_centroid[g_idx])
        for i in g_idx:
            SS_within += (distances_to_centroid[i] - g_mean)**2

    df_between = k - 1
    df_within = n - k
    if df_within <= 0 or SS_within == 0:
        return None

    F_obs = (SS_between / df_between) / (SS_within / df_within)

    # Permutation test
    rng = np.random.default_rng(42)
    count = 1
    for _ in range(n_perm):
        perm_dists = rng.permutation(distances_to_centroid)
        ss_b = 0
        ss_w = 0
        for g in unique_groups:
            g_idx = [i for i, x in enumerate(groups) if x == g]
            g_mean = np.mean(perm_dists[g_idx])
            ss_b += len(g_idx) * (g_mean - grand_mean)**2
            for i in g_idx:
                ss_w += (perm_dists[i] - g_mean)**2
        F_perm = (ss_b / df_between) / (ss_w / df_within) if ss_w > 0 else 0
        if F_perm >= F_obs:
            count += 1

    p_value = count / (n_perm + 1)

    return {
        'F_stat': round(float(F_obs), 4),
        'p_value': round(p_value, 4),
        'n_groups': k,
        'n_obs': n,
        'group_dispersions': group_dispersions,
        'interpretation': 'PASS (homogeneous)' if p_value > 0.05 else 'FAIL (heterogeneous dispersions)',
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)
    results = {}

    # ─── Analysis 1: PERMDISP on original 7-dim vector (30 corpora) ───────
    print("=" * 60)
    print("ANALYSIS 1: PERMDISP on 7-dim vector (30 corpora)")
    print("=" * 60)

    data_7d = load_vectors(conn, [1, 2, 4, 5, 6, 7, 8])
    names_7d = sorted(data_7d.keys())
    vectors_7d = np.array([data_7d[n] for n in names_7d])
    D_7d = euclidean_distance_matrix(vectors_7d, names_7d)

    # Build group labels
    protocol_binary = ['natural' if n in NATURAL_CORPORA else 'ai' for n in names_7d]
    domain_labels = [DOMAIN_MAP.get(n, 'unknown') for n in names_7d]
    protocol_fine = [PROTOCOL_FINE_MAP.get(n, 'unknown') for n in names_7d]

    print(f"\nCorpora included ({len(names_7d)}): {names_7d}")
    print(f"Protocol binary groups: {dict(zip(names_7d, protocol_binary))}")

    # PERMDISP for each grouping
    print("\n--- PERMDISP: Binary protocol ---")
    pd_binary = permdisp(D_7d, protocol_binary, N_PERM)
    print(f"  F={pd_binary['F_stat']}, p={pd_binary['p_value']}")
    print(f"  Dispersions: {pd_binary['group_dispersions']}")
    print(f"  Verdict: {pd_binary['interpretation']}")

    print("\n--- PERMDISP: Domain ---")
    pd_domain = permdisp(D_7d, domain_labels, N_PERM)
    print(f"  F={pd_domain['F_stat']}, p={pd_domain['p_value']}")
    print(f"  Dispersions: {pd_domain['group_dispersions']}")
    print(f"  Verdict: {pd_domain['interpretation']}")

    print("\n--- PERMDISP: Fine-grained protocol ---")
    pd_fine = permdisp(D_7d, protocol_fine, N_PERM)
    print(f"  F={pd_fine['F_stat']}, p={pd_fine['p_value']}")
    print(f"  Dispersions: {pd_fine['group_dispersions']}")
    print(f"  Verdict: {pd_fine['interpretation']}")

    results['permdisp_7dim'] = {
        'description': 'PERMDISP (Anderson 2006) on 7-dim z-scored vector, 30 corpora',
        'binary_protocol': pd_binary,
        'domain': pd_domain,
        'fine_protocol': pd_fine,
    }

    # ─── Analysis 2: Safe-vector PERMANOVA (dims 1,2,4,5 only) ───────────
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Safe-vector PERMANOVA (dims 1,2,4,5 only)")
    print("=" * 60)

    data_4d = load_vectors(conn, [1, 2, 4, 5])
    names_4d = sorted(data_4d.keys())
    vectors_4d = np.array([data_4d[n] for n in names_4d])
    D_4d = euclidean_distance_matrix(vectors_4d, names_4d)

    # Rebuild group labels for potentially different corpus set
    protocol_binary_4d = ['natural' if n in NATURAL_CORPORA else 'ai' for n in names_4d]
    domain_labels_4d = [DOMAIN_MAP.get(n, 'unknown') for n in names_4d]
    protocol_fine_4d = [PROTOCOL_FINE_MAP.get(n, 'unknown') for n in names_4d]

    print(f"\nCorpora included ({len(names_4d)}): {names_4d}")

    print("\n--- PERMANOVA: Binary protocol (4-dim) ---")
    perm_4d_binary = permanova(D_4d, protocol_binary_4d, N_PERM)
    print(f"  F={perm_4d_binary['pseudo_F']}, R2={perm_4d_binary['R2']}, p={perm_4d_binary['p_value']}")

    print("\n--- PERMANOVA: Domain (4-dim) ---")
    perm_4d_domain = permanova(D_4d, domain_labels_4d, N_PERM)
    print(f"  F={perm_4d_domain['pseudo_F']}, R2={perm_4d_domain['R2']}, p={perm_4d_domain['p_value']}")

    print("\n--- PERMANOVA: Fine-grained protocol (4-dim) ---")
    perm_4d_fine = permanova(D_4d, protocol_fine_4d, N_PERM)
    print(f"  F={perm_4d_fine['pseudo_F']}, R2={perm_4d_fine['R2']}, p={perm_4d_fine['p_value']}")

    # Also PERMDISP on 4-dim
    print("\n--- PERMDISP: Binary protocol (4-dim) ---")
    pd_4d_binary = permdisp(D_4d, protocol_binary_4d, N_PERM)
    print(f"  F={pd_4d_binary['F_stat']}, p={pd_4d_binary['p_value']}")
    print(f"  Verdict: {pd_4d_binary['interpretation']}")

    results['safe_vector_4dim'] = {
        'description': 'PERMANOVA + PERMDISP on 4-dim z-scored vector (dims 1,2,4,5 only — no size-confounded dims)',
        'n_corpora': len(names_4d),
        'corpora': names_4d,
        'permanova_binary': perm_4d_binary,
        'permanova_domain': perm_4d_domain,
        'permanova_fine': perm_4d_fine,
        'permdisp_binary': pd_4d_binary,
    }

    # ─── Analysis 3: PERMANOVA restricted to >= 150K word corpora ─────────
    print("\n" + "=" * 60)
    print("ANALYSIS 3: PERMANOVA on 7-dim, corpora >= 150K words only")
    print("=" * 60)

    large_corpora = {c for c, w in WORD_COUNTS.items() if w >= 150000}
    data_7d_large = {k: v for k, v in data_7d.items() if k in large_corpora}
    names_7d_large = sorted(data_7d_large.keys())

    if len(names_7d_large) >= 5:
        vectors_7d_large = np.array([data_7d_large[n] for n in names_7d_large])
        D_7d_large = euclidean_distance_matrix(vectors_7d_large, names_7d_large)

        protocol_binary_lg = ['natural' if n in NATURAL_CORPORA else 'ai' for n in names_7d_large]
        domain_labels_lg = [DOMAIN_MAP.get(n, 'unknown') for n in names_7d_large]
        protocol_fine_lg = [PROTOCOL_FINE_MAP.get(n, 'unknown') for n in names_7d_large]

        print(f"\nCorpora included ({len(names_7d_large)}): {names_7d_large}")
        n_natural = sum(1 for x in protocol_binary_lg if x == 'natural')
        n_ai = sum(1 for x in protocol_binary_lg if x == 'ai')
        print(f"  Natural: {n_natural}, AI: {n_ai}")

        print("\n--- PERMANOVA: Binary protocol (7-dim, >=150K) ---")
        perm_lg_binary = permanova(D_7d_large, protocol_binary_lg, N_PERM)
        print(f"  F={perm_lg_binary['pseudo_F']}, R2={perm_lg_binary['R2']}, p={perm_lg_binary['p_value']}")

        # Only run fine-grained if enough groups
        unique_fine = set(protocol_fine_lg)
        print(f"\n  Fine-grained protocol levels present: {unique_fine}")
        if len(unique_fine) >= 3:
            print("\n--- PERMANOVA: Fine-grained protocol (7-dim, >=150K) ---")
            perm_lg_fine = permanova(D_7d_large, protocol_fine_lg, N_PERM)
            print(f"  F={perm_lg_fine['pseudo_F']}, R2={perm_lg_fine['R2']}, p={perm_lg_fine['p_value']}")
        else:
            perm_lg_fine = {'note': f'Only {len(unique_fine)} protocol levels — too few for fine-grained test'}
            print(f"  Skipping fine-grained (only {len(unique_fine)} levels)")

        results['large_corpora_7dim'] = {
            'description': 'PERMANOVA on 7-dim z-scored vector, restricted to corpora >= 150K words',
            'threshold': '150K words',
            'n_corpora': len(names_7d_large),
            'corpora': names_7d_large,
            'permanova_binary': perm_lg_binary,
            'permanova_fine': perm_lg_fine,
        }
    else:
        print(f"  Only {len(names_7d_large)} corpora >= 150K with complete vectors — too few")
        results['large_corpora_7dim'] = {'note': 'Too few corpora with >= 150K words and complete vectors'}

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nPERMDISP (7-dim, binary protocol): F={pd_binary['F_stat']}, p={pd_binary['p_value']} → {pd_binary['interpretation']}")
    print(f"PERMDISP (7-dim, domain): F={pd_domain['F_stat']}, p={pd_domain['p_value']} → {pd_domain['interpretation']}")
    print(f"PERMDISP (7-dim, fine protocol): F={pd_fine['F_stat']}, p={pd_fine['p_value']} → {pd_fine['interpretation']}")

    print(f"\nSafe-vector PERMANOVA (4-dim, binary): R2={perm_4d_binary['R2']}, p={perm_4d_binary['p_value']}")
    print(f"Safe-vector PERMANOVA (4-dim, domain): R2={perm_4d_domain['R2']}, p={perm_4d_domain['p_value']}")
    print(f"Safe-vector PERMANOVA (4-dim, fine):   R2={perm_4d_fine['R2']}, p={perm_4d_fine['p_value']}")

    print(f"\nOriginal PERMANOVA (7-dim) for reference:")
    print(f"  Binary: R2=0.233, p=0.0002")
    print(f"  Domain: R2=0.210, p=0.024")
    print(f"  Fine:   R2=0.716, p=0.0001")

    # ─── Save ─────────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, 'permdisp_sensitivity_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    conn.close()


if __name__ == '__main__':
    main()
