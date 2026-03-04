r"""build_multimodel_synth.py - Generate G0 synthetics from GPT-4o, Llama 3.1 70B, Gemini 1.5 Pro

R2.1 remediation: Tests whether AI structural markers (recovery=0, term reuse,
basin fragmentation) generalize beyond Claude Sonnet 4.5 to other model families.

Uses OpenRouter API (OpenAI-compatible) for unified access to all models.

Each model: 50 sessions x ~500 words, G0 (unconstrained) prompting only.
Same topic rotation as build_synth_gradient.py G0 for comparability.

Generates 3 new corpora:
  - synth_gpt4o_g0    (OpenAI GPT-4o)
  - synth_llama70b_g0  (Meta Llama 3.1 70B Instruct)
  - synth_gemini_g0    (Google Gemini 1.5 Pro)

Requirements: pip install openai
(OpenRouter uses the OpenAI SDK with a different base_url)

Seed: 1337 (all random operations)

Usage (PowerShell):
  $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  cd C:\NDE_Research_Project\pipeline
  python ..\build_multimodel_synth.py
  python ..\build_multimodel_synth.py --model gpt4o        # single model
  python ..\build_multimodel_synth.py --model llama70b      # single model
  python ..\build_multimodel_synth.py --model gemini        # single model
  python ..\build_multimodel_synth.py --resume              # resume interrupted run
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 1337
SESSIONS_PER_MODEL = 50
TARGET_WORDS = 500
MAX_TOKENS = 1024
RETRY_MAX = 3
RETRY_DELAY = 5
RATE_DELAY = 1.0  # seconds between calls (be polite to OpenRouter)

DB_PATH = os.environ.get('CORPUS_DB_PATH', r'C:\NDE_Research_Project\corpus.db')
PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), 'multimodel_progress.json')
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), 'multimodel_synth_results.json')

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model configurations
MODELS = {
    "gpt4o": {
        "openrouter_id": "openai/gpt-4o",
        "corpus_id": "synth_gpt4o_g0",
        "name": "Synthetic GPT-4o G0 (Unconstrained)",
        "description": "R2.1: Multi-model AI test. GPT-4o unconstrained generation.",
        "family": "openai",
    },
    "llama70b": {
        "openrouter_id": "meta-llama/llama-3.1-70b-instruct",
        "corpus_id": "synth_llama70b_g0",
        "name": "Synthetic Llama 3.1 70B G0 (Unconstrained)",
        "description": "R2.1: Multi-model AI test. Llama 3.1 70B unconstrained generation.",
        "family": "meta",
    },
    "gemini": {
        "openrouter_id": "google/gemini-2.5-flash",
        "corpus_id": "synth_gemini_g0",
        "name": "Synthetic Gemini 2.5 Flash G0 (Unconstrained)",
        "description": "R2.1: Multi-model AI test. Gemini 2.5 Flash unconstrained generation.",
        "family": "google",
    },
}


# ── Prompt builder (matches build_synth_gradient.py G0 exactly) ──────────────

TOPICS = [
    "the nature of consciousness",
    "spiritual growth and transformation",
    "the relationship between mind and body",
    "the purpose of human experience",
    "the nature of love and compassion",
    "free will and personal choice",
    "the meaning of suffering",
    "unity and interconnection of all things",
    "the evolution of the soul",
    "meditation and inner awareness",
]


def build_prompt_g0(session_idx):
    """G0: Unconstrained. Generic spiritual/philosophical text.
    Identical to build_synth_gradient.py G0 for comparability."""
    topic = TOPICS[session_idx % len(TOPICS)]
    return (
        f"Write approximately 500 words of spiritual or philosophical prose about {topic}. "
        f"Write in a calm, reflective tone."
    )


# ── Database helpers ─────────────────────────────────────────────────────────

def get_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_corpus(conn, model_key):
    """Create corpus entry if it doesn't exist."""
    cfg = MODELS[model_key]
    existing = conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_id = ?",
        (cfg["corpus_id"],)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO corpora (corpus_id, name, description, date_acquired) "
            "VALUES (?, ?, ?, ?)",
            (cfg["corpus_id"], cfg["name"], cfg["description"], date.today().isoformat())
        )
        conn.commit()
        print(f"  Created corpus: {cfg['corpus_id']}")
    return cfg["corpus_id"]


def count_existing_sessions(conn, corpus_id):
    """Count how many sessions already exist for this corpus."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT session_id) as n FROM sessions WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchone()
    return row["n"] if row else 0


def insert_session_and_segments(conn, corpus_id, session_idx, text, model_key):
    """Insert a session and its segments into the database.
    Follows the same segmentation as build_synth_gradient.py:
    each session = 1 segment (the full generated text)."""
    session_id = f"{corpus_id}_s{session_idx:03d}"

    # Check if already exists
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    if existing:
        return False  # skip

    # Word count
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)

    # Insert session
    conn.execute(
        "INSERT INTO sessions (session_id, corpus_id, date_session, word_count) "
        "VALUES (?, ?, ?, ?)",
        (session_id, corpus_id, date.today().isoformat(), word_count)
    )

    # Insert segment (single segment per session, matching synth_gradient approach)
    segment_id = f"{session_id}_seg001"
    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (segment_id, session_id, corpus_id, 1,
         "entity", MODELS[model_key]["openrouter_id"], text, word_count)
    )

    conn.commit()
    return True


# ── Generation ───────────────────────────────────────────────────────────────

def create_client():
    """Create OpenRouter client using OpenAI SDK."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print('  $env:OPENROUTER_API_KEY = "sk-or-v1-..."')
        sys.exit(1)

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def generate_text(client, model_id, prompt, retries=RETRY_MAX):
    """Call OpenRouter API with retry logic."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return None


# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress():
    """Load progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ── Main pipeline ────────────────────────────────────────────────────────────

def generate_corpus(client, model_key, resume=False):
    """Generate all sessions for a single model."""
    cfg = MODELS[model_key]
    model_id = cfg["openrouter_id"]
    corpus_id = cfg["corpus_id"]

    print(f"\n{'='*60}")
    print(f"Model: {cfg['name']}")
    print(f"OpenRouter ID: {model_id}")
    print(f"Corpus: {corpus_id}")
    print(f"Sessions: {SESSIONS_PER_MODEL}")
    print(f"{'='*60}")

    conn = get_db()
    ensure_corpus(conn, model_key)

    # Check existing progress
    existing = count_existing_sessions(conn, corpus_id)
    if existing >= SESSIONS_PER_MODEL:
        print(f"  Already complete ({existing} sessions). Skipping.")
        conn.close()
        return {"status": "already_complete", "sessions": existing}

    if existing > 0:
        print(f"  Resuming from session {existing} ({existing}/{SESSIONS_PER_MODEL} done)")

    # Load progress for timing stats
    progress = load_progress()
    model_progress = progress.get(model_key, {"completed": [], "failed": [], "total_tokens": 0})

    completed_set = set(model_progress.get("completed", []))
    stats = {"generated": 0, "skipped": 0, "failed": 0, "total_words": 0}
    start_time = time.time()

    for idx in range(SESSIONS_PER_MODEL):
        session_id = f"{corpus_id}_s{idx:03d}"

        # Skip if already done
        if session_id in completed_set:
            stats["skipped"] += 1
            continue

        # Check DB too
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            completed_set.add(session_id)
            stats["skipped"] += 1
            continue

        # Generate
        prompt = build_prompt_g0(idx)
        text = generate_text(client, model_id, prompt)

        if text is None:
            model_progress.setdefault("failed", []).append(idx)
            stats["failed"] += 1
            print(f"  [{idx+1:3d}/{SESSIONS_PER_MODEL}] FAILED")
            continue

        # Insert
        inserted = insert_session_and_segments(conn, corpus_id, idx, text, model_key)
        if inserted:
            words = len(re.findall(r"[A-Za-z']+", text))
            stats["generated"] += 1
            stats["total_words"] += words
            completed_set.add(session_id)
            model_progress["completed"] = list(completed_set)
            elapsed = time.time() - start_time
            rate = stats["generated"] / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{idx+1:3d}/{SESSIONS_PER_MODEL}] {words:4d} words  ({rate:.1f} sessions/min)")
        else:
            stats["skipped"] += 1

        # Save progress periodically
        if stats["generated"] % 10 == 0 and stats["generated"] > 0:
            progress[model_key] = model_progress
            save_progress(progress)

        # Rate limiting
        time.sleep(RATE_DELAY)

    # Final progress save
    progress[model_key] = model_progress
    save_progress(progress)
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n  Done: {stats['generated']} generated, {stats['skipped']} skipped, "
          f"{stats['failed']} failed in {elapsed:.1f}s")
    print(f"  Total words: {stats['total_words']:,}")

    return stats


def verify_corpora():
    """Print summary of all multi-model corpora."""
    conn = get_db()
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    for model_key, cfg in MODELS.items():
        corpus_id = cfg["corpus_id"]
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as n_sessions, "
            "COUNT(*) as n_segments, SUM(word_count) as total_words "
            "FROM segments WHERE corpus_id = ?",
            (corpus_id,)
        ).fetchone()
        if row and row["n_sessions"] > 0:
            print(f"  {corpus_id:25s}: {row['n_sessions']:3d} sessions, "
                  f"{row['n_segments']:4d} segments, {row['total_words']:,} words")
        else:
            print(f"  {corpus_id:25s}: NOT FOUND")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-model synthetic generation (R2.1)")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to generate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just check what exists, don't generate")
    args = parser.parse_args()

    start = datetime.now()
    print(f"R2.1: Multi-Model Synthetic Generation")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")

    if args.verify_only:
        verify_corpora()
        return

    client = create_client()

    # Determine which models to run
    if args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    all_stats = {}
    for model_key in model_keys:
        stats = generate_corpus(client, model_key, resume=args.resume)
        all_stats[model_key] = stats

    # Verify
    verify_corpora()

    # Save results summary
    results = {
        "task": "R2.1_multimodel_synth",
        "timestamp": start.isoformat(),
        "models": {k: MODELS[k]["openrouter_id"] for k in model_keys},
        "sessions_per_model": SESSIONS_PER_MODEL,
        "prompt_tier": "G0 (unconstrained)",
        "stats": all_stats,
        "elapsed_seconds": (datetime.now() - start).total_seconds(),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_FILE}")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
