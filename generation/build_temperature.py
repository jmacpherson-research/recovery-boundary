r"""build_temp_sweep.py - Generate AI text at different temperatures

Test whether recovery=0 finding is temperature-dependent. Generates synthetics
at T=0.2 (deterministic) and T=1.5 (high variability) alongside existing
T=1.0 baselines (synth_g0, synth_gpt4o_g0).

Uses OpenRouter API (OpenAI-compatible) for unified access to Claude and GPT-4o.

Corpora generated:
  - synth_claude_t02     (Claude Sonnet 4.5 @ T=0.2)
  - synth_claude_t15     (Claude Sonnet 4.5 @ T=1.5)
  - synth_gpt4o_t02      (GPT-4o @ T=0.2)
  - synth_gpt4o_t15      (GPT-4o @ T=1.5)

NOTE: Existing baselines (T=1.0):
  - synth_g0 (Claude)
  - synth_gpt4o_g0 (GPT-4o)

Requirements: pip install openai
(OpenRouter uses the OpenAI SDK with a different base_url)

Seed: 1337 (all random operations)

Usage (PowerShell):
  $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  cd C:\NDE_Research_Project\pipeline
  python .\build_temp_sweep.py --corpus all
  python .\build_temp_sweep.py --corpus claude_t02
  python .\build_temp_sweep.py --resume
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
SESSIONS_PER_CORPUS = 50
TARGET_WORDS = 500
MAX_TOKENS = 1024
RETRY_MAX = 3
RETRY_DELAY = 5
RATE_DELAY = 1.0  # seconds between calls (be polite to OpenRouter)

DB_PATH = os.environ.get('CORPUS_DB_PATH', r'C:\NDE_Research_Project\corpus.db')
PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), 'temp_sweep_progress.json')
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), 'temp_sweep_results.json')

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Corpus configurations (model × temperature)
CORPORA = {
    "claude_t02": {
        "openrouter_id": "anthropic/claude-sonnet-4.5",
        "corpus_id": "synth_claude_t02",
        "temperature": 0.2,
        "name": "Synthetic Claude Sonnet 4.5 T=0.2 (Low Variability)",
        "description": "Temperature sweep: Claude @ T=0.2 (deterministic). Tests recovery temperature-dependency.",
        "family": "anthropic",
    },
    "claude_t15": {
        "openrouter_id": "anthropic/claude-sonnet-4.5",
        "corpus_id": "synth_claude_t15",
        "temperature": 1.5,
        "name": "Synthetic Claude Sonnet 4.5 T=1.5 (High Variability)",
        "description": "Temperature sweep: Claude @ T=1.5 (high variance). Tests recovery temperature-dependency.",
        "family": "anthropic",
    },
    "gpt4o_t02": {
        "openrouter_id": "openai/gpt-4o",
        "corpus_id": "synth_gpt4o_t02",
        "temperature": 0.2,
        "name": "Synthetic GPT-4o T=0.2 (Low Variability)",
        "description": "Temperature sweep: GPT-4o @ T=0.2 (deterministic). Tests recovery temperature-dependency.",
        "family": "openai",
    },
    "gpt4o_t15": {
        "openrouter_id": "openai/gpt-4o",
        "corpus_id": "synth_gpt4o_t15",
        "temperature": 1.5,
        "name": "Synthetic GPT-4o T=1.5 (High Variability)",
        "description": "Temperature sweep: GPT-4o @ T=1.5 (high variance). Tests recovery temperature-dependency.",
        "family": "openai",
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


def ensure_corpus(conn, corpus_key):
    """Create corpus entry if it doesn't exist."""
    cfg = CORPORA[corpus_key]
    corpus_id = cfg["corpus_id"]

    # Delete existing corpus for idempotency
    conn.execute("DELETE FROM segments WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM sessions WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM corpora WHERE corpus_id = ?", (corpus_id,))
    conn.commit()

    # Create fresh corpus entry
    conn.execute(
        "INSERT INTO corpora (corpus_id, name, description, date_acquired) "
        "VALUES (?, ?, ?, ?)",
        (corpus_id, cfg["name"], cfg["description"], date.today().isoformat())
    )
    conn.commit()
    print(f"  Created corpus: {corpus_id}")
    return corpus_id


def count_existing_sessions(conn, corpus_id):
    """Count how many sessions already exist for this corpus."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT session_id) as n FROM sessions WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchone()
    return row["n"] if row else 0


def insert_session_and_segments(conn, corpus_id, session_idx, text, corpus_key, temperature):
    """Insert a session and its segments into the database.
    Each session = 1 segment (the full generated text).
    Store temperature in metadata_json."""
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

    # Insert segment (single segment per session)
    segment_id = f"{session_id}_seg001"
    model_id = CORPORA[corpus_key]["openrouter_id"]
    metadata = {"temperature": temperature}
    metadata_json = json.dumps(metadata)

    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (segment_id, session_id, corpus_id, 1,
         "entity", model_id, text, word_count, metadata_json)
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


def generate_text(client, model_id, prompt, temperature, retries=RETRY_MAX):
    """Call OpenRouter API with retry logic and temperature parameter."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=temperature,  # KEY PARAMETER
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

def generate_corpus(client, corpus_key, resume=False):
    """Generate all sessions for a single corpus (model × temperature)."""
    cfg = CORPORA[corpus_key]
    model_id = cfg["openrouter_id"]
    corpus_id = cfg["corpus_id"]
    temperature = cfg["temperature"]

    print(f"\n{'='*60}")
    print(f"Corpus: {cfg['name']}")
    print(f"OpenRouter ID: {model_id}")
    print(f"Temperature: {temperature}")
    print(f"Corpus ID: {corpus_id}")
    print(f"Sessions: {SESSIONS_PER_CORPUS}")
    print(f"{'='*60}")

    conn = get_db()
    ensure_corpus(conn, corpus_key)

    # Check existing progress
    existing = count_existing_sessions(conn, corpus_id)
    if existing >= SESSIONS_PER_CORPUS:
        print(f"  Already complete ({existing} sessions). Skipping.")
        conn.close()
        return {"status": "already_complete", "sessions": existing}

    if existing > 0:
        print(f"  Resuming from session {existing} ({existing}/{SESSIONS_PER_CORPUS} done)")

    # Load progress for timing stats
    progress = load_progress()
    corpus_progress = progress.get(corpus_key, {"completed": [], "failed": [], "total_tokens": 0})

    completed_set = set(corpus_progress.get("completed", []))
    stats = {"generated": 0, "skipped": 0, "failed": 0, "total_words": 0}
    start_time = time.time()

    for idx in range(SESSIONS_PER_CORPUS):
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
        text = generate_text(client, model_id, prompt, temperature)

        if text is None:
            corpus_progress.setdefault("failed", []).append(idx)
            stats["failed"] += 1
            print(f"  [{idx+1:3d}/{SESSIONS_PER_CORPUS}] FAILED")
            continue

        # Insert
        inserted = insert_session_and_segments(conn, corpus_id, idx, text, corpus_key, temperature)
        if inserted:
            words = len(re.findall(r"[A-Za-z']+", text))
            stats["generated"] += 1
            stats["total_words"] += words
            completed_set.add(session_id)
            corpus_progress["completed"] = list(completed_set)
            elapsed = time.time() - start_time
            rate = stats["generated"] / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{idx+1:3d}/{SESSIONS_PER_CORPUS}] {words:4d} words  ({rate:.1f} sessions/min)")
        else:
            stats["skipped"] += 1

        # Save progress periodically
        if stats["generated"] % 10 == 0 and stats["generated"] > 0:
            progress[corpus_key] = corpus_progress
            save_progress(progress)

        # Rate limiting
        time.sleep(RATE_DELAY)

    # Final progress save
    progress[corpus_key] = corpus_progress
    save_progress(progress)
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n  Done: {stats['generated']} generated, {stats['skipped']} skipped, "
          f"{stats['failed']} failed in {elapsed:.1f}s")
    print(f"  Total words: {stats['total_words']:,}")

    return stats


def verify_corpora():
    """Print summary of all temperature-sweep corpora."""
    conn = get_db()
    print(f"\n{'='*60}")
    print("TEMPERATURE SWEEP VERIFICATION")
    print(f"{'='*60}")

    # Also show existing T=1.0 baselines for comparison
    all_corpora = {
        "synth_claude_t02": "Claude T=0.2",
        "synth_claude_t15": "Claude T=1.5",
        "synth_gpt4o_t02": "GPT-4o T=0.2",
        "synth_gpt4o_t15": "GPT-4o T=1.5",
        "synth_g0": "Claude T=1.0 (baseline)",
        "synth_gpt4o_g0": "GPT-4o T=1.0 (baseline)",
    }

    for corpus_id, label in all_corpora.items():
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as n_sessions, "
            "COUNT(*) as n_segments, SUM(word_count) as total_words "
            "FROM segments WHERE corpus_id = ?",
            (corpus_id,)
        ).fetchone()
        if row and row["n_sessions"] > 0:
            print(f"  {label:30s}: {row['n_sessions']:3d} sessions, "
                  f"{row['n_segments']:4d} segments, {row['total_words']:,} words")
        else:
            print(f"  {label:30s}: NOT FOUND")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Temperature sweep synthetic generation")
    parser.add_argument("--corpus", choices=list(CORPORA.keys()) + ["all"], default="all",
                        help="Which corpus to generate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just check what exists, don't generate")
    args = parser.parse_args()

    start = datetime.now()
    print(f"Temperature Sweep: AI Recovery @ T=0.2, 1.0, 1.5")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")

    if args.verify_only:
        verify_corpora()
        return

    client = create_client()

    # Determine which corpora to run
    if args.corpus == "all":
        corpus_keys = list(CORPORA.keys())
    else:
        corpus_keys = [args.corpus]

    all_stats = {}
    for corpus_key in corpus_keys:
        stats = generate_corpus(client, corpus_key, resume=args.resume)
        all_stats[corpus_key] = stats

    # Verify
    verify_corpora()

    # Save results summary
    results = {
        "task": "temperature_sweep",
        "timestamp": start.isoformat(),
        "corpora": {k: {"model": CORPORA[k]["openrouter_id"], "temperature": CORPORA[k]["temperature"]}
                    for k in corpus_keys},
        "sessions_per_corpus": SESSIONS_PER_CORPUS,
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
