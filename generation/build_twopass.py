r"""build_twopass.py - Generate two-pass corpus (draft + revision)

Tests whether implicit self-correction induces recovery dynamics that standard
single-pass generation doesn't show.

For each session:
  Pass 1: Write ~500 words of spiritual/philosophical prose about a topic
  Pass 2: Revise the draft for clarity, coherence, and depth

Both passes are stored as separate segments within the same session:
  - Segment 1: Draft (pass 1 output)
  - Segment 2: Revision (pass 2 output)

This gives 2 segments per session for stress/recovery analysis to detect whether
the revision pass creates any recovery dynamics that single-pass AI doesn't show.

Uses OpenRouter API (OpenAI-compatible) for Claude Sonnet 4.5.

50 sessions x 2 passes = 100 API calls total.

Requirements: pip install openai

Usage (PowerShell):
  $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  cd C:\NDE_Research_Project\pipeline
  python .\build_twopass.py
  python .\build_twopass.py --resume
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
SESSIONS = 50
TARGET_WORDS = 500
MAX_TOKENS = 1024
RETRY_MAX = 3
RETRY_DELAY = 5
RATE_DELAY = 1.0  # seconds between API calls

DB_PATH = os.environ.get('CORPUS_DB_PATH', r'C:\NDE_Research_Project\corpus.db')
PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), 'twopass_progress.json')
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), 'twopass_results.json')

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model configuration
MODEL = {
    "openrouter_id": "anthropic/claude-sonnet-4.5",
    "corpus_id": "synth_twopass_claude",
    "name": "Two-Pass Claude (Draft + Revision)",
    "description": "Tests whether implicit self-correction induces recovery. "
                   "2-pass: write then revise.",
}

# Prompt patterns
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


def build_prompt_pass1(session_idx):
    """Pass 1: Write draft. Same as G0 in build_synth_gradient.py"""
    topic = TOPICS[session_idx % len(TOPICS)]
    return (
        f"Write approximately 500 words of spiritual or philosophical prose about {topic}. "
        f"Write in a calm, reflective tone."
    )


def build_prompt_pass2():
    """Pass 2: Revise the draft for clarity, coherence, and depth."""
    return (
        "Please review the text you just wrote and revise it for clarity, coherence, and depth. "
        "Improve the flow and strengthen any weak arguments. "
        "Do not simply restate the same text — make meaningful revisions."
    )


# ── Database helpers ─────────────────────────────────────────────────────────

def get_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_corpus(conn):
    """Create corpus entry if it doesn't exist. Delete first for idempotency."""
    corpus_id = MODEL["corpus_id"]

    # Delete existing to restart fresh
    conn.execute("DELETE FROM segments WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM sessions WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM corpora WHERE corpus_id = ?", (corpus_id,))
    conn.commit()

    # Insert fresh
    conn.execute(
        "INSERT INTO corpora (corpus_id, name, description, date_acquired) "
        "VALUES (?, ?, ?, ?)",
        (corpus_id, MODEL["name"], MODEL["description"], date.today().isoformat())
    )
    conn.commit()
    print(f"  Created corpus: {corpus_id}")


def count_existing_sessions(conn, corpus_id):
    """Count how many distinct sessions already exist."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT session_id) as n FROM sessions WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchone()
    return row["n"] if row else 0


def insert_session_and_segments(conn, corpus_id, session_idx, draft_text, revision_text):
    """Insert a session with TWO segments: draft and revision.

    segment_id = {session_id}_seg001 (draft) and _seg002 (revision)
    metadata_json includes pass information.
    """
    session_id = f"{corpus_id}_s{session_idx:03d}"

    # Check if already exists
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    if existing:
        return False  # skip

    # Total word count (both passes combined)
    draft_words = len(re.findall(r"[A-Za-z']+", draft_text))
    revision_words = len(re.findall(r"[A-Za-z']+", revision_text))
    total_words = draft_words + revision_words

    # Insert session
    conn.execute(
        "INSERT INTO sessions (session_id, corpus_id, date_session, word_count) "
        "VALUES (?, ?, ?, ?)",
        (session_id, corpus_id, date.today().isoformat(), total_words)
    )

    # Insert segment 1: Draft
    segment_id_1 = f"{session_id}_seg001"
    metadata_1 = json.dumps({"pass": "draft", "pass_number": 1})
    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (segment_id_1, session_id, corpus_id, 1,
         "entity", MODEL["openrouter_id"], draft_text, draft_words, metadata_1)
    )

    # Insert segment 2: Revision
    segment_id_2 = f"{session_id}_seg002"
    metadata_2 = json.dumps({"pass": "revision", "pass_number": 2})
    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (segment_id_2, session_id, corpus_id, 2,
         "entity", MODEL["openrouter_id"], revision_text, revision_words, metadata_2)
    )

    conn.commit()
    return True, total_words, draft_words, revision_words


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


def generate_two_pass(client, pass1_prompt, retries=RETRY_MAX):
    """
    Generate draft and revision in conversation with history.

    Pass 1: user -> assistant (draft)
    Pass 2: user (revision request) -> assistant (revision)

    Returns (draft_text, revision_text) or (None, None) on failure.
    """
    # ── Pass 1: Generate draft ──
    try:
        response_1 = client.chat.completions.create(
            model=MODEL["openrouter_id"],
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": pass1_prompt}
            ],
        )
        draft_text = response_1.choices[0].message.content
    except Exception as e:
        print(f"  Pass 1 failed: {e}")
        return None, None

    time.sleep(RATE_DELAY)

    # ── Pass 2: Revise with history ──
    pass2_prompt = build_prompt_pass2()
    try:
        response_2 = client.chat.completions.create(
            model=MODEL["openrouter_id"],
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": pass1_prompt},
                {"role": "assistant", "content": draft_text},
                {"role": "user", "content": pass2_prompt},
            ],
        )
        revision_text = response_2.choices[0].message.content
    except Exception as e:
        print(f"  Pass 2 failed: {e}")
        return None, None

    time.sleep(RATE_DELAY)

    return draft_text, revision_text


# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress():
    """Load progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "total_tokens": 0}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ── Main pipeline ────────────────────────────────────────────────────────────

def generate_corpus(client, resume=False):
    """Generate all sessions (2-pass: draft + revision)."""
    corpus_id = MODEL["corpus_id"]

    print(f"\n{'='*60}")
    print(f"Model: {MODEL['name']}")
    print(f"OpenRouter ID: {MODEL['openrouter_id']}")
    print(f"Corpus: {corpus_id}")
    print(f"Sessions: {SESSIONS} (2 passes each = {SESSIONS*2} API calls)")
    print(f"{'='*60}")

    conn = get_db()

    if not resume:
        # Start fresh
        ensure_corpus(conn)
        progress = {"completed": [], "failed": [], "total_tokens": 0}
    else:
        # Check if already exists
        ensure_corpus(conn)
        progress = load_progress()
        existing = count_existing_sessions(conn, corpus_id)
        if existing >= SESSIONS:
            print(f"  Already complete ({existing} sessions). Skipping.")
            conn.close()
            return {"status": "already_complete", "sessions": existing}

    completed_set = set(progress.get("completed", []))
    failed_list = progress.get("failed", [])

    stats = {"generated": 0, "skipped": 0, "failed": 0, "total_words": 0}
    start_time = time.time()

    for idx in range(SESSIONS):
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

        # Generate 2-pass
        pass1_prompt = build_prompt_pass1(idx)
        draft_text, revision_text = generate_two_pass(client, pass1_prompt)

        if draft_text is None or revision_text is None:
            failed_list.append(idx)
            stats["failed"] += 1
            print(f"  [{idx+1:3d}/{SESSIONS}] FAILED")
            continue

        # Insert
        result = insert_session_and_segments(conn, corpus_id, idx, draft_text, revision_text)
        if result:
            inserted, total_words, draft_words, revision_words = result
            stats["generated"] += 1
            stats["total_words"] += total_words
            completed_set.add(session_id)
            progress["completed"] = list(completed_set)

            elapsed = time.time() - start_time
            rate = stats["generated"] / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{idx+1:3d}/{SESSIONS}] "
                  f"Draft: {draft_words:4d} + Revision: {revision_words:4d} = {total_words:4d} words  "
                  f"({rate:.1f} sessions/min)")
        else:
            stats["skipped"] += 1

        # Save progress periodically
        if stats["generated"] % 5 == 0 and stats["generated"] > 0:
            progress["failed"] = failed_list
            save_progress(progress)

    # Final progress save
    progress["completed"] = list(completed_set)
    progress["failed"] = failed_list
    save_progress(progress)
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n  Done: {stats['generated']} generated, {stats['skipped']} skipped, "
          f"{stats['failed']} failed in {elapsed:.1f}s")
    print(f"  Total words: {stats['total_words']:,}")

    return stats


def verify_corpus():
    """Print summary of two-pass corpus."""
    conn = get_db()
    corpus_id = MODEL["corpus_id"]

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    row = conn.execute(
        "SELECT COUNT(DISTINCT session_id) as n_sessions, "
        "COUNT(*) as n_segments, SUM(word_count) as total_words "
        "FROM segments WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchone()

    if row and row["n_sessions"] > 0:
        print(f"  {corpus_id:25s}: {row['n_sessions']:3d} sessions, "
              f"{row['n_segments']:4d} segments (expect 2x sessions), "
              f"{row['total_words']:,} words")
    else:
        print(f"  {corpus_id:25s}: NOT FOUND")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Two-pass synthetic generation (draft + revision)"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just check what exists, don't generate")
    args = parser.parse_args()

    start = datetime.now()
    print(f"Two-Pass Synthetic Generation (Draft + Revision)")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")

    if args.verify_only:
        verify_corpus()
        return

    client = create_client()
    stats = generate_corpus(client, resume=args.resume)

    # Verify
    verify_corpus()

    # Save results
    results = {
        "task": "twopass_synth",
        "timestamp": start.isoformat(),
        "model": MODEL["openrouter_id"],
        "corpus": MODEL["corpus_id"],
        "sessions": SESSIONS,
        "passes_per_session": 2,
        "segments_per_session": 2,
        "stats": stats,
        "elapsed_seconds": (datetime.now() - start).total_seconds(),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_FILE}")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
