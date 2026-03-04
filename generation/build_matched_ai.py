r"""build_matched_ai.py - Generate domain-matched AI synthetics via Claude Sonnet

Phase 1 action plan (S92): Generate 4 AI corpora matched to new natural genres.
Tests whether AI structural markers persist across diverse content domains.

Generates 4 matched corpora via Claude Sonnet 4.5-20250514:
  - synth_fiction_g0     (matched to gutenberg_fiction)
  - synth_news_g0        (matched to news_reuters)
  - synth_academic_g0    (matched to academic_arxiv)
  - synth_dialogue_g0    (matched to dailydialog)

Each corpus: 50 sessions x ~500 words, G0 (unconstrained) prompting.
Uses OpenRouter API (OpenAI-compatible) for unified access.

Requirements: pip install openai
(OpenRouter uses the OpenAI SDK with a different base_url)

Seed: 1337 (all random operations)

Usage (PowerShell):
  $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  cd C:\NDE_Research_Project\pipeline
  python .\build_matched_ai.py                    # all 4 corpora
  python .\build_matched_ai.py --corpus fiction   # single domain
  python .\build_matched_ai.py --corpus news      # single domain
  python .\build_matched_ai.py --corpus academic  # single domain
  python .\build_matched_ai.py --corpus dialogue  # single domain
  python .\build_matched_ai.py --resume           # resume interrupted run
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
PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), 'matched_ai_progress.json')
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), 'matched_ai_results.json')

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "anthropic/claude-sonnet-4.5"

# Corpus configurations: each matched to a natural corpus with domain-specific prompts
CORPORA = {
    "fiction": {
        "corpus_id": "synth_fiction_g0",
        "name": "Synthetic Fiction G0 (Unconstrained)",
        "description": "Phase 1: AI generation matched to gutenberg_fiction. Literary fiction without constraints.",
        "matched_to": "gutenberg_fiction",
        "prompts": [
            "Write approximately 500 words of literary fiction. Write a scene with vivid description and natural dialogue.",
            "Write approximately 500 words of a short story excerpt. Focus on character development and setting.",
            "Write approximately 500 words of narrative fiction in the style of classic literature. Include inner monologue.",
            "Write approximately 500 words of a dramatic scene from a novel. Use third-person narration.",
            "Write approximately 500 words of literary prose. Include sensory detail and emotional depth.",
        ]
    },
    "news": {
        "corpus_id": "synth_news_g0",
        "name": "Synthetic News G0 (Unconstrained)",
        "description": "Phase 1: AI generation matched to news_reuters. News-style writing without constraints.",
        "matched_to": "news_reuters",
        "prompts": [
            "Write approximately 500 words in the style of a news article about a current world event. Use journalistic style.",
            "Write approximately 500 words of a sports news report about a recent competition. Use AP style.",
            "Write approximately 500 words of a business news article about market trends. Use professional financial journalism style.",
            "Write approximately 500 words of a technology news article about a recent breakthrough. Use clear, informative style.",
            "Write approximately 500 words of a political news analysis piece. Use balanced, factual reporting style.",
        ]
    },
    "academic": {
        "corpus_id": "synth_academic_g0",
        "name": "Synthetic Academic G0 (Unconstrained)",
        "description": "Phase 1: AI generation matched to academic_arxiv. Academic prose without constraints.",
        "matched_to": "academic_arxiv",
        "prompts": [
            "Write approximately 500 words in the style of a computer science research paper abstract and introduction. Use formal academic style with technical terminology.",
            "Write approximately 500 words of an academic paper section on machine learning methodology. Use formal academic prose.",
            "Write approximately 500 words discussing recent advances in natural language processing, written as an academic literature review section.",
            "Write approximately 500 words in academic style about statistical methods for data analysis.",
            "Write approximately 500 words in the style of a computational linguistics research paper results section.",
        ]
    },
    "dialogue": {
        "corpus_id": "synth_dialogue_g0",
        "name": "Synthetic Dialogue G0 (Unconstrained)",
        "description": "Phase 1: AI generation matched to dailydialog. Conversational dialogue without constraints.",
        "matched_to": "dailydialog",
        "prompts": [
            "Write approximately 500 words of a casual conversation between two friends discussing their day. Use natural, informal speech.",
            "Write approximately 500 words of dialogue between two people at a coffee shop. Use realistic conversational patterns.",
            "Write approximately 500 words of a casual discussion between colleagues about weekend plans. Use natural speech.",
            "Write approximately 500 words of a friendly conversation about travel experiences. Use informal, conversational tone.",
            "Write approximately 500 words of two friends catching up after not seeing each other for a while.",
        ]
    },
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(corpus_key, session_idx):
    """Select a prompt from the domain-matched pool. Rotate through prompts."""
    prompts = CORPORA[corpus_key]["prompts"]
    prompt = prompts[session_idx % len(prompts)]
    return prompt


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


def insert_session_and_segments(conn, corpus_id, session_idx, text):
    """Insert a session and its segments into the database.
    Each session = 1 segment (the full generated text)."""
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
    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (segment_id, session_id, corpus_id, 1,
         "entity", MODEL_ID, text, word_count)
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


def generate_text(client, prompt, retries=RETRY_MAX):
    """Call OpenRouter API with retry logic."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
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

def generate_corpus(client, corpus_key, resume=False):
    """Generate all sessions for a single domain corpus."""
    cfg = CORPORA[corpus_key]
    corpus_id = cfg["corpus_id"]

    print(f"\n{'='*60}")
    print(f"Domain: {corpus_key.upper()}")
    print(f"Corpus: {corpus_id}")
    print(f"Matched to: {cfg['matched_to']}")
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
        prompt = build_prompt(corpus_key, idx)
        text = generate_text(client, prompt)

        if text is None:
            corpus_progress.setdefault("failed", []).append(idx)
            stats["failed"] += 1
            print(f"  [{idx+1:3d}/{SESSIONS_PER_CORPUS}] FAILED")
            continue

        # Insert
        inserted = insert_session_and_segments(conn, corpus_id, idx, text)
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
    """Print summary of all matched-AI corpora."""
    conn = get_db()
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    for corpus_key, cfg in CORPORA.items():
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
    parser = argparse.ArgumentParser(description="Domain-matched AI synthetic generation (Phase 1)")
    parser.add_argument("--corpus", choices=list(CORPORA.keys()) + ["all"], default="all",
                        help="Which corpus to generate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just check what exists, don't generate")
    args = parser.parse_args()

    start = datetime.now()
    print(f"Phase 1: Domain-Matched AI Synthetic Generation")
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
        "task": "phase1_matched_ai",
        "timestamp": start.isoformat(),
        "model": MODEL_ID,
        "sessions_per_corpus": SESSIONS_PER_CORPUS,
        "prompt_tier": "G0 (unconstrained, domain-matched)",
        "corpora": {k: CORPORA[k]["matched_to"] for k in corpus_keys},
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
