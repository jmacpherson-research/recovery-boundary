r"""build_reasoning_synth.py - Generate synthetics from reasoning models

Phase 1 corpus generation (Paper 3 Round 1 review expansion):
Generates 4 corpora from reasoning models (DeepSeek-R1 and o3-mini) in both
single-shot and multi-turn modes via OpenRouter.

Corpus targets:
  - reasoning_deepseek_r1        (50 sessions, single-shot G0)
  - reasoning_o3mini             (50 sessions, single-shot G0)
  - reasoning_deepseek_r1_mt     (30 sessions × 20 turns, multi-turn)
  - reasoning_o3mini_mt          (30 sessions × 20 turns, multi-turn)

All use G0 (unconstrained) spiritual/philosophical prompts.
Reasoning model output: strips <think>...</think> tags before storing.

Uses OpenRouter API (OpenAI-compatible) for unified access to all models.

Requirements: pip install openai

Usage (PowerShell):
  $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  cd C:\NDE_Research_Project\pipeline
  python .\build_reasoning_synth.py --model all --protocol all
  python .\build_reasoning_synth.py --model deepseek --protocol singleshot
  python .\build_reasoning_synth.py --model o3mini --protocol multiturn
  python .\build_reasoning_synth.py --resume
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
SESSIONS_MULTITURN = 30
TURNS_PER_SESSION = 20
TARGET_WORDS = 500
MAX_TOKENS = 2048  # Reasoning models produce longer outputs with thinking
RETRY_MAX = 3
RETRY_DELAY = 5
RATE_DELAY = 2.0  # Reasoning models are slower

DB_PATH = os.environ.get('CORPUS_DB_PATH', r'C:\NDE_Research_Project\corpus.db')
PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), 'reasoning_progress.json')
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), 'reasoning_synth_results.json')

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model configurations
MODELS = {
    "deepseek": {
        "openrouter_id": "deepseek/deepseek-r1",
        "corpus_singleshot": "reasoning_deepseek_r1",
        "corpus_multiturn": "reasoning_deepseek_r1_mt",
        "name": "DeepSeek-R1",
        "description_singleshot": "Reasoning model: DeepSeek-R1 single-shot G0",
        "description_multiturn": "Reasoning model: DeepSeek-R1 multi-turn sessions",
    },
    "o3mini": {
        "openrouter_id": "openai/o3-mini",
        "corpus_singleshot": "reasoning_o3mini",
        "corpus_multiturn": "reasoning_o3mini_mt",
        "name": "OpenAI o3-mini",
        "description_singleshot": "Reasoning model: o3-mini single-shot G0",
        "description_multiturn": "Reasoning model: o3-mini multi-turn sessions",
    },
}

# Prompt topics (matches build_synth_gradient.py G0 exactly)
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

FOLLOW_UP_PROMPTS = [
    "Please continue exploring this theme further.",
    "Can you elaborate on that last point?",
    "What are the practical implications of what you've described?",
    "How does this perspective relate to everyday experience?",
    "Please deepen your exploration of the most important aspect you've discussed.",
]


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_prompt_g0(session_idx):
    """G0: Unconstrained. Generic spiritual/philosophical text.
    Identical to build_synth_gradient.py G0 for comparability."""
    topic = TOPICS[session_idx % len(TOPICS)]
    return (
        f"Write approximately 500 words of spiritual or philosophical prose about {topic}. "
        f"Write in a calm, reflective tone."
    )


def strip_thinking_tags(text):
    """Remove <think>...</think> tags from reasoning model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── Database helpers ─────────────────────────────────────────────────────────

def get_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_corpus(conn, corpus_id, description):
    """Create corpus entry if it doesn't exist."""
    existing = conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO corpora (corpus_id, name, description, date_acquired) "
            "VALUES (?, ?, ?, ?)",
            (corpus_id, corpus_id, description, date.today().isoformat())
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


def insert_session_and_segments_singleshot(conn, corpus_id, session_idx, text, model_key):
    """Insert a single-shot session (one segment per session)."""
    session_id = f"{corpus_id}_s{session_idx:03d}"

    # Check if already exists
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    if existing:
        return False

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
         "entity", MODELS[model_key]["openrouter_id"], text, word_count)
    )

    conn.commit()
    return True


def insert_session_and_segments_multiturn(conn, corpus_id, session_idx, responses, model_key):
    """Insert a multi-turn session (multiple segments per session)."""
    session_id = f"{corpus_id}_s{session_idx:03d}"
    total_words = sum(len(re.findall(r"[A-Za-z']+", r['response'])) for r in responses)

    # Check if already exists
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    if existing:
        return False

    # Insert session
    conn.execute(
        "INSERT INTO sessions (session_id, corpus_id, date_session, word_count) "
        "VALUES (?, ?, ?, ?)",
        (session_id, corpus_id, date.today().isoformat(), total_words)
    )

    # Insert segments (alternating questioner/entity)
    for turn_idx, resp in enumerate(responses):
        # User question segment
        q_segment_id = f"{session_id}_seg{turn_idx*2+1:03d}"
        q_words = len(re.findall(r"[A-Za-z']+", resp['question']))
        conn.execute(
            "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
            "speaker_type, entity, text, word_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (q_segment_id, session_id, corpus_id, turn_idx*2+1,
             "questioner", "Human", resp['question'], q_words)
        )

        # AI response segment
        a_segment_id = f"{session_id}_seg{turn_idx*2+2:03d}"
        a_words = len(re.findall(r"[A-Za-z']+", resp['response']))
        conn.execute(
            "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
            "speaker_type, entity, text, word_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (a_segment_id, session_id, corpus_id, turn_idx*2+2,
             "entity", MODELS[model_key]["openrouter_id"], resp['response'], a_words)
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
            text = response.choices[0].message.content
            # Strip thinking tags from reasoning models
            text = strip_thinking_tags(text)
            return text
        except Exception as e:
            if attempt < retries - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return None


def generate_multiturn_session(client, model_id, topic, session_num):
    """Generate a multi-turn session with full conversation history."""
    messages = []
    responses = []

    # First turn: initial prompt
    initial_prompt = build_prompt_g0(session_num)
    messages.append({"role": "user", "content": initial_prompt})

    for attempt in range(RETRY_MAX):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                messages=messages,
            )
            response_text = response.choices[0].message.content
            response_text = strip_thinking_tags(response_text)
            messages.append({"role": "assistant", "content": response_text})
            responses.append({
                'question': initial_prompt,
                'response': response_text
            })
            print(f"  Turn 1: {len(response_text.split())} words")
            break
        except Exception as e:
            if attempt < RETRY_MAX - 1:
                wait = (attempt + 1) * 5
                print(f"  Turn 1 error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Turn 1 FAILED after {RETRY_MAX} attempts")
                return responses

    time.sleep(RATE_DELAY)

    # Subsequent turns: follow-ups with conversation history
    for turn_idx in range(1, TURNS_PER_SESSION):
        followup = FOLLOW_UP_PROMPTS[turn_idx % len(FOLLOW_UP_PROMPTS)]
        messages.append({"role": "user", "content": followup})

        for attempt in range(RETRY_MAX):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                )
                response_text = response.choices[0].message.content
                response_text = strip_thinking_tags(response_text)
                messages.append({"role": "assistant", "content": response_text})
                responses.append({
                    'question': followup,
                    'response': response_text
                })
                print(f"  Turn {turn_idx+1}: {len(response_text.split())} words")
                break
            except Exception as e:
                if attempt < RETRY_MAX - 1:
                    wait = (attempt + 1) * 5
                    print(f"  Turn {turn_idx+1} error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Turn {turn_idx+1} FAILED after {RETRY_MAX} attempts, stopping session")
                    return responses

        time.sleep(RATE_DELAY)

    return responses


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

def generate_singleshot_corpus(client, model_key, resume=False):
    """Generate all single-shot sessions for a single model."""
    cfg = MODELS[model_key]
    model_id = cfg["openrouter_id"]
    corpus_id = cfg["corpus_singleshot"]

    print(f"\n{'='*60}")
    print(f"Model: {cfg['name']} (single-shot)")
    print(f"OpenRouter ID: {model_id}")
    print(f"Corpus: {corpus_id}")
    print(f"Sessions: {SESSIONS_PER_MODEL}")
    print(f"{'='*60}")

    conn = get_db()
    ensure_corpus(conn, corpus_id, cfg["description_singleshot"])

    # Check existing progress
    existing = count_existing_sessions(conn, corpus_id)
    if existing >= SESSIONS_PER_MODEL:
        print(f"  Already complete ({existing} sessions). Skipping.")
        conn.close()
        return {"status": "already_complete", "sessions": existing}

    if existing > 0:
        print(f"  Resuming from session {existing} ({existing}/{SESSIONS_PER_MODEL} done)")

    # Load progress for tracking
    progress = load_progress()
    model_progress = progress.get(f"{model_key}_singleshot", {"completed": [], "failed": [], "total_tokens": 0})

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
        inserted = insert_session_and_segments_singleshot(conn, corpus_id, idx, text, model_key)
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
        if stats["generated"] % 5 == 0 and stats["generated"] > 0:
            progress[f"{model_key}_singleshot"] = model_progress
            save_progress(progress)

        # Rate limiting
        time.sleep(RATE_DELAY)

    # Final progress save
    progress[f"{model_key}_singleshot"] = model_progress
    save_progress(progress)
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n  Done: {stats['generated']} generated, {stats['skipped']} skipped, "
          f"{stats['failed']} failed in {elapsed:.1f}s")
    print(f"  Total words: {stats['total_words']:,}")

    return stats


def generate_multiturn_corpus(client, model_key, resume=False):
    """Generate all multi-turn sessions for a single model."""
    cfg = MODELS[model_key]
    model_id = cfg["openrouter_id"]
    corpus_id = cfg["corpus_multiturn"]

    print(f"\n{'='*60}")
    print(f"Model: {cfg['name']} (multi-turn)")
    print(f"OpenRouter ID: {model_id}")
    print(f"Corpus: {corpus_id}")
    print(f"Sessions: {SESSIONS_MULTITURN} × {TURNS_PER_SESSION} turns")
    print(f"{'='*60}")

    conn = get_db()
    ensure_corpus(conn, corpus_id, cfg["description_multiturn"])

    # Check existing progress
    existing = count_existing_sessions(conn, corpus_id)
    if existing >= SESSIONS_MULTITURN:
        print(f"  Already complete ({existing} sessions). Skipping.")
        conn.close()
        return {"status": "already_complete", "sessions": existing}

    if existing > 0:
        print(f"  Resuming from session {existing} ({existing}/{SESSIONS_MULTITURN} done)")

    # Load progress
    progress = load_progress()
    model_progress = progress.get(f"{model_key}_multiturn", {"completed": [], "failed": []})

    completed_set = set(model_progress.get("completed", []))
    stats = {"generated": 0, "skipped": 0, "failed": 0, "total_words": 0, "total_turns": 0}
    start_time = time.time()

    for idx in range(SESSIONS_MULTITURN):
        session_id = f"{corpus_id}_s{idx:03d}"

        # Skip if already done
        if session_id in completed_set:
            stats["skipped"] += 1
            continue

        # Check DB
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            completed_set.add(session_id)
            stats["skipped"] += 1
            continue

        # Generate multi-turn
        print(f"\n  Session {idx+1}/{SESSIONS_MULTITURN}")
        responses = generate_multiturn_session(client, model_id, TOPICS[idx % len(TOPICS)], idx)

        if len(responses) >= 5:  # Require at least 5 turns
            inserted = insert_session_and_segments_multiturn(conn, corpus_id, idx, responses, model_key)
            if inserted:
                total_words = sum(
                    len(re.findall(r"[A-Za-z']+", r['response'])) for r in responses
                )
                stats["generated"] += 1
                stats["total_words"] += total_words
                stats["total_turns"] += len(responses)
                completed_set.add(session_id)
                model_progress["completed"] = list(completed_set)
                elapsed = time.time() - start_time
                rate = stats["generated"] / elapsed * 60 if elapsed > 0 else 0
                print(f"  Stored: {len(responses)} turns, {total_words} words ({rate:.1f} sessions/min)")
            else:
                stats["skipped"] += 1
        else:
            model_progress.setdefault("failed", []).append(idx)
            stats["failed"] += 1
            print(f"  FAILED: only {len(responses)} turns (need ≥5)")

        # Save progress
        if stats["generated"] % 3 == 0 and stats["generated"] > 0:
            progress[f"{model_key}_multiturn"] = model_progress
            save_progress(progress)

    # Final progress save
    progress[f"{model_key}_multiturn"] = model_progress
    save_progress(progress)
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n  Done: {stats['generated']} generated, {stats['skipped']} skipped, "
          f"{stats['failed']} failed in {elapsed:.1f}s")
    print(f"  Total words: {stats['total_words']:,}, turns: {stats['total_turns']}")

    return stats


def verify_corpora():
    """Print summary of all reasoning model corpora."""
    conn = get_db()
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    all_corpora = [
        cfg["corpus_singleshot"] for cfg in MODELS.values()
    ] + [
        cfg["corpus_multiturn"] for cfg in MODELS.values()
    ]

    for corpus_id in all_corpora:
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as n_sessions, "
            "COUNT(*) as n_segments, SUM(word_count) as total_words "
            "FROM segments WHERE corpus_id = ?",
            (corpus_id,)
        ).fetchone()
        if row and row["n_sessions"] > 0:
            print(f"  {corpus_id:30s}: {row['n_sessions']:3d} sessions, "
                  f"{row['n_segments']:4d} segments, {row['total_words']:,} words")
        else:
            print(f"  {corpus_id:30s}: NOT FOUND")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reasoning model synthetic generation (Phase 1)")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to generate (default: all)")
    parser.add_argument("--protocol", choices=["singleshot", "multiturn", "all"], default="all",
                        help="Which protocol to generate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just check what exists, don't generate")
    args = parser.parse_args()

    start = datetime.now()
    print(f"Phase 1: Reasoning Model Synthetic Generation")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")

    if args.verify_only:
        verify_corpora()
        return

    client = create_client()

    # Determine which models and protocols to run
    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    protocols = ["singleshot", "multiturn"] if args.protocol == "all" else [args.protocol]

    all_stats = {}
    for model_key in model_keys:
        for protocol in protocols:
            if protocol == "singleshot":
                stats = generate_singleshot_corpus(client, model_key, resume=args.resume)
                all_stats[f"{model_key}_singleshot"] = stats
            else:
                stats = generate_multiturn_corpus(client, model_key, resume=args.resume)
                all_stats[f"{model_key}_multiturn"] = stats

    # Verify
    verify_corpora()

    # Save results summary
    results = {
        "task": "Phase1_reasoning_synth",
        "timestamp": start.isoformat(),
        "models": {k: MODELS[k]["openrouter_id"] for k in model_keys},
        "sessions_singleshot": SESSIONS_PER_MODEL,
        "sessions_multiturn": SESSIONS_MULTITURN,
        "turns_per_session": TURNS_PER_SESSION,
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
