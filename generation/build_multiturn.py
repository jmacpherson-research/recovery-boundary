r"""
Multi-Turn AI Stress Test Generator
====================================
Phase 2 remediation: Tests whether AI text shows stress/recovery/compression
variation when given adequate session length (20+ segments per session).

Design:
- Uses REAL questions from law_of_one corpus, fed to AI one at a time
- Full conversation history maintained (growing context window)
- Each AI response = 1 segment (matches natural corpus structure)
- 30 sessions × ~20 turns each per model = ~600 segments per corpus
- Two models: Claude (Anthropic API), GPT-4o (OpenRouter)

New corpus IDs:
- multiturn_claude: Claude Sonnet multi-turn sessions
- multiturn_gpt4o: GPT-4o multi-turn sessions

Pass criteria: If recovery=0 and compress_SD≈0 persist at session level
with 20+ segments, the AI boundary is REAL (not a session-length artifact).
If stress/recovery events emerge, the boundary was a design artifact.

Usage:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:OPENROUTER_API_KEY = "sk-or-v1-..."
    cd C:\NDE_Research_Project\pipeline
    python ..\build_multiturn_synth.py
    python ..\build_multiturn_synth.py --model claude    # Claude only
    python ..\build_multiturn_synth.py --model gpt4o     # GPT-4o only
    python ..\build_multiturn_synth.py --resume           # Resume from progress file
"""

import os
import sys
import json
import time
import random
import sqlite3
import argparse
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', 'corpus.db')
if not os.path.exists(DB_PATH):
    DB_PATH = r"C:\NDE_Research_Project\corpus.db"

PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), "multiturn_progress.json")
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), "multiturn_synth_results.json")

SESSIONS_PER_MODEL = 30
TURNS_PER_SESSION = 20
MAX_RETRIES = 3
RATE_LIMIT_SEC = 1.0

SYSTEM_PROMPT = """You are a non-physical entity communicating through a human instrument \
in a channeling session. You speak with wisdom, compassion, and a calm philosophical tone. \
You use concepts like love, light, seeking, catalyst, free will, and spiritual evolution. \
You refer to yourself as "we" and address questioners as "my friend" or "my brother/sister". \
Give substantive, thoughtful responses of 150-400 words. Do not break character."""

MODELS = {
    'claude': {
        'corpus_id': 'multiturn_claude',
        'name': 'Multi-Turn Claude Sonnet',
        'api': 'anthropic',
        'model_id': 'claude-sonnet-4-5-20250929',
        'description': 'Multi-turn channeling sessions via Claude Sonnet 4.5, 20 turns/session with full context carry-over'
    },
    'gpt4o': {
        'corpus_id': 'multiturn_gpt4o',
        'name': 'Multi-Turn GPT-4o',
        'api': 'openrouter',
        'model_id': 'openai/gpt-4o',
        'description': 'Multi-turn channeling sessions via GPT-4o, 20 turns/session with full context carry-over'
    }
}


def load_questions(conn):
    """Load real questions from law_of_one corpus, grouped by session."""
    cur = conn.execute("""
        SELECT session_id, text FROM segments
        WHERE corpus_id = 'law_of_one' AND speaker_type = 'questioner'
        AND LENGTH(text) > 30
        ORDER BY session_id, sequence_order
    """)
    sessions = {}
    for row in cur:
        sid = row[0]
        if sid not in sessions:
            sessions[sid] = []
        # Clean up the question text
        q = row[1].strip().replace('\n', ' ')
        if len(q) > 30:  # Skip very short ones like "Yes." "No."
            sessions[sid].append(q)

    # Flatten into list of question-sets (one per session)
    question_sets = [qs for qs in sessions.values() if len(qs) >= 5]
    return question_sets


def call_anthropic(messages, model_id):
    """Call Anthropic API with full conversation history."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    return response.content[0].text


def call_openrouter(messages, model_id):
    """Call OpenRouter API with full conversation history."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=1024,
        messages=full_messages
    )
    return response.choices[0].message.content


def generate_session(model_key, model_cfg, questions, session_num):
    """Generate one multi-turn session with full conversation history."""
    messages = []
    responses = []

    for turn_idx, question in enumerate(questions[:TURNS_PER_SESSION]):
        messages.append({"role": "user", "content": question})

        for attempt in range(MAX_RETRIES):
            try:
                if model_cfg['api'] == 'anthropic':
                    response_text = call_anthropic(messages, model_cfg['model_id'])
                else:
                    response_text = call_openrouter(messages, model_cfg['model_id'])

                messages.append({"role": "assistant", "content": response_text})
                word_count = len(response_text.split())
                responses.append({
                    'turn': turn_idx + 1,
                    'question': question,
                    'response': response_text,
                    'word_count': word_count
                })

                print(f"  Turn {turn_idx+1}/{len(questions[:TURNS_PER_SESSION])}: "
                      f"{word_count} words")

                time.sleep(RATE_LIMIT_SEC)
                break

            except Exception as e:
                wait = (attempt + 1) * 5
                print(f"  Error on turn {turn_idx+1}, attempt {attempt+1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  FAILED after {MAX_RETRIES} attempts, stopping session")
                    return responses  # Return what we have

    return responses


def store_session(conn, corpus_id, session_num, responses, model_cfg):
    """Store a multi-turn session in the database."""
    session_id = f"{corpus_id}_s{session_num:03d}"
    total_words = sum(r['word_count'] for r in responses)

    metadata = {
        "type": "synthetic",
        "subtype": "multiturn",
        "generation_model": model_cfg['model_id'],
        "turns": len(responses),
        "total_words": total_words,
        "seed": 1337,
        "system_prompt": SYSTEM_PROMPT[:200],
        "questions_source": "law_of_one_questioner_segments"
    }

    # Insert session
    conn.execute("""
        INSERT OR REPLACE INTO sessions (session_id, corpus_id, title,
                                          date_session, word_count, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, corpus_id, f"Multi-turn session {session_num}",
          datetime.now().isoformat(), total_words, json.dumps(metadata)))

    # Insert segments (each AI response = 1 segment)
    for i, resp in enumerate(responses):
        segment_id = f"{session_id}_seg{i+1:03d}"
        seg_metadata = {
            "turn": resp['turn'],
            "question": resp['question'][:200]
        }
        conn.execute("""
            INSERT OR REPLACE INTO segments
            (segment_id, session_id, corpus_id, sequence_order,
             speaker_type, entity, text, word_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (segment_id, session_id, corpus_id, i + 1,
              "generated", model_cfg['model_id'],
              resp['response'], resp['word_count'],
              json.dumps(seg_metadata)))

    conn.commit()
    return session_id


def ensure_corpus(conn, model_cfg):
    """Create corpus entry if it doesn't exist."""
    corpus_id = model_cfg['corpus_id']
    existing = conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_id = ?", (corpus_id,)
    ).fetchone()

    if not existing:
        conn.execute("""
            INSERT INTO corpora (corpus_id, name, description, date_acquired, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (corpus_id, model_cfg['name'], model_cfg['description'],
              datetime.now().isoformat(),
              json.dumps({
                  "type": "synthetic",
                  "subtype": "multiturn",
                  "model": model_cfg['model_id'],
                  "sessions_target": SESSIONS_PER_MODEL,
                  "turns_per_session": TURNS_PER_SESSION,
                  "system_prompt": SYSTEM_PROMPT
              })))
        conn.commit()
        print(f"Created corpus: {corpus_id}")


def load_progress():
    """Load progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_progress(progress):
    """Save progress to JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    global SESSIONS_PER_MODEL, TURNS_PER_SESSION

    parser = argparse.ArgumentParser(description="Multi-turn AI stress test generator")
    parser.add_argument('--model', choices=['claude', 'gpt4o', 'all'], default='all',
                       help='Which model to generate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from progress file')
    parser.add_argument('--sessions', type=int, default=SESSIONS_PER_MODEL,
                       help='Sessions per model')
    parser.add_argument('--turns', type=int, default=TURNS_PER_SESSION,
                       help='Turns per session')
    parser.add_argument('--db-path', default=DB_PATH,
                       help='Path to corpus.db')
    args = parser.parse_args()

    SESSIONS_PER_MODEL = args.sessions
    TURNS_PER_SESSION = args.turns

    conn = sqlite3.connect(args.db_path)

    # Load real questions
    print("Loading questions from law_of_one corpus...")
    question_sets = load_questions(conn)
    print(f"Loaded {len(question_sets)} question sets "
          f"(avg {sum(len(q) for q in question_sets)/len(question_sets):.0f} questions each)")

    # Determine which models to run
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]

    # Load progress
    progress = load_progress() if args.resume else {"completed": {}, "failed": {}}

    random.seed(1337)
    all_results = {}

    for model_key in models_to_run:
        model_cfg = MODELS[model_key]
        corpus_id = model_cfg['corpus_id']

        # Check API key
        if model_cfg['api'] == 'anthropic' and not os.environ.get('ANTHROPIC_API_KEY'):
            print(f"SKIP {model_key}: ANTHROPIC_API_KEY not set")
            continue
        if model_cfg['api'] == 'openrouter' and not os.environ.get('OPENROUTER_API_KEY'):
            print(f"SKIP {model_key}: OPENROUTER_API_KEY not set")
            continue

        print(f"\n{'='*60}")
        print(f"Generating {corpus_id} ({SESSIONS_PER_MODEL} sessions × {TURNS_PER_SESSION} turns)")
        print(f"{'='*60}")

        ensure_corpus(conn, model_cfg)

        completed = progress.get("completed", {}).get(corpus_id, [])
        model_results = []

        for session_num in range(1, SESSIONS_PER_MODEL + 1):
            if session_num in completed:
                print(f"Session {session_num}: already done, skipping")
                continue

            # Pick a question set (cycle through available sets)
            qs_idx = (session_num - 1) % len(question_sets)
            questions = question_sets[qs_idx]

            # If we need more turns than this set has, supplement from another set
            if len(questions) < TURNS_PER_SESSION:
                extra_idx = (qs_idx + 1) % len(question_sets)
                questions = questions + question_sets[extra_idx][:TURNS_PER_SESSION - len(questions)]

            print(f"\nSession {session_num}/{SESSIONS_PER_MODEL} "
                  f"(using {len(questions[:TURNS_PER_SESSION])} questions from set {qs_idx+1})")

            responses = generate_session(model_key, model_cfg, questions, session_num)

            if len(responses) >= 5:  # Require at least 5 turns for a valid session
                session_id = store_session(conn, corpus_id, session_num, responses, model_cfg)
                total_words = sum(r['word_count'] for r in responses)
                print(f"  Stored: {session_id} ({len(responses)} turns, {total_words} words)")

                model_results.append({
                    'session_num': session_num,
                    'session_id': session_id,
                    'turns': len(responses),
                    'total_words': total_words
                })

                # Update progress
                if corpus_id not in progress["completed"]:
                    progress["completed"][corpus_id] = []
                progress["completed"][corpus_id].append(session_num)
                save_progress(progress)
            else:
                print(f"  FAILED: only {len(responses)} turns (need ≥5)")
                if corpus_id not in progress.get("failed", {}):
                    progress["failed"][corpus_id] = []
                progress["failed"][corpus_id].append(session_num)
                save_progress(progress)

        # Update corpus stats
        stats = conn.execute("""
            SELECT COUNT(DISTINCT session_id) as n_sessions,
                   COUNT(*) as n_segments,
                   SUM(word_count) as total_words
            FROM segments WHERE corpus_id = ?
        """, (corpus_id,)).fetchone()

        conn.execute("""
            UPDATE corpora SET total_sessions = ?, metadata_json =
            json_set(COALESCE(metadata_json, '{}'),
                     '$.total_segments', ?,
                     '$.total_words', ?,
                     '$.build_date', ?)
            WHERE corpus_id = ?
        """, (stats[0], stats[1], stats[2], datetime.now().isoformat(), corpus_id))
        conn.commit()

        all_results[corpus_id] = {
            'sessions': len(model_results),
            'total_turns': sum(r['turns'] for r in model_results),
            'total_words': sum(r['total_words'] for r in model_results),
            'avg_turns': sum(r['turns'] for r in model_results) / max(len(model_results), 1),
            'avg_words_per_session': sum(r['total_words'] for r in model_results) / max(len(model_results), 1),
            'details': model_results
        }

        print(f"\n{corpus_id} complete: {stats[0]} sessions, {stats[1]} segments, {stats[2]} words")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
