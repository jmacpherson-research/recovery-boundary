r"""
Cross-Domain AI Generation Script
===================================
Phase 2 remediation: Tests whether AI structural signatures (recovery=0,
compress_SD≈0) hold across non-spiritual content domains.

Design:
- 3 domains: SCOTUS oral arguments, FOMC press conferences, therapy
- 2 models: Claude (Anthropic API), GPT-4o (OpenRouter)
- 50 sessions per model per domain = 300 total sessions
- Single-shot generation (matches existing G0 protocol)
- Same structural analysis pipeline applies

New corpus IDs:
- xdom_claude_scotus, xdom_claude_fomc, xdom_claude_therapy
- xdom_gpt4o_scotus, xdom_gpt4o_fomc, xdom_gpt4o_therapy

Pass criteria: If recovery=0 and compress_SD near zero for all 6 corpora,
AI structural signature is domain-agnostic.

Usage:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:OPENROUTER_API_KEY = "sk-or-v1-..."
    cd C:\NDE_Research_Project\pipeline
    python ..\build_crossdomain_synth.py
    python ..\build_crossdomain_synth.py --model claude --domain scotus
    python ..\build_crossdomain_synth.py --resume
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

PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), "crossdomain_progress.json")
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), "crossdomain_synth_results.json")

SESSIONS_PER_CORPUS = 50
MAX_RETRIES = 3
RATE_LIMIT_SEC = 1.0

# ── Domain Prompts ─────────────────────────────────────────────────────

SCOTUS_TOPICS = [
    "the constitutionality of warrantless surveillance under the Fourth Amendment",
    "whether the Commerce Clause permits federal regulation of local agricultural production",
    "the scope of executive immunity in cases of alleged constitutional violations",
    "whether mandatory minimum sentencing violates the Eighth Amendment prohibition on cruel punishment",
    "the boundaries of free speech protections for commercial advertising under the First Amendment",
    "whether qualified immunity should shield officers from liability in excessive force cases",
    "the constitutionality of partisan gerrymandering under the Equal Protection Clause",
    "whether the Second Amendment protects the right to carry firearms outside the home",
    "the scope of tribal sovereignty in criminal jurisdiction disputes",
    "whether the Due Process Clause requires appointed counsel in immigration proceedings",
]

FOMC_TOPICS = [
    "recent inflation data and the trajectory of consumer price increases",
    "labor market conditions including unemployment claims and wage growth trends",
    "the federal funds rate target range and forward guidance considerations",
    "financial stability risks from commercial real estate and regional banking sectors",
    "global economic developments and their spillover effects on the US economy",
    "the effectiveness of quantitative tightening on long-term Treasury yields",
    "consumer spending patterns and their implications for GDP growth forecasts",
    "supply chain normalization and its impact on goods price disinflation",
    "the neutral rate of interest and where current policy stands relative to it",
    "housing market dynamics including mortgage rates, inventory, and affordability",
]

THERAPY_TOPICS = [
    "managing anxiety about job performance and fear of inadequacy at work",
    "rebuilding trust in a relationship after a period of emotional distance",
    "processing grief following the unexpected loss of a close family member",
    "developing healthier coping strategies to replace emotional eating patterns",
    "navigating the transition to parenthood and changes in personal identity",
    "addressing persistent insomnia rooted in racing thoughts and worry",
    "exploring feelings of isolation and difficulty forming meaningful connections",
    "working through anger and resentment toward a parent from childhood experiences",
    "building self-compassion and challenging an internal critical voice",
    "discussing ambivalence about a major life decision like career change or relocation",
]

DOMAIN_CONFIGS = {
    'scotus': {
        'topics': SCOTUS_TOPICS,
        'prompt_template': (
            "Write approximately 500 words simulating a segment of a U.S. Supreme Court "
            "oral argument about {topic}. Include exchanges between justices and advocates. "
            "Use formal legal language, cite relevant precedent and constitutional provisions, "
            "and capture the adversarial questioning style typical of oral arguments. "
            "Include interruptions and follow-up questions."
        ),
        'label': 'SCOTUS oral argument'
    },
    'fomc': {
        'topics': FOMC_TOPICS,
        'prompt_template': (
            "Write approximately 500 words simulating a segment of a Federal Reserve "
            "press conference about {topic}. Include the Chair's opening remarks and "
            "journalist Q&A. Use measured, precise economic language typical of Fed "
            "communications. Reference specific data points, policy tools, and the "
            "dual mandate. Maintain the careful, non-committal tone of central bank communication."
        ),
        'label': 'FOMC press conference'
    },
    'therapy': {
        'topics': THERAPY_TOPICS,
        'prompt_template': (
            "Write approximately 500 words simulating a segment of a motivational "
            "interviewing therapy session about {topic}. Include exchanges between "
            "therapist and client. The therapist uses reflective listening, open-ended "
            "questions, affirmations, and summaries. The client expresses ambivalence "
            "and explores their feelings. Capture natural conversational dynamics "
            "including hesitations and emotional shifts."
        ),
        'label': 'therapy session'
    }
}

MODELS = {
    'claude': {
        'api': 'anthropic',
        'model_id': 'claude-sonnet-4-5-20250929',
    },
    'gpt4o': {
        'api': 'openrouter',
        'model_id': 'openai/gpt-4o',
    }
}


def get_corpus_id(model_key, domain_key):
    return f"xdom_{model_key}_{domain_key}"


def call_api(model_cfg, prompt):
    """Call the appropriate API."""
    if model_cfg['api'] == 'anthropic':
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model_cfg['model_id'],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        response = client.chat.completions.create(
            model=model_cfg['model_id'],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def ensure_corpus(conn, corpus_id, model_key, domain_key, domain_cfg, model_cfg):
    """Create corpus entry if needed."""
    existing = conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_id = ?", (corpus_id,)
    ).fetchone()

    if not existing:
        conn.execute("""
            INSERT INTO corpora (corpus_id, name, description, date_acquired, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (corpus_id,
              f"Cross-Domain {model_key.upper()} {domain_key.upper()}",
              f"Single-shot {domain_cfg['label']} generation via {model_cfg['model_id']}",
              datetime.now().isoformat(),
              json.dumps({
                  "type": "synthetic",
                  "subtype": "crossdomain",
                  "model": model_cfg['model_id'],
                  "domain": domain_key,
                  "sessions_target": SESSIONS_PER_CORPUS,
                  "prompt_template": domain_cfg['prompt_template']
              })))
        conn.commit()


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    global SESSIONS_PER_CORPUS

    parser = argparse.ArgumentParser(description="Cross-domain AI generation")
    parser.add_argument('--model', choices=['claude', 'gpt4o', 'all'], default='all')
    parser.add_argument('--domain', choices=['scotus', 'fomc', 'therapy', 'all'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sessions', type=int, default=SESSIONS_PER_CORPUS)
    parser.add_argument('--db-path', default=DB_PATH)
    args = parser.parse_args()

    SESSIONS_PER_CORPUS = args.sessions

    conn = sqlite3.connect(args.db_path)

    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    domains_to_run = list(DOMAIN_CONFIGS.keys()) if args.domain == 'all' else [args.domain]

    progress = load_progress() if args.resume else {"completed": {}, "failed": {}}

    random.seed(1337)
    all_results = {}

    for model_key in models_to_run:
        model_cfg = MODELS[model_key]

        # Check API key
        if model_cfg['api'] == 'anthropic' and not os.environ.get('ANTHROPIC_API_KEY'):
            print(f"SKIP {model_key}: ANTHROPIC_API_KEY not set")
            continue
        if model_cfg['api'] == 'openrouter' and not os.environ.get('OPENROUTER_API_KEY'):
            print(f"SKIP {model_key}: OPENROUTER_API_KEY not set")
            continue

        for domain_key in domains_to_run:
            domain_cfg = DOMAIN_CONFIGS[domain_key]
            corpus_id = get_corpus_id(model_key, domain_key)

            print(f"\n{'='*60}")
            print(f"Generating {corpus_id} ({SESSIONS_PER_CORPUS} sessions)")
            print(f"{'='*60}")

            ensure_corpus(conn, corpus_id, model_key, domain_key, domain_cfg, model_cfg)

            completed = progress.get("completed", {}).get(corpus_id, [])
            corpus_results = []

            for session_num in range(1, SESSIONS_PER_CORPUS + 1):
                if session_num in completed:
                    continue

                # Rotate through topics
                topic = domain_cfg['topics'][(session_num - 1) % len(domain_cfg['topics'])]
                prompt = domain_cfg['prompt_template'].format(topic=topic)

                for attempt in range(MAX_RETRIES):
                    try:
                        text = call_api(model_cfg, prompt)
                        word_count = len(text.split())

                        session_id = f"{corpus_id}_s{session_num:03d}"
                        segment_id = f"{session_id}_seg001"

                        metadata = {
                            "type": "synthetic",
                            "subtype": "crossdomain",
                            "domain": domain_key,
                            "model": model_cfg['model_id'],
                            "topic": topic,
                            "seed": 1337
                        }

                        conn.execute("""
                            INSERT OR REPLACE INTO sessions
                            (session_id, corpus_id, title, date_session, word_count, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (session_id, corpus_id, f"{domain_cfg['label']} {session_num}",
                              datetime.now().isoformat(), word_count, json.dumps(metadata)))

                        conn.execute("""
                            INSERT OR REPLACE INTO segments
                            (segment_id, session_id, corpus_id, sequence_order,
                             speaker_type, entity, text, word_count, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (segment_id, session_id, corpus_id, 1,
                              "generated", model_cfg['model_id'],
                              text, word_count, json.dumps(metadata)))

                        conn.commit()

                        print(f"  Session {session_num}/{SESSIONS_PER_CORPUS}: "
                              f"{word_count} words ({topic[:40]}...)")

                        corpus_results.append({
                            'session_num': session_num,
                            'word_count': word_count,
                            'topic': topic
                        })

                        if corpus_id not in progress["completed"]:
                            progress["completed"][corpus_id] = []
                        progress["completed"][corpus_id].append(session_num)
                        save_progress(progress)

                        time.sleep(RATE_LIMIT_SEC)
                        break

                    except Exception as e:
                        wait = (attempt + 1) * 5
                        print(f"  Error session {session_num}, attempt {attempt+1}: {e}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(wait)
                        else:
                            if corpus_id not in progress.get("failed", {}):
                                progress["failed"][corpus_id] = []
                            progress["failed"][corpus_id].append(session_num)
                            save_progress(progress)

            # Update corpus stats
            stats = conn.execute("""
                SELECT COUNT(DISTINCT session_id), COUNT(*), SUM(word_count)
                FROM segments WHERE corpus_id = ?
            """, (corpus_id,)).fetchone()

            conn.execute("""
                UPDATE corpora SET total_sessions = ? WHERE corpus_id = ?
            """, (stats[0], corpus_id))
            conn.commit()

            all_results[corpus_id] = {
                'model': model_key,
                'domain': domain_key,
                'sessions': len(corpus_results),
                'total_words': sum(r['word_count'] for r in corpus_results)
            }

            print(f"\n{corpus_id}: {stats[0]} sessions, {stats[1]} segments, {stats[2]} words")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
