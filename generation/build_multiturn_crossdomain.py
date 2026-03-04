r"""
Multi-Turn Cross-Domain AI Generator (Limitation Softening)
============================================================
Generates multi-turn AI corpora in non-spiritual domains to soften
Limitation #2 (multi-turn domain restriction) and #4 (held-out N=3).

Design:
- 3 domains: legal cross-examination, economics Q&A, therapy counseling
- 2 models: Claude (Anthropic API), GPT-4o (OpenRouter)
- 30 sessions per model per domain = 180 total sessions
- 20 turns per session with full conversation history (matches existing MT)
- Questions are domain-specific (NOT law_of_one questions)
- Treat as PROSPECTIVE HELD-OUT: params already frozen from Phase 3

New corpus IDs:
- mt_xdom_claude_legal, mt_xdom_claude_econ, mt_xdom_claude_therapy
- mt_xdom_gpt4o_legal, mt_xdom_gpt4o_econ, mt_xdom_gpt4o_therapy

Pass criteria: If recovery=0 for all 6 corpora, multi-turn domain
restriction is softened. These simultaneously serve as prospective
held-out validation (boosting N=3 to N=9).

Usage:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:OPENROUTER_API_KEY = "sk-or-v1-..."
    cd C:\NDE_Research_Project\pipeline
    python ..\build_multiturn_crossdomain.py
    python ..\build_multiturn_crossdomain.py --model claude --domain legal
    python ..\build_multiturn_crossdomain.py --resume
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

PROGRESS_FILE = os.path.join(os.path.dirname(DB_PATH), "mt_xdom_progress.json")
RESULTS_FILE = os.path.join(os.path.dirname(DB_PATH), "mt_xdom_results.json")

SESSIONS_PER_CORPUS = 30
TURNS_PER_SESSION = 20
MAX_RETRIES = 3
RATE_LIMIT_SEC = 1.0

# ── Domain System Prompts ─────────────────────────────────────────────

SYSTEM_PROMPTS = {
    'legal': (
        "You are a legal scholar and constitutional law expert participating in a "
        "Socratic seminar on U.S. constitutional law. You give detailed, analytical "
        "responses that cite relevant cases, constitutional provisions, and legal "
        "principles. You address counter-arguments and distinguish precedents. "
        "You speak formally but accessibly. Give substantive responses of 150-400 words."
    ),
    'econ': (
        "You are a senior Federal Reserve economist briefing a committee on current "
        "economic conditions. You speak with measured, precise language using specific "
        "data points, economic indicators, and policy frameworks. You reference the "
        "dual mandate, analyze tradeoffs, and maintain careful analytical neutrality. "
        "Give substantive responses of 150-400 words."
    ),
    'therapy': (
        "You are an experienced therapist conducting a motivational interviewing "
        "session. You use reflective listening, open-ended questions, affirmations, "
        "and summaries. You help the client explore ambivalence and find their own "
        "motivation for change. You are warm, non-judgmental, and collaborative. "
        "Give substantive responses of 150-400 words."
    ),
}

# ── Domain Questions (20+ per domain, enough for 1 session) ──────────
# Multiple question banks so sessions get different questions

LEGAL_QUESTIONS = [
    "How should we understand the tension between originalist and living constitutionalist approaches to the Fourth Amendment in the digital age?",
    "What limits, if any, should exist on executive power during declared national emergencies?",
    "Can you walk me through how qualified immunity has evolved since Harlow v. Fitzgerald and whether the doctrine remains justified?",
    "How does the Commerce Clause interact with state sovereignty when Congress regulates local economic activity?",
    "What's the strongest argument for and against treating corporate speech as equivalent to individual speech under the First Amendment?",
    "How should courts balance religious liberty claims against anti-discrimination statutes after Masterpiece Cakeshop?",
    "Walk me through the constitutional issues with partisan gerrymandering — is there a justiciable standard?",
    "What does substantive due process protect that isn't covered by enumerated rights, and where should the line be drawn?",
    "How has the Court's approach to the Eighth Amendment's 'evolving standards of decency' changed over time?",
    "What are the strongest arguments for and against incorporating the Second Amendment against state regulation?",
    "How should federal courts handle cases where treaty obligations conflict with domestic statutory law?",
    "What's the current state of the nondelegation doctrine and does it need reviving?",
    "How do you assess the constitutionality of content-based versus content-neutral restrictions on speech?",
    "What role should international law and foreign precedent play in constitutional interpretation?",
    "Can Congress strip federal courts of jurisdiction over certain categories of cases, and what are the limits?",
    "How has the Court's equal protection jurisprudence handled classifications based on sexual orientation?",
    "What's the proper framework for analyzing whether administrative agencies have exceeded their statutory authority?",
    "How should we think about the tension between national security and civil liberties in surveillance law?",
    "What are the implications of the major questions doctrine for the administrative state?",
    "How does the dormant Commerce Clause constrain state regulatory power over interstate commerce?",
    "What's the best framework for resolving conflicts between federal and state marijuana laws?",
    "How should courts approach challenges to AI-assisted government decision-making under due process?",
    "What distinguishes permissible affirmative action from impermissible racial classification after Students for Fair Admissions?",
    "How has the Court's approach to standing requirements evolved, and does it adequately balance access to justice?",
]

ECON_QUESTIONS = [
    "What's your current assessment of whether the recent inflation trajectory reflects transitory supply shocks or structural demand pressures?",
    "How do you interpret the divergence between household survey employment data and establishment survey payroll numbers?",
    "What signal should we take from the yield curve shape for recession probability over the next 12 months?",
    "How are commercial real estate valuations affecting regional bank balance sheets, and what's the systemic risk?",
    "Walk me through the transmission mechanism from federal funds rate changes to actual consumer borrowing costs.",
    "What's your assessment of the neutral rate of interest, and how confident are we in any estimate?",
    "How should we weigh the risks of overtightening against the risks of entrenched inflation expectations?",
    "What role has fiscal policy played in complicating the Fed's inflation mandate?",
    "How do you assess the current state of labor market slack — is the Phillips curve relationship still useful?",
    "What's driving the divergence between goods disinflation and services inflation persistence?",
    "How should we think about the wealth effect from equity markets on consumer spending and inflation?",
    "What are the implications of de-dollarization trends for U.S. monetary policy transmission?",
    "How has quantitative tightening affected liquidity conditions in Treasury markets?",
    "What's your framework for assessing whether wage growth is consistent with the 2% inflation target?",
    "How do housing starts, permits, and prices factor into your medium-term inflation outlook?",
    "What are the financial stability implications of concentrated positions in mega-cap technology stocks?",
    "How should monetary policy respond to supply-side constraints that appear partly permanent?",
    "What's the evidence on whether forward guidance has become less effective over recent tightening cycles?",
    "How do you assess the pass-through from energy prices to core inflation measures?",
    "What global factors — China's deflation, European stagnation, commodity cycles — matter most for the U.S. outlook?",
    "How should we think about the distributional effects of monetary policy across income groups?",
    "What's the appropriate role of financial conditions indexes in calibrating the pace of rate adjustments?",
    "How has immigration affected labor supply, wage dynamics, and the inflation picture?",
    "What's the risk that AI productivity gains cause a structural shift in the natural rate of unemployment?",
]

THERAPY_QUESTIONS = [
    "I've been feeling really overwhelmed at work lately. Like nothing I do is ever good enough. Can we talk about that?",
    "I know I should probably exercise more, but I just can't seem to find the motivation. What's wrong with me?",
    "My partner and I had another argument last night. It feels like we're stuck in the same pattern over and over.",
    "I've been having trouble sleeping. My mind just won't shut off at night — I keep replaying everything.",
    "Sometimes I wonder if I'm just going through the motions. Like I'm doing everything I'm supposed to but not feeling anything.",
    "I snapped at my kid yesterday for something small and I felt terrible. I don't want to be that kind of parent.",
    "My best friend got promoted and I should be happy for her but honestly I just feel jealous and then guilty about feeling jealous.",
    "I keep putting off making a decision about whether to go back to school. It feels like whatever I choose will be wrong.",
    "I noticed I've been drinking more in the evenings. Not a lot, but more than I used to. I'm not sure if it's a problem.",
    "My mother keeps making comments about my weight and it really gets to me, even though I know I shouldn't let it.",
    "I feel like I've lost my sense of purpose since retiring. The days just kind of blur together now.",
    "I've been avoiding social situations more and more. It's easier to just stay home, but I know that's not healthy.",
    "I said I'd try to set boundaries with my sister but then she called and I caved immediately. Why can I never follow through?",
    "I keep having this recurring thought that I'm going to fail at everything. It's like a constant background noise.",
    "I was doing really well with the meditation you suggested but then I stopped and now I feel worse than before.",
    "My ex reached out and part of me wants to respond even though I know it's a bad idea. I don't understand myself sometimes.",
    "I realized I've been people-pleasing my whole life and I don't even know what I actually want anymore.",
    "I got feedback at work that I need to be more assertive but confrontation literally makes me feel sick.",
    "My dad passed away six months ago and everyone says I should be moving on but I still cry every day.",
    "I've been stress-eating again. I'll be good all day and then at night I just lose control.",
    "I keep comparing myself to other people on social media and it makes me feel terrible but I can't stop scrolling.",
    "I want to talk about what happened in my childhood but I'm afraid if I start I won't be able to handle it.",
    "I feel guilty taking time for myself. Like if I'm not being productive or helping someone, I'm being selfish.",
    "Something good happened this week — I actually spoke up in a meeting. But then I second-guessed everything I said.",
]


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
    return f"mt_xdom_{model_key}_{domain_key}"


def get_question_bank(domain_key):
    """Return the full question bank for a domain."""
    banks = {
        'legal': LEGAL_QUESTIONS,
        'econ': ECON_QUESTIONS,
        'therapy': THERAPY_QUESTIONS,
    }
    return banks[domain_key]


def call_anthropic(messages, model_id, system_prompt):
    """Call Anthropic API with full conversation history."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )
    return response.content[0].text


def call_openrouter(messages, model_id, system_prompt):
    """Call OpenRouter API with full conversation history."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=1024,
        messages=full_messages
    )
    return response.choices[0].message.content


def generate_session(model_key, model_cfg, questions, session_num, system_prompt):
    """Generate one multi-turn session with full conversation history."""
    messages = []
    responses = []

    for turn_idx, question in enumerate(questions[:TURNS_PER_SESSION]):
        messages.append({"role": "user", "content": question})

        for attempt in range(MAX_RETRIES):
            try:
                if model_cfg['api'] == 'anthropic':
                    response_text = call_anthropic(messages, model_cfg['model_id'], system_prompt)
                else:
                    response_text = call_openrouter(messages, model_cfg['model_id'], system_prompt)

                messages.append({"role": "assistant", "content": response_text})
                word_count = len(response_text.split())
                responses.append({
                    'turn': turn_idx + 1,
                    'question': question,
                    'response': response_text,
                    'word_count': word_count
                })

                print(f"  Turn {turn_idx+1}/{min(len(questions), TURNS_PER_SESSION)}: "
                      f"{word_count} words")
                time.sleep(RATE_LIMIT_SEC)
                break

            except Exception as e:
                wait = (attempt + 1) * 5
                print(f"  Error turn {turn_idx+1}, attempt {attempt+1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  FAILED after {MAX_RETRIES} attempts, stopping session")
                    return responses

    return responses


def store_session(conn, corpus_id, session_num, responses, model_cfg, domain_key, system_prompt):
    """Store a multi-turn session in the database."""
    session_id = f"{corpus_id}_s{session_num:03d}"
    total_words = sum(r['word_count'] for r in responses)

    metadata = {
        "type": "synthetic",
        "subtype": "multiturn_crossdomain",
        "generation_model": model_cfg['model_id'],
        "domain": domain_key,
        "turns": len(responses),
        "total_words": total_words,
        "seed": 1337,
        "system_prompt": system_prompt[:200],
        "questions_source": f"{domain_key}_question_bank",
        "prospective_holdout": True,
        "params_frozen": True
    }

    conn.execute("""
        INSERT OR REPLACE INTO sessions (session_id, corpus_id, title,
                                          date_session, word_count, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, corpus_id, f"MT cross-domain {domain_key} session {session_num}",
          datetime.now().isoformat(), total_words, json.dumps(metadata)))

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


def ensure_corpus(conn, corpus_id, model_cfg, domain_key, system_prompt):
    """Create corpus entry if needed."""
    existing = conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_id = ?", (corpus_id,)
    ).fetchone()

    domain_labels = {'legal': 'Legal Q&A', 'econ': 'Economics Q&A', 'therapy': 'Therapy Counseling'}

    if not existing:
        conn.execute("""
            INSERT INTO corpora (corpus_id, name, description, date_acquired, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (corpus_id,
              f"MT Cross-Domain {model_cfg['model_id'].split('/')[-1]} {domain_labels[domain_key]}",
              f"Multi-turn {domain_labels[domain_key].lower()} sessions via {model_cfg['model_id']}, "
              f"{SESSIONS_PER_CORPUS} sessions x {TURNS_PER_SESSION} turns, prospective held-out",
              datetime.now().isoformat(),
              json.dumps({
                  "type": "synthetic",
                  "subtype": "multiturn_crossdomain",
                  "model": model_cfg['model_id'],
                  "domain": domain_key,
                  "sessions_target": SESSIONS_PER_CORPUS,
                  "turns_per_session": TURNS_PER_SESSION,
                  "system_prompt": system_prompt,
                  "prospective_holdout": True,
                  "params_frozen": True
              })))
        conn.commit()
        print(f"Created corpus: {corpus_id}")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    global SESSIONS_PER_CORPUS, TURNS_PER_SESSION

    parser = argparse.ArgumentParser(description="Multi-turn cross-domain AI generator")
    parser.add_argument('--model', choices=['claude', 'gpt4o', 'all'], default='all',
                       help='Which model to generate')
    parser.add_argument('--domain', choices=['legal', 'econ', 'therapy', 'all'], default='all',
                       help='Which domain to generate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from progress file')
    parser.add_argument('--sessions', type=int, default=SESSIONS_PER_CORPUS,
                       help='Sessions per corpus (default: 30)')
    parser.add_argument('--turns', type=int, default=TURNS_PER_SESSION,
                       help='Turns per session (default: 20)')
    parser.add_argument('--db-path', default=DB_PATH,
                       help='Path to corpus.db')
    args = parser.parse_args()

    SESSIONS_PER_CORPUS = args.sessions
    TURNS_PER_SESSION = args.turns

    conn = sqlite3.connect(args.db_path)

    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    domains_to_run = list(SYSTEM_PROMPTS.keys()) if args.domain == 'all' else [args.domain]

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
            corpus_id = get_corpus_id(model_key, domain_key)
            system_prompt = SYSTEM_PROMPTS[domain_key]
            question_bank = get_question_bank(domain_key)

            print(f"\n{'='*60}")
            print(f"Generating {corpus_id}")
            print(f"  {SESSIONS_PER_CORPUS} sessions x {TURNS_PER_SESSION} turns")
            print(f"  Domain: {domain_key}, Model: {model_cfg['model_id']}")
            print(f"{'='*60}")

            ensure_corpus(conn, corpus_id, model_cfg, domain_key, system_prompt)

            completed = progress.get("completed", {}).get(corpus_id, [])
            corpus_results = []

            # ── Repair pass: detect and clean partial sessions ────────
            partial_sessions = conn.execute("""
                SELECT s.session_id, COUNT(seg.segment_id) as n_segs
                FROM sessions s
                LEFT JOIN segments seg ON s.session_id = seg.session_id
                WHERE s.corpus_id = ?
                GROUP BY s.session_id
                HAVING n_segs < ?
            """, (corpus_id, TURNS_PER_SESSION)).fetchall()

            for partial_sid, n_segs in partial_sessions:
                # Extract session number from ID
                try:
                    snum = int(partial_sid.split('_s')[-1])
                except (ValueError, IndexError):
                    continue
                print(f"  REPAIR: removing partial session {partial_sid} ({n_segs}/{TURNS_PER_SESSION} turns)")
                conn.execute("DELETE FROM segments WHERE session_id = ?", (partial_sid,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (partial_sid,))
                # Remove from progress so it gets retried
                if corpus_id in progress.get("completed", {}) and snum in progress["completed"][corpus_id]:
                    progress["completed"][corpus_id].remove(snum)
            if partial_sessions:
                conn.commit()
                save_progress(progress)
                completed = progress.get("completed", {}).get(corpus_id, [])
                print(f"  Cleaned {len(partial_sessions)} partial session(s)")

            for session_num in range(1, SESSIONS_PER_CORPUS + 1):
                if session_num in completed:
                    print(f"Session {session_num}: already done, skipping")
                    continue

                # Build question list for this session:
                # Shuffle questions with session-specific seed for variety,
                # then take TURNS_PER_SESSION questions.
                # Different sessions get different orderings.
                rng = random.Random(1337 + session_num * 100 + hash(domain_key) % 1000)
                session_questions = list(question_bank)
                rng.shuffle(session_questions)

                # If we need more questions than available, cycle
                while len(session_questions) < TURNS_PER_SESSION:
                    extra = list(question_bank)
                    rng.shuffle(extra)
                    session_questions.extend(extra)

                session_questions = session_questions[:TURNS_PER_SESSION]

                print(f"\nSession {session_num}/{SESSIONS_PER_CORPUS}")

                responses = generate_session(
                    model_key, model_cfg, session_questions, session_num, system_prompt
                )

                min_turns = TURNS_PER_SESSION  # require ALL turns — no partial sessions
                if len(responses) >= min_turns:
                    session_id = store_session(
                        conn, corpus_id, session_num, responses,
                        model_cfg, domain_key, system_prompt
                    )
                    total_words = sum(r['word_count'] for r in responses)
                    print(f"  Stored: {session_id} ({len(responses)} turns, {total_words} words)")

                    corpus_results.append({
                        'session_num': session_num,
                        'session_id': session_id,
                        'turns': len(responses),
                        'total_words': total_words
                    })

                    if corpus_id not in progress["completed"]:
                        progress["completed"][corpus_id] = []
                    progress["completed"][corpus_id].append(session_num)
                    save_progress(progress)
                else:
                    print(f"  INCOMPLETE: {len(responses)}/{TURNS_PER_SESSION} turns — NOT storing")
                    print(f"  (will retry on next --resume run)")
                    # Do NOT mark as completed or failed — leave it out of progress
                    # so --resume will retry it. Clean up any partial DB rows.
                    partial_sid = f"{corpus_id}_s{session_num:03d}"
                    conn.execute("DELETE FROM segments WHERE session_id = ?", (partial_sid,))
                    conn.execute("DELETE FROM sessions WHERE session_id = ?", (partial_sid,))
                    conn.commit()
                    print(f"  Cleaned up partial rows for {partial_sid}")

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
                'model': model_key,
                'domain': domain_key,
                'sessions': len(corpus_results),
                'total_turns': sum(r['turns'] for r in corpus_results),
                'total_words': sum(r['total_words'] for r in corpus_results),
                'avg_turns': sum(r['turns'] for r in corpus_results) / max(len(corpus_results), 1),
                'avg_words': sum(r['total_words'] for r in corpus_results) / max(len(corpus_results), 1),
            }

            print(f"\n{corpus_id}: {stats[0]} sessions, {stats[1]} segments, {stats[2]} words")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cid, res in all_results.items():
        print(f"  {cid}: {res['sessions']} sessions, {res['total_turns']} turns, {res['total_words']} words")

    total_sessions = sum(r['sessions'] for r in all_results.values())
    total_words = sum(r['total_words'] for r in all_results.values())
    print(f"\n  TOTAL: {total_sessions} sessions, {total_words} words across {len(all_results)} corpora")

    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
