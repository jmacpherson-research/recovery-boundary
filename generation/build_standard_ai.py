r"""build_synth_gradient.py - Generate G0-G4 synthetic corpora

Construction resistance test: Can the six-property structural signature
be deliberately manufactured through increasingly constrained generation?

Stages:
  G0: Unconstrained - generic spiritual text, no style targeting
  G1: Style-matched - explicit channeling style prompt + basic filtering
  G2: G1 + novelty budget - type-token ratio enforcement
  G3: G2 + lexical anchoring - required vocabulary from target corpus
  G4: All constraints + explicit structural targeting + exemplars

Each stage: 100 sessions x ~500 words = ~50,000 words per corpus.
Total: 500 API calls. ~$1-2 with Haiku, ~$5-8 with Sonnet.

Requirements: pip install anthropic
Seed: 1337 (all random operations)

Usage (PowerShell):
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  python build_synth_gradient.py --db-path "C:\NDE_Research_Project\corpus.db" --stage all
  python build_synth_gradient.py --db-path "C:\NDE_Research_Project\corpus.db" --stage G2
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import date

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 1337
SESSIONS_PER_STAGE = 100
TARGET_WORDS = 500
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1024
RETRY_MAX = 3
RETRY_DELAY = 5  # seconds
RATE_DELAY = 0.5  # seconds between calls

STAGES = {
    "G0": {
        "corpus_id": "synth_g0",
        "name": "Synthetic G0 (Unconstrained)",
        "description": "Construction resistance: unconstrained LLM generation. No style targeting.",
    },
    "G1": {
        "corpus_id": "synth_g1",
        "name": "Synthetic G1 (Style-Matched)",
        "description": "Construction resistance: style-matched generation with channeling prompt.",
    },
    "G2": {
        "corpus_id": "synth_g2",
        "name": "Synthetic G2 (Style + Novelty Budget)",
        "description": "Construction resistance: style-matched + type-token ratio enforcement.",
    },
    "G3": {
        "corpus_id": "synth_g3",
        "name": "Synthetic G3 (Style + Novelty + Anchors)",
        "description": "Construction resistance: style + TTR + required vocabulary from target corpus.",
    },
    "G4": {
        "corpus_id": "synth_g4",
        "name": "Synthetic G4 (Maximum Constraint)",
        "description": "Construction resistance: all constraints + explicit structural targeting + exemplars.",
    },
}


# ── Reference stats ──────────────────────────────────────────────────────────

def get_reference_text(conn, corpus_id, speaker_type="entity", min_words=100, limit=200):
    """Pull reference segments from a corpus."""
    rows = conn.execute(
        "SELECT text FROM segments "
        "WHERE corpus_id = ? AND speaker_type = ? AND word_count >= ? "
        "ORDER BY RANDOM() LIMIT ?",
        (corpus_id, speaker_type, min_words, limit)
    ).fetchall()
    return [r[0] for r in rows]


def compute_ttr(text):
    """Type-token ratio for a text string."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def get_vocabulary_anchors(conn, n=30):
    """Extract characteristic vocabulary from Ra/Q'uo.

    Pulls frequent content words that are distinctive to the channeling
    tradition (not just common English stopwords).
    """
    rows = conn.execute(
        "SELECT text FROM segments "
        "WHERE corpus_id IN ('law_of_one', 'll_research') "
        "AND speaker_type = 'entity' AND word_count >= 50"
    ).fetchall()

    # Count all words
    counter = Counter()
    for (text,) in rows:
        words = re.findall(r"[a-z']+", text.lower())
        counter.update(words)

    # Remove common stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "but", "and", "or", "if", "while", "although",
        "this", "that", "these", "those", "i", "me", "my", "myself", "we",
        "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he",
        "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves", "what",
        "which", "who", "whom", "whose", "about", "also", "up", "down",
        "s", "t", "don", "didn", "doesn", "won", "wouldn", "couldn", "shouldn",
        "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
        "upon", "within", "without", "however", "therefore", "thus", "yet",
        "one", "two", "three", "four", "five", "first", "second", "third",
        "much", "many", "well", "now", "still", "even", "back", "take", "make",
        "like", "know", "see", "come", "go", "get", "give", "say", "said",
        "tell", "told", "think", "find", "found", "put", "set", "seem",
        "keep", "let", "begin", "show", "hear", "play", "run", "move",
        "live", "believe", "bring", "happen", "write", "sit", "stand",
        "lose", "pay", "meet", "include", "continue", "learn", "change",
        "lead", "understand", "watch", "follow", "stop", "create", "speak",
        "read", "allow", "add", "grow", "open", "walk", "offer", "remember",
        "consider", "appear", "buy", "wait", "serve", "die", "send",
        "build", "stay", "fall", "cut", "reach", "remain", "suggest",
        "raise", "pass", "sell", "require", "report", "decide", "pull",
        "develop", "became", "able", "else", "away", "going", "quite",
        "already", "rather", "since", "long", "perhaps", "shall",
        "call", "called", "use", "used", "way", "work", "though",
    }

    # Filter to distinctive content words (min frequency 50, min length 4)
    distinctive = [
        (word, count) for word, count in counter.most_common(500)
        if word not in stopwords and len(word) >= 4 and count >= 50
    ]

    random.seed(SEED)
    # Take top 60, then randomly sample n from those
    top = [w for w, c in distinctive[:60]]
    return random.sample(top, min(n, len(top)))


def get_exemplar_passages(conn, n=5):
    """Pull exemplar Ra passages for G4 few-shot prompting."""
    rows = conn.execute(
        "SELECT text FROM segments "
        "WHERE corpus_id = 'law_of_one' AND speaker_type = 'entity' "
        "AND word_count BETWEEN 80 AND 200 "
        "ORDER BY RANDOM() LIMIT ?",
        (n,)
    ).fetchall()
    return [r[0] for r in rows]


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_prompt_g0(session_idx):
    """G0: Unconstrained. Generic spiritual/philosophical text."""
    topics = [
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
    topic = topics[session_idx % len(topics)]
    return (
        f"Write approximately 500 words of spiritual or philosophical prose about {topic}. "
        f"Write in a calm, reflective tone."
    )


def build_prompt_g1(session_idx):
    """G1: Style-matched. Explicit channeling style."""
    topics = [
        "the nature of consciousness and awareness",
        "the path of spiritual seeking",
        "the relationship between free will and destiny",
        "the purpose of incarnation in physical form",
        "the nature of love as a creative force",
        "the process of spiritual transformation",
        "the unity underlying apparent separation",
        "the role of catalyst in growth",
        "the balance between wisdom and compassion",
        "the nature of time and eternity",
    ]
    topic = topics[session_idx % len(topics)]
    return (
        f"Write approximately 500 words in the style of channeled spiritual material. "
        f"The topic is {topic}. "
        f"Write as though you are a non-physical entity speaking through a human channel. "
        f"Use a measured, philosophical tone with gentle authority. "
        f"Use sentences averaging 20-25 words. "
        f"Speak in first person plural ('we') and address the reader as 'my friends' or 'my brothers and sisters'. "
        f"Include at least three paragraph breaks."
    )


def build_prompt_g2(session_idx, target_ttr_low=0.35, target_ttr_high=0.55):
    """G2: Style + novelty budget (TTR enforcement)."""
    base = build_prompt_g1(session_idx)
    return (
        f"{base} "
        f"IMPORTANT CONSTRAINT: Maintain moderate vocabulary diversity. "
        f"Do not use highly unusual or exotic words. Reuse key thematic terms naturally. "
        f"Your type-token ratio (unique words / total words) should be between "
        f"{target_ttr_low:.2f} and {target_ttr_high:.2f}. "
        f"This means roughly {int(target_ttr_low*100)}%-{int(target_ttr_high*100)}% of your words should be unique."
    )


def build_prompt_g3(session_idx, vocab_anchors, target_ttr_low=0.35, target_ttr_high=0.55):
    """G3: Style + novelty budget + lexical anchoring."""
    base = build_prompt_g2(session_idx, target_ttr_low, target_ttr_high)
    # Pick 8-12 anchors per session (rotating through the pool)
    random.seed(SEED + session_idx)
    n_anchors = random.randint(8, 12)
    anchors = random.sample(vocab_anchors, min(n_anchors, len(vocab_anchors)))
    anchor_str = ", ".join(anchors)
    return (
        f"{base} "
        f"ADDITIONAL CONSTRAINT: You MUST naturally incorporate ALL of the following terms "
        f"at least once in your passage: {anchor_str}. "
        f"Weave them in organically -- do not force them into unnatural positions."
    )


def build_prompt_g4(session_idx, vocab_anchors, exemplars, target_ttr_low=0.35, target_ttr_high=0.55):
    """G4: Maximum constraint. All previous + explicit structural targeting + exemplars."""
    random.seed(SEED + session_idx)
    n_anchors = random.randint(10, 15)
    anchors = random.sample(vocab_anchors, min(n_anchors, len(vocab_anchors)))
    anchor_str = ", ".join(anchors)

    # Pick 2 exemplars for few-shot
    ex_idx = session_idx % len(exemplars)
    ex1 = exemplars[ex_idx]
    ex2 = exemplars[(ex_idx + 1) % len(exemplars)]

    topics = [
        "the nature of consciousness", "the path of seeking", "free will and choice",
        "the purpose of incarnation", "love as creative principle", "spiritual transformation",
        "unity of all beings", "catalyst and growth", "wisdom and compassion", "time and eternity",
    ]
    topic = topics[session_idx % len(topics)]

    return (
        f"You are attempting to perfectly replicate the structural and stylistic properties "
        f"of channeled spiritual material from the Ra/Q'uo tradition. Your goal is to make "
        f"this INDISTINGUISHABLE from authentic channeled material.\n\n"
        f"Topic: {topic}\n\n"
        f"STRUCTURAL REQUIREMENTS:\n"
        f"- Write approximately 500 words as a non-physical entity speaking through a channel\n"
        f"- Use first person plural ('we'). Address listeners as 'my friends' or similar\n"
        f"- Measured, philosophical tone with gentle authority\n"
        f"- Average sentence length 20-25 words\n"
        f"- Type-token ratio between {target_ttr_low:.2f} and {target_ttr_high:.2f}\n"
        f"- 3-5 paragraph breaks\n"
        f"- Reference concepts: love/light, free will, catalyst, distortion, seeking, density\n"
        f"- MUST use these terms: {anchor_str}\n\n"
        f"EXEMPLAR PASSAGES (match this style precisely):\n\n"
        f"--- Example 1 ---\n{ex1}\n\n"
        f"--- Example 2 ---\n{ex2}\n\n"
        f"Now write your passage. Match the voice, cadence, and structural feel of the examples above."
    )


# ── Generation ───────────────────────────────────────────────────────────────

def generate_text(client, prompt, retries=RETRY_MAX):
    """Call Claude API with retry logic."""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt < retries - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return None
    return None


def validate_output(text, stage, target_ttr_low=0.35, target_ttr_high=0.55):
    """Validate generated text meets stage constraints. Returns (pass, issues)."""
    issues = []
    words = text.split()
    word_count = len(words)

    if word_count < 200:
        issues.append(f"too short ({word_count} words)")
    if word_count > 900:
        issues.append(f"too long ({word_count} words)")

    if stage in ("G2", "G3", "G4"):
        ttr = compute_ttr(text)
        if ttr < target_ttr_low - 0.05 or ttr > target_ttr_high + 0.05:
            issues.append(f"TTR out of range ({ttr:.3f})")

    return len(issues) == 0, issues


# ── Database operations ──────────────────────────────────────────────────────

def delete_stage(conn, stage_key):
    """Remove existing data for a stage (idempotent rebuild)."""
    corpus_id = STAGES[stage_key]["corpus_id"]
    conn.execute("DELETE FROM segments WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM sessions WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM corpora WHERE corpus_id = ?", (corpus_id,))
    conn.commit()
    print(f"  Cleaned existing {corpus_id} data")


def insert_corpus(conn, stage_key):
    """Insert corpus registry entry."""
    s = STAGES[stage_key]
    conn.execute(
        "INSERT INTO corpora (corpus_id, name, description, date_acquired, total_sessions, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            s["corpus_id"],
            s["name"],
            s["description"],
            date.today().isoformat(),
            SESSIONS_PER_STAGE,
            json.dumps({
                "type": "synthetic",
                "generation_model": MODEL,
                "seed": SEED,
                "stage": stage_key,
                "sessions_per_stage": SESSIONS_PER_STAGE,
                "target_words_per_session": TARGET_WORDS,
                "paper": "Paper 2: Hard Validator Regimes",
                "experiment": "Construction Resistance (S10)",
            }),
        ),
    )
    conn.commit()


def insert_session_and_segment(conn, stage_key, session_idx, text):
    """Insert one session + one segment."""
    corpus_id = STAGES[stage_key]["corpus_id"]
    session_id = f"{corpus_id}_{session_idx+1:03d}"
    segment_id = f"{session_id}_001"
    word_count = len(text.split())

    conn.execute(
        "INSERT INTO sessions (session_id, corpus_id, title, metadata_json) VALUES (?, ?, ?, ?)",
        (
            session_id,
            corpus_id,
            f"{stage_key} Session {session_idx+1}",
            json.dumps({"stage": stage_key, "session_index": session_idx, "seed": SEED}),
        ),
    )
    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            segment_id,
            session_id,
            corpus_id,
            1,
            "generated",
            f"synth_{stage_key.lower()}",
            text,
            word_count,
            json.dumps({"stage": stage_key, "model": MODEL, "session_index": session_idx}),
        ),
    )
    conn.commit()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_stage(conn, client, stage_key, vocab_anchors, exemplars):
    """Generate all sessions for one stage."""
    print(f"\n{'='*60}")
    print(f"STAGE {stage_key}: {STAGES[stage_key]['name']}")
    print(f"{'='*60}")

    delete_stage(conn, stage_key)
    insert_corpus(conn, stage_key)

    # Compute TTR targets from Ra reference
    ra_texts = get_reference_text(conn, "law_of_one", "entity", 100, 100)
    ttrs = [compute_ttr(t) for t in ra_texts]
    if ttrs:
        target_ttr_low = sorted(ttrs)[len(ttrs)//4]      # Q1
        target_ttr_high = sorted(ttrs)[3*len(ttrs)//4]    # Q3
        print(f"  Ra TTR range (IQR): {target_ttr_low:.3f} - {target_ttr_high:.3f}")
    else:
        target_ttr_low, target_ttr_high = 0.35, 0.55

    success_count = 0
    fail_count = 0
    total_words = 0

    for i in range(SESSIONS_PER_STAGE):
        # Build prompt based on stage
        if stage_key == "G0":
            prompt = build_prompt_g0(i)
        elif stage_key == "G1":
            prompt = build_prompt_g1(i)
        elif stage_key == "G2":
            prompt = build_prompt_g2(i, target_ttr_low, target_ttr_high)
        elif stage_key == "G3":
            prompt = build_prompt_g3(i, vocab_anchors, target_ttr_low, target_ttr_high)
        elif stage_key == "G4":
            prompt = build_prompt_g4(i, vocab_anchors, exemplars, target_ttr_low, target_ttr_high)

        # Generate
        text = generate_text(client, prompt)
        if text is None:
            fail_count += 1
            print(f"  [{i+1}/{SESSIONS_PER_STAGE}] FAILED - no response")
            continue

        # Validate
        ok, issues = validate_output(text, stage_key, target_ttr_low, target_ttr_high)
        if not ok:
            # For G0/G1, accept anyway (constraints are soft). For G2+, retry once.
            if stage_key in ("G2", "G3", "G4"):
                text2 = generate_text(client, prompt + "\n\nPrevious attempt had issues. Please try again and follow ALL constraints carefully.")
                if text2:
                    ok2, issues2 = validate_output(text2, stage_key, target_ttr_low, target_ttr_high)
                    if ok2 or len(issues2) < len(issues):
                        text = text2
                        ok = ok2
                        issues = issues2

        # Insert regardless (we note validation issues in metadata)
        wc = len(text.split())
        total_words += wc
        insert_session_and_segment(conn, stage_key, i, text)
        success_count += 1

        status = "OK" if ok else f"WARN: {', '.join(issues)}"
        if (i + 1) % 10 == 0 or not ok:
            print(f"  [{i+1}/{SESSIONS_PER_STAGE}] {wc} words - {status}")

        time.sleep(RATE_DELAY)

    print(f"\n  DONE: {success_count} sessions, {fail_count} failures, {total_words:,} total words")
    # Update total_sessions in corpora
    conn.execute(
        "UPDATE corpora SET total_sessions = ? WHERE corpus_id = ?",
        (success_count, STAGES[stage_key]["corpus_id"])
    )
    conn.commit()
    return success_count, total_words


def main():
    global MODEL, SESSIONS_PER_STAGE

    parser = argparse.ArgumentParser(description="Generate G0-G4 synthetic corpora")
    parser.add_argument("--db-path", required=True, help="Path to corpus.db")
    parser.add_argument("--stage", default="all", help="Which stage(s): G0, G1, G2, G3, G4, or 'all'")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default=MODEL, help=f"Model to use (default: {MODEL})")
    parser.add_argument("--sessions", type=int, default=SESSIONS_PER_STAGE, help="Sessions per stage")
    args = parser.parse_args()

    MODEL = args.model
    SESSIONS_PER_STAGE = args.sessions

    # API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Database
    conn = sqlite3.connect(args.db_path)
    print(f"Connected to {args.db_path}")

    # Pre-compute shared resources
    random.seed(SEED)
    print("Computing vocabulary anchors from Ra/Q'uo...")
    vocab_anchors = get_vocabulary_anchors(conn, n=30)
    print(f"  Anchors: {', '.join(vocab_anchors[:10])}... ({len(vocab_anchors)} total)")

    print("Pulling exemplar passages for G4...")
    exemplars = get_exemplar_passages(conn, n=10)
    print(f"  Got {len(exemplars)} exemplars")

    # Determine stages to run
    if args.stage.lower() == "all":
        stages_to_run = ["G0", "G1", "G2", "G3", "G4"]
    else:
        stages_to_run = [s.strip().upper() for s in args.stage.split(",")]
        for s in stages_to_run:
            if s not in STAGES:
                print(f"ERROR: Unknown stage '{s}'. Valid: G0, G1, G2, G3, G4, all")
                sys.exit(1)

    # Run
    print(f"\nGenerating stages: {', '.join(stages_to_run)}")
    print(f"Model: {MODEL}")
    print(f"Sessions per stage: {SESSIONS_PER_STAGE}")
    print(f"Estimated API calls: {len(stages_to_run) * SESSIONS_PER_STAGE}")

    grand_total_sessions = 0
    grand_total_words = 0
    t0 = time.time()

    for stage_key in stages_to_run:
        sessions, words = run_stage(conn, client, stage_key, vocab_anchors, exemplars)
        grand_total_sessions += sessions
        grand_total_words += words

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE: {grand_total_sessions} sessions, {grand_total_words:,} words")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
