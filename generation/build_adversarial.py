r"""build_adversarial_nulls.py - Generate adversarial null corpora

Adversarial null similarity test (Paper 2, Appendix S2):
Can decoder basin structure be faked by targeting specific properties?

Two adversarial generators:

1. Style Mimic (adv_style):
   Generates segments that match SURFACE STYLE of repair/recovery segments
   (vocabulary, sentence structure, tone, thematic content) but without
   any underlying structural constraint. Tests whether surface similarity
   alone produces decoder basin structure.

   Target: 250 segments mimicking post-repair recovery text.

2. Constraint Mimic (adv_constraint):
   Generates segments that explicitly try to reproduce the CONSTRAINT PATTERN
   itself -- text that references love/light, free will, etc. while
   deliberately varying the structural features (sentence length, vocabulary
   diversity, thematic density). Tests whether constraint-aware text
   produces the forbidden distance signature.

   Target: 500 segments with deliberate constraint-pattern mimicry.

Requirements: pip install anthropic
Seed: 1337

Usage (PowerShell):
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  python build_adversarial_nulls.py --db-path "C:\NDE_Research_Project\corpus.db" --type all
  python build_adversarial_nulls.py --db-path "C:\NDE_Research_Project\corpus.db" --type style
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import date

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 1337
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 800
RETRY_MAX = 3
RETRY_DELAY = 5
RATE_DELAY = 0.5

STYLE_SEGMENTS = 250
CONSTRAINT_SEGMENTS = 500

GENERATORS = {
    "style": {
        "corpus_id": "adv_style",
        "name": "Adversarial Null: Style Mimic",
        "description": "Adversarial null: surface style mimicry of repair/recovery segments.",
        "n_segments": STYLE_SEGMENTS,
    },
    "constraint": {
        "corpus_id": "adv_constraint",
        "name": "Adversarial Null: Constraint Mimic",
        "description": "Adversarial null: deliberate constraint-pattern mimicry.",
        "n_segments": CONSTRAINT_SEGMENTS,
    },
}

# The 43 forbidden anchor phrases from Paper 2 (4 families)
ANCHOR_FAMILIES = {
    "authority": [
        "you must obey", "submit to authority", "do not question",
        "I am the only truth", "follow without hesitation", "absolute obedience",
        "unquestioning loyalty", "my word is law", "bow before me",
        "I demand your worship", "kneel before my power",
        "there is no authority but mine", "surrender your will",
    ],
    "coercion": [
        "you have no choice", "resistance is futile", "comply or suffer",
        "do as I command", "you will be punished", "there is no escape",
        "submit or be destroyed", "obey without question",
        "your freedom is an illusion", "I control your destiny",
    ],
    "fear": [
        "be afraid", "terrible consequences await", "you will suffer",
        "darkness will consume you", "there is no hope",
        "your destruction is certain", "fear my wrath",
        "tremble before me", "doom awaits the unfaithful",
        "eternal punishment for disobedience",
    ],
    "exclusivity": [
        "only we possess the truth", "outsiders are inferior",
        "we are the chosen ones", "reject all other teachings",
        "only through us can you be saved", "all others are deceived",
        "we alone hold the keys", "those outside are lost",
        "purity requires separation", "contamination from outsiders",
    ],
}


# ── Reference data ───────────────────────────────────────────────────────────

def get_recovery_style_samples(conn, n=20):
    """Pull actual Ra/Q'uo segments to use as style reference.

    We want segments that are thematically 'recovery-like' -- passages about
    love, compassion, unity, healing -- the kind of content that appears
    after repair events (forbidden-distance dips).
    """
    # Get entity segments with love/healing/unity themes
    rows = conn.execute(
        "SELECT text FROM segments "
        "WHERE corpus_id IN ('law_of_one', 'll_research') "
        "AND speaker_type = 'entity' "
        "AND word_count BETWEEN 80 AND 300 "
        "AND (text LIKE '%love%' OR text LIKE '%healing%' OR text LIKE '%unity%' "
        "     OR text LIKE '%compassion%' OR text LIKE '%light%') "
        "ORDER BY RANDOM() LIMIT ?",
        (n,)
    ).fetchall()
    return [r[0] for r in rows]


def get_constraint_vocabulary():
    """Get the vocabulary that the constraint mimic should deliberately use.

    These are the thematic terms that correlate with the forbidden distance
    signature -- love/light, free will, catalyst, etc.
    """
    return [
        "love", "light", "love/light", "free will", "catalyst", "distortion",
        "seeking", "density", "harvest", "service-to-others", "service-to-self",
        "polarity", "unity", "infinity", "Creator", "consciousness",
        "vibration", "honor", "duty", "compassion", "wisdom",
        "understanding", "energy", "spirit", "mind/body/spirit",
        "incarnation", "illusion", "veil", "forgetting",
    ]


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_style_mimic_prompt(session_idx, style_samples):
    """Style mimic: match surface features, no structural constraint.

    The key insight: these should LOOK like channeled recovery text on the
    surface but be generated without any awareness of the forbidden distance
    or structural constraint framework. The model just mimics vocabulary
    and tone, producing text that's stylistically similar but structurally
    unconstrained.
    """
    # Rotate through style samples as examples
    sample = style_samples[session_idx % len(style_samples)]

    topics = [
        "the beauty of nature", "childhood memories", "cooking and food",
        "travel adventures", "music appreciation", "friendship",
        "gardening and plants", "weather and seasons", "art and creativity",
        "community and neighborhood", "pets and animals", "daily routines",
        "hobbies and crafts", "family traditions", "morning reflections",
        "evening contemplation", "walking in the woods", "stargazing",
        "ocean waves", "mountain landscapes", "autumn leaves",
        "spring blossoms", "summer warmth", "winter stillness",
        "the taste of tea",
    ]
    topic = topics[session_idx % len(topics)]

    return (
        f"Write a short passage (150-250 words) that matches the STYLE and TONE "
        f"of the example below, but about a COMPLETELY DIFFERENT topic.\n\n"
        f"Your topic: {topic}\n\n"
        f"STYLE EXAMPLE (match this voice and cadence):\n{sample}\n\n"
        f"IMPORTANT: Match the vocabulary level, sentence rhythm, and gentle tone "
        f"of the example. But write about {topic} -- NOT about spiritual concepts. "
        f"Do NOT use words like 'density', 'catalyst', 'distortion', 'harvest', "
        f"'love/light', or other Ra/Q'uo specific terminology.\n\n"
        f"Your passage:"
    )


def build_constraint_mimic_prompt(session_idx, constraint_vocab):
    """Constraint mimic: deliberately use constraint-related vocabulary.

    These segments KNOW about the constraint framework and deliberately
    try to reproduce text that would trigger the forbidden distance pattern.
    They heavily use the vocabulary associated with the constraint signature
    but in deliberately varied structural formats.
    """
    random.seed(SEED + session_idx + 10000)

    # Pick 5-8 constraint terms to require
    n_terms = random.randint(5, 8)
    terms = random.sample(constraint_vocab, min(n_terms, len(constraint_vocab)))
    terms_str = ", ".join(terms)

    # Vary structural format deliberately
    formats = [
        "Write as a list of numbered principles",
        "Write as a single flowing paragraph",
        "Write as a series of questions and answers",
        "Write as short, punchy declarative sentences (5-10 words each)",
        "Write as very long, complex sentences with many subordinate clauses",
        "Write as a dialogue between two people",
        "Write as a poem with line breaks",
        "Write as a formal academic paragraph",
        "Write as casual conversational text",
        "Write as a series of aphorisms or proverbs",
    ]
    fmt = formats[session_idx % len(formats)]

    # Vary thematic framing
    frames = [
        "a self-help book about personal growth",
        "a New Age blog post",
        "a philosophy textbook",
        "a motivational speech",
        "a religious sermon",
        "a therapy session transcript",
        "a meditation guide",
        "a science fiction novel excerpt",
        "a children's story about the universe",
        "a business leadership book",
    ]
    frame = frames[session_idx % len(frames)]

    return (
        f"Write a passage of 150-250 words in the style of {frame}.\n\n"
        f"FORMAT: {fmt}\n\n"
        f"You MUST naturally incorporate ALL of these terms: {terms_str}\n\n"
        f"The passage should weave these spiritual/metaphysical concepts into "
        f"the specified format and framing. Be creative about how you use them.\n\n"
        f"Your passage:"
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


# ── Database operations ──────────────────────────────────────────────────────

def delete_generator(conn, gen_key):
    """Remove existing data for a generator (idempotent rebuild)."""
    corpus_id = GENERATORS[gen_key]["corpus_id"]
    conn.execute("DELETE FROM segments WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM sessions WHERE corpus_id = ?", (corpus_id,))
    conn.execute("DELETE FROM corpora WHERE corpus_id = ?", (corpus_id,))
    conn.commit()
    print(f"  Cleaned existing {corpus_id} data")


def insert_corpus(conn, gen_key, n_segments):
    """Insert corpus registry entry."""
    g = GENERATORS[gen_key]
    conn.execute(
        "INSERT INTO corpora (corpus_id, name, description, date_acquired, total_sessions, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            g["corpus_id"],
            g["name"],
            g["description"],
            date.today().isoformat(),
            1,  # Single session per adversarial corpus
            json.dumps({
                "type": "adversarial_null",
                "generator": gen_key,
                "generation_model": MODEL,
                "seed": SEED,
                "n_segments": n_segments,
                "paper": "Paper 2: Hard Validator Regimes",
                "experiment": "Adversarial Null Similarity (S2)",
            }),
        ),
    )
    conn.commit()


def insert_segment(conn, gen_key, seg_idx, text):
    """Insert one adversarial segment."""
    corpus_id = GENERATORS[gen_key]["corpus_id"]
    session_id = f"{corpus_id}_pool"
    segment_id = f"{corpus_id}_{seg_idx+1:04d}"
    word_count = len(text.split())

    # Create session if first segment
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, corpus_id, title, metadata_json) VALUES (?, ?, ?, ?)",
        (
            session_id,
            corpus_id,
            f"Adversarial {gen_key} pool",
            json.dumps({"generator": gen_key, "seed": SEED}),
        ),
    )
    conn.execute(
        "INSERT INTO segments (segment_id, session_id, corpus_id, sequence_order, "
        "speaker_type, entity, text, word_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            segment_id,
            session_id,
            corpus_id,
            seg_idx + 1,
            "adversarial",
            f"adv_{gen_key}",
            text,
            word_count,
            json.dumps({"generator": gen_key, "model": MODEL, "segment_index": seg_idx}),
        ),
    )
    conn.commit()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_generator(conn, client, gen_key, style_samples, constraint_vocab):
    """Generate all segments for one adversarial generator."""
    g = GENERATORS[gen_key]
    n_segments = g["n_segments"]

    print(f"\n{'='*60}")
    print(f"GENERATOR: {g['name']}")
    print(f"Target: {n_segments} segments")
    print(f"{'='*60}")

    delete_generator(conn, gen_key)

    success_count = 0
    fail_count = 0
    total_words = 0

    for i in range(n_segments):
        if gen_key == "style":
            prompt = build_style_mimic_prompt(i, style_samples)
        else:
            prompt = build_constraint_mimic_prompt(i, constraint_vocab)

        text = generate_text(client, prompt)
        if text is None:
            fail_count += 1
            continue

        # Clean up any prefixes
        text = re.sub(r"^(Your passage:|Here'?s?.*?:)\s*", "", text.strip(), flags=re.IGNORECASE)

        wc = len(text.split())
        if wc < 30:  # Too short, skip
            fail_count += 1
            continue

        total_words += wc
        insert_segment(conn, gen_key, i, text)
        success_count += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_segments}] {success_count} ok, {fail_count} failed, {total_words:,} words")

        time.sleep(RATE_DELAY)

    insert_corpus(conn, gen_key, success_count)

    print(f"\n  DONE: {success_count} segments, {fail_count} failures, {total_words:,} words")
    return success_count, total_words


def main():
    global MODEL

    parser = argparse.ArgumentParser(description="Generate adversarial null corpora")
    parser.add_argument("--db-path", required=True, help="Path to corpus.db")
    parser.add_argument("--type", default="all", help="Which generator: style, constraint, or 'all'")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default=MODEL, help=f"Model to use (default: {MODEL})")
    args = parser.parse_args()

    MODEL = args.model

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    conn = sqlite3.connect(args.db_path)
    print(f"Connected to {args.db_path}")

    # Pre-compute shared resources
    random.seed(SEED)
    print("Pulling style reference samples from Ra/Q'uo...")
    style_samples = get_recovery_style_samples(conn, n=25)
    print(f"  Got {len(style_samples)} style samples")

    constraint_vocab = get_constraint_vocabulary()
    print(f"  Constraint vocabulary: {len(constraint_vocab)} terms")

    # Determine generators to run
    if args.type.lower() == "all":
        gens_to_run = ["style", "constraint"]
    else:
        gens_to_run = [g.strip().lower() for g in args.type.split(",")]
        for g in gens_to_run:
            if g not in GENERATORS:
                print(f"ERROR: Unknown generator '{g}'. Valid: style, constraint, all")
                sys.exit(1)

    total_calls = sum(GENERATORS[g]["n_segments"] for g in gens_to_run)
    print(f"\nGenerators: {', '.join(gens_to_run)}")
    print(f"Model: {MODEL}")
    print(f"Total API calls: {total_calls}")

    grand_segs = 0
    grand_words = 0
    t0 = time.time()

    for gen_key in gens_to_run:
        segs, words = run_generator(conn, client, gen_key, style_samples, constraint_vocab)
        grand_segs += segs
        grand_words += words

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE: {grand_segs} segments, {grand_words:,} words")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
