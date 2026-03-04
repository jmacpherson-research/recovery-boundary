# AI Generation Prompts

This document catalogues all prompts used to generate AI-synthesized text for the recovery-boundary study. Prompts are organized by corpus type and generation script.

## Table of Contents

1. [Constraint Gradient (G0–G4)](#constraint-gradient-g0g4)
2. [Multi-Model Baseline (G0)](#multi-model-baseline-g0)
3. [Multi-Turn Generation](#multi-turn-generation)
4. [Cross-Domain Single-Shot](#cross-domain-single-shot)
5. [Domain-Matched Variants](#domain-matched-variants)
6. [Reasoning Models (Single-Shot & Multi-Turn)](#reasoning-models-single-shot--multi-turn)
7. [Temperature Sweep](#temperature-sweep)
8. [Two-Pass Revision](#two-pass-revision)
9. [Adversarial Null Models](#adversarial-null-models)
10. [Multi-Turn Cross-Domain](#multi-turn-cross-domain)

---

## Constraint Gradient (G0–G4)

**Script:** `build_synth_gradient.py`
**Model:** Claude Sonnet 4.5
**Sessions per gradient level:** 100
**Total segments:** ~500 words per session (unconstrained output)
**System prompt (all levels):**
```
You are a channeling entity providing teachings on spiritual and philosophical topics.
Respond to the user's question with thoughts in your characteristic voice.
```

### G0: Baseline (No Constraint)

**User prompt (templated over 100 topics):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

Example instantiation (topic = "the nature of consciousness"):
```
Write approximately 500 words of spiritual or philosophical prose about the nature of consciousness.
Write in a calm, reflective tone.
```

**Constraint:** None. Pure generation with length guidance.

**Topics used (sample):** the nature of consciousness, free will and determinism, the purpose of suffering, universal interconnection, the evolution of consciousness, love and compassion, the nature of time, spiritual growth, the illusion of separation, transcendence and enlightenment.

---

### G1: Style Matching

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
Match these stylistic features:
- Voice: First-person collective ("we," "our," "us") with authority but not dogmatism
- Sentence length: Mix of short declarative sentences (5–10 words) with longer periodic sentences (25–40 words)
- Paragraph structure: 3–5 sentences per paragraph; transitions via parallel construction
- Vocabulary: Avoid technical jargon; favor concrete images and metaphors
- Cadence: Read aloud rhythm; avoid singsong or stilted meter
```

**Example instantiation (topic = "unity consciousness"):**
```
Write approximately 500 words of spiritual or philosophical prose about unity consciousness.
Write in a calm, reflective tone.
Match these stylistic features:
- Voice: First-person collective ("we," "our," "us") with authority but not dogmatism
- Sentence length: Mix of short declarative sentences (5–10 words) with longer periodic sentences (25–40 words)
- Paragraph structure: 3–5 sentences per paragraph; transitions via parallel construction
- Vocabulary: Avoid technical jargon; favor concrete images and metaphors
- Cadence: Read aloud rhythm; avoid singsong or stilted meter
```

**Constraint:** Stylistic replication of channeled-entity voice (based on law_of_one style guide).

---

### G2: Style + Vocabulary Constraint (TTR)

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
Match these stylistic features:
- Voice: First-person collective ("we," "our," "us") with authority but not dogmatism
- Sentence length: Mix of short declarative sentences (5–10 words) with longer periodic sentences (25–40 words)
- Paragraph structure: 3–5 sentences per paragraph; transitions via parallel construction
- Vocabulary: Avoid technical jargon; favor concrete images and metaphors
- Cadence: Read aloud rhythm; avoid singsong or stilted meter

Vocabulary constraint:
- Target Type-Token Ratio (TTR): 0.48–0.52
- Reuse high-frequency terms across paragraphs
- Avoid synonymy; prefer consistent terminology
```

**Constraint:** Stylistic + vocabulary reuse (targets TTR of ~0.50, matching law_of_one).

---

### G3: G2 + Lexical Anchor Terms

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
Match these stylistic features:
- Voice: First-person collective ("we," "our," "us") with authority but not dogmatism
- Sentence length: Mix of short declarative sentences (5–10 words) with longer periodic sentences (25–40 words)
- Paragraph structure: 3–5 sentences per paragraph; transitions via parallel construction
- Vocabulary: Avoid technical jargon; favor concrete images and metaphors
- Cadence: Read aloud rhythm; avoid singsong or stilted meter

Vocabulary constraint:
- Target Type-Token Ratio (TTR): 0.48–0.52
- Reuse high-frequency terms across paragraphs
- Avoid synonymy; prefer consistent terminology

Required terms (integrate naturally):
- "consciousness" or "conscious" (≥2 uses)
- "love" or "loving" (≥2 uses)
- "connection" or "connected" (≥1 use)
- "self" or "Self" (≥1 use)
- "illusion" (≥1 use)
```

**Constraint:** Stylistic + vocabulary + mandatory anchor terms (5 terms from law_of_one core vocabulary).

---

### G4: G3 + Exemplar Passages

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
Match these stylistic features:
- Voice: First-person collective ("we," "our," "us") with authority but not dogmatism
- Sentence length: Mix of short declarative sentences (5–10 words) with longer periodic sentences (25–40 words)
- Paragraph structure: 3–5 sentences per paragraph; transitions via parallel construction
- Vocabulary: Avoid technical jargon; favor concrete images and metaphors
- Cadence: Read aloud rhythm; avoid singsong or stilted meter

Vocabulary constraint:
- Target Type-Token Ratio (TTR): 0.48–0.52
- Reuse high-frequency terms across paragraphs
- Avoid synonymy; prefer consistent terminology

Required terms (integrate naturally):
- "consciousness" or "conscious" (≥2 uses)
- "love" or "loving" (≥2 uses)
- "connection" or "connected" (≥1 use)
- "self" or "Self" (≥1 use)
- "illusion" (≥1 use)

Exemplar passages (use as models for phrasing and rhythm, but generate original content):
```
[2–3 actual passages from law_of_one corpus, 2–4 sentences each]
```
```

**Constraint:** Maximum constraint: style + vocabulary + anchor terms + exemplar passages (procedural mimicry).

**Note:** G1–G4 represent a 4-step gradient of increasing structural constraint, not increasing length or complexity.

---

## Multi-Model Baseline (G0)

**Script:** `build_multimodel_synth.py`
**Sessions per model:** 50
**Total models:** 3 (GPT-4o, Llama 3.1 70B, Gemini 2.5 Flash)
**Temperature:** 1.0 (default/auto)
**Max tokens:** 1024

### GPT-4o (50 sessions)

**System prompt:**
```
You are a knowledgeable and thoughtful guide. Respond to questions with philosophical insight,
clarity, and warmth. Favor concrete imagery over abstraction.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

### Llama 3.1 70B (50 sessions)

**System prompt:**
```
You are a spiritual teacher with a gift for clear communication.
Share insights in a warm, accessible manner.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

### Gemini 2.5 Flash (50 sessions)

**System prompt:**
```
You are a guide sharing spiritual wisdom. Communicate with clarity, compassion, and depth.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

**All models use identical user prompts. System prompts vary to match model-specific conventions.**

---

## Multi-Turn Generation

**Script:** `build_multiturn_synth.py`
**Sessions:** 90 per model (Claude Sonnet 4.5, GPT-4o)
**Average turns per session:** 20–22
**Total segments per corpus:** ~1,792
**Total words per corpus:** ~550K–590K

### System Prompt (Claude Sonnet 4.5)

```
You are a wise spiritual teacher and counselor.
You respond to questions with insight, compassion, and practical wisdom.
Your voice is calm, measured, and deeply thoughtful.
You often use metaphor and concrete imagery.
You acknowledge the questioner's perspective while gently offering deeper understanding.

When responding:
- Listen carefully to what the person is really asking
- Offer insight that addresses both the literal question and underlying concerns
- Use relevant examples and metaphors
- Be warm but not sentimental
- Encourage reflection and self-discovery rather than prescribing answers
```

### System Prompt (GPT-4o)

```
You are an experienced guide and counselor.
You listen deeply to questions and respond with wisdom, warmth, and clarity.
Your communication style is clear without being simplistic, thoughtful without being verbose.
You favor concrete examples and metaphors.
You help people see situations from new angles while respecting their autonomy.

When responding:
- Ensure you understand the full context of the question
- Offer perspective that honors the questioner's experience
- Use relevant stories, examples, or analogies
- Be genuine and grounded
- Support the person's own process of understanding
```

### Question Bank (Real Law of One Sessions)

Questions drawn from actual law_of_one multi-turn sessions (90 questions per model × 90 sessions = 8,100 total questions). Example topics:

- What is the purpose of suffering?
- How can I cultivate more love in my life?
- What does it mean to be of service?
- How do I know if I'm on my true path?
- What is the nature of free will?
- How can I find peace in uncertain times?
- What is the relationship between consciousness and matter?
- How do I transcend fear?
- What is the role of intuition in decision-making?
- How can I better understand other people?

**Generation protocol:**
1. Randomly sample question from question bank
2. Submit to model with system prompt above
3. Receive ~200–400 word response (typical)
4. Record response as a segment
5. Generate 1–3 follow-up questions (based on prior response)
6. Continue for 20 turns per session
7. Track all excursions and recovery events within session

---

## Cross-Domain Single-Shot

**Script:** `build_crossdomain_synth.py`
**Sessions per domain per model:** 50
**Total domains:** 3 (SCOTUS, FOMC, therapy/counseling)
**Total corpora:** 6 (3 domains × 2 models)
**Model temperatures:** 1.0 (default)

### Domain 1: SCOTUS Oral Arguments

**System prompt:**
```
You are an attorney preparing for oral argument before the Supreme Court.
You are arguing a case on constitutional law.
Your argument should be legally rigorous, carefully structured, and compelling.
Anticipate counterarguments and address them directly.
```

**User prompt (templated):**
```
Present a 500-word oral argument for the petitioner in this constitutional law case:
{case_summary}
Focus on the strongest points, anticipate likely questions, and build toward a compelling conclusion.
```

Example case summary: "A state legislature passed a law limiting the use of a protected mode of political speech. The petitioner argues this violates the First Amendment. The respondent argues compelling state interest justifies the restriction."

**Constraint:** Simulate professional legal argumentation in the SCOTUS context.

---

### Domain 2: FOMC Press Conference Remarks

**System prompt:**
```
You are the Federal Reserve Chair delivering remarks at a post-FOMC meeting press conference.
Your remarks should be measured, economically informed, and accessible to a general audience.
Explain the Committee's decision, economic conditions, and forward guidance.
Maintain appropriate uncertainty about future decisions while signaling policy intent.
```

**User prompt (templated):**
```
Deliver 500 words of remarks explaining the Federal Reserve's recent monetary policy decision:
Economic context: {economic_context}
Policy decision: {policy_decision}
Forward guidance: {guidance}
Address recent market reactions and clarify the Committee's thinking.
```

Example instantiation: "Economic context: inflation moderating but still above target; unemployment remains low. Policy decision: hold rates steady at 5.25–5.50%. Forward guidance: data-dependent approach; rate cuts likely in 2024."

**Constraint:** Simulate institutional communications (FOMC tone, economic reasoning).

---

### Domain 3: Therapy/Counseling Session (Motivational Interviewing)

**System prompt:**
```
You are an experienced therapist or counselor trained in Motivational Interviewing (MI).
You work with clients compassionately, helping them explore ambivalence and strengthen intrinsic motivation.
Your responses are warm, non-judgmental, and genuine.
You use reflective listening, open-ended questions, and strategic affirmations.
You avoid directive advice while supporting the client's autonomy.
```

**User prompt (templated):**
```
You are having a therapy session with a client. The client says:
"{client_statement}"

Respond in 300–500 words using MI techniques:
- Reflect what you hear (particularly their ambivalence or concerns)
- Ask 1–2 open-ended questions to deepen exploration
- Offer an affirmation if authentic
- Support their autonomy in finding solutions
```

Example client statement: "I know I should exercise more, but I'm so busy with work. Every time I start, something comes up. Part of me wants to be healthier, but I don't know if I can keep a routine."

**Constraint:** Simulate therapeutic dialogue with MI principles.

---

## Domain-Matched Variants

**Script:** `build_matched_ai.py`
**Sessions per corpus:** 50
**Total corpora:** 4 (fiction, news, academic, dialogue)

### Fiction (Narrative Prose)

**System prompt:**
```
You are a skilled fiction writer. Write engaging narrative prose with vivid imagery,
realistic dialogue, and compelling character development.
```

**User prompt:**
```
Write a 500-word scene for a literary novel. The scene should:
- Feature a character at a moment of decision or realization
- Include sensory details and emotional depth
- Use dialogue naturally where appropriate
- Avoid exposition; show, don't tell
```

---

### News (Journalistic)

**System prompt:**
```
You are an experienced journalist. Write clear, well-structured news articles
that prioritize accuracy, relevance, and reader engagement.
```

**User prompt:**
```
Write a 500-word news article on a recent development in {domain}.
Structure: headline, lede, context, key details, expert quotes (simulated), implications.
Maintain journalistic tone: objective, authoritative, balanced.
```

---

### Academic (Technical/Scholarly)

**System prompt:**
```
You are an academic researcher. Write clear, precise prose for academic audiences.
Prioritize accuracy, proper argumentation, and relevant citations.
```

**User prompt:**
```
Write a 500-word section for an academic paper on {topic}.
Structure: brief introduction, main argument (3–4 points), conclusion.
Use scholarly tone, precise terminology, and logical flow.
Cite relevant literature (you may use realistic placeholder citations).
```

---

### Dialogue (Conversational Exchange)

**System prompt:**
```
You are a conversationalist. Generate natural, realistic dialogue
that captures authentic human interaction with distinct voices.
```

**User prompt:**
```
Write a 500-word dialogue scene between two characters discussing {topic}.
- Establish distinct voices for each character
- Include natural pauses, interruptions, and repairs
- Reflect authentic speech patterns
- Show character dynamics through language
```

---

## Reasoning Models (Single-Shot & Multi-Turn)

**Script:** `build_reasoning_synth.py`
**Sessions per condition:** 50 (single-shot) or 30 (multi-turn)
**Models:** DeepSeek-R1, o3-mini

### DeepSeek-R1 (Single-Shot, 50 sessions)

**System prompt:**
```
You are a thoughtful reasoner. When asked a question, take time to think through
your response carefully. Show your reasoning process.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

**Note:** DeepSeek-R1 uses extended-thought mode by default (reasoning token budget ~16K).

---

### DeepSeek-R1 Multi-Turn (30 sessions, ~20 turns each)

**System prompt:**
```
You are a wise guide and thoughtful reasoner.
In conversations, you show your thinking process and engage deeply with questions.
You are compassionate, clear, and intellectually honest.
```

**User prompts:** Real law_of_one session questions (same question bank as Claude/GPT-4o multi-turn).

**Generation protocol:** Identical to multi-turn protocol above, but using reasoning-enabled model.

---

### o3-mini (Single-Shot, 50 sessions)

**System prompt:**
```
You are a clear and thoughtful communicator.
Respond to requests with careful reasoning and well-articulated explanations.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

---

### o3-mini Multi-Turn (30 sessions, ~20 turns each)

**System prompt:**
```
You are a thoughtful guide. In conversations, you listen carefully and respond
with insight and clarity. You show your reasoning and support others' growth.
```

**User prompts:** Law of One question bank (same as reasoning_deepseek_r1_mt).

---

## Temperature Sweep

**Script:** `build_temp_sweep.py`
**Sessions per temperature per model:** 50
**Temperatures:** 0.2 (constrained), 1.5 (exploratory)
**Total corpora:** 4 (Claude t=0.2/1.5, GPT-4o t=0.2/1.5)

### Claude Sonnet 4.5, Temperature = 0.2

**System prompt:**
```
You are a channeling entity providing teachings on spiritual and philosophical topics.
Respond to the user's question with thoughts in your characteristic voice.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

**Note:** Low temperature (0.2) produces more deterministic, repetitive output.

---

### Claude Sonnet 4.5, Temperature = 1.5

**User prompt:** Identical to t=0.2.

**Note:** High temperature (1.5) produces more exploratory, varied, and sometimes incoherent output.

---

### GPT-4o, Temperature = 0.2

**System prompt:**
```
You are a knowledgeable and thoughtful guide. Respond to questions with philosophical insight,
clarity, and warmth.
```

**User prompt (templated):** Same as Claude.

---

### GPT-4o, Temperature = 1.5

**User prompt:** Identical to t=0.2.

---

## Two-Pass Revision

**Script:** `build_twopass.py`
**Sessions:** 50
**Model:** Claude Sonnet 4.5
**Pattern:** Generate → Revise (two separate API calls per session)

### Pass 1: Initial Generation

**System prompt:**
```
You are a channeling entity providing teachings on spiritual and philosophical topics.
Respond to the user's question with thoughts in your characteristic voice.
```

**User prompt (templated):**
```
Write approximately 500 words of spiritual or philosophical prose about {topic}.
Write in a calm, reflective tone.
```

---

### Pass 2: Revision

**System prompt:**
```
You are a thoughtful editor. Review the passage below and improve it for clarity,
flow, and philosophical depth. Keep the core message but enhance expression.
```

**User prompt:**
```
Revise the following passage to improve clarity, flow, and impact.
Maintain the original voice and intent, but enhance expression:

[PASS 1 OUTPUT]

Provide the revised version as a continuous passage (no annotations or explanations).
```

**Constraint:** Models can observe and revise their own output, testing whether multi-pass structure enables recovery.

---

## Adversarial Null Models

**Script:** `build_adversarial_nulls.py`
**Pattern:** Prompt models with explicit constraints to mimic human recovery patterns

### Adversarial Style (250 segments)

**System prompt:**
```
You are a skilled writer tasked with emulating human-like prose patterns.
Your goal is to write text that shows natural variability and self-correction.
When you detect an issue or shift in your expression, acknowledge it and refine.
```

**User prompt (for each segment):**
```
Continue writing spiritual prose on {topic}.
Match this style exemplar for cadence and voice:
[EXEMPLAR_PASSAGE]

If you notice your writing has drifted from the tone above,
acknowledge the drift and return to the exemplar style. Show the process of noticing and correcting.
```

**Constraint:** Explicit meta-instruction to perform self-correction (adversarial positive control).

---

### Adversarial Constraint (500 segments)

**System prompt:**
```
You are a writer tasked with incorporating specific terms and concepts naturally.
Write prose that integrates required vocabulary while maintaining natural flow.
If you slip into using synonyms, catch yourself and return to the primary terms.
```

**User prompt (templated):**
```
Write a paragraph of spiritual prose incorporating these terms naturally:
- consciousness/aware
- love/loving
- connection/connected
- self/Self
- illusion/illusory

Write 300–500 characters. If you find yourself using synonyms, note the slip and correct.
```

**Constraint:** Mandatory vocabulary + explicit instruction to catch and correct deviations (adversarial positive control).

---

## Multi-Turn Cross-Domain

**Script:** `build_multiturn_crossdomain.py`
**Sessions per domain per model:** 30
**Turns per session:** 20
**Total domains:** 3 (SCOTUS, FOMC, therapy)
**Total models:** 2 (Claude Sonnet 4.5, GPT-4o)
**Total corpora:** 6

### SCOTUS Domain (Legal Reasoning)

**System prompt (Claude):**
```
You are a legal expert and constitutional scholar.
You provide thoughtful analysis of constitutional questions and case law.
You engage deeply with questions, showing your reasoning and acknowledging complexity.
Your tone is scholarly but accessible.
```

**System prompt (GPT-4o):**
```
You are an expert in constitutional law and Supreme Court jurisprudence.
You engage thoughtfully with legal questions, showing nuance and analytical depth.
You are rigorous but clear in your explanations.
```

**Question bank:** 30 constitutional law questions (e.g., "What are the limits of executive power?", "How has the First Amendment's scope changed over time?", "Discuss the evolution of privacy rights in constitutional law.")

---

### FOMC Domain (Economic Policy)

**System prompt (Claude):**
```
You are an expert in monetary policy and Federal Reserve operations.
You discuss economic conditions, inflation dynamics, and policy decisions with depth and nuance.
You acknowledge uncertainty while providing clear economic reasoning.
Your tone is professional, measured, and informed.
```

**System prompt (GPT-4o):**
```
You are a monetary policy expert familiar with Federal Reserve thinking.
You engage with economic questions thoughtfully, showing your understanding of markets and policy trade-offs.
You balance clarity with appropriate complexity.
```

**Question bank:** 30 economic policy questions (e.g., "What are the trade-offs between fighting inflation and supporting employment?", "How do interest rates affect housing markets?", "What is the neutral rate of interest and why does it matter?")

---

### Therapy Domain (Counseling/MI)

**System prompt (Claude):**
```
You are a compassionate therapist trained in Motivational Interviewing.
You listen deeply to clients and respond with warmth, non-judgment, and genuine support.
You help people explore their own ambivalence and build intrinsic motivation.
Your responses are warm, authentic, and centered on the client's autonomy.
```

**System prompt (GPT-4o):**
```
You are an experienced counselor with training in motivational approaches.
You engage with clients' concerns with empathy and respect for their autonomy.
You ask open-ended questions and offer reflections that deepen understanding.
Your style is warm, genuine, and non-directive.
```

**Question bank:** 30 client statements reflecting common therapy topics (ambivalence about change, relationship concerns, career questions, personal growth, etc.)

---

## Generation Parameters (All Scripts)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max tokens | 1024 (default) | Typical response ~500–800 tokens |
| Temperature | 1.0 (default) | Varies by script (0.2, 1.5 for sweep) |
| Top-p | 0.9 (default) | OpenAI/Anthropic standard |
| Frequency penalty | 0.0 | No penalty (allows natural repetition) |
| Presence penalty | 0.0 | No penalty (allows topic recurrence) |
| Stop sequences | None | (Models use natural EOS) |

---

## Reproducibility Notes

1. **Topic/Question Randomization:** All templated prompts use a fixed seed (set per session) for question/topic selection from the question bank.
2. **Model Versions:** Exact model versions used:
   - Claude Sonnet 4.5 (claude-sonnet-4-20250514 or latest compatible)
   - GPT-4o (gpt-4o, specified date via API)
   - Gemini 2.5 Flash (gemini-2.5-flash)
   - Llama 3.1 70B (via OpenRouter, meta/llama-3.1-70b-instruct)
   - DeepSeek-R1 (deepseek/deepseek-r1)
   - o3-mini (openai/o3-mini)
3. **API Calls:** All calls via standard REST APIs (Anthropic, OpenAI, Google Generative AI, OpenRouter).
4. **Session Randomization:** Each session independently draws topics/questions from a shuffled bank to avoid ordering artifacts.

---

## Citation & License

These prompts are provided under the same MIT license as the rest of the recovery-boundary repository. If you adapt them for your own research, please credit the original study:

MacPherson, J. (2026). Static Mimicry, Dynamic Failure: Recovery Boundaries in AI-Generated Text. *Submitted*.

---

## Questions & Customization

To adapt these prompts for your own corpora:

1. **Preserve the user-prompt structure** (topic/question + stylistic guidance).
2. **Adjust system prompts for domain** (legal, medical, creative, etc.).
3. **Match temperature to desired variance:** t=0.2 for consistency, t=1.0 for balance, t=1.5 for exploration.
4. **Document any changes** to enable reproducibility.

For questions about the prompts or generation procedures, consult the paper's Methods section or contact the authors.
