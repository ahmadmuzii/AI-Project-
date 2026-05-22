from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

# ──────────────────────────────────────────────
#  Grok & Groq LLM helpers
# ──────────────────────────────────────────────

_GROQ_CLIENT = None


def _groq() -> Any | None:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is not None:
        return _GROQ_CLIENT
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq

        _GROQ_CLIENT = Groq(api_key=key)
        return _GROQ_CLIENT
    except Exception:
        return None


def _call_grok_http(prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str | None:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        return None
    import httpx
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-2-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        # Timeout at 25 seconds to keep mock interviews responsive
        resp = httpx.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=25.0)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Grok API error: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        print(f"Grok HTTP call failed: {e}")
        return None


def _call_llm(prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str | None:
    # Try Grok first if configured
    grok_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if grok_key:
        res = _call_grok_http(prompt, max_tokens, temperature)
        if res:
            return res
        print("Grok LLM failed, falling back to Groq...")

    client = _groq()
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq LLM call failed: {e}")
        return None


# ──────────────────────────────────────────────
#  Rule-based helpers (existing)
# ──────────────────────────────────────────────

POSITIVE_WORDS = {
    "confident", "strong", "achieved", "improved", "optimized", "delivered",
    "led", "built", "solved", "success",
}
NEGATIVE_WORDS = {
    "nervous", "maybe", "unsure", "difficult", "struggle", "failed",
    "confused", "worried", "problem",
}
TECH_KEYWORDS = {
    "backend": {"api", "database", "sql", "docker", "python", "fastapi", "microservice", "cache", "redis"},
    "data science": {"python", "pandas", "numpy", "model", "regression", "classification", "feature", "sql"},
    "frontend": {"react", "javascript", "typescript", "css", "state", "component", "ui", "ux"},
    "general": {"team", "project", "impact", "result", "improved", "delivery"},
}
STAR_HINTS = {
    "situation": {"at my previous", "in my last", "during a project", "context"},
    "task": {"my task", "responsible", "objective", "goal"},
    "action": {"i implemented", "i built", "i designed", "i created", "i led"},
    "result": {"result", "outcome", "improved", "reduced", "increased", "achieved"},
}

QUESTION_BANK = {
    "general": [
        "Tell me about yourself in one minute.",
        "Describe a challenge you solved recently.",
        "Why should we hire you for this role?",
        "How do you prioritize tasks under pressure?",
    ],
    "backend": [
        "How would you design a rate-limited API?",
        "Explain database indexing and trade-offs.",
        "How do you handle consistency in distributed systems?",
        "Describe Docker usage in deployment pipelines.",
    ],
    "data science": [
        "How do you handle data leakage in training?",
        "Explain bias-variance tradeoff with an example.",
        "How do you validate a regression model?",
        "What metrics would you track for model drift?",
    ],
}

from app.services.company_service import get_company_profile, get_company_style_prompt

RESUME_SKILLS = {
    "backend": {"python", "sql", "docker", "api", "fastapi", "redis", "git", "linux"},
    "data science": {"python", "pandas", "numpy", "scikit-learn", "sql", "statistics", "ml"},
    "frontend": {"react", "javascript", "typescript", "css", "html", "state management"},
}


# ──────────────────────────────────────────────
#  NLP analysis (unchanged)
# ──────────────────────────────────────────────


@dataclass
class NlpResult:
    sentiment: str
    sentiment_score: float
    star_score: float
    coherence_score: float
    keyword_relevance: float
    weak_topics: list[str]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]*", text.lower())


def analyze_answer_nlp(answer: str, role: str = "general") -> NlpResult:
    tokens = _tokens(answer)
    if not tokens:
        return NlpResult("neutral", 0.0, 0.0, 0.0, 0.0, ["clarity"])

    counts = Counter(tokens)
    pos = sum(counts[w] for w in POSITIVE_WORDS if w in counts)
    neg = sum(counts[w] for w in NEGATIVE_WORDS if w in counts)
    raw = (pos - neg) / max(1, len(tokens))
    sentiment_score = max(-1.0, min(1.0, raw * 10))
    sentiment = (
        "confident"
        if sentiment_score > 0.1
        else "nervous"
        if sentiment_score < -0.1
        else "neutral"
    )

    lower = answer.lower()
    star_hits = 0
    for _, phrases in STAR_HINTS.items():
        if any(p in lower for p in phrases):
            star_hits += 1
    star_score = round(star_hits / 4, 3)

    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    avg_len = sum(len(_tokens(s)) for s in sentences) / max(1, len(sentences))
    len_penalty = abs(18 - avg_len) / 18
    coherence_score = round(max(0.0, min(1.0, 1 - len_penalty)), 3)

    role_key = role.lower()
    role_keywords = TECH_KEYWORDS.get(role_key, TECH_KEYWORDS["general"])
    covered = len([k for k in role_keywords if k in counts])
    keyword_relevance = round(covered / max(1, len(role_keywords)), 3)

    weak_topics = []
    if star_score < 0.5:
        weak_topics.append("behavioral_storytelling")
    if keyword_relevance < 0.3:
        weak_topics.append("technical_depth")
    if coherence_score < 0.55:
        weak_topics.append("answer_structure")
    if sentiment == "nervous":
        weak_topics.append("confidence")
    if not weak_topics:
        weak_topics.append("general")

    return NlpResult(
        sentiment=sentiment,
        sentiment_score=round(sentiment_score, 3),
        star_score=star_score,
        coherence_score=coherence_score,
        keyword_relevance=keyword_relevance,
        weak_topics=weak_topics,
    )


def predict_readiness_days(overall_scores: list[float], target_score: float = 0.8) -> int | None:
    if len(overall_scores) < 3:
        return None
    xs = list(range(len(overall_scores)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(overall_scores) / len(overall_scores)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, overall_scores))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    if slope <= 0:
        return None
    days_to_target = (target_score - intercept) / slope
    import math
    if math.isnan(days_to_target) or days_to_target < 0:
        return 0
    return int(round(days_to_target))


# ──────────────────────────────────────────────
#  Feature 1: LLM Question Generation
# ──────────────────────────────────────────────


def generate_questions_llm(
    role: str,
    weak_topics: list[str],
    previous_questions: list[str],
    max_items: int = 4,
    resume_text: str = "",
) -> list[str]:
    prompt = f"""You are an expert technical interviewer for a "{role}" position.

The candidate's weak areas are: {', '.join(weak_topics) if weak_topics else 'general'}.
{"Their resume mentions: " + resume_text[:1000] if resume_text else ""}

Generate {max_items} specific, non-generic interview questions that:
1. Target the candidate's weak areas
2. Are specific to the {role} role
3. Require detailed, structured answers
4. Are different from typical "tell me about yourself" questions

Previous questions asked (DO NOT repeat):
{chr(10).join('- ' + q for q in previous_questions[-6:])}

Return ONLY the questions as a simple numbered list. One question per line.
No explanations, no headers, no extra text."""

    result = _call_llm(prompt, max_tokens=600)
    if result:
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        questions = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
        if questions:
            return questions[:max_items]

    return suggest_questions(role, weak_topics, previous_questions, max_items)


def generate_company_questions_llm(company: str, role: str) -> dict:
    profile = get_company_profile(company)
    style_prompt = get_company_style_prompt(company)
    focus = profile["focus"] if profile else "general interview readiness"

    prompt = f"""You are an interviewer at {company}, hiring for a "{role}" role.

{style_prompt}

Generate 4 interview questions that are authentic to how {company} interviews.
Make each question feel like it was pulled from a real {company} interview at that company.

Questions should:
- Be specific to {company}'s known interview process
- Target {role} skills
- Range from behavioral to technical
- NOT be generic (avoid "Tell me about yourself")

Return ONLY the questions as a simple numbered list. One per line."""

    result = _call_llm(prompt, max_tokens=500)
    if result:
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        questions = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
        if questions:
            return {"company": company, "focus": focus, "questions": questions[:4]}

    return company_mode_questions(company, role)


# ──────────────────────────────────────────────
#  Existing question fallbacks (unchanged)
# ──────────────────────────────────────────────


def suggest_questions(
    role: str, weak_topics: list[str], previous_questions: list[str], max_items: int = 4
) -> list[str]:
    role_key = role.lower()
    pool = list(QUESTION_BANK.get(role_key, QUESTION_BANK["general"]))
    if "technical_depth" in weak_topics and role_key != "general":
        pool.extend(QUESTION_BANK.get(role_key, []))
    if "answer_structure" in weak_topics:
        pool.insert(0, "Answer this using STAR format: Tell me about a high-impact project.")
    selected = []
    seen = set(q.strip().lower() for q in previous_questions)
    for q in pool:
        k = q.strip().lower()
        if k not in seen and k not in (s.lower() for s in selected):
            selected.append(q)
        if len(selected) >= max_items:
            break
    return selected


def company_mode_questions(company: str, role: str) -> dict:
    profile = get_company_profile(company)
    role_key = role.lower()
    base = QUESTION_BANK.get(role_key, QUESTION_BANK["general"])[:2]
    if not profile:
        return {"company": company, "focus": "general interview readiness", "questions": base}
    known = profile.get("known_questions", [])
    return {
        "company": company,
        "focus": profile["focus"],
        "questions": base + known[:2],
    }


# ──────────────────────────────────────────────
#  Feature 2: Resume-Aware Feedback via LLM
# ──────────────────────────────────────────────


def analyze_resume_text_llm(resume_text: str, role: str) -> dict:
    prompt = f"""You are a hiring manager for a "{role}" position.
Extract and analyze the following resume text.

Resume:
{resume_text[:2000]}

Return a JSON object with exactly these keys:
- "score": integer from 0-100 for resume fit
- "matched_skills": list of strings matching the resume to {role}
- "missing_skills": list of important skills NOT found in the resume
- "summary": one-sentence assessment

Return ONLY valid JSON. No markdown. No explanation."""

    result = _call_llm(prompt, max_tokens=400, temperature=0.3)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "score" in data:
                return {
                    "score": max(0, min(100, int(data["score"]))),
                    "matched_skills": data.get("matched_skills", []),
                    "missing_skills": data.get("missing_skills", []),
                    "summary": data.get("summary", f"Resume analysis for {role} profile."),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return analyze_resume_text(resume_text, role)


def analyze_resume_text(resume_text: str, role: str) -> dict:
    role_key = role.lower()
    required = RESUME_SKILLS.get(role_key, RESUME_SKILLS["backend"])
    text_tokens = set(_tokens(resume_text))
    matched = sorted([s for s in required if s.lower() in text_tokens])
    missing = sorted([s for s in required if s.lower() not in text_tokens])
    score = int(round((len(matched) / max(1, len(required))) * 100))
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "summary": f"Resume score {score}/100 for {role} profile.",
    }


def build_study_plan(weak_topics: list[str]) -> list[dict]:
    plan = []
    day = 1
    topic_to_plan = {
        "technical_depth": "Solve 2 role-specific problems and explain trade-offs aloud.",
        "answer_structure": "Practice STAR for 3 behavioral answers and self-review.",
        "confidence": "Do 10-minute mock with slower pace and filler-word control.",
        "behavioral_storytelling": "Prepare 4 stories: conflict, leadership, failure, impact.",
        "general": "Run one full mock interview and review transcript.",
    }
    for topic in weak_topics:
        plan.append(
            {"day": day, "focus": topic, "task": topic_to_plan.get(topic, topic_to_plan["general"])}
        )
        day += 1
    return plan


# ──────────────────────────────────────────────
#  Feature 3: Session Summary
# ──────────────────────────────────────────────


def generate_session_summary_llm(recordings_data: list[dict]) -> dict:
    if not recordings_data:
        return {
            "summary": "No recordings found in this session.",
            "trend": "stable",
            "top_improvements": [],
            "average_scores": {},
        }

    avg_scores = {}
    score_keys = ["fluency", "confidence", "composure", "overall"]
    for key in score_keys:
        vals = [r.get("scores", {}).get(key, 0) for r in recordings_data if r.get("scores")]
        avg_scores[key] = round(sum(vals) / max(1, len(vals)), 3) if vals else 0

    # Detect trend
    overalls = [r.get("scores", {}).get("overall", 0) for r in recordings_data if r.get("scores")]
    if len(overalls) >= 2:
        trend = "improving" if overalls[-1] > overalls[0] else "declining" if overalls[-1] < overalls[0] else "stable"
    else:
        trend = "stable"

    # Build prompt
    transcripts = []
    for i, r in enumerate(recordings_data):
        t = r.get("transcript", "")
        if t:
            transcripts.append(f"Answer {i+1}: {t[:300]}")

    prompt = f"""You are an interview coach reviewing a practice session with {len(recordings_data)} answers.

Average scores — Fluency: {avg_scores.get('fluency', 0)}, Confidence: {avg_scores.get('confidence', 0)}, Overall: {avg_scores.get('overall', 0)}
Trend: {trend}

Transcripts:
{chr(10).join(transcripts)}

Provide:
1. Overall performance summary (2-3 sentences)
2. Specific top 3 things to improve with actionable tips

Return as plain text. No JSON. No headers. Just natural paragraphs."""

    result = _call_llm(prompt, max_tokens=500)
    summary = result or f"Session with {len(recordings_data)} answers. Average overall score: {avg_scores.get('overall', 0)}."

    return {
        "summary": summary,
        "trend": trend,
        "average_scores": avg_scores,
        "recording_count": len(recordings_data),
    }


# ──────────────────────────────────────────────
#  Feature 4: Follow-up Questions
# ──────────────────────────────────────────────


def generate_follow_up_llm(question: str, answer: str, role: str) -> str:
    prompt = f"""You are an interviewer for a "{role}" position.

You asked: "{question}"
The candidate answered: "{answer[:1500]}"

Generate ONE relevant follow-up question that:
- Probes deeper into something specific the candidate mentioned
- Is NOT generic — reference their actual answer
- Sounds like a real interviewer following up naturally

Return ONLY the follow-up question. No explanation."""

    result = _call_llm(prompt, max_tokens=150, temperature=0.8)
    if result and len(result) > 10:
        return result

    # Fallback
    return f"Can you elaborate on that with a specific example from your experience?"


# ──────────────────────────────────────────────
#  Feature 5: Content Scoring via LLM
# ──────────────────────────────────────────────


def score_answer_content_llm(question: str, answer: str, role: str) -> dict:
    prompt = f"""You are evaluating an interview answer for a "{role}" position.

Question: "{question}"
Answer: "{answer[:1500]}"

Rate the answer on these criteria (0.0 to 1.0):
1. relevance: How directly does it answer the question? (not off-topic)
2. content_quality: Depth, specificity, examples, structure
3. star_usage: Does it use Situation-Task-Action-Result format?

Return ONLY valid JSON: {{"relevance": 0.X, "content_quality": 0.X, "star_usage": 0.X}}
No markdown. No explanation."""

    result = _call_llm(prompt, max_tokens=150, temperature=0.3)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                return {
                    "relevance": max(0.0, min(1.0, float(data.get("relevance", 0)))),
                    "content_quality": max(0.0, min(1.0, float(data.get("content_quality", 0)))),
                    "star_usage": max(0.0, min(1.0, float(data.get("star_usage", 0)))),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {"relevance": 0.0, "content_quality": 0.0, "star_usage": 0.0}


# ──────────────────────────────────────────────
#  Existing helpers (unchanged)
# ──────────────────────────────────────────────


def evaluate_stress(
    eye_contact_score: float, movement_score: float, voice_energy: float
) -> dict:
    stress = 1 - (
        0.45 * eye_contact_score + 0.35 * movement_score + 0.20 * min(1.0, voice_energy)
    )
    stress = max(0.0, min(1.0, stress))
    label = "high" if stress > 0.67 else "moderate" if stress > 0.4 else "low"
    return {"stress_score": round(stress, 3), "stress_level": label}


def topic_heatmap(metrics_rows: list[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for topic, overall in metrics_rows:
        grouped[topic or "general"].append(overall or 0.0)
    return {
        topic: round(1 - (sum(vals) / max(1, len(vals))), 3)
        for topic, vals in grouped.items()
    }


# ──────────────────────────────────────────────
#  Guided Interview Functions
# ──────────────────────────────────────────────


def generate_greeting_and_clarifying_questions(
    profile: dict,
    aim: str,
    company: str,
) -> dict:
    role = profile.get("target_role", "general")
    resume_text = profile.get("resume_text", "")
    name = profile.get("display_name") or "there"
    company_style = get_company_style_prompt(company)

    company_section = ""
    if company_style:
        company_section = f"\nCompany context:\n{company_style}\n"

    prompt = f"""You are an expert interviewer conducting a guided interview for a "{role}" position.

Candidate profile:
- Name: {name}
- Target role: {role}
- Years of experience: {profile.get("years_of_experience", "unknown")}
- Seniority: {profile.get("seniority_level", "unknown")}
- Focus areas: {profile.get("focus_areas", "general")}

{"Resume: " + resume_text[:1000] if resume_text else ""}

Interview context:
- Purpose: {aim or "General interview preparation"}
- {"Target company: " + company if company else ""}
{company_section}

Generate a warm greeting and 2-3 short clarifying questions for the candidate.
The clarifying questions should be quick to answer (1-2 sentences each) and help tailor the interview based on the candidate's profile and goals.

Return ONLY valid JSON with this exact structure:
{{
  "greeting": "A warm 1-2 sentence greeting from the interviewer, introducing themselves and setting a comfortable tone.",
  "clarifying_questions": [
    "Short clarifying question 1 based on the candidate's background",
    "Short clarifying question 2 based on the candidate's goals",
    "Short clarifying question 3 based on the candidate's profile"
  ]
}}

The questions should be brief, friendly, and help the interviewer understand what to focus on.
No markdown. No extra text. Only valid JSON."""

    result = _call_llm(prompt, max_tokens=400, temperature=0.7)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "greeting" in data and "clarifying_questions" in data:
                qs = data["clarifying_questions"]
                if isinstance(qs, list) and len(qs) >= 1:
                    return {
                        "greeting": data["greeting"],
                        "clarifying_questions": qs[:3],
                    }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback
    return {
        "greeting": f"Hi {name}, welcome! I'm thrilled to be your interviewer today. Let's start with a few quick questions to make sure I tailor this session perfectly for you.",
        "clarifying_questions": [
            f"What specific aspects of the {role} role would you like to focus on most?",
            "Are there any particular question types you find challenging?",
            "What's your main goal for this practice session?",
        ],
    }


def generate_first_interview_question(
    profile: dict,
    aim: str,
    company: str,
) -> str:
    role = profile.get("target_role", "general")
    resume_text = profile.get("resume_text", "")
    company_style = get_company_style_prompt(company)

    company_section = ""
    if company_style:
        company_section = f"\nCompany context:\n{company_style}\n"

    prompt = f"""You are an expert interviewer conducting a guided interview for a "{role}" position.

Candidate profile:
- Target role: {role}
- Years of experience: {profile.get("years_of_experience", "unknown")}
- Seniority: {profile.get("seniority_level", "unknown")}
- Focus areas: {profile.get("focus_areas", "general")}

{"Resume: " + resume_text[:1000] if resume_text else ""}

Interview context:
- Purpose: {aim or "General interview preparation"}
- {"Target company: " + company if company else ""}
{company_section}
Generate ONE opening interview question that:
1. Is specific to the {role} role
2. References the candidate's background where relevant
3. {"Matches " + company + "'s interview style as described above" if company else "Is open-ended and encourages a detailed answer"}
4. Is NOT "Tell me about yourself" — be more specific and engaging

Return ONLY the question. No explanation, no prefix, no quotes."""

    result = _call_llm(prompt, max_tokens=200, temperature=0.8)
    if result and len(result) > 15:
        return result

    # Fallback — use suggest_questions for a fresh opening question
    fallback_qs = suggest_questions(role, [], [], 1)
    if fallback_qs:
        return fallback_qs[0]
    return "Walk me through a recent project you're proud of and the impact you made."


def generate_next_question(
    interview_context: dict,
    qa_history: list[dict],
    profile: dict,
) -> str:
    role = profile.get("target_role", "general")

    # Build QA history summary
    history_lines = []
    for i, qa in enumerate(qa_history):
        history_lines.append(f"Q{i+1}: {qa.get('question', '')}")
        transcript = qa.get('transcript', '')[:200]
        if transcript:
            history_lines.append(f"A{i+1}: {transcript}")
        scores = []
        for k in ("content_score", "relevance_score", "fluency_score"):
            v = qa.get(k, 0)
            if v:
                scores.append(f"{k}={v}")
        if scores:
            history_lines.append(f"  Scores: {', '.join(scores)}")
        history_lines.append("")

    prompt = f"""You are an expert interviewer for a "{role}" position. Conducting a live guided interview.

Interview context:
- {"Target company: " + interview_context.get("target_company", "") if interview_context.get("target_company") else ""}
- Difficulty: {interview_context.get("difficulty", "intermediate")}

Previous Q&A history:
{chr(10).join(history_lines) if history_lines else "This is the first follow-up question."}

Generate ONE follow-up question that:
1. References something specific from the candidate's last answer
2. Probes deeper into a weak area or unexplored aspect
3. Is appropriate for the {role} role
4. Gets harder if previous answers were strong, or provides an easier angle if struggling
5. Is different from all previous questions

Return ONLY the question. No explanation, no prefix, no quotes."""

    result = _call_llm(prompt, max_tokens=200, temperature=0.8)
    if result and len(result) > 15:
        return result

    # Improved fallback — use the last answer to generate a contextual question
    if qa_history:
        last = qa_history[-1]
        last_q = last.get("question", "")
        last_a = last.get("transcript", "")
        if last_a:
            fallback = f"Can you elaborate on the key point you just mentioned about '{last_a[:100].strip()}' — walk me through your specific thought process and how you arrived at that conclusion?"
        elif last_q:
            fallback = f"Let me ask that differently: {last_q} Can you provide a specific example from your experience?"
        else:
            fallback = "Tell me about a specific challenge you faced in this area and how you overcame it."
    else:
        fallback = "Tell me about a specific challenge you faced in this area and how you overcame it."
    return fallback


def generate_interview_summary(
    qa_pairs: list[dict],
    profile: dict,
    scores: dict,
) -> dict:
    if not qa_pairs:
        return {
            "summary": "No questions were answered during this interview.",
            "strengths": [],
            "top_improvements": [],
            "action_plan": [],
            "readiness_estimate": "N/A",
        }

    # Build QA summary for prompt
    qa_lines = []
    for i, qa in enumerate(qa_pairs):
        q = qa.get("question", "")[:150]
        a = qa.get("transcript", "")[:300]
        fb = qa.get("feedback", "")[:200]
        qa_lines.append(f"Q{i+1}: {q}")
        qa_lines.append(f"A{i+1}: {a}")
        if fb:
            qa_lines.append(f"Feedback: {fb}")
        qa_lines.append(f"Scores: content={qa.get('content_score', 0)}, relevance={qa.get('relevance_score', 0)}, "
                        f"fluency={qa.get('fluency_score', 0)}, confidence={qa.get('confidence_score', 0)}")
        qa_lines.append("")

    role = profile.get("target_role", "general")

    prompt = f"""You are an expert interview coach reviewing a completed mock interview for a "{role}" position.

Number of questions answered: {len(qa_pairs)}
Average scores — Content: {scores.get('avg_content', 0)}, Relevance: {scores.get('avg_relevance', 0)}, Fluency: {scores.get('avg_fluency', 0)}, Confidence: {scores.get('avg_confidence', 0)}
Overall score: {scores.get('overall', 0)}/100

Q&A Transcripts with feedback:
{chr(10).join(qa_lines)}

Provide your analysis as a JSON object with exactly these keys:
- "summary": A comprehensive 3-4 sentence overall performance assessment (natural paragraph, NOT bullet points)
- "strengths": Array of 2-3 specific strengths demonstrated
- "top_improvements": Array of 3-4 specific, actionable improvements with concrete tips
- "action_plan": Array of 3-5 specific steps the candidate should take to prepare (each step is a string)
- "readiness_estimate": One of "Not ready", "Needs work", "Almost ready", "Ready", "Highly prepared"

Return ONLY valid JSON. No markdown. No explanation."""

    result = _call_llm(prompt, max_tokens=800, temperature=0.7)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "summary" in data:
                return {
                    "summary": data.get("summary", "Interview completed."),
                    "strengths": data.get("strengths", []),
                    "top_improvements": data.get("top_improvements", []),
                    "action_plan": data.get("action_plan", []),
                    "readiness_estimate": data.get("readiness_estimate", "Needs work"),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback
    avg_overall = scores.get("overall", 0)
    readiness = "Not ready" if avg_overall < 30 else "Needs work" if avg_overall < 50 else "Almost ready" if avg_overall < 70 else "Ready" if avg_overall < 90 else "Highly prepared"
    return {
        "summary": f"Completed {len(qa_pairs)} questions with an overall score of {avg_overall}/100. "
                   f"Content average: {scores.get('avg_content', 0):.1f}, Relevance: {scores.get('avg_relevance', 0):.1f}, "
                   f"Fluency: {scores.get('avg_fluency', 0):.1f}, Confidence: {scores.get('avg_confidence', 0):.1f}.",
        "strengths": ["Engaged throughout the interview", "Completed all questions"],
        "top_improvements": ["Review technical fundamentals for your role",
                            "Practice structuring answers with STAR format",
                            "Reduce filler words and hesitations",
                            "Provide more specific metrics in answers"],
        "action_plan": ["Review role-specific technical concepts daily",
                       "Practice 3 STAR-format answers per day",
                       "Record yourself answering questions and review",
                       "Take at least 2 more mock interviews before your real interview"],
        "readiness_estimate": readiness,
    }
