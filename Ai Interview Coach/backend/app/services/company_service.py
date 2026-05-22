import csv
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "swe_questions.csv"

# ──────────────────────────────────────────────
#  Dataset loader
# ──────────────────────────────────────────────

_question_cache = None


def load_question_bank() -> dict[str, list[dict]]:
    global _question_cache
    if _question_cache is not None:
        return _question_cache

    bank = defaultdict(list)
    if not CSV_PATH.exists():
        print(f"Warning: SWE question CSV not found at {CSV_PATH}")
        _question_cache = dict(bank)
        return _question_cache

    with open(CSV_PATH, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("Category", "General").strip()
            bank[cat].append({
                "question": row.get("Question", "").strip(),
                "answer": row.get("Answer", "").strip(),
                "difficulty": row.get("Difficulty", "Medium").strip(),
            })

    _question_cache = dict(bank)
    print(f"Loaded {sum(len(v) for v in bank.values())} questions across {len(bank)} categories")
    return _question_cache


def get_questions_by_category(category: str, difficulty: str | None = None, limit: int = 5) -> list[str]:
    bank = load_question_bank()
    # Fuzzy category match
    matched = []
    for cat, qs in bank.items():
        if category.lower() in cat.lower() or cat.lower() in category.lower():
            matched.extend(qs)
    if not matched:
        # Fallback: use all categories
        for qs in bank.values():
            matched.extend(qs)

    if difficulty:
        filtered = [q for q in matched if q["difficulty"].lower() == difficulty.lower()]
        if filtered:
            matched = filtered

    random.shuffle(matched)
    return [q["question"] for q in matched[:limit]]


# ──────────────────────────────────────────────
#  Company Style Profiles
# ──────────────────────────────────────────────

COMPANY_PROFILES = {
    "google": {
        "focus": "problem-solving and thinking aloud",
        "tone": "warm but rigorous — expects structured reasoning and collaboration",
        "format": "series of back-to-back technical interviews (coding, system design, behavioral) with increasing difficulty",
        "values": ["innovation", "user focus", "collaboration", "scale"],
        "categories": ["System Design", "Algorithms", "Distributed Systems", "General Programming"],
        "known_questions": [
            "Design a scalable URL shortener.",
            "How would you debug a production latency spike?",
            "Tell me about a time you disagreed with your manager.",
            "Design a web crawler that downloads billions of pages.",
        ],
    },
    "meta": {
        "focus": "execution, product thinking, and impact",
        "tone": "direct, fast-paced — cares about results and trade-offs",
        "format": "coding (LeetCode-style), system design, behavioral (product sense)",
        "values": ["move fast", "impact", "focus on users", "be direct"],
        "categories": ["System Design", "Front-end", "Algorithms", "Database and SQL"],
        "known_questions": [
            "How do you measure success for a new feature?",
            "How would you scale a feed ranking service?",
            "Design Facebook Messenger end-to-end.",
            "Tell me about a product you shipped and its impact.",
        ],
    },
    "amazon": {
        "focus": "Leadership Principles and customer obsession",
        "tone": "high bar, metric-driven — expects STAR answers with quantified results",
        "format": "behavioral (LP-based, 4+ rounds), system design, bar raiser round",
        "values": ["customer obsession", "ownership", "bias for action", "deliver results", "hire and develop the best"],
        "categories": ["System Design", "Distributed Systems", "General Programming"],
        "known_questions": [
            "Tell me about a time you took a calculated risk.",
            "How do you handle a difficult stakeholder?",
            "Describe the most complex project you've owned end-to-end.",
            "Design a product recommendation system.",
        ],
    },
    "apple": {
        "focus": "craftsmanship, attention to detail, and product excellence",
        "tone": "precise, design-conscious — expects deep technical knowledge and pride in work",
        "format": "multiple rounds covering domain depth, system design, and cultural fit",
        "values": ["simplicity", "quality over quantity", "innovation", "ownership"],
        "categories": ["Low-level Systems", "Front-end", "Security", "System Design"],
        "known_questions": [
            "Walk me through the full lifecycle of a feature you built.",
            "How would you optimize rendering performance on a constrained device?",
            "Tell me about a time you pushed for quality over speed.",
            "Design a secure file storage system.",
        ],
    },
    "netflix": {
        "focus": "high-performance engineering and freedom with responsibility",
        "tone": "confident, no-nonsense — expects ownership and strong opinions held loosely",
        "format": "deep-dive technical rounds, system design, culture fit (freedom & responsibility)",
        "values": ["judgment", "communication", "impact", "curiosity", "inclusion"],
        "categories": ["System Design", "DevOps", "Distributed Systems", "Database Systems"],
        "known_questions": [
            "How would you design a recommendation engine at Netflix scale?",
            "Tell me about a time you had to make a controversial technical decision.",
            "How do you handle service dependencies in a microservice architecture?",
            "Design a fault-tolerant video streaming pipeline.",
        ],
    },
    "stripe": {
        "focus": "API design, reliability, and developer experience",
        "tone": "thoughtful, detail-oriented — expects clarity and precision in communication",
        "format": "coding, system design, API design, debugging scenarios",
        "values": ["developer first", "precision", "reliability", "incremental improvement"],
        "categories": ["System Design", "Security", "Back-end", "Database and SQL"],
        "known_questions": [
            "Design a payment system that handles idempotency.",
            "How would you design an idempotent API endpoint?",
            "Tell me about a time you improved a system's reliability.",
            "Design a rate limiter for a public API.",
        ],
    },
    "microsoft": {
        "focus": "well-rounded engineering with growth mindset",
        "tone": "collaborative, mentoring — expects technical depth and team orientation",
        "format": "coding, system design, behavioral (growth mindset), final round with manager",
        "values": ["growth mindset", "diversity & inclusion", "customer focus", "innovation"],
        "categories": ["System Design", "Algorithms", "General Programming", "Front-end"],
        "known_questions": [
            "How would you design a collaborative editing tool like Google Docs?",
            "Tell me about a time you learned a new technology for a project.",
            "Design a distributed file system.",
            "How do you handle competing priorities from different teams?",
        ],
    },
    "mckinsey": {
        "focus": "structured thinking, communication, and problem-solving under pressure",
        "tone": "professional, structured — expects clarity, logic, and business acumen",
        "format": "case interviews, behavioral (personal impact), problem-solving tests",
        "values": ["client impact", "structured thinking", "leadership", "professional development"],
        "categories": ["General Programming", "System Design"],
        "known_questions": [
            "Walk me through a market-sizing framework.",
            "How would you advise a client with falling revenue?",
            "Tell me about a time you led a team through a difficult situation.",
            "Estimate the number of gas stations in the US.",
        ],
    },
    "goldman sachs": {
        "focus": "risk awareness, technical rigor, and business alignment",
        "tone": "formal, precise — expects reliability, security awareness, and composure",
        "format": "technical (coding, data structures), behavioral (integrity, team fit), system design",
        "values": ["integrity", "partnership", "client focus", "excellence"],
        "categories": ["Security", "Database and SQL", "Algorithms", "General Programming"],
        "known_questions": [
            "How would you design a high-frequency trading system?",
            "Tell me about a time you maintained integrity under pressure.",
            "Design a system that detects fraudulent transactions in real-time.",
            "Explain how you'd handle a production incident in a financial system.",
        ],
    },
    "deloitte": {
        "focus": "client delivery, adaptability, and technical consulting",
        "tone": "consultative, clear — expects structured thinking and client empathy",
        "format": "behavioral (client stories), case study, technical discussion",
        "values": ["client service", "integrity", "collaboration", "inclusion"],
        "categories": ["General Programming", "Software Testing", "DevOps", "Security"],
        "known_questions": [
            "Tell me about a time you managed a difficult client relationship.",
            "How would you approach migrating a legacy system to the cloud?",
            "Describe a project where you had to adapt to changing requirements.",
            "Walk me through how you'd estimate effort for a large engagement.",
        ],
    },
}


def get_company_profile(company: str) -> dict | None:
    if not company:
        return None
    for key, profile in COMPANY_PROFILES.items():
        if key in company.lower() or company.lower() in key:
            return profile
    return None


def get_company_style_prompt(company: str) -> str:
    profile = get_company_profile(company)
    if not profile:
        return ""

    prompt_parts = [
        f"Interview focus: {profile['focus']}.",
        f"Tone: {profile['tone']}.",
        f"Format: {profile['format']}.",
    ]
    if profile.get("values"):
        prompt_parts.append(f"Core values: {', '.join(profile['values'])}.")

    # Add known questions as examples
    if profile.get("known_questions"):
        examples = profile["known_questions"][:3]
        prompt_parts.append("Example questions from previous candidates:")
        for q in examples:
            prompt_parts.append(f"- {q}")

    return "\n".join(prompt_parts)


def get_company_categories(company: str) -> list[str]:
    profile = get_company_profile(company)
    return profile["categories"] if profile else []


def get_dataset_questions_for_company(company: str, limit: int = 3) -> list[str]:
    categories = get_company_categories(company)
    questions = []
    for cat in categories:
        qs = get_questions_by_category(cat, limit=limit)
        questions.extend(qs)
        if len(questions) >= limit:
            break
    return questions[:limit]
