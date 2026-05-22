from __future__ import annotations

import json
import os
import re
from collections import Counter

from app.services.intelligence_service import _call_llm, _tokens

# ──────────────────────────────────────────────
#  Role skill taxonomies (extended)
# ──────────────────────────────────────────────

ROLE_SKILLS: dict[str, set[str]] = {
    "backend": {
        "python", "java", "go", "rust", "c++", "c#", "node.js", "typescript",
        "sql", "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes",
        "aws", "gcp", "azure", "api", "rest", "graphql", "fastapi", "django",
        "flask", "spring", "microservice", "grpc", "kafka", "rabbitmq",
        "git", "linux", "ci/cd", "terraform", "jenkins",
    },
    "data science": {
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "keras", "matplotlib", "seaborn", "statistics", "probability",
        "machine learning", "deep learning", "nlp", "computer vision",
        "regression", "classification", "clustering", "feature engineering",
        "a/b testing", "sql", "spark", "hadoop", "airflow", "tableau",
        "docker", "git", "mlops",
    },
    "frontend": {
        "javascript", "typescript", "react", "vue", "angular", "next.js",
        "html", "css", "scss", "tailwind", "redux", "graphql", "rest",
        "jest", "cypress", "webpack", "vite", "responsive design",
        "accessibility", "git", "figma",
    },
    "general": {
        "teamwork", "communication", "leadership", "project management",
        "agile", "scrum", "problem solving", "critical thinking",
    },
}

ATS_SECTIONS = {
    "experience", "education", "skills", "projects", "summary",
    "objective", "certifications", "publications", "achievements",
    "leadership", "volunteering", "languages",
}

ACTION_VERBS = {
    "achieved", "developed", "implemented", "designed", "built",
    "led", "managed", "created", "improved", "optimized", "reduced",
    "increased", "delivered", "launched", "architected", "engineered",
    "configured", "integrated", "migrated", "automated",
}

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_PATTERN = re.compile(rf"({MONTHS})\s*\.?\s*(\d{{4}})", re.IGNORECASE)


# ──────────────────────────────────────────────
#  Structured extraction
# ──────────────────────────────────────────────


def extract_structured_data_llm(text: str) -> dict:
    prompt = f"""You are a resume parser. Extract structured data from the following resume text.

Resume:
{text[:2500]}

Return ONLY valid JSON with these keys:
- "skills": list of strings (technical + soft skills mentioned)
- "experience_years": integer (total years of professional experience, or 0 if unclear)
- "education": list of objects with "degree" and "institution" keys
- "sections_found": list of section headers detected (e.g. "experience", "education", "skills")

No markdown. No explanation."""
    result = _call_llm(prompt, max_tokens=500, temperature=0.2)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                return {
                    "skills": data.get("skills", []),
                    "experience_years": max(0, int(data.get("experience_years", 0))),
                    "education": data.get("education", []),
                    "sections_found": data.get("sections_found", []),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return extract_structured_data(text)


def extract_structured_data(text: str) -> dict:
    tokens = _tokens(text)
    text_lower = text.lower()

    years_exp = 0
    dates = DATE_PATTERN.findall(text)
    if dates:
        years = sorted(int(y) for _, y in dates if y.isdigit())
        if len(years) >= 2:
            years_exp = max(1, years[-1] - years[0])
        elif years:
            years_exp = 1

    all_skills = set()
    for role, skills in ROLE_SKILLS.items():
        for s in skills:
            if s.lower() in text_lower:
                all_skills.add(s)

    sections_found = [s for s in ATS_SECTIONS if s in text_lower]

    education = []
    edu_patterns = [
        r"(b\.?\s*s\.?\s*(?:in|of)?\s+\w+)",
        r"(m\.?\s*s\.?\s*(?:in|of)?\s+\w+)",
        r"(ph\.?\s*d\.?\s*(?:in|of)?\s+\w+)",
        r"(bachelor[^\n.]*)",
        r"(master[^\n.]*)",
        r"(phd[^\n.]*)",
        r"(mba[^\n.]*)",
    ]
    for pat in edu_patterns:
        m = re.search(pat, text_lower)
        if m:
            education.append({"degree": m.group(1).strip(), "institution": ""})

    return {
        "skills": sorted(all_skills),
        "experience_years": years_exp,
        "education": education,
        "sections_found": sections_found,
    }


# ──────────────────────────────────────────────
#  ATS Scoring
# ──────────────────────────────────────────────


def score_resume_ats(text: str, target_role: str) -> dict:
    text_lower = text.lower()
    tokens = _tokens(text)
    word_count = len(tokens) if tokens else 1
    role_skills = ROLE_SKILLS.get(target_role.lower(), ROLE_SKILLS["general"])

    keyword_matches = sum(1 for s in role_skills if s.lower() in text_lower)
    keyword_density_val = round(keyword_matches / max(1, len(role_skills)), 3)

    sections_found = sum(1 for s in ATS_SECTIONS if s in text_lower)
    section_score = min(1.0, sections_found / 6)

    action_verbs_found = sum(1 for v in ACTION_VERBS if v in text_lower)
    action_verb_score = min(1.0, action_verbs_found / 8)

    # Count numbers (quantifiable achievements)
    numbers = len(re.findall(r"\d+%|\d+x|\$\d+|\d+[kKmM]", text))
    quantification_score = min(1.0, numbers / 5)

    format_issues = 0
    if word_count < 100:
        format_issues += 1
    if word_count > 2000:
        format_issues += 1
    if not re.search(r"(education|experience|skills)", text_lower):
        format_issues += 1
    if text.count("\n\n") < 3:
        format_issues += 1
    format_score = max(0, 1 - format_issues * 0.25)

    ats_score = int(round((
        0.30 * keyword_density_val +
        0.20 * section_score +
        0.15 * action_verb_score +
        0.15 * quantification_score +
        0.20 * format_score
    ) * 100))

    return {
        "ats_score": max(0, min(100, ats_score)),
        "keyword_density": round(keyword_density_val, 3),
        "keyword_matches": keyword_matches,
        "total_keywords": len(role_skills),
        "section_score": round(section_score, 3),
        "sections_found": sections_found,
        "action_verb_score": round(action_verb_score, 3),
        "action_verbs_found": action_verbs_found,
        "quantification_score": round(quantification_score, 3),
        "format_score": round(format_score, 3),
    }


def analyze_skills_gap_llm(skills: list[str], target_role: str) -> dict:
    prompt = f"""You are a hiring manager for a "{target_role}" position.
The candidate has the following skills: {', '.join(skills)}

Analyze the skills gap for a {target_role} role.
Return ONLY valid JSON:
{{
  "matched": ["skill1", "skill2"],
  "missing": ["skill3", "skill4"],
  "irrelevant": ["skill5"],
  "suggestions": ["suggestion1", "suggestion2"]
}}
No markdown. No explanation."""
    result = _call_llm(prompt, max_tokens=300, temperature=0.3)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                return {
                    "matched": data.get("matched", []),
                    "missing": data.get("missing", []),
                    "irrelevant": data.get("irrelevant", []),
                    "suggestions": data.get("suggestions", []),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return analyze_skills_gap(skills, target_role)


def analyze_skills_gap(skills: list[str], target_role: str) -> dict:
    role = target_role.lower()
    required = ROLE_SKILLS.get(role, ROLE_SKILLS["general"])
    skills_lower = set(s.lower() for s in skills)

    matched = sorted(s for s in required if s.lower() in skills_lower)
    missing = sorted(s for s in required if s.lower() not in skills_lower)
    irrelevant = sorted(s for s in skills if s.lower() not in required and s.lower() not in {m.lower() for m in matched})

    suggestions = []
    if missing:
        top = missing[:3]
        suggestions.append(f"Consider adding: {', '.join(top)}")
        if "docker" in missing or "kubernetes" in missing:
            suggestions.append("Containerization skills are increasingly expected for this role")
        if "python" in missing:
            suggestions.append("Python is a core requirement — add projects demonstrating proficiency")

    return {
        "matched": matched,
        "missing": missing,
        "irrelevant": irrelevant,
        "suggestions": suggestions,
    }


# ──────────────────────────────────────────────
#  Resume General Profile
# ──────────────────────────────────────────────


ROLE_CATEGORIES = ["backend", "frontend", "data science", "devops", "fullstack", "mobile", "product management", "data engineering", "ml engineering", "security", "general"]


def generate_resume_profile_llm(text: str) -> dict:
    prompt = f"""You are a senior career coach and resume reviewer. Analyze the following resume and provide a general profile WITHOUT anchoring to a specific target role.

Resume:
{text[:2500]}

Return ONLY valid JSON with these exact keys:
- "description": A 2-3 sentence summary describing the candidate — experience level, key areas of expertise, what their background indicates.
- "suggested_roles": A list of 2-4 role categories from this set that best match the resume: {', '.join(ROLE_CATEGORIES)}. Pick the ones that fit best.
- "structure": An object with:
   - "sections_found": list of section headers detected (e.g. "experience", "education", "skills", "projects", "certifications")
   - "format_quality": one of "poor", "fair", "good", "excellent"
   - "length": one of "too short", "appropriate", "too long"
   - "readability": one of "poor", "fair", "good", "excellent"
   - "strengths": list of 2-3 structural strengths
   - "issues": list of 2-3 structural issues or missing elements
- "overall_structure_score": integer 0-100 rating the resume structure alone (sections, formatting, readability, completeness)

No markdown. No explanation."""
    result = _call_llm(prompt, max_tokens=600, temperature=0.3)
    if result:
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                return {
                    "description": data.get("description", ""),
                    "suggested_roles": data.get("suggested_roles", []),
                    "structure": data.get("structure", {}),
                    "overall_structure_score": max(0, min(100, int(data.get("overall_structure_score", 50)))),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {
        "description": "",
        "suggested_roles": [],
        "structure": {"sections_found": [], "format_quality": "fair", "length": "appropriate", "readability": "fair", "strengths": [], "issues": []},
        "overall_structure_score": 50,
    }


def generate_resume_summary_llm(text: str, role: str, ats_score: int, skills_gap: dict) -> str:
    prompt = f"""You are a career coach. Based on the resume analysis below, provide brief actionable feedback.

Target Role: {role}
ATS Score: {ats_score}/100
Matched Skills: {', '.join(skills_gap.get('matched', []))}
Missing Skills: {', '.join(skills_gap.get('missing', []))}

Resume:
{text[:1500]}

Provide 2-3 sentences of specific, actionable advice to improve this resume for the {role} role.
Focus on skills gaps, formatting, and impact. Be direct and practical."""
    result = _call_llm(prompt, max_tokens=300, temperature=0.5)
    if result:
        return result
    return f"Your resume scores {ats_score}/100 for a {role} role. {'Focus on adding: ' + ', '.join(skills_gap.get('missing', [])[:3]) if skills_gap.get('missing') else 'Your skills align well with the target role.'}"
