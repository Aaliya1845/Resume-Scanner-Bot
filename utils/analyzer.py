import re
from utils.text_cleaner import (
    get_keyword_set,
    matching_keywords,
    missing_keywords,
)

# Common technical skills database
TECHNICAL_SKILLS = {
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "vue", "node", "express",
    "django", "flask", "fastapi", "php", "laravel",
    "mysql", "postgresql", "mongodb", "sqlite", "oracle",
    "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp",
    "tensorflow", "keras", "pytorch",
    "machine", "learning", "deep", "ai",
    "nlp", "opencv", "pandas", "numpy",
    "scikit", "sklearn", "excel", "powerbi",
    "tableau", "linux", "rest", "api"
}

# Keywords used for job-role prediction
ROLE_KEYWORDS = {
    "Data Scientist": {
        "python", "machine", "learning", "pandas",
        "numpy", "tensorflow", "keras", "sql"
    },
    "Machine Learning Engineer": {
        "python", "tensorflow", "keras", "pytorch",
        "machine", "learning", "docker"
    },
    "Python Developer": {
        "python", "flask", "django", "fastapi",
        "api", "mysql"
    },
    "Frontend Developer": {
        "html", "css", "javascript",
        "react", "angular", "vue"
    },
    "Backend Developer": {
        "node", "express", "django",
        "flask", "api", "sql"
    },
    "Full Stack Developer": {
        "html", "css", "javascript",
        "react", "node", "express",
        "mongodb", "mysql"
    }
}


def extract_skills(text):
    """
    Extract technical skills from resume text.
    """
    words = set(re.findall(r"[a-zA-Z+#]+", text.lower()))
    return sorted(list(words.intersection(TECHNICAL_SKILLS)))


def analyze_resume(resume_text, job_text):
    """
    Main analysis function.
    """

    matched = matching_keywords(resume_text, job_text)
    missing = missing_keywords(resume_text, job_text)
    skills = extract_skills(resume_text)

    return {
        "matched_keywords": matched,
        "missing_keywords": missing,
        "skills_found": skills,
        "skills_count": len(skills),
        "matched_count": len(matched),
        "missing_count": len(missing)
    }


def predict_job_role(resume_text):
    """
    Predict the best matching job role.
    """

    resume_words = get_keyword_set(resume_text)

    best_role = "General Software Engineer"
    highest_score = 0

    for role, keywords in ROLE_KEYWORDS.items():

        score = len(resume_words.intersection(keywords))

        if score > highest_score:
            highest_score = score
            best_role = role

    return best_role


def resume_strengths(resume_text):
    """
    Return resume strengths.
    """

    skills = extract_skills(resume_text)

    strengths = []

    if "python" in skills:
        strengths.append("Strong Python Programming")

    if "machine" in skills or "learning" in skills:
        strengths.append("Machine Learning Knowledge")

    if "sql" in skills or "mysql" in skills:
        strengths.append("Database Skills")

    if "react" in skills:
        strengths.append("Frontend Development")

    if "flask" in skills or "django" in skills:
        strengths.append("Backend Development")

    if "git" in skills:
        strengths.append("Version Control Experience")

    return strengths


def improvement_suggestions(resume_text, missing_skills):
    """
    Generate resume improvement suggestions.
    """

    suggestions = []

    if len(resume_text.split()) < 250:
        suggestions.append(
            "Increase resume content with projects and achievements."
        )

    if "github" not in resume_text.lower():
        suggestions.append(
            "Add your GitHub profile."
        )

    if "linkedin" not in resume_text.lower():
        suggestions.append(
            "Include your LinkedIn profile."
        )

    if "project" not in resume_text.lower():
        suggestions.append(
            "Mention academic or personal projects."
        )

    if "intern" not in resume_text.lower():
        suggestions.append(
            "Add internship experience if available."
        )

    if missing_skills:
        suggestions.append(
            "Include relevant missing technical skills."
        )

    return suggestions


def resume_grade(score):
    """
    Grade based on ATS score.
    """

    if score >= 90:
        return "A+"

    elif score >= 80:
        return "A"

    elif score >= 70:
        return "B+"

    elif score >= 60:
        return "B"

    elif score >= 50:
        return "C"

    return "Needs Improvement"
