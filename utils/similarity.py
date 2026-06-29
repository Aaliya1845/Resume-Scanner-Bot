from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_cleaner import clean_text


def calculate_similarity(resume_text, job_text):
    """
    Calculate cosine similarity between resume and job description.
    Returns percentage (0–100).
    """

    if not resume_text.strip() or not job_text.strip():
        return 0.0

    resume = clean_text(resume_text)
    job = clean_text(job_text)

    vectorizer = CountVectorizer()

    vectors = vectorizer.fit_transform([resume, job])

    similarity = cosine_similarity(vectors)[0][1]

    return round(similarity * 100, 2)


def get_grade(score):
    """
    Return resume grade.
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

    elif score >= 40:
        return "D"

    return "F"


def get_eligibility(score):
    """
    Eligibility status.
    """

    if score >= 60:
        return "Eligible"

    elif score >= 40:
        return "Partially Eligible"

    return "Not Eligible"


def score_color(score):
    """
    Color used in dashboard.
    """

    if score >= 80:
        return "green"

    elif score >= 60:
        return "orange"

    return "red"


def ats_score(score):
    """
    Convert similarity score into ATS score.
    """

    ats = score

    if ats > 100:
        ats = 100

    if ats < 0:
        ats = 0

    return round(ats)


def score_summary(score):
    """
    Summary shown to the user.
    """

    if score >= 90:
        return (
            "Excellent match! Your resume is highly aligned with "
            "the job description."
        )

    elif score >= 75:
        return (
            "Very good match. A few improvements can further "
            "increase your chances."
        )

    elif score >= 60:
        return (
            "Good match. Consider adding more relevant skills "
            "and project experience."
        )

    elif score >= 40:
        return (
            "Average match. Update your resume with missing "
            "technical skills."
        )

    return (
        "Low match. Your resume requires significant improvements "
        "to fit this role."
    )


def matched_percentage(match_count, total_required):
    """
    Percentage of required skills matched.
    """

    if total_required == 0:
        return 0

    return round((match_count / total_required) * 100, 2)


def resume_quality(score):
    """
    Resume quality level.
    """

    if score >= 90:
        return "Outstanding"

    elif score >= 80:
        return "Excellent"

    elif score >= 70:
        return "Very Good"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Average"

    return "Needs Improvement"
