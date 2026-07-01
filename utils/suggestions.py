"""
AI Suggestions Module
Resume Job Scanner V2
"""


def get_learning_resources(skill):
    """
    Return learning resources for a missing skill.
    """

    resources = {
        "python": "Python Official Docs | https://docs.python.org/3/",
        "java": "Oracle Java Tutorials",
        "sql": "SQLBolt | W3Schools SQL",
        "mysql": "MySQL Documentation",
        "mongodb": "MongoDB University",
        "html": "MDN HTML Guide",
        "css": "MDN CSS Guide",
        "javascript": "JavaScript.info",
        "react": "React Official Documentation",
        "node": "Node.js Official Documentation",
        "django": "Django Documentation",
        "flask": "Flask Documentation",
        "docker": "Docker Official Documentation",
        "kubernetes": "Kubernetes Documentation",
        "aws": "AWS Skill Builder",
        "azure": "Microsoft Learn",
        "tensorflow": "TensorFlow Tutorials",
        "pytorch": "PyTorch Tutorials",
        "machine": "Coursera Machine Learning",
        "learning": "Google Machine Learning Crash Course",
        "git": "Pro Git Book",
        "github": "GitHub Skills",
        "excel": "Microsoft Excel Training",
        "powerbi": "Microsoft Learn Power BI",
        "tableau": "Tableau Learning"
    }

    return resources.get(skill.lower(), "Search official documentation or a trusted online course.")


def certification_recommendations(role):
    """
    Certifications based on predicted role.
    """

    role = role.lower()

    if "data scientist" in role:
        return [
            "IBM Data Science Professional Certificate",
            "Google Advanced Data Analytics",
            "Microsoft Azure AI Engineer"
        ]

    elif "machine learning" in role:
        return [
            "TensorFlow Developer Certificate",
            "AWS Machine Learning Specialty",
            "Google ML Crash Course"
        ]

    elif "python" in role:
        return [
            "PCAP - Python Associate",
            "Python Institute Certification"
        ]

    elif "frontend" in role:
        return [
            "Meta Front-End Developer",
            "Google UX Design"
        ]

    elif "backend" in role:
        return [
            "AWS Developer Associate",
            "Node.js Certification"
        ]

    elif "full stack" in role:
        return [
            "Meta Full Stack Developer",
            "IBM Full Stack Cloud Developer"
        ]

    return [
        "Google Career Certificates",
        "AWS Cloud Practitioner"
    ]


def career_tips(score):
    """
    Career guidance based on ATS score.
    """

    if score >= 85:
        return [
            "Excellent resume.",
            "Start applying to top companies.",
            "Practice coding interviews."
        ]

    elif score >= 70:
        return [
            "Your resume is competitive.",
            "Improve projects and achievements.",
            "Keep your GitHub updated."
        ]

    elif score >= 50:
        return [
            "Add more technical skills.",
            "Mention internships and certifications.",
            "Improve project descriptions."
        ]

    return [
        "Rewrite your resume.",
        "Include relevant projects.",
        "Learn missing technical skills.",
        "Add measurable achievements."
    ]


def resume_improvement_checklist():
    """
    General checklist.
    """

    return [
        "Use a one-page resume.",
        "Keep formatting consistent.",
        "Add measurable achievements.",
        "Mention internships.",
        "Include certifications.",
        "Add GitHub profile.",
        "Add LinkedIn profile.",
        "Use ATS-friendly fonts.",
        "Avoid tables and excessive graphics.",
        "Customize the resume for each job."
    ]


def recommend_skills(missing_skills):
    """
    Prioritize the first 10 missing skills.
    """

    recommendations = []

    for skill in missing_skills[:10]:
        recommendations.append({
            "skill": skill,
            "resource": get_learning_resources(skill)
        })

    return recommendations


def motivational_message(score):
    """
    Display encouragement.
    """

    if score >= 80:
        return "🎉 Excellent! Your resume is highly competitive."

    elif score >= 60:
        return "👍 Good work! A few improvements can make your resume even stronger."

    elif score >= 40:
        return "💡 You're on the right track. Focus on adding relevant skills and projects."

    return "🚀 Every great career starts somewhere. Improve your resume and keep learning!"
