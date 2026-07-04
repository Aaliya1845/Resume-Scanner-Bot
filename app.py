
import streamlit as st
import os

from utils.pdf_reader import extract_text_from_pdf, get_pdf_statistics, validate_pdf
from utils.text_cleaner import resume_statistics, extract_keywords
from utils.similarity import (
    calculate_similarity, ats_score, get_grade,
    get_eligibility, score_summary
)
from utils.analyzer import (
    analyze_resume, predict_job_role,
    resume_strengths, improvement_suggestions
)
from utils.charts import (
    ats_gauge, skill_bar_chart,
    pie_chart, keyword_chart, create_wordcloud
)
from utils.suggestions import (
    career_tips, certification_recommendations,
    recommend_skills, motivational_message,
    resume_improvement_checklist
)
from utils.report_generator import generate_pdf_report

st.set_page_config(page_title="Resume Scanner Bot", layout="wide")

if os.path.exists("style.css"):
    with open("style.css","r",encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="main-title">Resume Scanner Bot </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Powered ATS Resume Analyzer</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Resume Scanner Bot")
    st.write("Upload a resume, paste a job description and generate an ATS style report.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=220)

if st.button("Analyze Resume", type="primary"):
    if not resume_file or not job_description.strip():
        st.error("Please upload a resume and enter a job description.")
        st.stop()

    ok, msg = validate_pdf(resume_file)
    if not ok:
        st.error(msg)
        st.stop()

    resume_text = extract_text_from_pdf(resume_file)
    pdf_stats = get_pdf_statistics(resume_file)
    text_stats = resume_statistics(resume_text)

    score = calculate_similarity(resume_text, job_description)
    ats = ats_score(score)
    grade = get_grade(ats)
    eligibility = get_eligibility(ats)
    summary = score_summary(ats)

    analysis = analyze_resume(resume_text, job_description)
    role = predict_job_role(resume_text)
    strengths = resume_strengths(resume_text)
    suggestions = improvement_suggestions(
        resume_text, analysis["missing_keywords"]
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ATS Score", f"{ats}/100")
    c2.metric("Grade", grade)
    c3.metric("Pages", pdf_stats["pages"])
    c4.metric("Words", text_stats["total_words"])

    st.progress(ats/100)

    col1,col2=st.columns(2)
    with col1:
        st.plotly_chart(ats_gauge(ats), use_container_width=True)
        fig = keyword_chart(text_stats["keywords"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(skill_bar_chart(
            analysis["matched_keywords"],
            analysis["missing_keywords"]), use_container_width=True)
        st.plotly_chart(pie_chart(
            analysis["matched_keywords"],
            analysis["missing_keywords"]), use_container_width=True)

    wc = create_wordcloud(resume_text)
    if wc:
        st.pyplot(wc)

    st.subheader("Predicted Job Role")
    st.success(role)
    st.info(summary)
    st.write("**Eligibility:**", eligibility)

    st.subheader("Resume Strengths")
    st.write(strengths if strengths else ["No major strengths detected."])

    st.subheader("Missing Skills")
    st.write(analysis["missing_keywords"])

    st.subheader("Recommendations")
    for item in recommend_skills(analysis["missing_keywords"]):
        st.write(f"**{item['skill']}** - {item['resource']}")

    st.subheader("Career Tips")
    for tip in career_tips(ats):
        st.write("- ", tip)

    st.subheader("Suggested Certifications")
    for cert in certification_recommendations(role):
        st.write("- ", cert)

    st.subheader("Resume Checklist")
    for chk in resume_improvement_checklist():
        st.write("- ", chk)

    st.success(motivational_message(ats))

    pdf_path = generate_pdf_report(
        "ATS_Report.pdf",
        ats,
        grade,
        role,
        analysis["matched_keywords"],
        analysis["missing_keywords"],
        strengths,
        suggestions,
        summary
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download PDF Report",
            f,
            file_name="ATS_Report.pdf",
            mime="application/pdf"
        )
