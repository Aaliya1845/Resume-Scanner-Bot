import streamlit as st
import os
from utils.pdf_reader import extract_text_from_pdf, get_pdf_statistics, validate_pdf
from utils.text_cleaner import resume_statistics
from utils.similarity import calculate_similarity, ats_score, get_grade, get_eligibility, score_summary
from utils.analyzer import analyze_resume, predict_job_role, resume_strengths, improvement_suggestions
from utils.charts import ats_gauge
from utils.report_generator import generate_pdf_report

st.set_page_config(page_title="Resume Scanner Bot", layout="wide")

if os.path.exists("style.css"):
    with open("style.css","r",encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Resume Scanner Bot")
st.caption("AI Powered ATS Resume Analyzer")

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

    sim = calculate_similarity(resume_text, job_description)
    ats = ats_score(sim)
    grade = get_grade(ats)
    eligibility = get_eligibility(ats)
    summary = score_summary(ats)

    analysis = analyze_resume(resume_text, job_description)
    role = predict_job_role(resume_text)
    strengths = resume_strengths(resume_text)
    suggestions = improvement_suggestions(resume_text, analysis["missing_keywords"])

    st.plotly_chart(ats_gauge(ats), use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("ATS Score", f"{ats}%")
    c2.metric("Grade", grade)
    c3.metric("Eligibility", eligibility)

    st.info(summary)
    st.subheader("Predicted Job Role")
    st.success(role)

    st.subheader("Resume Statistics")
    st.write(pdf_stats)
    st.write(text_stats)

    col1,col2=st.columns(2)
    with col1:
        st.subheader("Matched Skills")
        if analysis["matched_keywords"]:
            st.success(", ".join(analysis["matched_keywords"]))
        else:
            st.write("No matched skills found.")
    with col2:
        st.subheader("Missing Skills")
        if analysis["missing_keywords"]:
            st.error(", ".join(analysis["missing_keywords"]))
        else:
            st.success("No missing skills.")

    st.subheader("Resume Strengths")
    for s in strengths:
        st.write("•", s)

    st.subheader("Improvement Suggestions")
    for s in suggestions:
        st.write("•", s)

    with st.expander("Extracted Resume Text"):
        st.text(resume_text)

    pdf_path = generate_pdf_report(
        "resume_report.pdf",
        ats,
        grade,
        role,
        analysis["matched_keywords"],
        analysis["missing_keywords"],
        strengths,
        suggestions,
        summary,
    )
    with open(pdf_path,"rb") as f:
        st.download_button("Download PDF Report",f,"Resume_Report.pdf","application/pdf")

