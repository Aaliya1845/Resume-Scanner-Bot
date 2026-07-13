import streamlit as st
import os

from utils.pdf_reader import (
    extract_text_from_pdf,
    validate_pdf
)

from utils.similarity import (
    calculate_similarity,
    ats_score,
    get_grade,
    get_eligibility,
    score_summary
)

from utils.analyzer import (
    analyze_resume,
    predict_job_role,
    improvement_suggestions
)

from utils.charts import ats_gauge

from utils.report_generator import generate_pdf_report


st.set_page_config(
    page_title="Resume Scanner Bot",
    layout="wide",
    page_icon="📄"
)


if os.path.exists("style.css"):

    with open("style.css", "r", encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


st.markdown(
    '<div class="main-title">Resume Scanner Bot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered ATS Resume Analyzer</div>',
    unsafe_allow_html=True
)


with st.sidebar:

    st.header("Resume Scanner")

    st.write(
        "Upload your Resume PDF and paste the Job Description to check ATS compatibility."
    )


resume_file = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "💼 Paste Job Description",
    height=220
)


if st.button(
    "Analyze Resume",
    type="primary"
):

    if not resume_file:

        st.error("Please upload your Resume.")

        st.stop()

    if not job_description.strip():

        st.error("Please paste the Job Description.")

        st.stop()

    valid, message = validate_pdf(resume_file)

    if not valid:

        st.error(message)

        st.stop()

    resume_text = extract_text_from_pdf(
        resume_file
    )

    similarity = calculate_similarity(
        resume_text,
        job_description
    )

    ats = ats_score(similarity)

    grade = get_grade(ats)

    eligibility = get_eligibility(ats)

    summary = score_summary(ats)

    analysis = analyze_resume(
        resume_text,
        job_description
    )

    predicted_role = predict_job_role(
        resume_text
    )

    suggestions = improvement_suggestions(
        resume_text,
        analysis["missing_keywords"]
    )

    st.plotly_chart(
        ats_gauge(ats),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "ATS Score",
            f"{ats}%"
        )

    with col2:

        st.metric(
            "Resume Grade",
            grade
        )

    with col3:

        st.metric(
            "Eligibility",
            eligibility
        )
    
    st.subheader("✅ Matched Skills")

    if analysis["matched_keywords"]:

        for skill in analysis["matched_keywords"]:

            st.markdown(f"✔ **{skill}**")

    else:

        st.info("No matched skills found.")


    st.divider()

    st.subheader("❌ Missing Skills")

    if analysis["missing_keywords"]:

        for skill in analysis["missing_keywords"]:

            st.markdown(f"✖ **{skill}**")

    else:

        st.success("No missing skills found.")


    st.divider()


    st.subheader("📄 Extracted Resume")

    st.text_area(
        "Resume Content",
        value=resume_text,
        height=700,
        disabled=True
    )


    st.divider()


    pdf_path = generate_pdf_report(

        "resume_report.pdf",

        ats,

        grade,

        predicted_role,

        analysis["matched_keywords"],

        analysis["missing_keywords"],

        [],

        suggestions,

        summary

    )


    with open(pdf_path, "rb") as pdf_file:

        st.download_button(

            label="📥 Download ATS Report",

            data=pdf_file,

            file_name="Resume_Report.pdf",

            mime="application/pdf",

            use_container_width=True

)
