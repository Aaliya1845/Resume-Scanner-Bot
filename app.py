import streamlit as st
import os
from utils.resume_parser import extract_resume_text
from utils.ats_engine import analyze_resume

# Page configuration
st.set_page_config(
    page_title="AI ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# Header
st.markdown("""
<div class="hero">
    <h1>AI ATS Resume Analyzer</h1>
    <p>Analyze your resume, improve ATS score and get AI-powered suggestions.</p>
</div>
""", unsafe_allow_html=True)


# Upload Section
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Upload Resume")

resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Enter the job description here..."
)

st.markdown('</div>', unsafe_allow_html=True)


# Analyze Button
if st.button("Analyze Resume"):

    if resume_file is None:
        st.warning("Please upload your resume.")

    elif job_description.strip() == "":
        st.warning("Please enter job description.")

    else:

        with st.spinner("Analyzing resume..."):

            resume_text = extract_resume_text(resume_file)

            result = analyze_resume(
                resume_text,
                job_description
            )


        st.success("Analysis Completed!")


        # Score Card
        st.markdown("""
        <div class="card">
        <h2>ATS Score</h2>
        </div>
        """, unsafe_allow_html=True)


        score = result["score"]

        st.progress(score / 100)

        st.markdown(
            f"""
            <div class="score">
                {score}%
            </div>
            """,
            unsafe_allow_html=True
        )


        # Match Keywords
        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="card">
            <h3>Matched Skills</h3>
            """,
            unsafe_allow_html=True)

            for skill in result["matched_skills"]:
                st.write("✅", skill)

            st.markdown("</div>",
            unsafe_allow_html=True)



        with col2:

            st.markdown("""
            <div class="card">
            <h3>Missing Skills</h3>
            """,
            unsafe_allow_html=True)

            for skill in result["missing_skills"]:
                st.write("❌", skill)

            st.markdown("</div>",
            unsafe_allow_html=True)



        # Suggestions

        st.markdown("""
        <div class="card">
        <h3>AI Improvement Suggestions</h3>
        """,
        unsafe_allow_html=True)


        for suggestion in result["suggestions"]:
            st.write("💡", suggestion)


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# Footer
st.markdown("""
<div class="footer">
AI ATS Resume Analyzer | Built with Python + Streamlit
</div>
""",
unsafe_allow_html=True)
