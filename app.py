
import streamlit as st
import PyPDF2
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download nltk punkt if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def calculate_similarity(resume, job_desc):
    resume = clean_text(resume)
    job_desc = clean_text(job_desc)

    vectorizer = CountVectorizer().fit_transform([resume, job_desc])
    vectors = vectorizer.toarray()

    similarity = cosine_similarity(vectors)[0][1]
    return similarity * 100

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.set_page_config(layout="wide")
st.title("Resume-Job Description Matcher")

st.write("Upload a resume (PDF) and enter a job description to find the match percentage.")

# Resume Upload
st.header("1. Upload Your Resume")
resume_file = st.file_uploader("Choose a PDF file", type=["pdf"])

resume_text = ""
if resume_file is not None:
    resume_text = extract_text_from_pdf(resume_file)
    st.success("Resume uploaded and text extracted successfully!")
    with st.expander("View Extracted Resume Text"):
        st.write(resume_text)

# Job Description Input
st.header("2. Enter Job Description")
job_description = st.text_area("Paste the job description here:", height=200)

# Calculate Similarity
st.header("3. Calculate Match")
if st.button("Calculate Match Percentage"):
    if resume_text and job_description:
        match_percentage = calculate_similarity(resume_text, job_description)

        st.subheader("Match Result:")
        st.metric(label="Match Percentage", value=f"{match_percentage:.2f}%")

        if match_percentage >= 60:
            st.balloons()
            st.success("✅ Eligible 🎉")
        elif match_percentage >= 40:
            st.info("⚠️ Partially Eligible 🙂")
        else:
            st.warning("❌ Not Eligible 😢")

        st.subheader("Skills Breakdown (Missing from Resume):")
        required_skills = set(clean_text(job_description).split(','))
        resume_words = set(clean_text(resume_text).split())

        missing_skills = required_skills - resume_words

        if missing_skills:
            for skill in sorted(list(missing_skills)):
                if skill.strip(): # Avoid printing empty strings from extra commas
                    st.markdown(f"- {skill.strip().capitalize()}")
        else:
            st.success("All required skills found in resume!")

    else:
        st.error("Please upload a resume and enter a job description to calculate the match.")
