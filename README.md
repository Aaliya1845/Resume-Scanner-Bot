Resume Job Scanner: An AI-Based Resume and Job Description Matching System

Abstract

Recruitment has become increasingly competitive, with organizations receiving thousands of resumes for a single job opening. Manually reviewing each application is time-consuming and prone to human error. This project, Resume Job Scanner, presents an AI-assisted resume screening system that compares a candidate's resume with a job description and calculates a similarity score based on textual analysis. The system extracts text from PDF resumes, preprocesses the content by removing unnecessary characters and converting it into a standard format, and applies the Count Vectorizer and Cosine Similarity algorithms to measure the relevance between the resume and the job description. Additionally, it identifies missing skills from the candidate's resume and provides eligibility feedback based on the calculated score. Developed using Python and Streamlit, the application provides an interactive and user-friendly interface for both recruiters and job seekers. The proposed system improves the efficiency of the recruitment process while reducing manual effort and increasing consistency in resume evaluation.

Keywords: Resume Screening, Artificial Intelligence, Natural Language Processing, Cosine Similarity, Streamlit, Recruitment Automation.

---

I. Introduction

The hiring process is an essential activity in every organization. With the increasing number of job applications, recruiters often face difficulties in evaluating resumes efficiently. Manual resume screening is time-consuming and may lead to inconsistent decision-making due to human bias.

Artificial Intelligence (AI) and Natural Language Processing (NLP) have introduced automated techniques for analyzing textual documents. These technologies enable the comparison of resumes with job descriptions, allowing recruiters to identify suitable candidates more quickly.

The objective of the Resume Job Scanner is to automate the initial screening process by calculating the similarity between a candidate's resume and a job description. The system also identifies missing skills and provides eligibility recommendations, making the recruitment process faster, more accurate, and transparent.

---

II. Literature Survey

Several researchers have proposed AI-based resume screening systems using Natural Language Processing techniques.

1. Resume parsing techniques have been widely adopted to extract structured information from resumes in PDF and document formats.

2. Natural Language Processing has been successfully applied to compare resumes with job descriptions by extracting keywords and analyzing semantic similarity.

3. Cosine Similarity is one of the most popular algorithms used for text similarity because of its simplicity, computational efficiency, and reliable performance in document comparison.

4. Machine Learning and AI-powered Applicant Tracking Systems (ATS) are increasingly being used by companies to rank resumes automatically before human evaluation.

Although existing systems provide accurate matching, many commercial solutions are expensive and inaccessible for educational purposes. The proposed Resume Job Scanner offers a lightweight, open-source alternative suitable for students, researchers, and small organizations.

---

III. Methodology

The proposed system follows the following workflow:

1. The user uploads a resume in PDF format.

2. The system extracts text from the PDF using the PyPDF2 library.

3. The extracted resume text and job description undergo preprocessing, including lowercase conversion and removal of punctuation and special characters.

4. Count Vectorizer converts both texts into numerical feature vectors.

5. Cosine Similarity computes the similarity between the resume and job description.

6. The similarity score is displayed as a percentage.

7. Missing skills are identified by comparing the processed resume text with the job description.

8. Based on predefined thresholds, the system categorizes candidates as:
   
   - Eligible (≥60%)
   - Partially Eligible (40–59%)
   - Not Eligible (<40%)

---

IV. Implementation

The Resume Job Scanner is implemented using Python and Streamlit.

Software Requirements

- Python 3.x
- Streamlit
- PyPDF2
- Scikit-learn
- NLTK
- Regular Expressions (re)

Modules Used

1. PDF Text Extraction

The PyPDF2 library extracts textual content from uploaded PDF resumes.

2. Text Preprocessing

The resume and job description are cleaned by:

- Converting text to lowercase.
- Removing special characters.
- Eliminating unnecessary symbols.

3. Feature Extraction

Count Vectorizer converts textual data into numerical vectors representing word frequencies.

4. Similarity Calculation

Cosine Similarity measures the closeness between the resume vector and the job description vector.

5. Skill Analysis

The application identifies keywords present in the job description but missing from the resume.

6. User Interface

Streamlit provides an interactive web interface for uploading resumes, entering job descriptions, calculating similarity scores, and displaying results.

---

V. Conclusion

The Resume Job Scanner demonstrates how Artificial Intelligence and Natural Language Processing can simplify the recruitment process. The system automatically evaluates resumes against job descriptions, computes similarity scores, identifies missing skills, and provides eligibility recommendations. The application minimizes manual effort, improves consistency, and serves as an effective educational project demonstrating practical applications of AI in recruitment.

---

VI. Future Aspects

The project can be further enhanced by incorporating advanced AI techniques and additional features such as:

- Integration with Large Language Models (LLMs).
- Semantic similarity using BERT or Sentence Transformers.
- OCR support for scanned resumes.
- Automatic resume improvement suggestions.
- Multi-language resume analysis.
- ATS compatibility score prediction.
- Recruiter dashboard with candidate ranking.
- Cloud deployment with user authentication.
- Skill recommendation using Generative AI.
- Integration with LinkedIn and online job portals.

---

References

[1] G. Salton and M. J. McGill, Introduction to Modern Information Retrieval. McGraw-Hill, 1983.

[2] C. D. Manning, P. Raghavan, and H. Schütze, Introduction to Information Retrieval. Cambridge University Press, 2008.

[3] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781, 2013.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[5] Streamlit Documentation. https://docs.streamlit.io/

[6] PyPDF2 Documentation. https://pypdf2.readthedocs.io/

[7] Bird, S., Klein, E., and Loper, E., Natural Language Processing with Python. O'Reilly Media, 2009.

[8] Jurafsky, D., and Martin, J. H., Speech and Language Processing, 3rd Edition, Pearson.
