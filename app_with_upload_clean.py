
import os
import re
import io

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resume-to-Job Matching Platform",
    page_icon="📄",
    layout="wide"
)

st.title("Resume-to-Job Matching Platform")
st.write(
    "Match resumes with job descriptions using text similarity "
    "and skill coverage."
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Resume_Job_Matching",
    "data",
    "data",
    "all_resumes.csv"
)

SKILLS = [
    "python",
    "machine learning",
    "sql",
    "pandas",
    "numpy",
    "scikit-learn",
    "data analysis",
    "data visualization"
]




def clean_text(text):
    text = str(text or "")

    # Fix common text-encoding problems
    replacements = {
        "â€“": "-",
        "â€”": "-",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€™": "'",
        "Â": "",
        "ï¼\u200b": "",
        "ï¼": "",
        "ΓÇö": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Normalize spaces
    return re.sub(r"\s+", " ", text).strip()



def extract_text(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        unique_pages = []
        seen_pages = set()

        for page in reader.pages:
            page_text = page.extract_text() or ""
            normalized_page = clean_text(page_text)

            if normalized_page and normalized_page not in seen_pages:
                unique_pages.append(page_text)
                seen_pages.add(normalized_page)

        return clean_text(" ".join(unique_pages))


    if name.endswith(".docx"):
        document = Document(io.BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs]
        return clean_text(" ".join(paragraphs))

    if name.endswith(".txt"):
        return clean_text(data.decode("utf-8", errors="ignore"))

    raise ValueError("Unsupported file type.")


def load_resumes():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Resume dataset was not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "resume_text" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'resume_text' column."
        )

    df["resume_text"] = df["resume_text"].fillna("").map(clean_text)
    df = df[df["resume_text"].str.len() > 0].copy()
    return df.reset_index(drop=True)


def calculate_matches(df, job_description):
    """Use the same TF-IDF, cosine similarity, and 50/50 score."""
    texts = df["resume_text"].fillna("").map(clean_text).tolist()

    if not texts or not clean_text(job_description):
        raise ValueError("Resume text and job description are required.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    resume_vectors = vectorizer.fit_transform(texts)
    job_vector = vectorizer.transform([job_description])

    similarities = cosine_similarity(
        job_vector, resume_vectors
    ).flatten()

    results = df.copy()
    results["match_percentage"] = similarities * 100


    def find_skills(text):
        text = clean_text(text).lower()
        return [
            skill for skill in SKILLS
            if re.search(r"\b" + re.escape(skill) + r"\b", text)
        ]

    required_skills = find_skills(job_description)

    results["skills_found"] = results["resume_text"].map(find_skills)

    if required_skills:
        results["matched_skills"] = results["resume_text"].map(
            lambda text: [
                skill for skill in required_skills
                if re.search(
                    r"\b" + re.escape(skill) + r"\b",
                    clean_text(text).lower()
                )
            ]
        )
        results["skill_count"] = results["matched_skills"].map(len)
        results["skill_coverage_percentage"] = (
            results["skill_count"] / len(required_skills) * 100
        )
    else:
        results["matched_skills"] = [[] for _ in range(len(results))]
        results["skill_count"] = 0
        results["skill_coverage_percentage"] = 0.0


    results["skills_found"] = results["resume_text"].map(find_skills)
    results["skill_count"] = results["skills_found"].map(len)
    results["skill_coverage_percentage"] = (
        results["skill_count"] / len(SKILLS) * 100
    )

    results["combined_score"] = (
        0.5 * results["match_percentage"]
        + 0.5 * results["skill_coverage_percentage"]
    )

    return results.sort_values(
        "combined_score", ascending=False
    ).reset_index(drop=True)


def show_results(results, title="Top Matching Resumes"):
    st.subheader(title)

    display = results.head(10).copy()
    display["skills_found"] = display["skills_found"].map(
        lambda skills: ", ".join(skills)
    )

    display_columns = [
        c for c in [
            "filename",
            "category",
            "match_percentage",
            "skills_found",
            "skill_coverage_percentage",
            "combined_score"
        ] if c in display.columns
    ]

    st.dataframe(
        display[display_columns].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Scores measure textual similarity and coverage of eight "
        "predefined skills. They are not probabilities of being hired."
    )

    csv_data = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download top matching results (CSV)",
        data=csv_data,
        file_name="matching_resumes.csv",
        mime="text/csv"
    )

    st.subheader("Why These Resumes Matched")

    for index, row in display.iterrows():
        category = row.get("category", "Resume")
        filename = row.get("filename", f"Resume {index + 1}")

        with st.expander(
            f"{filename} — {row['match_percentage']:.2f}% text similarity"
        ):
            st.write(f"**Category:** {category}")
            st.write(
                f"**Skill coverage:** "
                f"{row['skill_coverage_percentage']:.1f}%"
            )
            st.write(
                "**Skills found:** "
                + (", ".join(row["skills_found"]) or "None of the listed skills")
            )
            st.write("**Resume text:**")
            st.write(row["resume_text"][:5000] or "No text available.")


# Load dataset only when a mode needs it.
tab1, tab2, tab3 = st.tabs([
    "Stored Resume Matching",
    "Upload a Resume",
    "Upload a Job Description"
])

with tab1:
    st.subheader("Match against stored resumes")

    job_title = st.text_input(
        "Job Title",
        key="stored_job_title"
    )
    job_description = st.text_area(
        "Job Description",
        height=180,
        key="stored_job_description"
    )

    if st.button("Find Matching Resumes", key="stored_match"):
        if not job_title.strip() or not job_description.strip():
            st.warning("Please enter both the job title and description.")
        else:
            try:
                with st.spinner("Matching resumes..."):
                    resumes = load_resumes()
                    matches = calculate_matches(resumes, job_description)
                st.success(f"Found the top matches for: {job_title}")
                show_results(matches)
            except Exception as exc:
                st.error(f"Matching failed: {exc}")


with tab2:
    st.subheader("Upload a resume and match it to a job")

    resume_file = st.file_uploader(
        "Upload resume (PDF or DOCX)",
        type=["pdf", "docx"],
        key="resume_upload"
    )

    uploaded_job_title = st.text_input(
        "Job Title",
        key="uploaded_job_title"
    )

    job_description_file = st.file_uploader(
        "Upload job description (PDF, DOCX, or TXT) — optional",
        type=["pdf", "docx", "txt"],
        key="resume_job_file"
    )

    uploaded_job_description = st.text_area(
        "Or paste the job description here",
        height=150,
        key="resume_job_text"
    )

    if st.button("Match Uploaded Resume", key="uploaded_resume_match"):
        try:
            if resume_file is None:
                st.warning("Please upload a resume first.")
            else:
                resume_text = extract_text(resume_file)

                if job_description_file is not None:
                    job_text = extract_text(job_description_file)
                else:
                    job_text = clean_text(uploaded_job_description)

                if not resume_text:
                    st.error(
                        "No text could be extracted from the resume. "
                        "Scanned PDFs may need OCR."
                    )
                elif not job_text:
                    st.warning(
                        "Please upload a job description or paste its text."
                    )
                else:
                    one_resume = pd.DataFrame({
                        "filename": [resume_file.name],
                        "category": ["Uploaded Resume"],
                        "resume_text": [resume_text]
                    })

                    st.write("DEBUG — Job description text length:", len(job_text))
                    st.write("DEBUG — Resume text length:", len(resume_text))
                    result = calculate_matches(one_resume, job_text)
                    row = result.iloc[0]

                    st.success("Resume matched successfully.")
                    st.metric(
                        "Text Similarity",
                        f"{row['match_percentage']:.2f}%"
                    )
                    st.metric(
                        "Skill Coverage",
                        f"{row['skill_coverage_percentage']:.1f}%"
                    )
                    st.metric(
                        "Combined Score",
                        f"{row['combined_score']:.2f}"
                    )
                    st.write("**Skills found:**")
                    st.write(", ".join(row["skills_found"]) or "None of the listed skills")
                    st.write("**Extracted resume text:**")
                    st.text(resume_text[:5000])

        except Exception as exc:
            st.error(f"Resume matching failed: {exc}")


with tab3:
    st.subheader("Upload a job description and rank stored resumes")

    jd_file = st.file_uploader(
        "Upload job description (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="jd_upload"
    )

    jd_title = st.text_input(
        "Job Title",
        key="uploaded_jd_title"
    )

    if st.button("Find Resumes for Uploaded Job", key="uploaded_jd_match"):
        if jd_file is None:
            st.warning("Please upload a job description.")
        else:
            try:
                jd_text = extract_text(jd_file)

                if not jd_text:
                    st.error(
                        "No text could be extracted from the job description."
                    )
                else:
                    with st.spinner("Matching stored resumes..."):
                        resumes = load_resumes()
                        matches = calculate_matches(resumes, jd_text)

                    st.success(
                        "Matching completed"
                        + (f" for {jd_title}" if jd_title.strip() else "")
                    )
                    show_results(matches, "Top Resumes for Uploaded Job")

            except Exception as exc:
                st.error(f"Job-description matching failed: {exc}")
