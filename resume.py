#STEP 1: Install required libraries

!pip install scikit-learn pandas python-docx

#STEP 2: Import libraries

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files
from docx import Document

# STEP 3: Helper functions

def clean_text(text):
    """Cleans text by removing symbols, numbers, and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def read_docx(file_path):
    """Reads text content from a .docx file."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def calculate_match(job_description, resume_text):
    """Calculates cosine similarity between job description and resume."""
    texts = [job_description, resume_text]
    cv = CountVectorizer()
    matrix = cv.fit_transform(texts)
    similarity = cosine_similarity(matrix)[0][1]
    return similarity * 100

#STEP 4: Upload Job Description

print("Please upload the Job Description file (.txt or .docx):")
uploaded_job = files.upload()
job_file = list(uploaded_job.keys())[0]

# STEP 5: Upload Resumes (.docx)

print("Now upload all the Resume files (.docx):")
uploaded_resumes = files.upload()

#STEP 6: Create folder to store resumes

resume_folder = "resumes"
os.makedirs(resume_folder, exist_ok=True)

#Save uploaded resumes to the folder
for resume_name, resume_content in uploaded_resumes.items():
    with open(os.path.join(resume_folder, resume_name), "wb") as f:
        f.write(resume_content)

print(f"\n All resumes saved to folder: '{resume_folder}'")

# STEP 7: Read and clean Job Description

if job_file.endswith(".txt"):
    with open(job_file, "r", encoding="utf-8") as f:
        job_description = clean_text(f.read())
elif job_file.endswith(".docx"):
    job_description = clean_text(read_docx(job_file))
else:
    raise ValueError("Please upload job description in .txt or .docx format")

#STEP 8: Process each resume

results = []
for resume_name in os.listdir(resume_folder):
    resume_path = os.path.join(resume_folder, resume_name)

    # Read .docx resumes
    if resume_name.endswith(".docx"):
        resume_text = read_docx(resume_path)
    else:
        print(f"Skipping unsupported file: {resume_name}")
        continue

    # Clean and calculate match
    resume_text = clean_text(resume_text)
    match_score = calculate_match(job_description, resume_text)
    eligibility = "Eligible" if match_score >= 50 else "Not Eligible"

    results.append({
        "Resume File": resume_name,
        "Match Percentage": round(match_score, 2),
        "Status": eligibility
    })

# STEP 9: Display and Save Results

df = pd.DataFrame(results)
df.sort_values(by="Match Percentage", ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

print("\n Resume Analysis Completed!\n")
print(df)

# Save as CSV
output_file = "shortlisted.csv"
df.to_csv(output_file, index=False)

print("\n Results saved as 'shortlisted.csv'")
files.download(output_file)