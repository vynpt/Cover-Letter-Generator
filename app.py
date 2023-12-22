import streamlit as st
import fitz  # PyMuPDF
import spacy
import pandas as pd
import re 

from gpt2_generate import *

# Load our trained NER model
from NER import *
model_dir = "NER/best_model"
nlp_ner = spacy.load(model_dir)

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="Cover Letter Generator", layout='wide') 
st.title('Cover Letter Generator')
st.markdown("Please upload your resume in .pdf and select a job you want to apply for. We will generate a cover letter that suits the job you chose based on your resume.")


jobs = pd.read_csv("Database/cleaned_job.csv", encoding='ISO-8859-1')

# Find the selected job details
# Add a search bar for users to search for a job
search_query = st.text_input("Search for a job title:")
filtered_jobs_df = jobs[jobs["title"].str.contains(search_query, case=False, na=False)]
selected_job_title = st.selectbox("Select a job title:", filtered_jobs_df["title"].astype(str).unique())
selected_job_df = filtered_jobs_df[filtered_jobs_df['title'] == selected_job_title]
selected_job = selected_job_df.iloc[0] if not selected_job_df.empty else None

# selected_job_title = st.selectbox("Select a job title:", jobs["title"].astype(str).unique())
# selected_job_df = jobs[jobs['title'] == selected_job_title]
# selected_job = selected_job_df.iloc[0] if not selected_job_df.empty else None

# Placeholder for the uploaded file path
# def extract_text_from_first_page(pdf_path):
#     # Open the PDF file
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         # Extract text from the first page
#         page = pdf[0]
#         text = page.get_text()
#     return text

# Create a file uploader widget
st.write("###")
uploaded_file = st.file_uploader("Upload your resume here", type="pdf")

text = ""
if uploaded_file is not None:
#     # Read the content of the file as bytes
    pdf_data = uploaded_file.read()
    
    # Open the PDF file
    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    
# Now you have the text content of the PDF and you can process it
text_uploaded = str(text)
st.text(text)  # Disp


# Extract skills from the uploaded resume and selected job description
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skill(text):
    unique_skills = set()

    doc = nlp(text)
    for sentence in doc.sents:
        sentence_doc = nlp_ner(sentence.text)
        for ent in sentence_doc.ents:
            # Check if the entity is a "SKILL" and not already extracted
            if ent.label_ == "SKILL" and ent.text not in unique_skills:
                # print(ent.text)
                unique_skills.add(ent.text)
    return unique_skills 

def clean_and_split_skills(skill_string):
    # Define a set of delimiters that can separate skills
    delimiters = '|', ',', '\\', '\n', '(', ')'
    regex_pattern = '|'.join(map(re.escape, delimiters))
    
    # Split the string based on the regex pattern and remove empty strings
    skills = [skill.strip() for skill in re.split(regex_pattern, skill_string) if skill.strip()]
    return skills

def common_skill(resume_skills, job_skills):
    # Split and clean resume and job skills
    all_resume_skills = set()
    for resume_skill in resume_skills:
        all_resume_skills.update(clean_and_split_skills(resume_skill))
    
    all_job_skills = set()
    for job_skill in job_skills:
        all_job_skills.update(clean_and_split_skills(job_skill))

    common = all_resume_skills.intersection(all_job_skills)
    return list(common)

skill_resume = extract_skill(text_uploaded)
skill_job = extract_skill(selected_job['description'])  

# Generate text with the fine-tuned model
list_skill = ' '.join(common_skill(skill_resume, skill_job))
prompt_text = f"Dear Hiring Managers,\n\n I am writing to express my interest in the {selected_job['title']} position. My skills in {list_skill} align well with this role and make me an excellent candidate for this role."
# prompt_text = f"Create a cover letter for Data Science job"
generated = generate_text(prompt_text)

body_container = st.container()
with body_container:
    submit_button = st.button(label='Generate Cover Letter')
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    if submit_button:
        
        with col1:
            st.subheader("Candidate:")
            st.write(f"Name: {extract_name(text_uploaded)}")
            if common_skill(skill_resume, skill_job) != set():
                st.write(f"Your skills relevant to this job:")
                for skill in common_skill(skill_resume, skill_job):
                    st.write(f"  {skill}")
            
        with col2:
            st.subheader("Your selected job:")
            st.write(f"Title: {selected_job['title']}")
            st.write(f"Job Description: {selected_job['description']}")
            st.write(f"Experience Level: {selected_job['formatted_experience_level']}")
            st.write(f"Location: {selected_job['location']}")
            st.write(f"Job Posting: {selected_job['job_posting_url']}")
                
        with col3:
            st.subheader("Generated Cover Letter")
            st.write(generated)
