import streamlit as st
import fitz  # PyMuPDF
import spacy
import pandas as pd
from io import StringIO
from spacy.matcher import Matcher

# from selected_resume import *
# from NER import *

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="Cover Letter Generator", layout='wide') 
st.title('Cover Letter Generator')
st.markdown("Please upload your resume in .pdf and select a job you want to apply for. We will generate a cover letter that suits the job you chose based on your resume.")


jobs = pd.read_csv("Database/Job_cleaned.csv", encoding='ISO-8859-1')

# Find the selected job details
selected_job_title = st.selectbox("Select a job title:", sorted(jobs["title"].astype(str).unique()))
selected_job_df = jobs[jobs['title'] == selected_job_title]
selected_job = selected_job_df.iloc[0] if not selected_job_df.empty else None

# Placeholder for the uploaded file path
def extract_text_from_first_page(pdf_path):
    # Open the PDF file
    with fitz.open(pdf_path) as pdf:
        text = ""
        # Extract text from the first page
        page = pdf[0]
        text = page.get_text()
    return text

def extract_resume_info(text):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Placeholder for extracted info
    resume_info = {
        "name": [],
        "skills": [],
        "projects": [],
        "education": []
    }
    
    # Assume the first named entity of type PERSON is the name
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            resume_info["name"].append(ent.text)
        elif ent.label_ == "SKILL":
            resume_info["skills"].append(ent.text)
        elif ent.label_ == "EXPERIENCES":
            resume_info["projects"].append(ent.text)
        elif ent.label_ == "EDUCATION":
            resume_info["Education"].append(ent.text)
    return resume_info

# Function to handle the uploaded file
def handle_uploaded_file(text):
    # text = extract_text_from_first_page(uploaded_file_path)
    info = extract_resume_info(text)
    return info


# Create a file uploader widget
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
    
extracted_info = handle_uploaded_file(text_uploaded)
# print(extracted_info)

body_container = st.container()
with body_container:
    submit_button = st.button(label='Generate Cover Letter')
    col1, col2, col3 = st.columns([1, 1, 1.5])

    if submit_button:
        # with st.spinner("Working hard..."):
            # try:
            #     response, response_tokens, prompt_tokens, completion_tokens, response_cost = generate_response(resume_content)
           
        with col1:
            st.subheader("Candidate:")
            st.write(f"Name: {extracted_info['name'][0]}")
            st.write(f"Skills relevant to this job: {extracted_info['skills']}")
            
                
        with col2:
            st.subheader("Your selected job:")
            st.write(f"Company: ")
            st.write(f"Title: {selected_job['title']}")
            st.write(f"Job Description: {selected_job['description']}")
            st.write(f"Experience Level: {selected_job['formatted_experience_level']}")
            st.write(f"Location: {selected_job['location']}")
            st.write(f"Job Posting: {selected_job['job_posting_url']}")
    
                
        with col3:
            st.subheader("Generated Cover Letter")