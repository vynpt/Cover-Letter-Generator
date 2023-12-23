import streamlit as st
import fitz  # PyMuPDF
import spacy
import pandas as pd
import re 

from gpt2_generate import *
# from llama2_generate import *

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

# Generate text with the gpt2 fine-tuned model
list_skill = ' '.join(common_skill(skill_resume, skill_job))
prompt_text1 = f"Dear Hiring Managers,\n\n I am writing to express my interest in the {selected_job['title']} position. My skills in {list_skill} align well with this role and make me an excellent candidate."
# prompt_text = f"Create a cover letter for Data Science job"
generated = generate_text(prompt_text1)

# Generate text with the gpt2 fine-tuned model
prompt_text2 = f"Dear Hiring Managers,\n\n I am writing to express my interest in the {selected_job['title']} position. My skills in {list_skill} align well with this role and make me an excellent candidate."
# prompt_text = f"Create a cover letter for Data Science job"
generated2 = regular_gpt2(prompt_text2)

# Generate text using the pipeline of llama2 model
# generated_llama2 = generate_cv(list_skill, selected_job['title'], selected_job['description'])


body_container = st.container()
with body_container:
    submit_button = st.button(label='Generate Cover Letter')
    col1, col2, col3, col4 = st.columns([0.75, 1.25, 1.5, 1.5])

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
            st.subheader("Generated Cover Letter (GPT-2 finetuned)")
            st.write(generated)
            st.write("Perplexity: ", compute_perplexity())
            
        # with col4: 
        #     st.subheader("Generated Cover Letter (GPT-2 regular)")
        #     st.write(generated2)
            
        with col4: 
            st.subheader("Generated Cover Letter (LLAMA-2)")
        #     # st.write(generated_llama2)
            
        #     st.write("""Dear Hiring Manager, \n\n I am writing to apply for the position of Data Scientist at WEX, as advertised on [website]. With over 3 years of experience in statistical modeling and quantitative analysis, I am confident that my skills and expertise align with the requirements of this role. 
        #                 As a certified data scientist with a Bachelor's degree in Data Science and a strong background in machine learning, statistics, and data visualization, I possess a unique combination of technical and domain expertise that makes me an ideal candidate for this position. My experience includes \n\n
                        
        #                 * Experience in Python programming, including data analysis and machine learning tasks \n
        #                 * Experience in data science, including data visualization and tableau \n
        #                 * Strong understanding of graduate-level multivariate statistical techniques and sampling methods, including but not limited to multivariate regression, ANOVA, factor analysis, cluster analysis, and principal components analysis \n\n
        #                 * Proficient in SQL and excel, with the ability to effectively utilize advanced functions and features \n
        #                 * Experience in mentoring and developing junior analysts roven track record of collaborating closely with cross-functional teams to strategize, plan, and analyze A/B tests \n
        #                 * Strong communication skills, with the ability to translate, communicate, and present results and recommendations to a non-technical audience \n
                        
        #                 In my current role as a Data Science Consultant at Datamites, I have been responsible for analyzing and processing complex data sets using advanced querying, visualization, and analytics tools. I have worked extensively with Python, R, and SQL, 
        #                 and have experience in machine learning tools and statistical techniques. I have also demonstrated a strong ability to work independently and collaboratively, and to effectively communicate technical findings to both technical and non-technical stakeholders.
        #                 As a strong advocator of the augmented era, I am passionate about bringing business concepts in areas of machine learning, AI, and robotics to real-life solutions. 
        #                 I believe that data science can be used to drive business growth and improve operational efficiency, and I am excited about the opportunity to join NRG and contribute my skills and expertise to the success of the company.
        #                 In particular, I am drawn to WEX's focus on promoting customized offerings and leveraging data to optimize marketing efforts.""")