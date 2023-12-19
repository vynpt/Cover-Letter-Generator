import spacy
import pandas as pd
import streamlit as st
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

df = pd.read_csv("Database/resume_data.csv")

# Load the pre-trained spaCy model
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

skills_list = [
    "Python", "JavaScript", "React", "SQL", "Git",
    "R", "Tableau", "Machine Learning",
    "HTML", "CSS", "PHP", "WordPress",
    "Excel", "Power BI",
    "Data Cleaning", "Data Visualization", "Data Analysis",
    "Statistics", "Big Data", "Hadoop", "Spark"
]

df = pd.read_csv("Database/resume_data.csv")
selected_resume = df[df['JOB NAME'] == 'Data Science Job']
resume1 = selected_resume.iloc[3].astype(str)
resume2 = selected_resume.iloc[4].astype(str)
resume3 = selected_resume.iloc[5].astype(str)


# # Extract entities
# for entity in doc.ents:
#     if entity.label_ in skills_list:  # These labels might vary
#         print(entity.text)
        
skill_patterns = list(nlp.pipe(skills_list))
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("SKILL", skill_patterns)

@Language.component("skill_component")
def skill_component(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="SKILL") for match_id, start, end in matches]
    doc.ents = spans
    return doc

nlp.add_pipe("skill_component", after="ner")
print(nlp.pipe_names)

skills = []
resume_texts = [resume1]
for idx, text in enumerate(resume_texts):
    doc = nlp(str(text))
    unique_skills = set()
    print(f"Skills in Resume {idx + 1}:")
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            unique_skills.add(ent.text)
    for skill in unique_skills:
        skills += skill


# st.set_page_config(page_title="Cover Letter Generator", layout='wide') 
# st.title('Cover Letter Generator')
# st.markdown("This app will generate a cover letter for you based on your resume.")   
# answer_container = st.container()
# with answer_container:
#     submit_button = st.button(label='Generate Cover Letter')
#     # col1, col2 = st.columns(2, gap='medium')
#     col1, col2, col3 = st.columns([1, 1, 1.5])

#     if submit_button:
#         with st.spinner("Working hard..."):
#             # try:
#             #     response, response_tokens, prompt_tokens, completion_tokens, response_cost = generate_response(resume_content)
           
#                 with col1:
#                     st.subheader("Candidate:")
#                     st.write(f"Name: {resume1['First Name']} {resume1['Last Name']}")
#                     st.write(f"Skills: {unique_skills}")
                
#                 with col2:
#                     st.subheader("Job applied for:")
#                     st.write(f"Job Name: {resume1['JOB NAME']}")
#                     st.write(f"Skills: {unique_skills}")
                
#                 with col3:
#                     st.subheader("Generated Cover Letter")
                    # st.write(response)

                # with col3:
                #     st.subheader("Evaluation Metrics")
            #         st.write(f"Total Tokens: {response_tokens}")
            #         st.write(f"Prompt Tokens: {prompt_tokens}")
            #         st.write(f"Completion Tokens: {completion_tokens}")
            #         st.write(f"Total Cost (USD): ${response_cost}")

            # except Exception as e:
            #     st.write(f"No repsonse. Error: {e}")