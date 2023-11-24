import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


st.set_page_config(page_title="Generating Cover Letter from a resume", layout='wide')
st.title('Cover Letter Generator')
#st.markdown("This app generates a cover letter based on your resume. Please upload your resume and fill in the required information. The app will generate a cover letter for you. You can also download the cover letter as a .docx file.")
st.markdown("This app generates a cover letter based on your resume. The app will generate a cover letter for you. You can also download the cover letter as a .docx file.")

API_KEY = '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'

template = """"You are a generator model that writes a cover letter based on a resume. This is the resume:\n{context}\n\n
I want you to produce a cover letter that is professional but concise with 2 paragraphs. The first paragraph summarizes the skills and tech stack of this candidate. The second paragraph describes a most outstanding project or experience.
"""

# load the dataset
df = pd.read_csv("UI/resume.csv")

# Displaying the selectbox
option = st.selectbox(
    'Select a Bill',
    ('Resume 1', 'Resume 2','Resume 3')
)

# Extracting the resume content from the selected option
selected_num = int(option[7:]) - 1
resume_content = df['cleaned_resume'].iloc[selected_num]

# Function to generate response
def generate_response(text):
    prompt = PromptTemplate(input_variables=["context"], template=template)

    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY,
                    temperature=0.01, model="gpt-3.5-turbo-1106"), prompt=prompt)
        
        response = llm.predict(context=text)
        
        #response = llm.predict(context=text, max_length=100, num_return_sequences=1, stop_sequences=["\n\n"])[0]
        return response, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost
    

answer_container = st.container()
with answer_container:
    submit_button = st.button(label='Generate Cover Letter')
    # col1, col2 = st.columns(2, gap='medium')
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    if submit_button:
        with st.spinner("Working hard..."):
            try:
                response, response_tokens, prompt_tokens, completion_tokens, response_cost = generate_response(resume_content)
           
                with col1:
                    st.subheader("Resume to generate cover letter for")
                    st.write(resume_content)
                with col2:
                    st.subheader("Generated Cover Letter")
                    st.write(response)

                with col3:
                    st.subheader("Evaluation Metrics")
                    st.write(f"Total Tokens: {response_tokens}")
                    st.write(f"Prompt Tokens: {prompt_tokens}")
                    st.write(f"Completion Tokens: {completion_tokens}")
                    st.write(f"Total Cost (USD): ${response_cost}")

            except Exception as e:
                st.write(f"No repsonse. Error: {e}")