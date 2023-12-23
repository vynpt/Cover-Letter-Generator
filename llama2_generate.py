from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time 

# Load the model and tokenizer from the saved directory
model_directory = 'llama2-finetuned'
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# Create a text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

def generate_cv(skill, position, jd):
    # prompt = f"Create a cover letter  based on the applicant's skillset {skill}. Tailor the resume to fit the {position} position of this job description: \n{jd}\n"
    prompt = "Tell me about AI"
    system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    
    prompt_template = f"""[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {prompt} [/INST]"""

    response = text_generator(prompt_template, max_length=200)
    if response and "generated_text" in response[0]:
        return response[0]["generated_text"].split("[/INST]")[1]
    else:
        # Handle the case where the expected output isn't generated
        return "No generated text found. Please check the input prompt and model configuration."


# name = 'Vy'
# content = 'resume'
# position = 'data science'
# jd = 'job description'

# print(generate_cv(name, content, position, jd))

