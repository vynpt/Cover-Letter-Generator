from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config 
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from tqdm.auto import tqdm
import torch


model_path = 'gpt2-finetuned'

def generate_text(prompt):
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the prompt
    # inputs = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
    
    # Generate the cover letter text
    output = model.generate(
            input_ids=inputs,
            max_length=200,  
            # max_new_tokens=100,
            temperature=0.3,  
            top_p=0.8,
            top_k=40,
            repetition_penalty=2.0,
            )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# prompt_text = "I am writing to express my interest in the [Job Title] position. My background in [Your Background] and my skills in [Your Skills] make me an excellent candidate."
# generated = generate_text(prompt_text)
# print(generated)


def regular_gpt2(prompt):
    generator = pipeline('text-generation', model='gpt2')
    output = generator(prompt)
    # model = GPT2LMHeadModel.from_pretrained(model_path)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # tokenizer.padding_side = "left"
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # # Tokenize the prompt
    # # inputs = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
    # inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
    
    # # Generate the cover letter text
    # output = model.generate(
    #         )

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output[0]['generated_text']


def compute_perplexity():
    model_finetuned = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer_finetuned = GPT2Tokenizer.from_pretrained(model_path)
    text =  """
        Dear Hiring Manager,
        I am writing to express my interest in the Data Scientist position at XYZ Corporation. With a strong background in proogramming and extensive experience in developing high-quality model, I am confident in my ability to contribute effectively to your team.
        In my previous role at ABC Inc., I successfully led a team to deliver a complex project that improved our client's workflow by 30%. My skills in Python and SQL, along with my ability to work collaboratively in fast-paced environments, make me a strong candidate for this position.
        Thank you for considering my application. I look forward to the opportunity to discuss how my skills and experiences align with the needs of XYZ Corporation.

        Sincerely,
        [Your Name]
        """
    
    encodings = tokenizer_finetuned(text, return_tensors='pt', truncation=True, max_length=1024)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model_finetuned(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss

    # Calculating average negative log-likelihood
    avg_neg_log_likelihood = loss.item()

    # Calculating Perplexity
    perplexity = torch.exp(torch.tensor(avg_neg_log_likelihood))
    
    return perplexity.item()

