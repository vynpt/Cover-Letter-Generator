from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config 
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm


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
            top_p=0.3,
            top_k=40,
            repetition_penalty=2.0,
            )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

prompt_text = "I am writing to express my interest in the [Job Title] position. My background in [Your Background] and my skills in [Your Skills] make me an excellent candidate."
generated = generate_text(prompt_text)
print(generated)




