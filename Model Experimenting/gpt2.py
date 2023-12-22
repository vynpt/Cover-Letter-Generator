from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import pipeline
from transformers import AdamW
import torch
import pandas as pd
from datasets import load_dataset, list_datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from app import text_uploaded, selected_job

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def load_dataset(file_path, tokenizer):
    # Load the dataset into a pandas dataframe
    df = pd.read_csv(file_path)
    text_data = df['Resume'].tolist()
    
    # Tokenize the text data
    encodings = tokenizer(text_data, add_special_tokens=True, 
                          padding=True, truncation=True, 
                          return_tensors="pt")
    return encodings

# Use the function to load the data
file_path = 'Database/resume_data.csv'  # Correct this path as needed
encodings = load_dataset(file_path, tokenizer)

# Define the custom Dataset class
class GPT2Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.encodings[idx], dtype=torch.long)}

# Create an instance of the dataset and data loader
dataset = GPT2Dataset(encodings)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set the number of epochs for training
epochs = 1

# Training loop
for epoch in range(epochs):
    model.train()
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
    for batch in progress_bar:
        inputs = batch['input_ids'].to(device)
        outputs = model(input_ids=inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update with the loss value
        progress_bar.set_postfix({'loss': loss.item()})

# Text generation 
prompt = "Create a cover letter based on the applicant's resume: /n{text_uploaded}/n Tailor the resume to fit this job description: /n{selected_job}/n"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
output_sequences = model.generate(input_ids=inputs, max_length=50)
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)





