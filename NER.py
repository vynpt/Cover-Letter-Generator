import spacy
import random
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define a pattern for the Matcher
# Here's an example pattern that looks for two tokens: "Bachelor" followed by "of"
education_pattern = [{"LOWER": "bachelor"}, {"LOWER": "of"}]
matcher.add("EDUCATION", [education_pattern])

# Process some text
text = "I have a Bachelor of Science in Computer Science."
doc = nlp(text)

# Find matches in the text
matches = matcher(doc)
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)

TRAIN_DATA = [
    ("I have worked on project Alpha.", {"entities": [(21, 32, "PROJECT")]}),
    ("I know Python, Java, and SQL.", {"entities": [(7, 13, "SKILL"), (15, 19, "SKILL"), (24, 27, "SKILL")]}),
    # More examples...
]

# Load the model you want to train
nlp = spacy.load("en_core_web_sm")

# Get the NER component
ner = nlp.get_pipe("ner")

# Add new entity labels to the NER
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other components to speed up training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(10):  # Number of training iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update([text], [annotations], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)
