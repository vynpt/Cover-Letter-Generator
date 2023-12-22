import spacy

model_dir = "NER/best_model"

nlp = spacy.load(model_dir)

test_texts = [
    "I have expertise in Python, Java, and SQL.",
    "I am proficient in multiple programming languages, including Java, Python, C++, and JavaScript, and have experience in both front-end and back-end development, allowing me to create full-stack web applications and software solutions. I have a deep understanding of data structures and algorithms, enabling me to design efficient and scalable solutions to complex problems.",
    "Proficient in JavaScript and familiar with React framework."
]

# Perform entity recognition on each test text
for text in test_texts:
    doc = nlp(text)
    print(f"Entities in '{text}':")
    for ent in doc.ents:
        print(f"  {ent.label_}: {ent.text}")
    print()

# Test a single text
# test_text = "I developed a new algorithm in Python and optimized SQL queries."
# doc = nlp(test_text)
# print("Entities in the test text:")
# for ent in doc.ents:
#     print(f"{ent.label_}: {ent.text}")

# print(ent.label_)
