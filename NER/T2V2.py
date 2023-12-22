import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json

# 加载预训练的SpaCy模型
model_path = "C:/Users/Texas2020/Desktop/NLP/NER/V2SPACY"
nlp = spacy.load(model_path)

# 加载增强的训练数据
training_data_path = "C:/Users/Texas2020/Desktop/NLP/NER/V3SPACY/V3training_data.json"
with open(training_data_path, 'r', encoding='utf-8') as f:
    TRAIN_DATA = json.load(f)

# 分割数据为训练集和验证集
random.shuffle(TRAIN_DATA)
split = int(len(TRAIN_DATA) * 0.8)
train_data = TRAIN_DATA[:split]
valid_data = TRAIN_DATA[split:]

# 将数据转换为SpaCy的Example对象
train_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in train_data]
valid_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in valid_data]

# 定义一个函数来评估模型性能
def evaluate_model(nlp_model, examples):
    correct_preds, total_preds, total_true = 0, 0, 0
    for example in examples:
        doc = nlp_model(example.text)
        gold = example.reference
        true_ents = set([(ent.start_char, ent.end_char, ent.label_) for ent in gold.ents])
        pred_ents = set([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
        correct_preds += len(true_ents & pred_ents)
        total_preds += len(pred_ents)
        total_true += len(true_ents)
    precision = correct_preds / total_preds if total_preds > 0 else 0
    recall = correct_preds / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1-score": f1}

# 训练模型
with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
    optimizer = nlp.resume_training()
    best_f1_score = 0.0
    for itn in range(100):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=compounding(4., 32., 1.001)):
            nlp.update(batch, drop=0.5, losses=losses, sgd=optimizer)
        print(f"Iteration {itn}, Losses: {losses}")

        # 评估模型在验证集上的性能
        scores = evaluate_model(nlp, valid_examples)
        print(f"Iteration {itn}, Precision: {scores['precision']}, Recall: {scores['recall']}, F1-score: {scores['f1-score']}")

        # 保存性能最佳的模型
        if scores['f1-score'] > best_f1_score:
            best_f1_score = scores['f1-score']
            nlp.to_disk("C:/Users/Texas2020/Desktop/NLP/NER/V3SPACY/best_model")

print("Training completed.")
