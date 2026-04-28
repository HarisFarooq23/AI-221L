# Install if needed:
# pip install transformers datasets torch

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

############################################
# 1. CREATE SYNTHETIC NER DATA
############################################

data = [
    {"tokens": ["Haris", "studies", "at", "GIKI", "in", "Pakistan"],
     "ner_tags": [1, 0, 0, 2, 0, 3]},

    {"tokens": ["Ali", "works", "at", "Google"],
     "ner_tags": [1, 0, 0, 2]},

    {"tokens": ["Sara", "lives", "in", "Lahore"],
     "ner_tags": [1, 0, 0, 3]},

    {"tokens": ["Ahmed", "studied", "at", "MIT"],
     "ner_tags": [1, 0, 0, 2]},

    {"tokens": ["Fatima", "is", "from", "Karachi"],
     "ner_tags": [1, 0, 0, 3]},
]

# Label mapping
label_list = ["O", "B-PER", "B-ORG", "B-LOC"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

dataset = Dataset.from_list(data)

############################################
# 2. TOKENIZATION + ALIGN LABELS
############################################

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=32
    )

    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(-100)

        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels)

############################################
# 3. LOAD MODEL (BERT FOR NER)
############################################

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

############################################
# 4. TRAINING SETUP
############################################

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

############################################
# 5. TRAIN MODEL
############################################

trainer.train()

############################################
# 6. TEST PREDICTION
############################################

def predict(sentence):
    tokens = sentence.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    predictions = torch.argmax(logits, dim=2)

    predicted_labels = [
        id2label[p.item()] for p in predictions[0]
    ]

    print("\nSentence:", sentence)
    print("Tokens:", tokens)
    print("Predictions:", predicted_labels)

predict("Haris works at Microsoft in Islamabad")

############################################
# 7. POSITIONAL ENCODING DEMO
############################################

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return PE

# Example
pe = positional_encoding(seq_len=10, d_model=8)

print("\nPositional Encoding Matrix (10 positions, 8 dims):")
print(pe)