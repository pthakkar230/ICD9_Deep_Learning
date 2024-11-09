import pandas as pd
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np

DATA_FILE = 'sample.csv'
LABELS_OUTPUT = 'labels_one_hot.pt'
EMBEDDED_VISIT_NOTES_OUTPUT = 'embedded_visit_notes.pt'
VECTOR_SIZE = 100
EPOCHS = 10
MAXIMUM_LENGTH = 700
BATCH_SIZE = 10  # Process 10 sentences at a time

# Read MIMIC-3 data
print(f"Reading file {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE)

# One hot encode ICD-9 Codes
print("Producing one-hot encoded ICD-9 code labels...")
one_hot_encoded = pd.get_dummies(df['icd9_code'])
labels = torch.tensor(one_hot_encoded.values, dtype=torch.float)
torch.save(labels, LABELS_OUTPUT)

# Preprocess visit notes
print("Preprocessing visit notes...")

# Tokenize text and create TaggedDocument objects
tagged_data = [TaggedDocument(words=simple_preprocess(row['text']), tags=[str(index)]) for index, row in df.iterrows()]

# Create document vectors using Doc2Vec
print("Creating document vectors...")
model = Doc2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS, min_count=1, dm=1, workers=4)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Embed MIMIC visit notes data
print("Embedding visit notes...")
embedded_sentences = []

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE]
    for i, row in batch.iterrows():
        vector = model.dv[str(i)]
        embedded_words = torch.tensor(vector).repeat(MAXIMUM_LENGTH, 1)
        embedded_sentences.append(embedded_words)
    
    # Save batch to file to avoid keeping everything in memory
    if start == 0:
        torch.save(torch.stack(embedded_sentences).permute(0, 2, 1), EMBEDDED_VISIT_NOTES_OUTPUT)
    else:
        existing_data = torch.load(EMBEDDED_VISIT_NOTES_OUTPUT, weights_only=True)
        combined_data = torch.cat((existing_data, torch.stack(embedded_sentences).permute(0, 2, 1)), dim=0)
        torch.save(combined_data, EMBEDDED_VISIT_NOTES_OUTPUT)
    
    embedded_sentences = []  # Clear the list to save memory

print(f"\n2 Files Produced:\n{LABELS_OUTPUT}\n{EMBEDDED_VISIT_NOTES_OUTPUT}")
