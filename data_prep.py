import pandas as pd
import torch
from gensim import utils
from gensim.models import Word2Vec
import numpy as np

DATA_FILE = 'data/sample.csv'
LABELS_OUTPUT = 'data/labels_one_hot.pt'
EMBEDDED_VISIT_NOTES_OUTPUT = 'data/embedded_visit_notes.pt'
VECTOR_SIZE = 100
EPOCHS = 10
MAXIMUM_LENGTH = 700

# Read MIMIC-3 data
print("Reading file ", DATA_FILE, "...")
df = pd.read_csv(DATA_FILE)

# One hot encode ICD-9 Codes
print("Producing one-hot encoded ICD-9 code labels...")
one_hot_encoded = pd.get_dummies(df['icd9_code'])
labels = torch.tensor(one_hot_encoded.values, dtype=torch.float)
torch.save(labels, LABELS_OUTPUT)

# Preprocess visit notes
print("Preprocessing visit notes...")
sentences = []
for index, row in df.iterrows():
    sentences.append(utils.simple_preprocess(row['text']))

# Create word vectors
print("Creating word vectors...")
model = Word2Vec(sentences=sentences, sg=1, vector_size=VECTOR_SIZE, epochs=EPOCHS)

# Embed MIMIC visit notes data
print("Embedding visit notes...")
embedded_sentences = []
for s in sentences:
    embedded_words = torch.zeros((MAXIMUM_LENGTH, VECTOR_SIZE))
    for i in range(len(s)):
        try:
            vector = model.wv[s[i]]
        except:
            vector = model.wv['unk']
        embedded_words[i] = torch.tensor(vector)
    embedded_sentences.append(embedded_words)

# Data Shape: (# of patients, vector_size, maximum_length)
embedded_sentences = torch.tensor(np.array(embedded_sentences)).permute(0, 2, 1)
torch.save(embedded_sentences, EMBEDDED_VISIT_NOTES_OUTPUT)

print("\n2 Files Produced:\n", LABELS_OUTPUT, "\n", EMBEDDED_VISIT_NOTES_OUTPUT)

