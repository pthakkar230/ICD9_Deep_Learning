import pandas as pd
import torch
from gensim import utils
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

DATA_FILE = 'data/sample.csv'
LABELS_OUTPUT = 'data/labels_one_hot.pt'
W2V_EMBEDDED_VISIT_NOTES_OUTPUT = 'data/w2v_embedded_visit_notes.pt'
D2V_EMBEDDED_VISIT_NOTES_OUTPUT = 'data/d2v_embedded_visit_notes.pt'
W2V_VECTOR_SIZE = 100
D2V_VECTOR_SIZE = 128
EPOCHS = 10
MAXIMUM_LENGTH = 700
BATCH_SIZE = 10  # Process 10 sentences at a time
W2V_MIN_COUNT = 1
D2V_MIN_COUNT = 1

# Read MIMIC-3 data
print("Reading file ", DATA_FILE, "...")
df = pd.read_csv(DATA_FILE)
# df = df[0:20000]

# One hot encode ICD-9 Codes
print("Producing one-hot encoded ICD-9 code labels...")
one_hot_encoded = pd.get_dummies(df['icd9_code'])
labels = torch.tensor(one_hot_encoded.values, dtype=torch.float)
torch.save(labels, LABELS_OUTPUT)

# Preprocess visit notes
print("\nW2V")
print("Preprocessing visit notes...")
sentences = []
for index, row in df.iterrows():
    sentences.append(utils.simple_preprocess(row['text']))

# Create word vectors
print("Creating word vectors...")
model = Word2Vec(sentences=sentences, sg=1, vector_size=W2V_VECTOR_SIZE, epochs=EPOCHS, min_count=W2V_MIN_COUNT)

# Embed MIMIC visit notes data using W2V
print("Embedding visit notes...")
embedded_sentences = []
for s in sentences:
    embedded_words = torch.zeros((MAXIMUM_LENGTH, W2V_VECTOR_SIZE))
    for i in range(len(s)):
        try:
            vector = model.wv[s[i]]
        except:
            vector = torch.zeros(W2V_VECTOR_SIZE)
        embedded_words[i] = torch.tensor(vector)
    embedded_sentences.append(embedded_words)

# Data Shape: (# of patients, vector_size, maximum_length)
embedded_sentences = torch.tensor(np.array(embedded_sentences)).permute(0, 2, 1)
torch.save(embedded_sentences, W2V_EMBEDDED_VISIT_NOTES_OUTPUT)

print("\nD2V")
# Tokenize text and create TaggedDocument objects
tagged_data = [TaggedDocument(words=utils.simple_preprocess(row['text']), tags=[str(index)]) for index, row in df.iterrows()]

# Create document vectors using Doc2Vec
print("Creating document vectors...")
model = Doc2Vec(vector_size=D2V_VECTOR_SIZE, epochs=EPOCHS, min_count=D2V_MIN_COUNT, dm=1, workers=4)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Embed MIMIC visit notes data using Doc2Vec
print("Embedding visit notes...")
doc_vectors = np.array([model.dv[i] for i in range(len(model.dv))])
doc_vectors = torch.tensor(doc_vectors)
torch.save(doc_vectors, D2V_EMBEDDED_VISIT_NOTES_OUTPUT)


print("\n3 Files Produced:\n", LABELS_OUTPUT, "\n", W2V_EMBEDDED_VISIT_NOTES_OUTPUT, "\n", D2V_EMBEDDED_VISIT_NOTES_OUTPUT)

