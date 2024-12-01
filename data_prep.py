import pandas as pd
import torch
from gensim import utils
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import google.auth
from google.cloud import bigquery
from queries import LABELS_QUERY, DATA_QUERY

# Set to false if you don't want to recreate Gensim models
RETRAIN = False

SAMPLE_DATA_FILE = 'data/sample.csv'
QUERY_DATA = 'data/visit_notes.csv'
LABELS_DATA = 'data/labels.csv'
LABELS_OUTPUT = 'data2/labels_multi_hot.pt'
W2V_EMBEDDED_VISIT_NOTES_OUTPUT = 'data2/w2v_embedded_visit_notes.pt'
D2V_EMBEDDED_VISIT_NOTES_OUTPUT = 'data2/d2v_embedded_visit_notes.pt'
W2V_VECTOR_SIZE = 100
D2V_VECTOR_SIZE = 128
EPOCHS = 10
MAXIMUM_LENGTH = 1500 #700
BATCH_SIZE = 1000  # Process 1000 sentences at a time
W2V_MIN_COUNT = 1
D2V_MIN_COUNT = 1
PROJECT = 'serious-water-441620-d1'


# https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to
# ./google-cloud-sdk/bin/gcloud auth application-default login
# Set up auth to Google Big Query using connection from Physionet
print("Setting up Google Auth")
credentials, project = google.auth.default()
client = bigquery.Client(credentials=credentials, project=PROJECT)

# Read MIMIC-3 data
# print("Reading file ", SAMPLE_DATA_FILE, "...")
# df = pd.read_csv(SAMPLE_DATA_FILE)
# df = df[0:20000]

print("Reading visit notes data from BigQuery ", "...")
# data_query = client.query(DATA_QUERY)
# df = data_query.to_dataframe()
# df.to_csv(QUERY_DATA, index=False)

# Read already saved data
df = pd.read_csv(QUERY_DATA)  


print("Reading labels data from BigQuery ", "...")
# labels_query = client.query(LABELS_QUERY)
# labels_df = labels_query.to_dataframe()
# labels_df.to_csv(LABELS_DATA, index=False)

# Read already saved data
labels_df = pd.read_csv(LABELS_DATA)  

# Multi hot encode ICD-9 Codes
print("Producing multi-hot encoded ICD-9 code labels...")
multi_hot = pd.crosstab(labels_df['HADM_ID'], labels_df['diagnosis'])

labels = torch.tensor(multi_hot.values, dtype=torch.float)
torch.save(labels, LABELS_OUTPUT)

# Preprocess visit notes
print("\nW2V")
print("Preprocessing visit notes...")
sentences = []
for index, row in df.iterrows():
    sentences.append(utils.simple_preprocess(row['text']))

# Create word vectors
if RETRAIN:
    print("Creating word vectors...")
    model = Word2Vec(sentences=sentences, sg=1, vector_size=W2V_VECTOR_SIZE, epochs=EPOCHS, min_count=W2V_MIN_COUNT)
    model.save("data/word2vec.model")

model = Word2Vec.load("data/word2vec.model")

# Embed MIMIC visit notes data using W2V
print("Embedding visit notes...")
embedded_sentences = torch.zeros((len(sentences), W2V_VECTOR_SIZE, MAXIMUM_LENGTH))

for i in range(len(sentences)):
    embedded_words = torch.zeros((MAXIMUM_LENGTH, W2V_VECTOR_SIZE))
    for j in range(MAXIMUM_LENGTH):
        try:
            vector = torch.tensor(model.wv[sentences[i][j]])
        except:
            vector = torch.zeros(W2V_VECTOR_SIZE)
        embedded_words[j] = vector
    embedded_words = np.transpose(embedded_words)
    embedded_sentences[i] = embedded_words

    if i % BATCH_SIZE == 0:
        if i + BATCH_SIZE > len(sentences):
            print("     Processed {} out of {} rows".format(len(sentences), len(sentences)))
        else:
            print("     Processed {} out of {} rows".format(i + BATCH_SIZE, len(sentences)))

print("Saving visit notes vectors...")
torch.save(embedded_sentences, W2V_EMBEDDED_VISIT_NOTES_OUTPUT) 

print("\nD2V")
# Tokenize text and create TaggedDocument objects
tagged_data = [TaggedDocument(words=utils.simple_preprocess(row['text']), tags=[str(index)]) for index, row in df.iterrows()]

# Create document vectors using Doc2Vec
if RETRAIN:
    print("Creating document vectors...")
    model = Doc2Vec(vector_size=D2V_VECTOR_SIZE, epochs=EPOCHS, min_count=D2V_MIN_COUNT, dm=1, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("data/doc2vec.model")

model = Doc2Vec.load("data/doc2vec.model")

# Embed MIMIC visit notes data using Doc2Vec
print("Embedding visit notes...")
doc_vectors = np.array([model.dv[i] for i in range(len(model.dv))])
doc_vectors = torch.tensor(doc_vectors)
torch.save(doc_vectors, D2V_EMBEDDED_VISIT_NOTES_OUTPUT)


print("\n3 Files Produced:\n", LABELS_OUTPUT, "\n", W2V_EMBEDDED_VISIT_NOTES_OUTPUT, "\n", D2V_EMBEDDED_VISIT_NOTES_OUTPUT)

