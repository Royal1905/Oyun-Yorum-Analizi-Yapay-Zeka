import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

input_text = "Oyunun atmosferi muhteşemdi ama bazı kontroller zorluydu."

with open('oyun/models/tfidf_lemmatized.pkl', 'rb') as f:
    vectorizer_lem, matrix_lem = pickle.load(f)

with open('oyun/models/tfidf_stemmed.pkl', 'rb') as f:
    vectorizer_stem, matrix_stem = pickle.load(f)

df_lem = pd.read_csv('oyun/data/lemmatized.csv')
df_stem = pd.read_csv('oyun/data/stemmed.csv')

input_vec_lem = vectorizer_lem.transform([input_text])
input_vec_stem = vectorizer_stem.transform([input_text])

scores_lem = cosine_similarity(input_vec_lem, matrix_lem).flatten()
scores_stem = cosine_similarity(input_vec_stem, matrix_stem).flatten()

top_lem = df_lem.iloc[scores_lem.argsort()[-5:][::-1]]
top_stem = df_stem.iloc[scores_stem.argsort()[-5:][::-1]]

print("TF-IDF Lemmatized:")
print(top_lem)
print("TF-IDF Stemmed:")
print(top_stem)