import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('oyun/data/lemmatized.csv')
comments = df['comment'].tolist()

input_text = "Oyunun atmosferi muhteşemdi ama bazı kontroller zorluydu."
input_tokens = input_text.lower().split()

model = Word2Vec.load('oyun/models/w2v_lemmatized_sg_w3_d100.model')
input_vecs = [model.wv[w] for w in input_tokens if w in model.wv]
input_avg = np.mean(input_vecs, axis=0)

results = []
for comment in comments:
    tokens = comment.lower().split()
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if vecs:
        avg_vec = np.mean(vecs, axis=0)
        score = cosine_similarity([input_avg], [avg_vec])[0][0]
        results.append((comment, score))

top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]
for text, score in top5:
    print(f"{score:.3f} - {text}")