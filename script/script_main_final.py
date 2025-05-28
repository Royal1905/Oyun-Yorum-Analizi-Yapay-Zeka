
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.metrics import jaccard_score

# --- VERİ YÜKLEME ---
lemma_df = pd.read_csv("data/lemmatized.csv")
stem_df = pd.read_csv("data/stemmed.csv")
input_text = "Oyunda zombileri çeşitli şekillerde kesebiliyorsunuz. Oldukça rahatlatıcı. Tavsiye ederim."

# --- TF-IDF ANALİZİ ---
def tfidf_analysis(df, name):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df["text"])
    input_vec = vectorizer.transform([input_text])
    scores = cosine_similarity(input_vec, matrix)[0]
    df["similarity"] = scores
    top5 = df.sort_values(by="similarity", ascending=False).head(5)
    print(f"\nTF-IDF ({name}) İlk 5 Sonuç:")
    for idx, row in top5.iterrows():
        score = top5['similarity'].iloc[top5.index.get_loc(idx)]
        print(f"- {row['text']} (score: {score:.4f})")

tfidf_analysis(lemma_df.copy(), "Lemmatized")
tfidf_analysis(stem_df.copy(), "Stemmed")

# --- WORD2VEC ANALİZİ ---
def avg_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def word2vec_analysis(df, model_dir, model_prefix):
    input_tokens = input_text.lower().split()
    print(f"\nWord2Vec ({model_prefix}) İlk 5 Benzerlik:")
    for i in range(1, 9):
        model_path = os.path.join(model_dir, f"{model_prefix}_model_{i}.model")
        if not os.path.exists(model_path):
            continue
        model = Word2Vec.load(model_path)
        input_vec = avg_vector(input_tokens, model)
        similarities = []
        for text in df["text"]:
            tokens = text.lower().split()
            vec = avg_vector(tokens, model)
            sim = cosine_similarity([input_vec], [vec])[0][0]
            similarities.append(sim)
        df["similarity"] = similarities
        top5 = df.sort_values(by="similarity", ascending=False).head(5)
        print(f"Model {i}:")
        for idx, row in top5.iterrows():
            score = top5['similarity'].iloc[top5.index.get_loc(idx)]
            print(f"  - {row['text']} (score: {score:.4f})")

word2vec_analysis(lemma_df.copy(), "models", "w2v_lemmatized")
word2vec_analysis(stem_df.copy(), "models", "w2v_stemmed")

# --- JACCARD BENZERLİK (MODEL TOP 5 İNDEXLERİNE GÖRE) ---
def jaccard_models():
    dummy_models = {
        "w2v_lemmatized_model_1": [0, 1, 2, 3, 4],
        "w2v_lemmatized_model_2": [0, 2, 3, 5, 7],
        "w2v_stemmed_model_1":    [6, 7, 8, 9, 10]
    }
    keys = list(dummy_models.keys())
    print("\nJaccard Benzerlik Skorları:")
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = np.zeros(25)
            b = np.zeros(25)
            a[dummy_models[keys[i]]] = 1
            b[dummy_models[keys[j]]] = 1
            score = jaccard_score(a, b)
            print(f"{keys[i]} vs {keys[j]}: {score:.2f}")

jaccard_models()
