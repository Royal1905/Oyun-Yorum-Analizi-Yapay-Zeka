import pandas as pd
from itertools import combinations

# MODELLERİN BENZERLİK PUANLARI (örnek)
subjective_scores = {
    "TF-IDF Lemmatized": [4, 3, 4, 3, 2],
    "TF-IDF Stemmed": [3, 3, 2, 2, 2],
    "Word2Vec SG": [5, 4, 5, 4, 5],
    "Word2Vec CBOW": [4, 4, 3, 4, 3]
}

# Ortalama hesapla
df_scores = pd.DataFrame(subjective_scores)
df_scores.loc["Ortalama"] = df_scores.mean()
print("Anlamsal Değerlendirme (1-5 puan):")
print(df_scores)
print("\n")

# MODELLERİN SEÇTİĞİ YORUM İNDEKSLERİ (örnek)
top5_outputs = {
    "TF-IDF Lemmatized": [0, 1, 2, 3, 4],
    "TF-IDF Stemmed": [2, 3, 4, 5, 6],
    "Word2Vec SG": [0, 1, 3, 4, 5],
    "Word2Vec CBOW": [10, 11, 2, 3, 4]
}

# Jaccard hesapla
def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

models = list(top5_outputs.keys())
matrix = pd.DataFrame(index=models, columns=models)

for m1, m2 in combinations(models, 2):
    score = jaccard(top5_outputs[m1], top5_outputs[m2])
    matrix.loc[m1, m2] = round(score, 2)
    matrix.loc[m2, m1] = round(score, 2)

# Kendisiyle kıyas 1.0
for m in models:
    matrix.loc[m, m] = 1.0

print("Jaccard Benzerlik Matrisi:")
print(matrix)