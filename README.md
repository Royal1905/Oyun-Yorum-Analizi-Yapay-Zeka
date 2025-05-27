# Oyun Yorum Analizi – NLP Projesi

Bu proje, Steam platformundan alınan oyun kullanıcı yorumlarının metin benzerliğini analiz etmeyi amaçlar. Kullanılan yöntemler:
- TF-IDF
- Word2Vec (Skip-Gram ve CBOW)

## Giriş
Proje kapsamında kullanıcıdan alınan bir giriş cümlesi ile veri setindeki yorumlar karşılaştırılır. TF-IDF ve Word2Vec modelleriyle cosine similarity hesaplanarak en benzer 5 yorum bulunur. Ardından modellerin performansı hem anlamsal puanlama (subjektif) hem de sıralama benzerliği (Jaccard) ile değerlendirilir.

## Dosya Yapısı
- `data/`: Ön işlenmiş yorumlar (lemmatized.csv, stemmed.csv)
- `models/`: Eğitilmiş TF-IDF ve Word2Vec modelleri
- `scripts/`: Kod dosyaları (benzerlik, değerlendirme)
- `rapor/`: PDF rapor ve analiz çıktıları

## Kurulum ve Çalıştırma
Gerekli kütüphaneler:
```bash
pip install pandas scikit-learn gensim
```

ZIP açıldıktan sonra:
```bash
unzip oyun-yorum-analizi.zip -d oyun
```

Benzerlik analizleri:
```bash
python oyun/scripts/tfidf_similarity.py         # TF-IDF ile ilk 5 benzer yorum
python oyun/scripts/word2vec_similarity.py      # Word2Vec SG modeli ile ilk 5 yorum
python oyun/scripts/evaluation.py               # Anlamsal puan ve Jaccard matrisi
```

## Yazar
Bu proje, Oğuzhan Şen tarafından Gümüşhane Üniversitesi Yapay Zeka dersi kapsamında geliştirilmiştir.