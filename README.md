
# Yapay Zekâ Dersi – Dying Light Oyun Yorum Analizi
## Eğitilen Modellerle Metin Benzerliği Hesaplama ve Değerlendirme

### 📌 Proje Açıklaması
Bu projede, "Dying Light" oyununa ait 25 adet Türkçe kullanıcı yorumu ile metin benzerliği analizi gerçekleştirilmiştir. Önceki ödevde hazırlanan veri setleri temel alınarak, TF-IDF ve Word2Vec modelleri eğitilmiş ve belirlenen giriş cümlesiyle metin benzerliği karşılaştırmaları yapılmıştır.

---

### 🗂️ Klasör Yapısı

```
project/
├── data/
│   ├── lemmatized.csv
│   ├── stemmed.csv

├── models/
│   ├── w2v_lemmatized_model_1.model
│   ├── ...
│   └── w2v_stemmed_model_8.model
├── script/
│   └── main.py
├── README.md
```

---

### 🧪 Çalıştırma Talimatları

1. Python 3.8+ kurulu olmalıdır.
2. Gerekli kütüphaneler (pip install):
   ```bash
   pip install pandas numpy scikit-learn gensim
   ```

3. Ana betiği çalıştırmak için:
   ```bash
   python script/main.py
   ```

4. Kod, aşağıdaki işlemleri otomatik olarak yapar:
   - TF-IDF modelleri ile ilk 5 benzer yorumun ve cosine benzerlik skorlarının bulunması
   - 16 adet Word2Vec modeli ile anlam benzerliği analizi yapılması
   - Her model için 1–5 arası anlamsal puanlama
   - Jaccard benzerlik matrisi ile model sıralama tutarlılığının hesaplanması

---

### 🔍 Kullanılan Giriş Cümlesi
> Oyunda zombileri çeşitli şekillerde kesebiliyorsunuz. Oldukça rahatlatıcı. Tavsiye ederim.

---

### 📊 Öne Çıkan Sonuçlar

- **TF-IDF Ortalama Anlamsal Puan:** 3.0
- **Word2Vec En Başarılı Model:** `w2v_lemmatized_model_2` (Ortalama puan: 4.2)
- **Jaccard En Yüksek Benzerlik:** `w2v_lemmatized_model_1` ve `model_3` (1.00)

---

### 📚 Kullanılan Kütüphaneler

| Kütüphane | Amaç |
|-----------|------|
| **pandas** | CSV dosyalarının okunması, veri çerçevesi işlemleri |
| **numpy** | Vektör işlemleri, ortalama alma |
| **scikit-learn** (`TfidfVectorizer`, `cosine_similarity`) | TF-IDF vektörleştirme ve benzerlik hesaplama |
| **gensim** (`Word2Vec`) | Word2Vec model eğitimi ve vektörleme |
| **os** | Dosya ve dizin erişimleri |

---

### ✍️ Hazırlayan
**Oğuzhan Şen**  
Yapay Zeka Dersi – Final Ödevi 
Gümüşhane Üniversitesi  
İktisadi ve İdari Bilimler Fakültesi  
Yönetim Bilişim Sistemleri Bölümü
