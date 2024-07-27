import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse.linalg import svds
from collections import defaultdict

class SistemRekomendasiKompleks:
    def __init__(self, n_pengguna, n_produk, embedding_dim=100, max_words=10000, max_len=100):
        self.graf_pengguna_produk = nx.Graph()
        self.model_sentimen = None
        self.model_faktorizasi_matriks = None
        self.data_pengguna = None
        self.data_produk = None
        self.n_pengguna = n_pengguna
        self.n_produk = n_produk
        self.embedding_dim = embedding_dim
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)

    def _buat_model_sentimen(self):
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            LSTM(64),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def inisialisasi_data(self, data_pengguna, data_produk):
        self.data_pengguna = data_pengguna
        self.data_produk = data_produk
        self._bangun_graf_pengguna_produk()

    def _bangun_graf_pengguna_produk(self):
        for _, row in self.data_pengguna.iterrows():
            self.graf_pengguna_produk.add_node(f"U{row['id_pengguna']}", tipe='pengguna')
        
        for _, row in self.data_produk.iterrows():
            self.graf_pengguna_produk.add_node(f"P{row['id_produk']}", tipe='produk')
        
        for _, row in self.data_pengguna.iterrows():
            for id_produk in row['produk_dilihat'].split(','):
                self.graf_pengguna_produk.add_edge(f"U{row['id_pengguna']}", f"P{id_produk}", bobot=1)

    def latih_model_sentimen(self, ulasan, label):
        self.tokenizer.fit_on_texts(ulasan)
        sequences = self.tokenizer.texts_to_sequences(ulasan)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)

        label = np.array(label)  # Konversi label ke numpy array

        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, label, test_size=0.2)

        self.model_sentimen = self._buat_model_sentimen()
        self.model_sentimen.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

    def analisis_sentimen(self, teks):
        sequences = self.tokenizer.texts_to_sequences([teks])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return self.model_sentimen.predict(padded_sequences)[0][0]

    def hitung_pagerank_termodifikasi(self):
        return nx.pagerank(self.graf_pengguna_produk)

    def latih_model_faktorizasi_matriks(self, matriks_interaksi, k=20):
        U, s, Vt = svds(matriks_interaksi, k=k)
        s_diag = np.diag(s)
        self.model_faktorizasi_matriks = {
            'U': U,
            's': s_diag,
            'Vt': Vt
        }

    def prediksi_rating(self, id_pengguna, id_produk):
        user_idx = self.data_pengguna[self.data_pengguna['id_pengguna'] == id_pengguna].index[0]
        item_idx = self.data_produk[self.data_produk['id_produk'] == id_produk].index[0]
        
        prediksi = np.dot(
            np.dot(self.model_faktorizasi_matriks['U'][user_idx, :], 
                   self.model_faktorizasi_matriks['s']),
            self.model_faktorizasi_matriks['Vt'][:, item_idx]
        )
        return prediksi

    def generate_rekomendasi(self, id_pengguna, n=10):
        skor_rekomendasi = defaultdict(float)
        
        pagerank_scores = self.hitung_pagerank_termodifikasi()
        
        for id_produk in self.data_produk['id_produk']:
            skor_rekomendasi[id_produk] += self.prediksi_rating(id_pengguna, id_produk)
            skor_rekomendasi[id_produk] += pagerank_scores.get(f"P{id_produk}", 0) * 10
        
        rekomendasi = sorted(skor_rekomendasi.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return rekomendasi

    def evaluasi_rekomendasi(self, rekomendasi_aktual, rekomendasi_prediksi):
        set_aktual = set(rekomendasi_aktual)
        set_prediksi = set(rekomendasi_prediksi)
        presisi = len(set_aktual.intersection(set_prediksi)) / len(set_prediksi)
        recall = len(set_aktual.intersection(set_prediksi)) / len(set_aktual)
        f1_score = 2 * (presisi * recall) / (presisi + recall) if (presisi + recall) > 0 else 0
        return {'presisi': presisi, 'recall': recall, 'f1_score': f1_score}

if __name__ == "__main__":
    np.random.seed(42)
    n_pengguna, n_produk = 1000, 500
    
    data_pengguna = pd.DataFrame({
        'id_pengguna': range(n_pengguna),
        'produk_dilihat': [','.join(map(str, np.random.choice(range(n_produk), size=10))) for _ in range(n_pengguna)],
        'ulasan': ['produk ini bagus' if np.random.random() > 0.5 else 'produk ini kurang bagus' for _ in range(n_pengguna)]
    })
    
    data_produk = pd.DataFrame({
        'id_produk': range(n_produk),
        'kategori': np.random.choice(['A', 'B', 'C'], size=n_produk),
        'popularitas': np.random.rand(n_produk)
    })

    sistem = SistemRekomendasiKompleks(n_pengguna, n_produk)
    sistem.inisialisasi_data(data_pengguna, data_produk)

    # Rahul2024
    ulasan = data_pengguna['ulasan'].tolist()
    label = [1 if 'bagus' in review else 0 for review in ulasan]
    sistem.latih_model_sentimen(ulasan, label)

    matriks_interaksi = np.random.rand(n_pengguna, n_produk)
    sistem.latih_model_faktorizasi_matriks(matriks_interaksi)

    rekomendasi = sistem.generate_rekomendasi(1)
    print("Rekomendasi untuk pengguna 1:", rekomendasi)

    rekomendasi_aktual = np.random.choice(range(n_produk), size=10, replace=False)
    metrik_evaluasi = sistem.evaluasi_rekomendasi(rekomendasi_aktual, [r[0] for r in rekomendasi])
    print("Metrik evaluasi:", metrik_evaluasi)
