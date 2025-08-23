from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.base import BaseEstimator, TransformerMixin
import re

class IndonesianTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        stopword_factory = StopWordRemoverFactory()
        custom_stopwords = {
            'yg', 'dg', 'rt', 'dgn', 'ny', 'dll', 'tsb', 'dr', 'pd',
            'scroll', 'resume', 'advertisement', 'iklan', 'sponsor', 'promo', 'baca',
            'klik', 'lanjut', 'selengkapnya', 'like', 'share', 'comment', 'subscribe',
            'follow', 'video', 'foto', 'gambar', 'infografis', 'caption', 'deskripsi',
            'sumber', 'reporter', 'editor', 'wartawan', 'penulis', 'kontributor', 'publisher',
            'simak', 'tonton', 'lihat', 'dengar', 'unduh', 'download',
            'republika', 'kompas', 'detik', 'tempo', 'cnn', 'bbc', 'rt', 'via',
            'twitter', 'facebook', 'instagram', 'youtube', 'tiktok',
            'halaman', 'kategori', 'narasi', 'verifikasi', 'referensi',
            'error', 'tagar', 'tulis', 'komen', 'read',
            'www', 'https', 'http', 'com', 'net', 'co', 'id'
        }
        self.stopwords = set(stopword_factory.get_stop_words()).union(custom_stopwords)

        self.excluded_from_stemming = {
            'politik', 'ekonomi', 'tokoh', 'jakarta', 'indonesia', 'pemerintah',
            'demokrasi', 'korupsi', 'hukum', 'budaya', 'sejarah', 'teknologi'
        }

        self.tokenizer = RegexpTokenizer(r'\w+')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        text = self._check_is_string(text)
        text = self._case_folding(text)
        text = self._remove_less_important_data(text)
        tokens = self._tokenize_text(text)
        tokens = self._filter_shorttoken(tokens)
        tokens = self._stemming(tokens)
        return ' '.join(tokens)

    def _check_is_string(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        return text
    
    def _case_folding(self, text):
        return text.lower().strip()
    
    def _remove_less_important_data(self, text):
        text = re.sub(r'(https?://\S+|www\.\S+|\S+\.(com|id|net|org|co)(/\S*)?)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\w+@\w+\.\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize_text(self, text):
        return self.tokenizer.tokenize(text)
    
    def _filter_shorttoken(self, tokens):
        filtered_token = [
            token for token in tokens if len(token) > 3 and token not in self.stopwords and not re.match(r'^(http|https|www|com|net|co|id)$', token)
        ]

        return filtered_token
    
    def _stemming(self, tokens):
        stemmed_token = [
            token if token in self.excluded_from_stemming 
            else self.stemmer.stem(token) for token in tokens
        ]

        return stemmed_token