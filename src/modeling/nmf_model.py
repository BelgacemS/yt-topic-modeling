"""Modèle NMF (Non-negative Matrix Factorization) via scikit-learn.

Basé sur Lee & Seung 1999 - on factorise la matrice TF-IDF :
V ≈ W * H  avec V=doc-term, W=doc-topic, H=topic-term
Tout est non-négatif, ce qui rend les topics plus interprétables.
"""

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import numpy as np
import pandas as pd

from src.modeling.base import BaseTopicModel


class NMFModel(BaseTopicModel):
    """Topic modeling avec NMF + TF-IDF.

    NMF décompose la matrice TF-IDF en deux matrices non-négatives :
    - W (documents × topics) : poids de chaque topic par document
    - H (topics × mots) : poids de chaque mot par topic
    """

    def __init__(self, nb_topics=10, max_iter=500, init='nndsvd',
                 solver='cd', random_state=42):
        super().__init__(nb_topics)
        self.max_iter = max_iter
        self.init = init
        self.solver = solver
        self.random_state = random_state

        self.model = None
        self.vectorizer = None
        self.W = None  # matrice document-topic
        self.H = None  # matrice topic-mot
        self.feature_names = None
        self.texts = None  # pour le calcul de cohérence

    def fit(self, documents: list[str]) -> None:
        """Entraîne NMF sur les documents pré-traités."""
        print(f"Entraînement NMF avec {self.nb_topics} topics...")

        self.texts = [doc.split() for doc in documents]

        # vectorisation TF-IDF (essentiel pour NMF)
        self.vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=2,
            max_features=5000,
        )
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"Matrice TF-IDF : {tfidf_matrix.shape}")

        # NMF
        self.model = NMF(
            n_components=self.nb_topics,
            init=self.init,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        self.W = self.model.fit_transform(tfidf_matrix)
        self.H = self.model.components_

        # extraction des topics
        self._extract_topics()
        self.doc_topics = list(np.argmax(self.W, axis=1))

        self.is_fitted = True
        coherence = self.get_coherence()
        print(f"NMF entraîné ! Cohérence Cv = {coherence:.4f}")

    def _extract_topics(self, top_n=10):
        self.topics = {}
        for topic_id in range(self.nb_topics):
            # indices des mots les plus importants pour ce topic
            top_indices = self.H[topic_id].argsort()[::-1][:top_n]
            words = [(self.feature_names[i], float(self.H[topic_id][i]))
                     for i in top_indices]
            self.topics[topic_id] = words

    def get_topics(self) -> dict[int, list[tuple[str, float]]]:
        if not self.is_fitted:
            print("Le modèle n'est pas encore entraîné !")
            return {}
        return self.topics

    def transform(self, documents: list[str]) -> list[int]:
        """Prédit le topic dominant pour de nouveaux documents."""
        if not self.is_fitted:
            return []

        tfidf = self.vectorizer.transform(documents)
        W_new = self.model.transform(tfidf)
        return list(np.argmax(W_new, axis=1))

    def get_topic_info(self) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame()

        rows = []
        for topic_id, words in self.topics.items():
            nb_docs = self.doc_topics.count(topic_id)
            top_words = ", ".join([w for w, _ in words[:5]])
            rows.append({
                "topic_id": topic_id,
                "nb_docs": nb_docs,
                "top_words": top_words,
            })

        return pd.DataFrame(rows)

    def get_coherence(self) -> float:
        """Calcule la cohérence Cv via Gensim pour pouvoir comparer avec LDA."""
        if not self.is_fitted:
            return 0.0

        # on construit un dictionnaire gensim pour le calcul
        dictionary = corpora.Dictionary(self.texts)

        # les topics sous forme de listes de mots
        topic_words = []
        for topic_id in range(self.nb_topics):
            words = [w for w, _ in self.topics[topic_id][:10]]
            topic_words.append(words)

        cm = CoherenceModel(
            topics=topic_words,
            texts=self.texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return cm.get_coherence()

    def find_best_nb_topics(self, documents, topic_range=range(5, 30, 5)):
        """Teste plusieurs K et retourne les scores de cohérence."""
        print("Recherche du nombre optimal de topics (NMF)...")

        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=5000)
        tfidf = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        texts = [doc.split() for doc in documents]
        dictionary = corpora.Dictionary(texts)

        scores = {}
        for k in topic_range:
            nmf = NMF(n_components=k, init=self.init, solver=self.solver,
                       max_iter=self.max_iter, random_state=self.random_state)
            nmf.fit(tfidf)

            # extraire les mots de chaque topic
            topic_words = []
            for i in range(k):
                top_idx = nmf.components_[i].argsort()[::-1][:10]
                words = [feature_names[j] for j in top_idx]
                topic_words.append(words)

            cm = CoherenceModel(topics=topic_words, texts=texts,
                                dictionary=dictionary, coherence='c_v')
            score = cm.get_coherence()
            scores[k] = score
            print(f"  K={k:3d} → Cv = {score:.4f}")

        best_k = max(scores, key=scores.get)
        print(f"Meilleur K = {best_k} (Cv = {scores[best_k]:.4f})")
        return scores
