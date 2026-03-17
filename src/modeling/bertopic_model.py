"""Modèle BERTopic - Grootendorst 2022.

Pipeline : sentence embeddings (SBERT) → UMAP → HDBSCAN → c-TF-IDF
c-TF-IDF opère au niveau des clusters (pas des documents),
ce qui donne les mots les plus représentatifs de chaque topic.

BERTopic est particulièrement adapté aux textes courts (commentaires YouTube)
car il s'appuie sur la sémantique des embeddings plutôt que sur les
co-occurrences de mots (contrairement à LDA).
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import numpy as np
import pandas as pd

from src.modeling.base import BaseTopicModel


class BERTopicModel(BaseTopicModel):
    """Topic modeling avec BERTopic.

    Utilise des embeddings de phrases pour capturer la sémantique,
    puis UMAP + HDBSCAN pour le clustering, et c-TF-IDF pour
    la représentation des topics.

    Le topic -1 correspond aux outliers (docs pas assignés).
    """

    def __init__(self, nb_topics=None, embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                 min_cluster_size=15, umap_n_neighbors=15, random_state=42, language="french"):
        # nb_topics=None → BERTopic décide automatiquement
        super().__init__(nb_topics if nb_topics else -1)
        self.embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.umap_n_neighbors = umap_n_neighbors
        self.random_state = random_state
        self.language = language

        self.model = None
        self.embeddings = None
        self.texts = None
        self._nr_topics = nb_topics  # None = auto, sinon on réduit

    def _build_model(self):
        """Construit le pipeline BERTopic avec les bons paramètres."""
        # UMAP : on réduit à 5 dimensions pour le clustering
        # min_dist=0 pour des clusters bien séparés (cf. best practices)
        umap_model = UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_state,
        )

        # HDBSCAN : clustering basé sur la densité
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5,
            metric='euclidean',
            prediction_data=True,
        )

        # CountVectorizer pour c-TF-IDF
        # TODO: passer les stopwords FR ici pour avoir des topics plus propres
        vectorizer = CountVectorizer(
            min_df=2,
            ngram_range=(1, 2),
        )

        # on charge le modèle d'embedding
        embedding_model = SentenceTransformer(self.embedding_model_name)

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=self._nr_topics,  # None = automatique
            verbose=True,
        )

    def fit(self, documents: list[str]) -> None:
        """Entraîne BERTopic sur les documents."""
        print(f"Entraînement BERTopic (embedding: {self.embedding_model_name})...")
        print(f"  {len(documents)} documents à traiter")

        self.texts = [doc.split() for doc in documents]
        self._build_model()

        # fit_transform retourne les topics et les probabilités
        topics, probs = self.model.fit_transform(documents)
        self.doc_topics = list(topics)

        # récupération des topics
        self._extract_topics()

        # le nombre de topics trouvés (sans compter -1)
        real_topics = [t for t in set(topics) if t != -1]
        self.nb_topics = len(real_topics)

        nb_outliers = topics.count(-1) if isinstance(topics, list) else int((np.array(topics) == -1).sum())

        self.is_fitted = True
        print(f"BERTopic entraîné ! {self.nb_topics} topics trouvés, {nb_outliers} outliers")

    def _extract_topics(self, top_n=10):
        self.topics = {}
        topic_info = self.model.get_topic_info()

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            topic_words = self.model.get_topic(topic_id)
            if topic_words:
                self.topics[topic_id] = [(w, float(s)) for w, s in topic_words[:top_n]]

    def get_topics(self) -> dict[int, list[tuple[str, float]]]:
        if not self.is_fitted:
            print("Le modèle n'est pas encore entraîné !")
            return {}
        return self.topics

    def transform(self, documents: list[str]) -> list[int]:
        """Prédit les topics pour de nouveaux documents."""
        if not self.is_fitted:
            return []
        topics, _ = self.model.transform(documents)
        return list(topics)

    def get_topic_info(self) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame()
        return self.model.get_topic_info()

    def get_coherence(self) -> float:
        """Calcule la cohérence Cv pour comparer avec LDA et NMF."""
        if not self.is_fitted or not self.texts:
            return 0.0

        dictionary = corpora.Dictionary(self.texts)

        # on prend les mots des topics (sans le -1)
        topic_words = []
        for topic_id, words in self.topics.items():
            if topic_id == -1:
                continue
            topic_words.append([w for w, _ in words[:10]])

        if not topic_words:
            return 0.0

        cm = CoherenceModel(
            topics=topic_words,
            texts=self.texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return cm.get_coherence()

    def reduce_outliers(self, documents):
        """Réduit le nombre d'outliers en les réassignant au topic le plus proche."""
        if not self.is_fitted:
            print("Le modèle n'est pas encore entraîné !")
            return

        new_topics = self.model.reduce_outliers(documents, self.doc_topics,
                                                  strategy="c-tf-idf")
        self.model.update_topics(documents, topics=new_topics)
        self.doc_topics = list(new_topics)
        self._extract_topics()

        nb_outliers = self.doc_topics.count(-1)
        print(f"Outliers réduits : {nb_outliers} restants")
