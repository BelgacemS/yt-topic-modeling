"""Modèle LDA (Latent Dirichlet Allocation) via Gensim.

Basé sur Blei et al. 2003 - on utilise l'inférence variationnelle
de Gensim avec optimisation automatique de alpha et eta.
"""

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd

from src.modeling.base import BaseTopicModel


class LDAModel(BaseTopicModel):
    """Topic modeling avec LDA (Gensim).

    Alpha contrôle la distribution des topics par document :
      - alpha bas → peu de topics par doc (sparse)
      - alpha haut → beaucoup de topics par doc

    Eta (beta) contrôle la distribution des mots par topic :
      - eta bas → topics spécifiques (peu de mots)
      - eta haut → topics généraux
    """

    def __init__(self, nb_topics=10, passes=20, iterations=400,
                 alpha='auto', eta='auto', random_state=42):
        super().__init__(nb_topics)
        self.passes = passes
        self.iterations = iterations
        self.alpha = alpha
        self.eta = eta
        self.random_state = random_state

        # seront remplis au fit
        self.model = None
        self.dictionary = None
        self.corpus_bow = None
        self.texts = None  # on garde les textes tokenisés pour la cohérence

    def fit(self, documents: list[str]) -> None:
        """Entraîne LDA sur les documents. Les docs doivent être pré-traités."""
        print(f"Entraînement LDA avec {self.nb_topics} topics...")

        # tokenisation simple (les docs devraient déjà être nettoyés)
        self.texts = [doc.split() for doc in documents]

        # construction du dictionnaire et du corpus BoW
        self.dictionary = corpora.Dictionary(self.texts)
        # on filtre les mots trop rares ou trop fréquents
        # no_below=2 ça vire les hapax, no_above=0.9 les mots trop communs
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        self.corpus_bow = [self.dictionary.doc2bow(text) for text in self.texts]

        print(f"Vocabulaire : {len(self.dictionary)} mots, {len(self.corpus_bow)} documents")

        # entraînement du modèle LDA
        self.model = models.LdaModel(
            corpus=self.corpus_bow,
            id2word=self.dictionary,
            num_topics=self.nb_topics,
            passes=self.passes,
            iterations=self.iterations,
            alpha=self.alpha,
            eta=self.eta,
            random_state=self.random_state,
            chunksize=2000,
            eval_every=None,  # désactivé pour aller plus vite
        )

        # on récupère les topics
        self._extract_topics()

        # assignation des topics aux documents
        self.doc_topics = self._get_doc_topics()

        self.is_fitted = True
        coherence = self.get_coherence()
        print(f"LDA entraîné ! Cohérence Cv = {coherence:.4f}")

    def _extract_topics(self, top_n=10):
        self.topics = {}
        for topic_id in range(self.nb_topics):
            words = self.model.show_topic(topic_id, topn=top_n)
            self.topics[topic_id] = words

    def _get_doc_topics(self):
        res = []
        for bow in self.corpus_bow:
            topic_dist = self.model.get_document_topics(bow)
            if topic_dist:
                # on prend le topic avec la plus grande probabilité
                best_topic = max(topic_dist, key=lambda x: x[1])[0]
            else:
                best_topic = 0
            res.append(best_topic)
        return res

    def get_topics(self) -> dict[int, list[tuple[str, float]]]:
        if not self.is_fitted:
            print("Le modèle n'est pas encore entraîné !")
            return {}
        return self.topics

    def transform(self, documents: list[str]) -> list[int]:
        """Prédit le topic pour de nouveaux documents."""
        if not self.is_fitted:
            print("Le modèle n'est pas encore entraîné !")
            return []

        res = []
        for doc in documents:
            bow = self.dictionary.doc2bow(doc.split())
            topic_dist = self.model.get_document_topics(bow)
            if topic_dist:
                best = max(topic_dist, key=lambda x: x[1])[0]
            else:
                best = 0
            res.append(best)
        return res

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
        """Calcule la cohérence Cv avec Gensim."""
        if not self.is_fitted:
            return 0.0

        cm = CoherenceModel(
            model=self.model,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return cm.get_coherence()

    def find_best_nb_topics(self, documents, topic_range=range(5, 30, 5)):
        """Teste plusieurs valeurs de K et retourne les scores de cohérence.

        Utile pour la méthode du coude (elbow method).
        """
        print("Recherche du nombre optimal de topics...")
        texts = [doc.split() for doc in documents]
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [dictionary.doc2bow(t) for t in texts]

        scores = {}
        for k in topic_range:
            lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=self.passes,
                iterations=self.iterations,
                alpha='auto',
                eta='auto',
                random_state=self.random_state,
                eval_every=None,
            )
            cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
            score = cm.get_coherence()
            scores[k] = score
            print(f"  K={k:3d} → Cv = {score:.4f}")

        best_k = max(scores, key=scores.get)
        print(f"Meilleur K = {best_k} (Cv = {scores[best_k]:.4f})")
        return scores
