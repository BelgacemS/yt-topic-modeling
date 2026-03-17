"""Classe abstraite pour les modèles de topic modeling."""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import pickle

import pandas as pd


class BaseTopicModel(ABC):
    """Interface commune pour tous les modèles de topic modeling.

    Chaque modèle (LDA, NMF, BERTopic) hérite de cette classe
    et implémente les méthodes abstraites.
    """

    def __init__(self, nb_topics=10):
        self.nb_topics = nb_topics
        self.is_fitted = False
        self.topics = {}  # {topic_id: [(mot, poids), ...]}
        self.doc_topics = []  # topic assigné à chaque doc

    @abstractmethod
    def fit(self, documents: list[str]) -> None:
        """Entraîne le modèle sur les documents."""
        pass

    @abstractmethod
    def get_topics(self) -> dict[int, list[tuple[str, float]]]:
        """Retourne les topics avec leurs mots-clés et poids."""
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[int]:
        """Assigne un topic à chaque document."""
        pass

    @abstractmethod
    def get_topic_info(self) -> pd.DataFrame:
        """Retourne un DataFrame résumant les topics."""
        pass

    def get_coherence(self) -> float:
        """Score de cohérence Cv. À override dans les sous-classes."""
        return 0.0

    def get_diversity(self, top_n=10) -> float:
        """Diversité des topics : proportion de mots uniques dans les top-N mots."""
        topics = self.get_topics()
        if not topics:
            return 0.0

        all_words = []
        for topic_id, words in topics.items():
            # on ignore le topic -1 (outliers de BERTopic)
            if topic_id == -1:
                continue
            all_words.extend([w for w, _ in words[:top_n]])

        if len(all_words) == 0:
            return 0.0

        return len(set(all_words)) / len(all_words)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # sauvegarde les métadonnées en JSON
        meta = {
            "nb_topics": self.nb_topics,
            "model_type": self.__class__.__name__,
            "is_fitted": self.is_fitted,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # le modèle lui-même en pickle
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self, f)

        print(f"Modèle sauvegardé dans {path}")

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            model = pickle.load(f)
        print(f"Modèle chargé depuis {path}")
        return model
