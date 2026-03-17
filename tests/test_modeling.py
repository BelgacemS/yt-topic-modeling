"""Tests pour le module de topic modeling.

On crée un corpus synthétique avec 3 thèmes bien distincts
pour vérifier que les modèles arrivent à les retrouver.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.modeling.base import BaseTopicModel
from src.modeling.lda_model import LDAModel
from src.modeling.nmf_model import NMFModel
from src.modeling.bertopic_model import BERTopicModel


# --- Corpus de test ---

def make_test_corpus():
    """Crée un corpus synthétique avec 3 thèmes distincts.

    Thème 1 : cuisine / nourriture
    Thème 2 : sport / football
    Thème 3 : musique / concert

    On génère assez de docs pour que les modèles puissent converger.
    """
    cuisine = [
        "recette gâteau chocolat délicieux facile préparer cuisine maison",
        "poulet rôti four pommes terre légumes saison cuisine familiale",
        "pâtes carbonara crème lardons parmesan recette italienne rapide",
        "soupe légumes carottes poireaux pommes terre hiver chaud réconfort",
        "tarte tatin pommes caramélisées pâte feuilletée dessert gourmand",
        "risotto champignons riz arborio bouillon parmesan crémeux cuisine",
        "salade composée tomates concombre feta olives huile olive fraîche",
        "crêpes sucrées farine oeufs lait beurre chandeleur dessert",
        "gratin dauphinois pommes terre crème four fromage gratiner délicieux",
        "blanquette veau carottes champignons sauce blanche riz tradition",
        "quiche lorraine lardons oeufs crème pâte brisée four entrée",
        "mousse chocolat noir oeufs sucre dessert léger aérien",
        "ratatouille courgettes aubergines tomates poivrons légumes provençal",
        "boeuf bourguignon vin rouge carottes oignons mijoter plat",
        "crème brûlée vanille sucre caraméliser dessert classique français",
    ]

    sport = [
        "match football équipe victoire but gardien stade supporters",
        "entraînement course running marathon préparation endurance sport",
        "championnat tennis raquette service revers coup droit balle",
        "natation piscine crawl brasse papillon longueurs entraînement eau",
        "basketball panier dribble équipe terrain match score points",
        "cyclisme vélo tour france étape montagne sprint peloton",
        "rugby mêlée essai transformation terrain équipe match passe",
        "athlétisme sprint saut hauteur longueur perche stade olympique",
        "boxe combat ring rounds gants arbitre combat catégorie",
        "handball gardien but tir équipe terrain match championnat",
        "volleyball filet smash service réception terrain équipe",
        "ski alpin descente slalom neige montagne hiver station",
        "football ligue champions groupe match victoire qualification",
        "musculation salle fitness exercices répétitions séries poids",
        "escalade grimpe falaise corde harnais voie difficile bloc",
    ]

    musique = [
        "concert salle spectacle musiciens scène public applaudissements",
        "guitare électrique solo riff amplificateur rock musique groupe",
        "piano classique sonate concerto orchestre symphonie interprétation",
        "batterie rythme tempo percussion cymbales caisse claire groove",
        "chanteur voix microphone scène performance live chanson paroles",
        "album studio enregistrement mixage production artiste sortie",
        "festival musique scène artistes public ambiance été plein air",
        "rap hip hop flow lyrics beat production freestyle",
        "violon archet cordes orchestre quatuor musique chambre classique",
        "jazz improvisation saxophone trompette contrebasse club swing",
        "synthétiseur électronique beats sampling production musique techno",
        "chorale chant polyphonie voix ensemble répétition concert église",
        "basse groove ligne basse funk reggae rythme section",
        "opéra aria ténor soprano orchestre mise scène théâtre",
        "playlist streaming écoute musique découverte artiste genre",
    ]

    # on duplique un peu pour avoir assez de données
    docs = (cuisine + sport + musique) * 3
    return docs


# ---- Fixtures ----

@pytest.fixture
def corpus():
    return make_test_corpus()


@pytest.fixture
def tmp_dir():
    """Dossier temporaire pour sauvegarder les modèles."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---- Tests BaseTopicModel ----

class TestBaseTopicModel:

    def test_cannot_instantiate(self):
        """On ne peut pas instancier la classe abstraite directement."""
        with pytest.raises(TypeError):
            BaseTopicModel(nb_topics=5)

    def test_diversity_empty(self):
        """La diversité retourne 0 si pas de topics."""
        # on crée une sous-classe minimale pour tester
        class DummyModel(BaseTopicModel):
            def fit(self, documents): pass
            def get_topics(self): return {}
            def transform(self, documents): return []
            def get_topic_info(self): return None

        model = DummyModel()
        assert model.get_diversity() == 0.0


# ---- Tests LDA ----

class TestLDAModel:

    def test_fit(self, corpus):
        """LDA doit s'entraîner sans erreur."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        assert model.is_fitted

    def test_get_topics(self, corpus):
        """Les topics doivent contenir des mots avec des poids."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        topics = model.get_topics()

        assert len(topics) == 3
        for topic_id, words in topics.items():
            assert len(words) > 0
            # chaque entrée est un tuple (mot, poids)
            for word, weight in words:
                assert isinstance(word, str)
                assert isinstance(weight, (float, int)) or hasattr(weight, '__float__')
                assert float(weight) > 0

    def test_transform(self, corpus):
        """Transform doit retourner un topic par document."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        topics = model.transform(corpus[:5])
        assert len(topics) == 5
        assert all(0 <= t < 3 for t in topics)

    def test_coherence(self, corpus):
        """La cohérence doit être un nombre positif."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        coh = model.get_coherence()
        assert isinstance(coh, float)
        # la cohérence Cv est entre 0 et 1 normalement
        assert 0 <= coh <= 1

    def test_diversity(self, corpus):
        """La diversité doit être entre 0 et 1."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        div = model.get_diversity()
        assert 0 <= div <= 1

    def test_topic_info(self, corpus):
        """get_topic_info doit retourner un DataFrame correct."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        info = model.get_topic_info()
        assert len(info) == 3
        assert "topic_id" in info.columns
        assert "nb_docs" in info.columns
        assert "top_words" in info.columns

    def test_save_load(self, corpus, tmp_dir):
        """On doit pouvoir sauvegarder et recharger un modèle."""
        model = LDAModel(nb_topics=3, passes=5, iterations=100)
        model.fit(corpus)
        save_path = Path(tmp_dir) / "lda_test"
        model.save(save_path)

        loaded = BaseTopicModel.load(save_path)
        assert loaded.is_fitted
        assert len(loaded.get_topics()) == 3


# ---- Tests NMF ----

class TestNMFModel:

    def test_fit(self, corpus):
        model = NMFModel(nb_topics=3)
        model.fit(corpus)
        assert model.is_fitted

    def test_get_topics(self, corpus):
        model = NMFModel(nb_topics=3)
        model.fit(corpus)
        topics = model.get_topics()
        assert len(topics) == 3
        for topic_id, words in topics.items():
            assert len(words) > 0

    def test_transform(self, corpus):
        model = NMFModel(nb_topics=3)
        model.fit(corpus)
        topics = model.transform(corpus[:5])
        assert len(topics) == 5

    def test_coherence(self, corpus):
        model = NMFModel(nb_topics=3)
        model.fit(corpus)
        coh = model.get_coherence()
        assert isinstance(coh, float)

    def test_topic_info(self, corpus):
        model = NMFModel(nb_topics=3)
        model.fit(corpus)
        info = model.get_topic_info()
        assert len(info) == 3

    def test_save_load(self, corpus, tmp_dir):
        model = NMFModel(nb_topics=3)
        model.fit(corpus)
        save_path = Path(tmp_dir) / "nmf_test"
        model.save(save_path)

        loaded = BaseTopicModel.load(save_path)
        assert loaded.is_fitted
        assert len(loaded.get_topics()) == 3


# ---- Tests BERTopic ----

class TestBERTopicModel:

    def test_fit(self, corpus):
        """BERTopic doit s'entraîner sans erreur."""
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        assert model.is_fitted

    def test_get_topics(self, corpus):
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        topics = model.get_topics()
        # BERTopic peut trouver n'importe quel nombre de topics
        assert len(topics) > 0

    def test_transform(self, corpus):
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        topics = model.transform(corpus[:5])
        assert len(topics) == 5

    def test_coherence(self, corpus):
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        coh = model.get_coherence()
        assert isinstance(coh, float)

    def test_topic_info(self, corpus):
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        info = model.get_topic_info()
        assert len(info) > 0

    def test_diversity(self, corpus):
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        div = model.get_diversity()
        assert 0 <= div <= 1

    def test_save_load(self, corpus, tmp_dir):
        model = BERTopicModel(min_cluster_size=10)
        model.fit(corpus)
        save_path = Path(tmp_dir) / "bertopic_test"
        model.save(save_path)

        loaded = BaseTopicModel.load(save_path)
        assert loaded.is_fitted
