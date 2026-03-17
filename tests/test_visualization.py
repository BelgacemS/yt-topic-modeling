"""Tests pour le module de visualisation Flask."""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np


# --- Test de l'app Flask ---

class TestFlaskApp:
    """Tests des routes de l'API."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        """Prépare l'app avec des données mockées."""
        from src.visualization.app import app, VIZ

        # on simule des données déjà chargées
        n = 30
        df = pd.DataFrame({
            "video_id": [f"vid_{i % 3}" for i in range(n)],
            "comment_id": [f"c_{i}" for i in range(n)],
            "raw_text": [f"commentaire test numéro {i}" for i in range(n)],
            "cleaned_text": [f"commentaire test numéro {i}" for i in range(n)],
            "tokens": [["commentaire", "test"] for _ in range(n)],
            "language": ["fr"] * n,
            "topic": [i % 3 for i in range(n)],
            "x": np.random.randn(n).tolist(),
            "y": np.random.randn(n).tolist(),
            "channel": [f"channel_{i % 2}" for i in range(n)],
            "video_title": [f"Vidéo {i % 3}" for i in range(n)],
            "upload_date": [f"2024-0{(i % 3) + 1}-15" for i in range(n)],
            "topic_label": [f"Topic {i % 3}" for i in range(n)],
        })

        topics = {
            0: [("mot1", 0.5), ("mot2", 0.3), ("mot3", 0.2)],
            1: [("word1", 0.6), ("word2", 0.25), ("word3", 0.15)],
            2: [("terme1", 0.4), ("terme2", 0.35), ("terme3", 0.25)],
        }

        topic_labels = {
            0: "Topic 0: mot1, mot2, mot3",
            1: "Topic 1: word1, word2, word3",
            2: "Topic 2: terme1, terme2, terme3",
        }

        VIZ['model'] = MagicMock()
        VIZ['model_name'] = "NMFModel"
        VIZ['df'] = df
        VIZ['topics'] = topics
        VIZ['topic_labels'] = topic_labels
        VIZ['channels'] = ["channel_0", "channel_1"]
        VIZ['videos'] = {"vid_0": "Vidéo 0", "vid_1": "Vidéo 1", "vid_2": "Vidéo 2"}

        app.config['TESTING'] = True
        self.client = app.test_client()

        yield

        # cleanup
        VIZ.clear()

    def test_index(self):
        """La page d'accueil doit retourner 200."""
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"YT Topic Modeling" in resp.data

    def test_api_topics(self):
        """L'API topics retourne la liste des topics."""
        resp = self.client.get("/api/topics")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data) == 3
        assert data[0]["id"] == 0
        assert len(data[0]["words"]) == 3

    def test_api_scatter(self):
        """L'API scatter retourne les coordonnées UMAP."""
        resp = self.client.get("/api/scatter")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "x" in data
        assert "y" in data
        assert "topic" in data
        assert len(data["x"]) == 30

    def test_api_scatter_filter_channel(self):
        """Le filtre par chaîne fonctionne."""
        resp = self.client.get("/api/scatter?channel=channel_0")
        data = json.loads(resp.data)
        assert len(data["x"]) == 15  # la moitié

    def test_api_scatter_filter_video(self):
        """Le filtre par vidéo fonctionne."""
        resp = self.client.get("/api/scatter?video=vid_0")
        data = json.loads(resp.data)
        assert len(data["x"]) == 10

    def test_api_wordcloud(self):
        """L'API wordcloud retourne une image PNG."""
        resp = self.client.get("/api/wordcloud/0")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        # vérifier que c'est bien un PNG (magic bytes)
        assert resp.data[:4] == b'\x89PNG'

    def test_api_wordcloud_404(self):
        """Topic inexistant → 404."""
        resp = self.client.get("/api/wordcloud/999")
        assert resp.status_code == 404

    def test_api_barchart(self):
        """L'API barchart retourne les mots et poids."""
        resp = self.client.get("/api/barchart/0")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "words" in data
        assert "weights" in data
        assert data["words"][0] == "mot1"

    def test_api_timeline(self):
        """L'API timeline retourne les séries temporelles."""
        resp = self.client.get("/api/timeline")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "dates" in data
        assert "series" in data
        assert len(data["series"]) == 3  # 3 topics


class TestComputeUMAP:
    """Tests pour le calcul UMAP."""

    def test_umap_small_corpus(self):
        """UMAP doit gérer les petits corpus."""
        from src.visualization.app import compute_umap_2d
        docs = ["bonjour le monde", "salut la terre", "hello world"]
        coords = compute_umap_2d(docs)
        assert coords.shape == (3, 2)

    def test_umap_very_small(self):
        """Avec < 5 docs, on retourne du random."""
        from src.visualization.app import compute_umap_2d
        coords = compute_umap_2d(["test", "hello"])
        assert coords.shape == (2, 2)


class TestLoadVideoMeta:
    """Tests pour le chargement des métadonnées vidéo."""

    def test_load_from_json(self, tmp_path):
        from src.visualization.app import load_video_meta

        data = {
            "video_id": "test123",
            "title": "Ma vidéo",
            "channel": "MaChaine",
            "upload_date": "2024-01-15",
            "comments": [],
        }
        with open(tmp_path / "test123.json", "w") as f:
            json.dump(data, f)

        meta = load_video_meta(tmp_path)
        assert "test123" in meta
        assert meta["test123"]["channel"] == "MaChaine"
        assert meta["test123"]["upload_date"] == "2024-01-15"

    def test_empty_dir(self, tmp_path):
        from src.visualization.app import load_video_meta
        meta = load_video_meta(tmp_path)
        assert meta == {}

    def test_nonexistent_dir(self):
        from src.visualization.app import load_video_meta
        meta = load_video_meta("/chemin/qui/existe/pas")
        assert meta == {}
