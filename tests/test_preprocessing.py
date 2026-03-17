"""Tests pour le module de prétraitement NLP."""

import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from src.preprocessing.preprocessor import (
    to_lowercase,
    remove_urls,
    remove_emojis,
    remove_mentions,
    remove_hashtags,
    normalize_repeated_chars,
    clean_extra_whitespace,
    detect_language,
    tokenize_and_lemmatize,
    TextPreprocessor,
    load_raw_comments,
    preprocess_comments,
    save_to_parquet,
    CUSTOM_STOPWORDS_FR,
)


# --- Tests des fonctions de nettoyage individuelles ---

class TestCleaningFunctions:

    def test_lowercase(self):
        assert to_lowercase("BONJOUR tout LE MONDE") == "bonjour tout le monde"

    def test_remove_urls_https(self):
        text = "regarde cette vidéo https://youtu.be/abc123 c'est ouf"
        assert "https://youtu.be/abc123" not in remove_urls(text)
        assert "regarde cette vidéo" in remove_urls(text)

    def test_remove_urls_www(self):
        text = "va voir www.example.com pour plus d'infos"
        assert "www.example.com" not in remove_urls(text)

    def test_remove_urls_preserves_normal_text(self):
        text = "pas d'url ici"
        assert remove_urls(text) == "pas d'url ici"

    def test_remove_emojis(self):
        text = "trop bien 🔥🔥🔥 j'adore 😍"
        cleaned = remove_emojis(text)
        assert "🔥" not in cleaned
        assert "😍" not in cleaned
        assert "trop bien" in cleaned

    def test_remove_emojis_keeps_text(self):
        text = "pas d'emoji ici"
        assert remove_emojis(text) == "pas d'emoji ici"

    def test_remove_mentions(self):
        text = "@JeanDupont t'as trop raison frère"
        cleaned = remove_mentions(text)
        assert "@JeanDupont" not in cleaned
        assert "trop raison" in cleaned

    def test_remove_hashtags_keeps_word(self):
        # on vire le # mais on garde le mot
        text = "cette vidéo est #incroyable"
        assert remove_hashtags(text) == "cette vidéo est incroyable"

    def test_normalize_repeated_chars(self):
        assert normalize_repeated_chars("trooooop") == "troop"
        assert normalize_repeated_chars("ouiiii") == "ouii"
        assert normalize_repeated_chars("normal") == "normal"
        # "hahahahaha" c'est pas un seul char répété, la regex gère pas ça
        # (et c'est ok, c'est un cas edge pas super important)
        assert normalize_repeated_chars("hahahahaha") == "hahahahaha"
        assert normalize_repeated_chars("noooon") == "noon"

    def test_clean_whitespace(self):
        text = "  trop   d'espaces   partout  "
        assert clean_extra_whitespace(text) == "trop d'espaces partout"

    def test_clean_whitespace_newlines(self):
        text = "ligne1\n\nligne2\t\tligne3"
        assert clean_extra_whitespace(text) == "ligne1 ligne2 ligne3"


# --- Tests détection de langue ---

class TestLanguageDetection:

    def test_detect_french(self):
        # langdetect est pas parfait sur les textes courts mais bon
        text = "Cette vidéo est vraiment intéressante, j'ai beaucoup appris"
        assert detect_language(text) == "fr"

    def test_detect_english(self):
        text = "This video is really interesting, I learned a lot from it"
        assert detect_language(text) == "en"

    def test_detect_empty_string(self):
        assert detect_language("") == "unknown"

    def test_detect_gibberish(self):
        # ça devrait pas planter au moins
        res = detect_language("asdfghjkl")
        assert isinstance(res, str)


# --- Tests tokenisation + lemmatisation ---

class TestTokenization:

    def test_tokenize_french(self):
        tokens = tokenize_and_lemmatize("les chats mangent des souris", lang="fr")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # "chats" devrait être lemmatisé en "chat"
        assert "chat" in tokens

    def test_tokenize_english(self):
        tokens = tokenize_and_lemmatize("the cats are eating mice", lang="en")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # "cats" → "cat"
        assert "cat" in tokens

    def test_removes_stopwords_french(self):
        tokens = tokenize_and_lemmatize("je suis très content de cette vidéo", lang="fr")
        # "je", "suis", "de", "cette" sont des stopwords spaCy
        assert "je" not in tokens
        assert "de" not in tokens

    def test_removes_custom_stopwords(self):
        # les mots YouTube custom doivent être filtrés aussi
        tokens = tokenize_and_lemmatize("mdr trop lol genre vidéo", lang="fr")
        assert "mdr" not in tokens
        assert "lol" not in tokens
        assert "genre" not in tokens

    def test_min_length_filter(self):
        # les tokens de moins de 3 caractères sont filtrés
        tokens = tokenize_and_lemmatize("le chat va au parc", lang="fr", remove_stops=False)
        # "le" (2), "va" (2), "au" (2) sont trop courts
        for t in tokens:
            assert len(t) >= 3

    def test_keeps_stopwords_when_disabled(self):
        tokens = tokenize_and_lemmatize(
            "je suis très content de cette vidéo",
            lang="fr",
            remove_stops=False
        )
        # les stopwords spaCy ne sont pas filtrés, mais les < 3 chars oui
        assert len(tokens) > 2

    def test_removes_punctuation(self):
        tokens = tokenize_and_lemmatize("wow!!! c'est génial...", lang="fr")
        assert "!" not in tokens
        assert "." not in tokens
        assert "..." not in tokens

    def test_empty_text(self):
        tokens = tokenize_and_lemmatize("", lang="fr")
        assert tokens == []


# --- Tests du pipeline complet ---

class TestTextPreprocessor:

    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_youtube_comment_fr(self):
        # commentaire YouTube typique bien crado
        text = "TROP BIEEEEN 🔥🔥🔥 @MecQuiCommente va voir https://spam.com #incroyable"
        cleaned = self.preprocessor.clean_text(text)

        assert cleaned.islower() or cleaned == ""  # lowercase
        assert "https://spam.com" not in cleaned    # urls virées
        assert "🔥" not in cleaned                  # emojis virés
        assert "@MecQuiCommente" not in cleaned     # mentions virées
        assert "#" not in cleaned                   # hashtag viré
        assert "bieen" in cleaned                   # répétitions normalisées

    def test_clean_youtube_comment_en(self):
        text = "OMG THIS IS SOOO GOOD!!! 😂😂 check out https://link.com @someone"
        cleaned = self.preprocessor.clean_text(text)
        assert "https://link.com" not in cleaned
        assert "@someone" not in cleaned
        assert "soo" in cleaned  # normalisé

    def test_process_text_full_pipeline(self):
        text = "Les chats sont vraiment les meilleurs animaux 🐱"
        cleaned, tokens, lang = self.preprocessor.process_text(text)

        assert isinstance(cleaned, str)
        assert isinstance(tokens, list)
        assert isinstance(lang, str)
        assert len(tokens) > 0
        # cleaned_text est maintenant les tokens rejoints
        # donc il contient que des mots propres sans stopwords
        assert "les" not in cleaned.split()
        assert "sont" not in cleaned.split()

    def test_process_empty_text(self):
        cleaned, tokens, lang = self.preprocessor.process_text("")
        assert cleaned == ""
        assert tokens == []
        assert lang == "unknown"

    def test_process_none(self):
        cleaned, tokens, lang = self.preprocessor.process_text(None)
        assert cleaned == ""
        assert tokens == []

    def test_custom_steps(self):
        # on désactive le lowercase
        preprocessor = TextPreprocessor(steps={"lowercase": False})
        cleaned = preprocessor.clean_text("HELLO World")
        assert "HELLO" in cleaned or "World" in cleaned

    def test_disable_lemmatize(self):
        preprocessor = TextPreprocessor(steps={"lemmatize": False})
        _, tokens, _ = preprocessor.process_text(
            "The cats are eating food",
            lang="en"
        )
        # sans lemmatisation, on devrait avoir les formes originales
        assert isinstance(tokens, list)


# --- Tests chargement et export ---

class TestDataIO:

    def _create_test_json(self, tmpdir, filename="video1.json"):
        """Helper pour créer un fichier JSON de test."""
        data = {
            "video_id": "test123",
            "title": "Test Video",
            "channel": "TestChannel",
            "upload_date": "2024-01-15",
            "comments": [
                {
                    "id": "c1",
                    "text": "Super vidéo, j'ai adoré le contenu !",
                    "author": "User1",
                    "timestamp": "2024-01-16T10:30:00",
                    "likes": 5,
                    "replies": [
                        {
                            "id": "r1",
                            "text": "Merci beaucoup !",
                            "author": "Creator",
                            "timestamp": "2024-01-16T11:00:00",
                            "likes": 2,
                        }
                    ]
                },
                {
                    "id": "c2",
                    "text": "Pas ouf honnêtement 😒",
                    "author": "User2",
                    "timestamp": "2024-01-17T08:00:00",
                    "likes": 0,
                    "replies": []
                }
            ]
        }
        fpath = Path(tmpdir) / filename
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return fpath

    def test_load_raw_comments(self, tmp_path):
        self._create_test_json(tmp_path)
        comments = load_raw_comments(tmp_path)

        # 2 commentaires + 1 réponse = 3
        assert len(comments) == 3
        assert comments[0]["video_id"] == "test123"
        assert comments[0]["raw_text"] == "Super vidéo, j'ai adoré le contenu !"

    def test_load_empty_dir(self, tmp_path):
        comments = load_raw_comments(tmp_path)
        assert len(comments) == 0

    def test_load_multiple_files(self, tmp_path):
        self._create_test_json(tmp_path, "video1.json")
        self._create_test_json(tmp_path, "video2.json")
        comments = load_raw_comments(tmp_path)
        assert len(comments) == 6  # 3 par fichier

    def test_preprocess_comments(self):
        comments = [
            {"video_id": "v1", "comment_id": "c1", "raw_text": "Super vidéo vraiment géniale !"},
            {"video_id": "v1", "comment_id": "c2", "raw_text": "This is really great content"},
        ]
        df = preprocess_comments(comments)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # vérifier les colonnes
        expected_cols = ["video_id", "comment_id", "raw_text", "cleaned_text", "tokens", "language"]
        for col in expected_cols:
            assert col in df.columns

    def test_save_and_load_parquet(self, tmp_path):
        df = pd.DataFrame({
            "video_id": ["v1"],
            "comment_id": ["c1"],
            "raw_text": ["test"],
            "cleaned_text": ["test"],
            "tokens": [["test"]],
            "language": ["fr"],
        })
        output = tmp_path / "test.parquet"
        save_to_parquet(df, output)

        assert output.exists()
        loaded = pd.read_parquet(output)
        assert len(loaded) == 1
        assert loaded["video_id"].iloc[0] == "v1"


# --- Tests avec des commentaires YouTube réalistes ---

class TestRealisticComments:
    """Tests avec des vrais exemples de commentaires YouTube (bruités)."""

    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    # commentaires typiques qu'on trouve sous les vidéos YouTube
    REALISTIC_COMMENTS = [
        "PREMIER !!!! 🎉🎉🎉",
        "Qui regarde en 2024 ? 👀",
        "Mdr trop drole 😂😂😂😂😂 j'arrive pas à respirer",
        "@BestOf regarde ça c'est trop bien",
        "like si t'es la avant 1M de vues",
        "Check my channel https://youtube.com/scam plzzz 🙏🙏",
        "Franchement c pas ouf, la qualité a baissé depuis qlq temps...",
        "THIS IS SOOOOOO GOOOOOOD OMGGGG",
        "j'ai rien compris mais j'ai like quand même 😅",
        "🔥🔥🔥🔥🔥🔥🔥",
        "",  # commentaire vide
        "Vidéo sponsorisée par NordVPN comme d'hab 🙄",
        "1:32 le meilleur moment de la vidéo 💯",
        "ratio",
        "W vidéo L commentaires",
    ]

    def test_all_realistic_comments_dont_crash(self):
        """Le pipeline doit pas planter sur ces commentaires."""
        for comment in self.REALISTIC_COMMENTS:
            cleaned, tokens, lang = self.preprocessor.process_text(comment)
            assert isinstance(cleaned, str)
            assert isinstance(tokens, list)
            assert isinstance(lang, str)

    def test_emoji_only_comment(self):
        cleaned, tokens, _ = self.preprocessor.process_text("🔥🔥🔥🔥🔥🔥🔥")
        assert cleaned == ""
        assert tokens == []

    def test_spam_url_comment(self):
        text = "Check my channel https://youtube.com/scam plzzz 🙏🙏"
        cleaned, tokens, _ = self.preprocessor.process_text(text)
        assert "https://youtube.com/scam" not in cleaned

    def test_repeated_chars_comment(self):
        text = "THIS IS SOOOOOO GOOOOOOD OMGGGG"
        cleaned = self.preprocessor.clean_text(text)
        # "soooooo" → "soo", "gooooood" → "good"
        assert "oooo" not in cleaned

    def test_short_slang_comment(self):
        # les commentaires courts/argot sont difficiles pour langdetect
        # au minimum ça doit pas planter
        text = "ratio"
        cleaned, tokens, lang = self.preprocessor.process_text(text)
        assert isinstance(lang, str)

    def test_timestamp_comment(self):
        text = "1:32 le meilleur moment de la vidéo"
        cleaned, tokens, _ = self.preprocessor.process_text(text)
        # "moment" devrait rester dans les tokens (pas un stopword, >= 3 chars)
        assert "moment" in tokens or any("moment" in t for t in tokens)

    def test_preprocess_batch_realistic(self):
        """Test le pipeline sur un batch de commentaires réalistes."""
        comments = [
            {"video_id": "v1", "comment_id": f"c{i}", "raw_text": text}
            for i, text in enumerate(self.REALISTIC_COMMENTS)
        ]
        df = preprocess_comments(comments)

        assert len(df) == len(self.REALISTIC_COMMENTS)
        assert "cleaned_text" in df.columns
        assert "tokens" in df.columns
        assert "language" in df.columns
