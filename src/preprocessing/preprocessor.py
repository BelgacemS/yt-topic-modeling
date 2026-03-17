"""
Module de prétraitement NLP pour les commentaires YouTube.
Pipeline : lowercase → nettoyage URLs → nettoyage emojis → tokenisation → stopwords → lemmatisation
"""

import re
import json
from pathlib import Path

import pandas as pd
import spacy
from langdetect import detect, LangDetectException


# on charge les modèles spaCy une seule fois (c'est lourd)
# on met ça en global pour pas les recharger à chaque appel
_NLP_MODELS = {}

# stopwords custom pour les commentaires YouTube
# spaCy en attrape déjà pas mal mais y'a du vocabulaire YouTube/internet
# que spaCy connaît pas
CUSTOM_STOPWORDS_FR = {
    # abréviations internet / YouTube
    "mdr", "lol", "ptdr", "omg", "wtf", "xd", "xdd", "lmao", "rofl",
    "svp", "stp", "tlm", "tkt", "jsp", "jpp", "oklm", "osef", "imo",
    # mots passe-partout trop fréquents dans les commentaires
    "genre", "truc", "machin", "chose", "faire", "avoir", "être", "aller",
    "plus", "tout", "très", "trop", "bien", "comme", "encore", "aussi",
    "même", "rien", "dire", "voir", "mettre", "venir", "prendre",
    "bon", "quand", "vraiment", "juste", "toujours", "super", "tellement",
    "déjà", "après", "avant", "depuis", "encore", "donc", "alors",
    # formes verbales fréquentes pas dans les stopwords spaCy
    "fait", "dit", "mis", "pris", "peut", "faut", "doit", "veut",
    "sais", "crois", "pense", "trouve", "aime", "adore",
    # mots de remplissage YouTube
    "vidéo", "video", "chaîne", "chaine", "merci", "svp",
    "abonne", "abonner", "like", "liker", "partage", "partager",
    "commentaire", "commenter",
}

# même chose pour l'anglais
CUSTOM_STOPWORDS_EN = {
    "lol", "lmao", "omg", "wtf", "btw", "imo", "smh", "tbh", "ngl",
    "like", "just", "really", "very", "much", "still", "also", "even",
    "thing", "stuff", "gonna", "gotta", "wanna", "kinda", "sorta",
    "video", "channel", "subscribe", "comment",
}

def _get_nlp(lang):
    """Charge le modèle spaCy pour la langue donnée (avec cache)."""
    if lang not in _NLP_MODELS:
        if lang == "fr":
            _NLP_MODELS[lang] = spacy.load("fr_core_news_md", disable=["ner", "parser"])
        elif lang == "en":
            _NLP_MODELS[lang] = spacy.load("en_core_web_md", disable=["ner", "parser"])
        else:
            # fallback sur l'anglais pour les langues pas supportées
            _NLP_MODELS[lang] = spacy.load("en_core_web_md", disable=["ner", "parser"])
    return _NLP_MODELS[lang]


# --- Fonctions de nettoyage individuelles ---

def to_lowercase(text):
    return text.lower()


def remove_urls(text):
    # on vire les URLs http(s) et les www.
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text.strip()


def remove_emojis(text):
    # regex qui matche la plupart des emojis unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symboles & pictographes
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # drapeaux
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"             # zero width joiner
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_mentions(text):
    # vire les @pseudo
    return re.sub(r'@\w+', '', text)


def remove_hashtags(text):
    # vire les #hashtag (juste le symbole, on garde le mot)
    return re.sub(r'#(\w+)', r'\1', text)


def normalize_repeated_chars(text):
    # "trooooop" → "troop", "hahahahaha" → "hahaha"
    # on limite à max 2 répétitions d'un même caractère
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def clean_extra_whitespace(text):
    # vire les espaces multiples, tabs, retours à la ligne
    return re.sub(r'\s+', ' ', text).strip()


def detect_language(text):
    """Détecte la langue du texte. Retourne 'fr', 'en', ou 'unknown'."""
    # TODO: langdetect est pas super fiable sur les textes courts,
    #       regarder fasttext-langdetect comme alternative ?
    try:
        lang = detect(text)
        if lang in ("fr", "en"):
            return lang
        # pour les autres langues, on renvoie quand même le code
        return lang
    except LangDetectException:
        return "unknown"


def tokenize_and_lemmatize(text, lang="fr", remove_stops=True):
    """Tokenise et lemmatise le texte avec spaCy.

    Retourne la liste des tokens lemmatisés.
    """
    nlp = _get_nlp(lang)
    doc = nlp(text)

    # on choisit les custom stopwords selon la langue
    custom_stops = CUSTOM_STOPWORDS_FR if lang == "fr" else CUSTOM_STOPWORDS_EN

    tokens = []
    for token in doc:
        # on garde pas la ponctuation, les espaces, ni les nombres seuls
        if token.is_punct or token.is_space or token.like_num:
            continue
        # stopwords spaCy + nos stopwords custom
        if remove_stops and (token.is_stop or token.text.lower() in custom_stops):
            continue

        lemma = token.lemma_.lower().strip()

        # on filtre aussi le lemme contre les custom stopwords
        # (ex: "fait" est le lemme de "faites", etc.)
        if remove_stops and lemma in custom_stops:
            continue

        # minimum 3 caractères — ça vire "de", "le", "la", "tu", "je", etc.
        if len(lemma) < 3:
            continue

        tokens.append(lemma)

    return tokens


# --- Pipeline principal ---

class TextPreprocessor:
    """Pipeline de prétraitement pour les commentaires YouTube.

    Chaque étape peut être activée/désactivée via le dict `steps`.
    """

    # config par défaut - on peut override en passant un dict au constructeur
    DEFAULT_STEPS = {
        "lowercase": True,
        "remove_urls": True,
        "remove_emojis": True,
        "remove_mentions": True,
        "remove_hashtags": True,       # garde le mot, vire juste le #
        "normalize_repeated": True,
        "clean_whitespace": True,
        "remove_stopwords": True,
        "lemmatize": True,
    }

    def __init__(self, steps=None):
        self.steps = {**self.DEFAULT_STEPS}
        if steps:
            self.steps.update(steps)

    def clean_text(self, text):
        """Applique les étapes de nettoyage (avant tokenisation)."""
        if not text or not isinstance(text, str):
            return ""

        if self.steps["lowercase"]:
            text = to_lowercase(text)
        if self.steps["remove_urls"]:
            text = remove_urls(text)
        if self.steps["remove_emojis"]:
            text = remove_emojis(text)
        if self.steps["remove_mentions"]:
            text = remove_mentions(text)
        if self.steps["remove_hashtags"]:
            text = remove_hashtags(text)
        if self.steps["normalize_repeated"]:
            text = normalize_repeated_chars(text)
        if self.steps["clean_whitespace"]:
            text = clean_extra_whitespace(text)

        return text

    def process_text(self, text, lang=None):
        """Pipeline complet pour un texte : nettoyage + tokenisation.

        Retourne (cleaned_text, tokens, detected_lang)
        cleaned_text = tokens rejoints en string (prêt pour le modeling)
        """
        if not text or not isinstance(text, str):
            return "", [], "unknown"

        # détection de langue sur le texte brut (avant nettoyage)
        if lang is None:
            lang = detect_language(text)

        surface_cleaned = self.clean_text(text)

        # tokenisation + lemmatisation
        if self.steps["lemmatize"]:
            tokens = tokenize_and_lemmatize(
                surface_cleaned,
                lang=lang if lang in ("fr", "en") else "en",
                remove_stops=self.steps["remove_stopwords"]
            )
        else:
            # tokenisation basique sans spaCy
            tokens = surface_cleaned.split()
            if self.steps["remove_stopwords"]:
                nlp = _get_nlp(lang if lang in ("fr", "en") else "en")
                stops = nlp.Defaults.stop_words
                custom = CUSTOM_STOPWORDS_FR if lang == "fr" else CUSTOM_STOPWORDS_EN
                all_stops = stops | custom
                tokens = [t for t in tokens if t not in all_stops and len(t) >= 3]

        # cleaned_text = les tokens rejoints, prêts pour le topic modeling
        # le texte original est toujours dispo dans raw_text
        cleaned_text = " ".join(tokens)

        return cleaned_text, tokens, lang


# --- Chargement et export des données ---

def load_raw_comments(raw_dir):
    """Charge tous les fichiers JSON de data/raw/ et retourne une liste de dicts."""
    raw_dir = Path(raw_dir)
    all_comments = []

    json_files = list(raw_dir.glob("*.json"))
    print(f"Trouvé {len(json_files)} fichiers JSON dans {raw_dir}")

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_id = data.get("video_id", fpath.stem)

        for comment in data.get("comments", []):
            all_comments.append({
                "video_id": video_id,
                "comment_id": comment.get("id", ""),
                "raw_text": comment.get("text", ""),
            })
            # on récupère aussi les réponses si y'en a
            for reply in comment.get("replies", []):
                all_comments.append({
                    "video_id": video_id,
                    "comment_id": reply.get("id", ""),
                    "raw_text": reply.get("text", ""),
                })

    print(f"Total : {len(all_comments)} commentaires chargés")
    return all_comments


def preprocess_comments(comments, steps=None, batch_size=1000):
    """Prétraite une liste de commentaires et retourne un DataFrame.

    Format de sortie : [video_id, comment_id, raw_text, cleaned_text, tokens, language]
    """
    preprocessor = TextPreprocessor(steps=steps)

    results = []
    total = len(comments)

    for i, comment in enumerate(comments):
        raw = comment["raw_text"]
        cleaned, tokens, lang = preprocessor.process_text(raw)

        results.append({
            "video_id": comment["video_id"],
            "comment_id": comment["comment_id"],
            "raw_text": raw,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "language": lang,
        })

        # petit log d'avancement tous les batch_size commentaires
        if (i + 1) % batch_size == 0:
            print(f"  Prétraitement : {i + 1}/{total} commentaires traités...")

    print(f"Prétraitement terminé : {total} commentaires")

    df = pd.DataFrame(results)
    return df


def save_to_parquet(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Sauvegardé dans {output_path} ({len(df)} lignes)")


# --- Point d'entrée CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prétraitement des commentaires YouTube")
    parser.add_argument("--input", default="data/raw", help="Dossier des JSON bruts")
    parser.add_argument("--output", default="data/processed/corpus.parquet", help="Fichier parquet de sortie")
    args = parser.parse_args()

    print("=== Prétraitement NLP des commentaires YouTube ===")

    # chargement
    comments = load_raw_comments(args.input)
    if not comments:
        print("Aucun commentaire trouvé, rien à faire.")
        return

    # prétraitement
    df = preprocess_comments(comments)

    # on vire les commentaires vides après nettoyage
    nb_before = len(df)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    nb_removed = nb_before - len(df)
    if nb_removed > 0:
        print(f"Supprimé {nb_removed} commentaires vides après nettoyage")

    # export
    save_to_parquet(df, args.output)

    # quelques stats rapides
    print(f"\n--- Stats ---")
    print(f"Langues détectées : {df['language'].value_counts().to_dict()}")
    print(f"Nb moyen de tokens par commentaire : {df['tokens'].apply(len).mean():.1f}")


if __name__ == "__main__":
    main()
