"""Script principal - orchestre le pipeline complet d'analyse thématique.

Usage :
    python main.py --demo                    # données synthétiques pour tester
    python main.py --channels channels.txt   # pipeline complet sur de vraies données
    python main.py --viz --model models/nmf  # juste la visualisation
"""

import argparse
import json
import random
import time
from pathlib import Path


# ---- Étapes du pipeline ----

def run_extraction(channels_file, output_dir="data/raw", max_videos=None, workers=4):
    """Lance l'extraction des commentaires via yt-dlp."""
    from src.extraction.extractor import CommentExtractor

    print("\n" + "=" * 60)
    print("ÉTAPE 1 : Extraction des commentaires")
    print("=" * 60)

    channels = Path(channels_file).read_text().strip().splitlines()
    extractor = CommentExtractor(output_dir=output_dir, max_workers=workers)
    extractor.run(channels=channels, max_videos=max_videos)


def run_preprocessing(raw_dir="data/raw", output_path="data/processed/corpus.parquet"):
    """Nettoyage + tokenisation + lemmatisation des commentaires."""
    from src.preprocessing.preprocessor import (
        load_raw_comments, preprocess_comments, save_to_parquet
    )

    print("\n--- Prétraitement NLP ---")

    comments = load_raw_comments(raw_dir)
    if not comments:
        print("Aucun commentaire trouvé, abandon.")
        return False

    df = preprocess_comments(comments)

    # virer les commentaires vides après nettoyage
    nb_before = len(df)
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    nb_removed = nb_before - len(df)
    if nb_removed > 0:
        print(f"Supprimé {nb_removed} commentaires vides")

    save_to_parquet(df, output_path)

    # quelques stats
    print(f"\nStats : {len(df)} commentaires, langues = {df['language'].value_counts().to_dict()}")
    return True


def run_modeling(corpus_path="data/processed/corpus.parquet", output_dir="models", nb_topics=10):
    """Lance LDA + NMF + BERTopic et compare."""
    from src.modeling.compare import load_corpus, run_comparison, print_report, save_report

    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Topic Modeling")
    print("=" * 60)

    docs_clean, docs_raw = load_corpus(corpus_path)
    results, models = run_comparison(docs_clean, docs_raw, nb_topics, output_dir)
    print_report(results, models)
    save_report(results, models, output_dir)


def run_modeling_quick(corpus_path="data/processed/corpus.parquet", output_dir="models", nb_topics=4):
    """Version rapide pour la démo - NMF + LDA seulement, pas BERTopic."""
    import pandas as pd
    from src.modeling.lda_model import LDAModel
    from src.modeling.nmf_model import NMFModel

    print("\n--- Topic Modeling (mode rapide, sans BERTopic) ---")

    df = pd.read_parquet(corpus_path)
    docs = df['cleaned_text'].dropna().tolist()
    print(f"{len(docs)} documents à traiter")

    output_dir = Path(output_dir)

    # NMF - rapide et bon pour la démo
    nmf = NMFModel(nb_topics=nb_topics)
    nmf.fit(docs)
    nmf.save(output_dir / "nmf")

    # LDA - pour avoir un deuxième modèle
    lda = LDAModel(nb_topics=nb_topics, passes=5, iterations=100)
    lda.fit(docs)
    lda.save(output_dir / "lda")

    print(f"\nModèles sauvegardés dans {output_dir}")


def run_visualization(model_dir="models/nmf", corpus_path="data/processed/corpus.parquet",
                      raw_dir="data/raw", port=8080):
    from src.visualization.app import app, init_app

    print("\n--- Lancement de la visu ---")

    init_app(model_dir, corpus_path, raw_dir)

    print(f"\nOuvre http://localhost:{port} dans ton navigateur")
    print("Ctrl+C pour arrêter\n")
    app.run(host="0.0.0.0", port=port)


# ---- Mode démo ----

def generate_demo_data(raw_dir="data/raw"):
    """Génère des données synthétiques pour tester le pipeline sans YouTube."""

    print("\n" + "=" * 60)
    print("Génération de données de démo")
    print("=" * 60)

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # vidéos fictives sur 4 thèmes
    videos = [
        {"video_id": "demo_cuisine_01", "title": "Ma recette de gâteau au chocolat",
         "channel": "Chef Michel", "upload_date": "2024-01-15", "topic": "cuisine"},
        {"video_id": "demo_sport_01", "title": "Résumé PSG vs Marseille",
         "channel": "Sport TV", "upload_date": "2024-02-20", "topic": "sport"},
        {"video_id": "demo_musique_01", "title": "Concert de jazz au Sunset",
         "channel": "Jazz FM", "upload_date": "2024-03-10", "topic": "musique"},
        {"video_id": "demo_tech_01", "title": "Test du nouveau MacBook Pro",
         "channel": "Tech Review", "upload_date": "2024-04-05", "topic": "tech"},
        {"video_id": "demo_cuisine_02", "title": "Tarte aux pommes traditionnelle",
         "channel": "Chef Michel", "upload_date": "2024-05-12", "topic": "cuisine"},
        {"video_id": "demo_sport_02", "title": "Les meilleurs buts de la saison",
         "channel": "Sport TV", "upload_date": "2024-06-18", "topic": "sport"},
        {"video_id": "demo_musique_02", "title": "Top 10 des albums de l'année",
         "channel": "Jazz FM", "upload_date": "2024-07-22", "topic": "musique"},
        {"video_id": "demo_tech_02", "title": "Intelligence artificielle en 2024",
         "channel": "Tech Review", "upload_date": "2024-08-30", "topic": "tech"},
    ]

    # templates de commentaires par thème
    templates = {
        "cuisine": [
            "Cette recette est délicieuse j'ai adoré le résultat final",
            "Super recette facile à préparer et vraiment bonne",
            "Le gâteau au chocolat est mon dessert préféré depuis toujours",
            "J'ai ajouté de la vanille et c'est encore meilleur que l'original",
            "Merci pour la recette ma famille a adoré le plat",
            "Les pâtes carbonara avec le parmesan c'est trop bon",
            "Je cuisine cette recette de poulet tous les weekends",
            "Le poulet rôti avec les légumes du jardin c'est un classique",
            "Tes recettes de cuisine sont toujours au top bravo",
            "J'ai raté la cuisson du gâteau mais le goût était bon",
            "Le fromage fondu sur le gratin dauphinois c'est incroyable",
            "Meilleure recette de crêpes que j'ai jamais trouvée en ligne",
            "La sauce tomate maison change complètement le goût du plat",
            "Un dessert parfait pour les fêtes de fin d'année en famille",
            "Les épices et les herbes donnent du caractère au plat",
        ],
        "sport": [
            "Quel match incroyable les joueurs étaient au top niveau",
            "Le gardien a fait des arrêts magnifiques hier soir au stade",
            "L'équipe de France a dominé le match du début à la fin",
            "Le but marqué en fin de match était absolument magnifique",
            "L'entraîneur devrait changer sa tactique pour le prochain match",
            "La course du marathon était épuisante mais tellement gratifiante",
            "Les supporters dans le stade mettaient une ambiance de folie",
            "Le tennis sur terre battue est un sport technique et mental",
            "Le transfert de ce joueur de football va changer la saison",
            "Le championnat de ligue cette année est vraiment disputé",
            "La défense de l'équipe était solide pendant tout le match",
            "Le sprint final du cycliste était incroyable de puissance",
            "Le classement du championnat de football est très serré",
            "L'arbitre a fait plusieurs erreurs d'arbitrage pendant le match",
            "La victoire en coupe d'Europe était largement méritée bravo",
        ],
        "musique": [
            "Ce concert de musique était absolument incroyable et émouvant",
            "Le solo de guitare électrique à la fin est magnifique",
            "J'écoute cet album de jazz en boucle depuis une semaine",
            "La voix de la chanteuse soprano est vraiment exceptionnelle",
            "Le rythme de cette chanson de funk donne envie de danser",
            "L'ambiance du festival de musique était magique cet été",
            "Le piano dans ce morceau classique apporte une émotion folle",
            "Cet artiste musicien mérite beaucoup plus de reconnaissance",
            "Le nouvel album du groupe est meilleur que le précédent",
            "La batterie et les percussions sur ce titre sont impressionnantes",
            "J'ai découvert ce groupe de rock grâce à cette vidéo",
            "Le jazz et le swing c'est vraiment de la musique pour l'âme",
            "Les paroles et le texte de cette chanson sont profonds",
            "Le son de la basse et du groove est tellement bon",
            "Vivement le prochain concert de musique live dans ma ville",
        ],
        "tech": [
            "Ce téléphone smartphone a une batterie vraiment impressionnante",
            "L'intelligence artificielle et le machine learning vont tout changer",
            "Le processeur de cet ordinateur est beaucoup plus rapide",
            "L'écran du nouveau laptop est magnifique en haute résolution",
            "La mise à jour du logiciel et de l'application corrige les bugs",
            "Le prix de ce gadget est élevé pour les fonctionnalités",
            "La caméra du smartphone prend des photos incroyables la nuit",
            "Le stockage cloud et le backup sont pratiques au quotidien",
            "La programmation en Python et le développement web sont populaires",
            "Ce robot aspirateur intelligent nettoie mieux que je pensais",
            "L'application mobile est intuitive et bien conçue niveau interface",
            "Le réseau fibre et la connexion internet changent l'expérience",
            "La sécurité des données et la vie privée sont cruciales",
            "Ce test du processeur GPU montre des performances impressionnantes",
            "Les objets connectés et la domotique transforment la maison",
        ],
    }

    for video_info in videos:
        vid_id = video_info["video_id"]
        topic = video_info["topic"]

        comments = []
        nb_comments = random.randint(25, 45)

        for i in range(nb_comments):
            # 80% du bon thème, 20% mélangé (c'est plus réaliste)
            if random.random() < 0.8:
                text = random.choice(templates[topic])
            else:
                other = random.choice([t for t in templates if t != topic])
                text = random.choice(templates[other])

            # un peu de bruit pour faire réaliste
            if random.random() < 0.15:
                text = text.upper()
            if random.random() < 0.2:
                text = text + " " + random.choice(["!", "!!", "trop bien", "vraiment", "franchement"])

            comments.append({
                "id": f"{vid_id}_c{i:03d}",
                "text": text,
                "author": f"User{random.randint(1, 200)}",
                "timestamp": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T{random.randint(8, 22):02d}:00:00",
                "likes": random.randint(0, 50),
                "replies": [],
            })

        data = {
            "video_id": vid_id,
            "title": video_info["title"],
            "channel": video_info["channel"],
            "upload_date": video_info["upload_date"],
            "comments": comments,
        }

        with open(raw_dir / f"{vid_id}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    total = sum(1 for _ in raw_dir.glob("*.json"))
    print(f"Données générées : {total} fichiers dans {raw_dir}")


# ---- Point d'entrée ----

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse thématique de commentaires YouTube"
    )

    # modes
    parser.add_argument("--demo", action="store_true",
                        help="Mode démo avec données synthétiques")
    parser.add_argument("--viz", action="store_true",
                        help="Lance uniquement la visualisation")

    # extraction
    parser.add_argument("--channels", type=str,
                        help="Fichier texte avec URLs des chaînes YouTube")
    parser.add_argument("--videos", type=str, nargs='+',
                        help="URLs ou IDs de vidéos individuelles")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Nb max de vidéos par chaîne")

    # options
    parser.add_argument("--nb-topics", type=int, default=10,
                        help="Nombre de topics (défaut: 10)")
    parser.add_argument("--model", type=str, default="models/nmf",
                        help="Dossier du modèle à visualiser (défaut: models/nmf)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port du serveur Flask (défaut: 8080)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Threads pour l'extraction (défaut: 4)")

    # skip steps
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-modeling", action="store_true")

    args = parser.parse_args()

    # --- mode visu seule ---
    if args.viz:
        run_visualization(args.model, "data/processed/corpus.parquet", "data/raw", args.port)
        return

    # --- mode démo ---
    if args.demo:
        t0 = time.time()
        generate_demo_data()
        run_preprocessing()
        run_modeling_quick(nb_topics=4)
        elapsed = time.time() - t0
        print(f"\nPipeline démo terminé en {elapsed:.0f}s")
        run_visualization(args.model, "data/processed/corpus.parquet", "data/raw", args.port)
        return

    # --- pipeline complet ---
    if not args.channels and not args.videos and not args.skip_extraction:
        print("Erreur : il faut --channels, --videos, --demo, ou --skip-extraction")
        print("Essaye : python main.py --demo")
        return

    t0 = time.time()

    # extraction
    if not args.skip_extraction:
        if args.channels:
            run_extraction(args.channels, max_videos=args.max_videos, workers=args.workers)
        if args.videos:
            from src.extraction.extractor import CommentExtractor
            ext = CommentExtractor(output_dir="data/raw", max_workers=args.workers)
            ext.run(videos=args.videos, max_videos=args.max_videos)

    # preprocessing
    if not args.skip_preprocessing:
        ok = run_preprocessing()
        if not ok:
            return

    # modeling
    if not args.skip_modeling:
        run_modeling(nb_topics=args.nb_topics)

    elapsed = time.time() - t0
    print(f"\nPipeline complet en {elapsed:.0f}s")

    # lancer la visu
    run_visualization(args.model, "data/processed/corpus.parquet", "data/raw", args.port)


if __name__ == "__main__":
    main()
