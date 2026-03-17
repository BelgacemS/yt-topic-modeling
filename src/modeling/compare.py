"""Script de comparaison des 3 modèles de topic modeling.

Compare LDA, NMF et BERTopic sur le même corpus,
avec des métriques quantitatives (cohérence Cv, diversité)
et un rapport récapitulatif.

Usage : python -m src.modeling.compare --input data/processed/corpus.parquet
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.modeling.lda_model import LDAModel
from src.modeling.nmf_model import NMFModel
from src.modeling.bertopic_model import BERTopicModel


def load_corpus(path):
    """Charge le corpus depuis un fichier parquet."""
    df = pd.read_parquet(path)
    # on utilise cleaned_text pour LDA/NMF et raw_text pour BERTopic
    docs_clean = df['cleaned_text'].dropna().tolist()
    docs_raw = df['raw_text'].dropna().tolist() if 'raw_text' in df.columns else docs_clean
    print(f"Corpus chargé : {len(docs_clean)} documents")
    return docs_clean, docs_raw


def run_comparison(docs_clean, docs_raw, nb_topics=10, output_dir="models"):
    """Lance les 3 modèles et compare les résultats."""
    output_dir = Path(output_dir)
    results = []

    # --- LDA ---
    print("\n" + "="*60)
    print("LDA (Gensim)")
    print("="*60)
    t0 = time.time()
    lda = LDAModel(nb_topics=nb_topics)
    lda.fit(docs_clean)
    lda_time = time.time() - t0

    lda_coherence = lda.get_coherence()
    lda_diversity = lda.get_diversity()
    lda.save(output_dir / "lda")

    results.append({
        "modèle": "LDA",
        "nb_topics": nb_topics,
        "cohérence_cv": round(lda_coherence, 4),
        "diversité": round(lda_diversity, 4),
        "temps_s": round(lda_time, 1),
    })

    # --- NMF ---
    print("\n" + "="*60)
    print("NMF (scikit-learn)")
    print("="*60)
    t0 = time.time()
    nmf = NMFModel(nb_topics=nb_topics)
    nmf.fit(docs_clean)
    nmf_time = time.time() - t0

    nmf_coherence = nmf.get_coherence()
    nmf_diversity = nmf.get_diversity()
    nmf.save(output_dir / "nmf")

    results.append({
        "modèle": "NMF",
        "nb_topics": nb_topics,
        "cohérence_cv": round(nmf_coherence, 4),
        "diversité": round(nmf_diversity, 4),
        "temps_s": round(nmf_time, 1),
    })

    # --- BERTopic ---
    # TODO: ajouter un try/except au cas où pas assez de docs pour HDBSCAN
    print("\n" + "="*60)
    print("BERTopic")
    print("="*60)
    t0 = time.time()
    # BERTopic marche mieux avec le texte brut (les embeddings capturent la sémantique)
    bertopic = BERTopicModel(nb_topics=nb_topics)
    bertopic.fit(docs_raw)
    bert_time = time.time() - t0

    bert_coherence = bertopic.get_coherence()
    bert_diversity = bertopic.get_diversity()
    bertopic.save(output_dir / "bertopic")

    results.append({
        "modèle": "BERTopic",
        "nb_topics": bertopic.nb_topics,  # peut être différent si auto
        "cohérence_cv": round(bert_coherence, 4),
        "diversité": round(bert_diversity, 4),
        "temps_s": round(bert_time, 1),
    })

    return results, {"lda": lda, "nmf": nmf, "bertopic": bertopic}


def print_report(results, models):
    print("\n" + "="*60)
    print("RAPPORT COMPARATIF")
    print("="*60)

    # tableau récapitulatif
    df = pd.DataFrame(results)
    print("\nMétriques globales :")
    print(df.to_string(index=False))

    # meilleur modèle
    best_idx = df['cohérence_cv'].idxmax()
    print(f"\nMeilleur modèle (cohérence Cv) : {df.iloc[best_idx]['modèle']}")

    best_div = df['diversité'].idxmax()
    print(f"Meilleur modèle (diversité) : {df.iloc[best_div]['modèle']}")

    # détails des topics par modèle
    for name, model in models.items():
        print(f"\n--- Topics {name.upper()} ---")
        info = model.get_topic_info()
        if not info.empty:
            print(info.to_string(index=False))

    return df


def save_report(results, models, output_dir="models"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # résultats quantitatifs
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # topics détaillés
    all_topics = {}
    for name, model in models.items():
        topics = model.get_topics()
        # conversion pour JSON (les tuples deviennent des listes)
        all_topics[name] = {
            str(k): [(w, round(float(s), 4)) for w, s in v]
            for k, v in topics.items()
        }

    with open(output_dir / "all_topics.json", "w") as f:
        json.dump(all_topics, f, indent=2, ensure_ascii=False)

    print(f"\nRapport sauvegardé dans {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare LDA, NMF et BERTopic")
    parser.add_argument("--input", type=str, default="data/processed/corpus.parquet",
                        help="Chemin du corpus parquet")
    parser.add_argument("--nb-topics", type=int, default=10,
                        help="Nombre de topics (défaut: 10)")
    parser.add_argument("--output", type=str, default="models",
                        help="Dossier de sortie")
    args = parser.parse_args()

    docs_clean, docs_raw = load_corpus(args.input)
    results, models = run_comparison(docs_clean, docs_raw, args.nb_topics, args.output)
    report_df = print_report(results, models)
    save_report(results, models, args.output)


if __name__ == "__main__":
    main()
