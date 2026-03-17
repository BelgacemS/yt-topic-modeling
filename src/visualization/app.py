"""App Flask pour visualiser les résultats du topic modeling.

Lance avec :
    python -m src.visualization.app --model models/nmf --corpus data/processed/corpus.parquet
"""

import json
import argparse
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from src.modeling.base import BaseTopicModel


app = Flask(__name__)

# données globales chargées au démarrage
VIZ = {}


def init_app(model_dir, corpus_path, raw_dir="data/raw"):
    """Charge le modèle + corpus et prépare tout pour la visu."""
    global VIZ

    print("Chargement du modèle...")
    model = BaseTopicModel.load(model_dir)

    print("Chargement du corpus...")
    df = pd.read_parquet(corpus_path)

    # on synchronise le nombre de docs (au cas où)
    doc_topics = model.doc_topics
    n = min(len(doc_topics), len(df))
    df = df.iloc[:n].reset_index(drop=True)
    df['topic'] = doc_topics[:n]

    # topics et mots-clés
    topics = model.get_topics()

    # projection UMAP 2D pour le scatter
    print("Calcul UMAP 2D (ça peut prendre un moment)...")
    coords = compute_umap_2d(df['cleaned_text'].tolist())
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]

    # métadonnées vidéos (channel, dates) depuis les JSON bruts
    meta = load_video_meta(raw_dir)
    df['channel'] = df['video_id'].map(lambda v: meta.get(v, {}).get('channel', 'inconnu'))
    df['video_title'] = df['video_id'].map(lambda v: meta.get(v, {}).get('title', ''))
    df['upload_date'] = df['video_id'].map(lambda v: meta.get(v, {}).get('upload_date', ''))

    # TODO: les labels sont moches quand les mots sont pas lemmatisés
    topic_labels = {}
    for tid, words in topics.items():
        if tid == -1:
            topic_labels[tid] = "Outliers"
        else:
            top3 = ", ".join([w for w, _ in words[:3]])
            topic_labels[tid] = f"Topic {tid}: {top3}"

    df['topic_label'] = df['topic'].map(topic_labels)

    # stocker tout
    VIZ['model'] = model
    VIZ['model_name'] = type(model).__name__
    VIZ['df'] = df
    VIZ['topics'] = topics
    VIZ['topic_labels'] = topic_labels
    VIZ['channels'] = sorted(df['channel'].dropna().unique().tolist())

    # dict video_id -> titre pour le menu déroulant
    video_titles = {}
    for vid in sorted(df['video_id'].unique()):
        title = meta.get(vid, {}).get('title', '')
        video_titles[vid] = title if title else vid
    VIZ['videos'] = video_titles

    nb = len([t for t in topics if t != -1])
    print(f"Prêt ! {len(df)} documents, {nb} topics ({VIZ['model_name']})")


def compute_umap_2d(documents):
    """Projette les documents en 2D via TF-IDF + UMAP."""
    from umap import UMAP

    n = len(documents)
    if n < 5:
        # trop peu de docs, on fait du random
        return np.column_stack([np.random.randn(n), np.random.randn(n)])

    vec = TfidfVectorizer(
        max_features=min(3000, n * 5),
        max_df=0.95,
        min_df=min(2, max(1, n // 50)),
    )
    tfidf = vec.fit_transform(documents)

    if tfidf.shape[1] == 0:
        return np.column_stack([np.random.randn(n), np.random.randn(n)])

    reducer = UMAP(
        n_components=2,
        random_state=42,
        metric='cosine',
        n_neighbors=min(15, n - 1),
    )
    return reducer.fit_transform(tfidf)


def load_video_meta(raw_dir):
    raw_dir = Path(raw_dir)
    meta = {}

    if not raw_dir.exists():
        return meta

    for fpath in raw_dir.glob("*.json"):
        if fpath.name.startswith('.'):
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            vid = data.get("video_id", fpath.stem)
            meta[vid] = {
                "title": data.get("title", ""),
                "channel": data.get("channel", ""),
                "upload_date": data.get("upload_date", ""),
            }
        except Exception:
            continue

    return meta


# ---- Routes ----

@app.route("/")
def index():
    if not VIZ:
        return "Pas de données. Lance d'abord le pipeline (python main.py --demo)", 500

    topic_ids = sorted([t for t in VIZ['topics'] if t != -1])
    first_topic = topic_ids[0] if topic_ids else 0

    return render_template("index.html",
        model_name=VIZ['model_name'],
        nb_docs=len(VIZ['df']),
        nb_topics=len(topic_ids),
        channels=VIZ['channels'],
        videos=VIZ['videos'],
        topic_labels=VIZ['topic_labels'],
        first_topic=first_topic,
    )


@app.route("/api/topics")
def api_topics():
    """Liste des topics avec leurs mots-clés et stats."""
    result = []
    for tid, words in VIZ['topics'].items():
        if tid == -1:
            continue
        result.append({
            "id": tid,
            "label": VIZ['topic_labels'].get(tid, f"Topic {tid}"),
            "words": [{"word": w, "weight": round(float(s), 4)} for w, s in words[:10]],
            "nb_docs": int((VIZ['df']['topic'] == tid).sum()),
        })
    return jsonify(result)


@app.route("/api/scatter")
def api_scatter():
    """Données pour le scatter plot UMAP, avec filtres optionnels."""
    df = VIZ['df']

    # filtres
    channel = request.args.get('channel')
    video = request.args.get('video')

    mask = pd.Series(True, index=df.index)
    if channel and channel != 'all':
        mask &= df['channel'] == channel
    if video and video != 'all':
        mask &= df['video_id'] == video

    filtered = df[mask]

    # limiter à 5000 points pour la perf du navigateur
    if len(filtered) > 5000:
        filtered = filtered.sample(5000, random_state=42)

    return jsonify({
        "x": filtered['x'].tolist(),
        "y": filtered['y'].tolist(),
        "topic": filtered['topic'].tolist(),
        "topic_label": filtered['topic_label'].tolist(),
        "text": filtered['raw_text'].str[:200].tolist(),
        "video_id": filtered['video_id'].tolist(),
    })


@app.route("/api/wordcloud/<int:topic_id>")
def api_wordcloud(topic_id):
    """Génère un nuage de mots PNG pour un topic."""
    topics = VIZ['topics']
    if topic_id not in topics:
        return "Topic introuvable", 404

    words = topics[topic_id]
    freq = {w: max(float(s), 0.001) for w, s in words}

    wc = WordCloud(
        width=600, height=400,
        background_color='white',
        colormap='viridis',
        max_words=50,
    ).generate_from_frequencies(freq)

    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)

    return Response(buf.getvalue(), mimetype='image/png')


@app.route("/api/barchart/<int:topic_id>")
def api_barchart(topic_id):
    """Top 10 mots d'un topic avec leurs poids."""
    topics = VIZ['topics']
    if topic_id not in topics:
        return jsonify({"words": [], "weights": []})

    data = topics[topic_id][:10]
    return jsonify({
        "words": [w for w, _ in data],
        "weights": [round(float(s), 4) for _, s in data],
    })


@app.route("/api/timeline")
def api_timeline():
    """Évolution des topics au fil du temps (par date d'upload vidéo)."""
    df = VIZ['df']

    # on a besoin de dates valides
    df_dated = df[df['upload_date'].str.len() > 0].copy()
    if df_dated.empty:
        return jsonify({"dates": [], "series": []})

    # proportion de chaque topic par date
    grouped = df_dated.groupby(['upload_date', 'topic']).size().reset_index(name='count')
    totals = df_dated.groupby('upload_date').size().reset_index(name='total')

    dates = sorted(totals['upload_date'].unique())
    topic_ids = sorted([t for t in df['topic'].unique() if t != -1])

    series = []
    for tid in topic_ids:
        values = []
        for d in dates:
            c = grouped[(grouped['upload_date'] == d) & (grouped['topic'] == tid)]['count'].sum()
            t = totals[totals['upload_date'] == d]['total'].iloc[0]
            values.append(round(c / t * 100, 1) if t > 0 else 0)
        series.append({
            "name": VIZ['topic_labels'].get(tid, f"Topic {tid}"),
            "values": values,
        })

    return jsonify({"dates": dates, "series": series})


# ---- Point d'entrée ----

def main():
    parser = argparse.ArgumentParser(description="Visualisation des topics")
    parser.add_argument("--model", default="models/nmf", help="Dossier du modèle")
    parser.add_argument("--corpus", default="data/processed/corpus.parquet", help="Corpus parquet")
    parser.add_argument("--raw", default="data/raw", help="Dossier des JSON bruts")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_app(args.model, args.corpus, args.raw)

    print(f"\nOuvre http://localhost:{args.port} dans ton navigateur")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
