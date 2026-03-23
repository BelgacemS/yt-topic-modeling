# YT Topic Modeling

Pipeline d'analyse thématique de commentaires YouTube. Extrait les commentaires de chaînes/vidéos YouTube, applique un prétraitement NLP, puis compare trois approches de topic modeling : LDA, NMF et BERTopic. Les résultats sont présentés via une interface web interactive.


## Installation

```bash
# cloner le repo
git clone <url-du-repo>
cd yt-topic-modeling

# installer les dépendances
pip install -r requirements.txt

# modèles spaCy (français + anglais)
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_md
```

Ou avec le Makefile :
```bash
make install
```

Nécessite Python 3.10+.


## Utilisation rapide

### Mode démo (sans YouTube)

Pour tester le pipeline avec des données synthétiques :
```bash
python main.py --demo
```

Ca génère de faux commentaires, lance le prétraitement + modélisation, et ouvre l'interface web sur `http://localhost:8080`.

### Pipeline complet

```bash
# 1. Créer un fichier channels.txt avec les URLs des chaînes
echo "https://www.youtube.com/@NomDeLaChaine" > channels.txt

# 2. Lancer le pipeline complet
python main.py --channels channels.txt --nb-topics 10

# 3. Ou étape par étape
python -m src.extraction.extractor --channels channels.txt --output data/raw/
python -m src.preprocessing.preprocessor --input data/raw --output data/processed/corpus.parquet
python -m src.modeling.compare --input data/processed/corpus.parquet --nb-topics 10 --output models

# 4. Visualisation seule
python main.py --viz --model models/nmf
```

### Options utiles

```bash
# limiter le nombre de vidéos par chaîne
python main.py --channels channels.txt --max-videos 20

# choisir le modèle pour la visu (lda, nmf, ou bertopic)
python main.py --viz --model models/lda

# changer le port
python main.py --viz --port 9000

# sauter certaines étapes
python main.py --skip-extraction --skip-preprocessing --nb-topics 15
```


## Architecture

```
yt-topic-modeling/
├── src/
│   ├── extraction/          # Module 1 : extraction YouTube (yt-dlp)
│   │   └── extractor.py
│   ├── preprocessing/       # Module 2 : prétraitement NLP (spaCy)
│   │   └── preprocessor.py
│   ├── modeling/            # Module 3 : topic modeling
│   │   ├── base.py          # classe abstraite BaseTopicModel
│   │   ├── lda_model.py     # LDA (Gensim, Blei et al. 2003)
│   │   ├── nmf_model.py     # NMF (scikit-learn, Lee & Seung 1999)
│   │   ├── bertopic_model.py # BERTopic (Grootendorst 2022)
│   │   └── compare.py       # comparaison des 3 modèles
│   └── visualization/       # Module 4 : interface web (Flask + Plotly)
│       ├── app.py
│       ├── templates/
│       └── static/
├── tests/                   # tests unitaires (pytest)
├── main.py                  # script principal du pipeline
├── Makefile
└── requirements.txt
```

### Flux de données

```
Extraction (yt-dlp)     →  data/raw/*.json          (JSON, un fichier par vidéo)
Prétraitement (spaCy)   →  data/processed/*.parquet  (nettoyage, tokenisation, lemmatisation)
Topic Modeling           →  models/                   (LDA, NMF, BERTopic en pickle)
Visualisation (Flask)    →  http://localhost:8080      (dashboard interactif)
```

### Modèles implémentés

| Modèle | Librairie | Approche | Référence |
|--------|-----------|----------|-----------|
| LDA | Gensim | Probabiliste (Dirichlet) | Blei et al. 2003 |
| NMF | scikit-learn | Factorisation matricielle | Lee & Seung 1999 |
| BERTopic | bertopic | Embeddings + clustering | Grootendorst 2022 |

Les trois modèles partagent la même interface (`BaseTopicModel`) et sont comparés sur la cohérence Cv et la diversité des topics.


## Visualisation

L'interface web affiche :
- **Scatter plot UMAP** : projection 2D des commentaires, colorés par topic. Hover pour voir le texte.
- **Nuage de mots** : mots les plus représentatifs de chaque topic.
- **Barchart** : top 10 mots par topic avec leurs poids.
- **Timeline** : évolution de la proportion de chaque topic au fil du temps.
- **Filtres** : sélection par chaîne ou par vidéo.


## Tests

```bash
python -m pytest tests/ -v

# ou
make test
```


## Dépendances principales

- yt-dlp : extraction YouTube
- spaCy + langdetect : NLP et détection de langue
- gensim : LDA et métriques de cohérence
- scikit-learn : NMF et TF-IDF
- bertopic + sentence-transformers : BERTopic
- umap-learn + hdbscan : réduction et clustering
- flask + plotly + wordcloud : visualisation
