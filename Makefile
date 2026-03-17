.PHONY: install test demo extract preprocess model viz clean

# Installation complète
install:
	pip install -r requirements.txt
	python3 -m spacy download fr_core_news_md
	python3 -m spacy download en_core_web_md

# Lancer les tests
test:
	python3 -m pytest tests/ -v

# Mode démo (données synthétiques + pipeline + visu)
demo:
	python3 main.py --demo

# Étapes individuelles du pipeline
extract:
	python3 -m src.extraction.extractor --channels channels.txt --output data/raw/

preprocess:
	python3 -m src.preprocessing.preprocessor --input data/raw --output data/processed/corpus.parquet

model:
	python3 -m src.modeling.compare --input data/processed/corpus.parquet --nb-topics 10 --output models

# Visualisation seule
viz:
	python3 main.py --viz --model models/nmf

# Nettoyage
clean:
	rm -rf data/ models/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
