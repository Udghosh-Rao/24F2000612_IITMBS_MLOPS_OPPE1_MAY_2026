.PHONY: install dev test lint data train-1 train-2 evaluate serve app docker

install:      ## full pipeline deps
	pip install -r requirements.txt
dev:          ## lightweight deps for app + tests
	pip install -r requirements-dev.txt
lint:
	ruff check .
test:
	pytest -q
data:         ## build engineered datasets from raw CSVs (needs `dvc pull`)
	python -m src.prepare_data --input-dirs data/v0 --output feature_repo/data/iteration1.parquet
	python -m src.prepare_data --input-dirs data/v0 data/v1 --output feature_repo/data/iteration2.parquet
	cd feature_repo && feast apply
train-1:
	python -m src.train --input feature_repo/data/iteration1.parquet --iteration 1
train-2:
	python -m src.train --input feature_repo/data/iteration2.parquet --iteration 2
evaluate:
	python -m src.ci_evaluate
serve:
	uvicorn src.serve:app --reload --port 8000
app:
	streamlit run app.py
docker:
	docker build -t stock-movement-api .
