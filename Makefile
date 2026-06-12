.PHONY: setup data features train evaluate run clean

setup:
	pip install -r requirements.txt

data:
	python src/data/preprocessing.py

features:
	python src/features/engineering.py

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

run: data features train evaluate
	@echo "Pipeline completo!"

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cache limpo!"