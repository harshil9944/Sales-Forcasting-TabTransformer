PYTHON ?= python
CONFIG ?= config/default.yaml
MODEL ?= tabtx
INPUT ?= data/raw/sample.csv
OUTPUT ?= artifacts/$(MODEL)/preds.csv

.PHONY: setup lint test prepare train eval predict clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m black tabtransformer_sales
	$(PYTHON) -m isort tabtransformer_sales
	$(PYTHON) -m mypy tabtransformer_sales/src

test:
	$(PYTHON) -m pytest -q

prepare:
	$(PYTHON) -m tabtransformer_sales.src.cli prepare --config $(CONFIG)

train:
	$(PYTHON) -m tabtransformer_sales.src.cli train --config $(CONFIG) --model $(MODEL)

eval:
	$(PYTHON) -m tabtransformer_sales.src.cli eval --config $(CONFIG) --model $(MODEL)

predict:
	$(PYTHON) -m tabtransformer_sales.src.cli predict --config $(CONFIG) --model $(MODEL) --input $(INPUT) --output $(OUTPUT)

clean:
	if exist artifacts (powershell -Command "Remove-Item -Recurse -Force artifacts")
	if exist .mypy_cache (powershell -Command "Remove-Item -Recurse -Force .mypy_cache")
	if exist .pytest_cache (powershell -Command "Remove-Item -Recurse -Force .pytest_cache")
	for /d %%d in (tabtransformer_sales\src\*\__pycache__) do (powershell -Command "Remove-Item -Recurse -Force '%%d'")
