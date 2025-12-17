# TabTransformer Sales Forecasting

Development-focused repository for forecasting Rossmann-style store sales using classical baselines and a TabTransformer deep model. The pipeline engineers temporal features, trains models with deterministic splits, and exports metrics, checkpoints, and diagnostic plots for experimentation.

## Features
- Data preparation: date parsing, lag/rolling features, calendar attributes, categorical casting, parquet + metadata export
- Models: Linear Regression, XGBoost, and a PyTorch TabTransformer with categorical embeddings & attention
- Training loop: early stopping on validation RMSE, gradient clipping, AMP flag, artifact logging per model
- Evaluation: MAE/RMSE/R2/MAPE metrics, scatter + residual plots, optional attention visualization
- CLI workflow: `prepare`, `train`, `eval`, `predict` commands backed by YAML configuration
- Tooling: Makefile targets, pytest suite, mypy/black/isort configuration

## Team
- Harshil Patel (hpatel17@stevens.edu)
- Christopher Meumann (cmeumann@stevens.edu)

## Contributions
- Harshil Patel
    - Setup data processing for dataset sales CSVs and split dataset into training, validation and testing.
    - Built logging for each process happens in system.
    - Built TabTransformer model.
    - Built training configuration for all models.
    - Built Command line interface for performing actions to run application.



- Christopher Meumann:
    - built linear regression & XGBoost models.
    - Wrote Tests for application.
    - built module for computing metrics, save metrics, load models for prediction, save plots too.

## Getting Started
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

## Configuration
Edit `config/default.yaml` to point to your raw CSV (Rossmann schema) and to control feature lists, splits, and training hyper-parameters. Key sections:
- `paths`: raw CSV, processed parquet directory, artifact root
- `features`: categorical/numeric columns, forecasting horizon, target name
- `splits`: either explicit date boundaries or ratios
- `training`: global hyper-parameters (batch size, lr, patience, AMP, device)
- `tabtransformer` / `xgboost`: model-specific knobs

## CLI Usage
```
python -m tabtransformer_sales.src.cli prepare --config config/default.yaml
python -m tabtransformer_sales.src.cli train   --config config/default.yaml --model tabtx
python -m tabtransformer_sales.src.cli eval    --config config/default.yaml --model tabtx
python -m tabtransformer_sales.src.cli predict --config config/default.yaml --model tabtx --input data/raw/sample.csv --output artifacts/tabtx/preds.csv
```
Each command saves outputs under `artifacts/{model_name}/` and logs metrics to stdout.

## Example Metrics (synthetic sample)
| Metric | Linear Regression | XGBoost | TabTransformer |
| ------ | ----------------- | ------- | -------------- |
| MAE    | 1350.2            | 1124.8  | **975.4**      |
| RMSE   | 1812.7            | 1520.5  | **1298.3**     |
| R2     | 0.62              | 0.71    | **0.78**       |

> Numbers are illustrative; expect variance with real Rossmann data.

## Development Workflow
- `make prepare` runs the CLI prepare step with the default config
- `make train MODEL=linreg` trains a chosen model (linreg|xgb|tabtx)
- `make eval MODEL=tabtx`
- `make predict MODEL=xgb INPUT=path/to.csv OUTPUT=preds.csv`
- `make lint` runs black, isort, mypy
- `make test` executes `pytest -q`

## Testing
The `tests/` directory contains:
- `test_data.py`: synthetic CSV → processed frame checks (shapes, NaNs)
- `test_metrics.py`: deterministic metric expectations
- `test_models.py`: smoke tests for each model
- `test_cli.py`: end-to-end CLI run on toy data

Run locally with:
```bash
pytest -q
```

## Dataset Notes
- Input CSV must include at minimum: `Store, DayOfWeek, Date, Promo, StateHoliday, SchoolHoliday, Customers, Sales, CompetitionDistance`
- `Date` must be YYYY-MM-DD; stores must be sortable (int or string)
- Missing `CompetitionDistance` is imputed with median; other numeric NaNs fall back to forward fill + global median
- Target defaults to next-day sales (`Sales` shifted by +1 day) but the `horizon` config overrides it.

## Outputs
- Processed parquet + metadata JSON in `data/processed/`
- Artifacts per model: `model.{pt,joblib}`, `metrics.json`, plots, prediction files, config snapshot
- Optional attention heatmaps when `plot_attention: true`

## Support & Extensions
- Swap in weekly forecasting by adjusting `features.horizon`
- Extend categorical lists or add exogenous regressors via the config lists
- Integrate additional models by extending `tabtransformer_sales/src/models`

Happy forecasting!
