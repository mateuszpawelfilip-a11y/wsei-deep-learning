import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_surrogates(y_true: np.ndarray, y_pred: np.ndarray, name: str = "model") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    corr, _ = spearmanr(y_true, y_pred)
    return {
        "name": name,
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "Spearman": float(corr) if not np.isnan(corr) else 0.0,
    }


def print_metrics_table(rows: list[dict]):
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Spearman':>10}")
    print("-" * 50)
    for r in rows:
        print(
            f"{r['name']:<12} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} "
            f"{r['R2']:>8.4f} {r['Spearman']:>10.4f}"
        )
