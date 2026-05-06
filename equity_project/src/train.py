import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()

project_path = current_file.parent.parent 

if str(project_path.parent) not in sys.path:
    sys.path.insert(0, str(project_path.parent))

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

project_path = Path(__file__).parent.parent


def instantiate_model():
    """Задаем параметры модели для обучения

    Returns:
        CatBoostClassifier: Инициированная ML модель
    """
    model = CatBoostClassifier(
        auto_class_weights="Balanced",
        loss_function="MultiClass",
        use_best_model=True,
        eval_metric="MultiClass",
        early_stopping_rounds=50,
        random_seed=6193,
    )
    return model

def train():
    os.makedirs(project_path.as_posix() + "/models", exist_ok=True)

    X = pd.read_parquet(project_path.as_posix() + "/data/processed/X_train.parquet")
    y = pd.read_parquet(project_path.as_posix() + "/data/processed/y_train.parquet")

    # Берем уникальные даты
    dates = X.index.get_level_values(0).unique().sort_values()
    split_idx = int(len(dates) * 0.7) # 70% на трейн
    

    # Отрезаем 20 дней с конца трейна, чтобы барьеры не пересекались с валидацией
    train_dates = dates[:split_idx - 20] 
    val_dates = dates[split_idx:]

    X_train, y_train = X.loc[train_dates], y.loc[train_dates]
    X_val, y_val = X.loc[val_dates], y.loc[val_dates]

    model = instantiate_model()
    model.set_params(depth=4, l2_leaf_reg=5)

    model.fit(y=y_train, X=X_train, eval_set=(X_val, y_val), verbose=25)

    joblib.dump(model, project_path.as_posix() + "/models/model.joblib")


if __name__ == "__main__":
    train()
