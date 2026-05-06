import warnings
import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()

project_path = current_file.parent.parent 

if str(project_path.parent) not in sys.path:
    sys.path.insert(0, str(project_path.parent))

import pandas as pd
import yfinance as yf

from equity_project.src.utils import load_config, three_barrier

warnings.filterwarnings("ignore")

project_path = Path(__file__).parent.parent


def generate_features(data):
    """Generate some features based on data

    Args:
        data (pd.DataFrame): raw OHLC dataset

    Returns:
        pd.DataFrame: features dataset
    """
    X = data.copy()

    close_col = "Close"

    # dealing with multiindex
    tickers = X[close_col].columns

    # price deviation from moving averages
    X[[(("dev5"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(5).mean()
    ) / X[close_col]
    X[[(("dev22"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(22).mean()
    ) / X[close_col]
    X[[(("dev252"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(252).mean()
    ) / X[close_col]
    X[[(("ma200vs50"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(200).mean() - X[close_col].rolling(50).mean()
    ) / X[close_col]

    # price momentum
    X[[(("mom5"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(5).rank(axis=1)
    )
    X[[(("mom22"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(22).rank(axis=1)
    )
    X[[(("mom252"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(252).rank(axis=1)
    )

    # volatility
    X[[(("vol5"), ticker) for ticker in tickers]] = (X[close_col].rolling(5).std()) / X[
        close_col
    ].rolling(5).mean()
    X[[(("vol22"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(22).std()
    ) / X[close_col].rolling(22).mean()
    X[[(("vol252"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(252).std()
    ) / X[close_col].rolling(252).mean()

    # drop unnecessary сols
    X.drop(columns=["Close", "High", "Low", "Open", "Volume"], inplace=True)

    # avoid forward-looking
    X = X.shift(1)

    # avoid cold start
    X = X.iloc[260:, :]

    return X


def get_label(train_data):
    # Ratio of take-profit to stop-loss
    target = train_data.Close.apply(lambda x: three_barrier(x, ptSl=[1, 1], horizon=15))
    return target



def get_raw_data():
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    TRAIN_START_DATE = cfg["train_start_date"]
    BACKTEST_END_DATE = cfg["backtest_end_date"]

    historical_components = pd.read_csv(
        project_path.as_posix() + "/data/pony/S&P_500_Historical_Components.csv",
        index_col=0,
        parse_dates=True
    )

    # Собираем все тикеры
    all_tickers = set()
    for row in historical_components['tickers']:
        all_tickers.update(str(row).split(','))
    
    TICKERS = sorted([t.replace('.', '-') for t in all_tickers])
    
    print(f"Downloading {len(TICKERS)} tickers (Sequential mode, please wait)...")
    
    # Отключаем потоки (threads=False), чтобы не злить сервера Yahoo
    data = yf.download(
        TICKERS, 
        start=TRAIN_START_DATE, 
        end=BACKTEST_END_DATE, 
        group_by="column", 
        auto_adjust=True,
        threads=False 
    )

    data.dropna(axis=1, how='all', inplace=True)
    
    # Приводим к единому формату
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_product([['Close'], data.columns])
        
    # --- БЛОК ВАЛИДАЦИИ КАЧЕСТВА ДАННЫХ ---
    downloaded_tickers = data.columns.get_level_values(1).unique()
    must_have_tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'AMZN']
    
    missing = [t for t in must_have_tickers if t not in downloaded_tickers]
    
    if missing:
        raise ValueError(f"ОШИБКА YAHOO: Не скачались ключевые тикеры: {missing}. Запустите скрипт еще раз через пару минут.")
    else:
        print(f"Успех! База скачана отлично. Доступно {len(downloaded_tickers)} тикеров.")
    # ---------------------------------------
    
    return data, historical_components



def get_data():
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    
    # Создаем папки
    for folder in ["/data/raw", "/data/processed", "/models", "/artifacts/metrics", "/artifacts/plots"]:
        os.makedirs(project_path.as_posix() + folder, exist_ok=True)

    data, historical_components = get_raw_data()

    # Генерируем фичи и таргеты
    X = generate_features(data)
    y = get_label(data)

    # Переводим в длинный формат (Stack)
    X = X.stack(level=1)
    y = y.stack() # Это превратит DF в Series с MultiIndex (Date, Ticker)

    # Оставляем только те (Дата, Тикер), которые есть в обоих датасетах
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    y.name = "target"

    # Создаем маску для борьбы с Survivorship Bias
    mask = pd.DataFrame(0, index=data.index, columns=data.columns.get_level_values(1).unique())
    for date, row in historical_components.iterrows():
        actual_dates = mask.index[mask.index >= date]
        if len(actual_dates) > 0:
            active = [t.replace('.', '-') for t in str(row['tickers']).split(',')]
            valid = [t for t in active if t in mask.columns]
            mask.loc[actual_dates[0]:, mask.columns] = 0
            mask.loc[actual_dates[0]:, valid] = 1
    
    mask.to_parquet(project_path.as_posix() + "/data/processed/universe_mask.parquet")

    # Сохранение данных
    TRAIN_END = cfg["train_end_date"]
    TEST_START = cfg["backtest_start_date"]

    # Сырые данные для бэктеста
    data.to_parquet(project_path.as_posix() + "/data/raw/backtest_data.parquet")

    # Обработанные данные для ML
    X.loc[:TRAIN_END].to_parquet(project_path.as_posix() + "/data/processed/X_train.parquet")
    y.loc[:TRAIN_END].to_frame().to_parquet(project_path.as_posix() + "/data/processed/y_train.parquet")
    
    X.loc[TEST_START:].to_parquet(project_path.as_posix() + "/data/processed/X_backtest.parquet")
    y.loc[TEST_START:].to_frame().to_parquet(project_path.as_posix() + "/data/processed/y_backtest.parquet")



if __name__ == "__main__":
    get_data()


