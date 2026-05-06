import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_path = current_file.parent.parent 

if str(project_path.parent) not in sys.path:
    sys.path.insert(0, str(project_path.parent))

import joblib
import pandas as pd
import vectorbt as vbt
import yfinance as yf
import numpy as np
import datetime

from equity_project.src.utils import load_config, save_dict


def generate_weights(preds, mask, close, vix, vix_sma, vix_std):
    preds_unstack = preds.unstack(level=1)
    # Сигнал: Вероятность роста минус вероятность падения
    signal = preds_unstack[2] - preds_unstack[0]

    # Синхронизируем индексы
    common_cols = signal.columns.intersection(mask.columns)
    common_idx = signal.index.intersection(mask.index)
    
    signal = signal.loc[common_idx, common_cols]
    mask_aligned = mask.loc[common_idx, common_cols]

    # Зануляем акции вне S&P 500
    signal = signal * mask_aligned

    # Считаем ранги
    ranks = signal.rank(axis=1, ascending=False, pct=False)
    
    # Синхронизируем VIX с нашими датами 
    vix_aligned = vix.reindex(signal.index).ffill()
    vix_sma_aligned = vix_sma.reindex(signal.index).ffill()
    vix_std_aligned = vix_std.reindex(signal.index).ffill()

    # Создаем серию лимитов: по умолчанию Топ-5 (для спокойного рынка)
    dynamic_top_n = pd.Series(5, index=signal.index)
    # Если VIX пробил SMA50 вверх (кризис, корреляция стремится к 1) -> переключаемся на Топ-15 для диверсификации
    dynamic_top_n[vix_aligned > (vix_sma_aligned + 0.5 * vix_std_aligned)] = 15

    # Векторное сравнение: оставляем только те акции, чей ранг <= нашему динамическому числу на этот день
    weights = ranks.le(dynamic_top_n, axis=0).astype(float)
    
    # Обнуляем неуверенные сигналы
    weights[signal <= 0] = 0 

    # Взвешиваем по уверенности
    confident_signals = signal * weights
    
    weights = (confident_signals.T / confident_signals.sum(axis=1)).T
    weights = weights.fillna(0)
    
    return weights

def run_backtest():
    os.makedirs(project_path.as_posix() + "/artifacts/plots", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/artifacts/metrics", exist_ok=True)

    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    # Считываем данные
    X_backtest = pd.read_parquet(project_path.as_posix() + "/data/processed/X_backtest.parquet")
    backtest_data = pd.read_parquet(project_path.as_posix() + "/data/raw/backtest_data.parquet", engine="pyarrow")

    # Инференс модели
    model = joblib.load(project_path.as_posix() + "/models/model.joblib")
    preds = model.predict_proba(X_backtest)
    preds = pd.DataFrame(preds, index=X_backtest.index)

    close = backtest_data.Close.dropna(axis=1, how="all")
    

    
    # Cold start для sma50
    vix_start = (close.index[0] - datetime.timedelta(days=100)).strftime('%Y-%m-%d')
    vix = yf.download("^VIX", start=vix_start, end=close.index[-1], progress=False)["Close"]
    
    if isinstance(vix, pd.DataFrame):
        vix = vix.squeeze()
        
    vix_sma = vix.rolling(window=252).mean()
    vix_std = vix.rolling(window=252).std()


    # Загружаем маску и генерируем веса (передаем close для расчета волатильности)
    mask = pd.read_parquet(project_path.as_posix() + "/data/processed/universe_mask.parquet")
    size = generate_weights(preds, mask, close, vix, vix_sma, vix_std)

    # Синхронизация цен и весов
    common_tickers = sorted(list(set(close.columns) & set(size.columns)))
    close = close.loc[size.index, common_tickers]
    size = size[common_tickers]
    
    # Формируем цены исполнения (Open следующего дня)
    price = backtest_data.Open.shift(-1).loc[size.index, common_tickers]
    price = price.fillna(close)


    bad_price_mask = (price <= 0) | price.isna()
    price = price.where(~bad_price_mask, 1.0)
    size = size.where(~bad_price_mask, 0.0)

    init_cash = cfg["init_cash"]
    fees = cfg["fees"]

    # Бэктест
    pf = vbt.Portfolio.from_orders(
        close=close,
        price=price,
        size=size,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq="1d",
        init_cash=init_cash,
        fees=fees,
    )

    stats = pf.stats()
    
    # Бенчмарк SPY
    spy = yf.download("SPY", start=close.index[0], end=close.index[-1], progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()
        
    spy_return = (spy.iloc[-1] / spy.iloc[0] - 1) * 100


    print("Backtest results")
    print("="*40)
    print(f"Start:                 {stats['Start']}")
    print(f"End:                   {stats['End']}")
    print(f"Total Return [%]:      {stats['Total Return [%]']:.2f}%")
    print(f"Benchmark (SPY) [%]:   {spy_return:.2f}%")
    print(f"Max Drawdown [%]:      {stats['Max Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio:          {stats['Sharpe Ratio']:.4f}")
    print(f"Calmar Ratio:          {stats['Calmar Ratio']:.4f}")
    print("="*40)
    
    metrics_dict = stats.to_dict()
    metrics_dict["Benchmark (SPY) [%]"] = spy_return
    save_dict(metrics_dict, project_path.as_posix() + "/artifacts/metrics/backtest_metrics.json")


    fig = pf.cumulative_returns().vbt.plot(
        title="Cumulative Returns (Strategy vs Benchmark)",
        yaxis_title="Return",
        xaxis_title="Date"
    )
    
    fig.write_image(project_path.as_posix() + "/artifacts/plots/pnl.png", engine="kaleido")

if __name__ == "__main__":
    run_backtest()



