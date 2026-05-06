import datetime as dt
import json
from typing import Dict


import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from yaml import safe_load


def load_config(config_path: str) -> Dict:
    """Загружает yaml конфиг в виде python словаря"""
    with open(config_path) as file:
        config = safe_load(file)
    return config


def save_dict(dict_, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict_, f, indent=4, default=str)


def three_barrier(close, ptSl=[1, 1], rolling_n=50, scaling_factor=2.0, horizon=20):
    """
    Векторизованная реализация метода Тройного барьера (Triple Barrier Method).

    Args:
        close (pd.Series): Цены закрытия ОДНОГО тикера.
        ptSl (list): Множители для верхнего [0] и нижнего [1] барьеров.
        rolling_n (int): Окно для расчета волатильности.
        scaling_factor (float): Масштабатор волатильности (ширина коридора).
        horizon (int): Горизонт вертикального барьера (дней удержания сделки).

    Returns:
        pd.Series: Классы (2 - Take Profit, 0 - Stop Loss, 1 - Flat).
    """
    #  Считаем дневную волатильность
    ret = close.pct_change()
    vol = ret.rolling(window=rolling_n).std()
    
    #  Определяем динамические барьеры для каждого дня
    trgt = vol * scaling_factor
    pt = trgt * ptSl[0]   # Верхний барьер
    sl = -trgt * ptSl[1]  # Нижний барьер
    
    #  Собираем матрицу будущих доходностей (Сдвиг векторов)
    # Вместо цикла по времени мы разом смотрим в будущее на h дней
    future_rets = {i: (close.shift(-i) / close) - 1 for i in range(1, horizon + 1)}
    df_future = pd.DataFrame(future_rets)
    
    # 4. Находим пробития барьеров
    # Сравниваем будущие доходности с нашими барьерами
    hit_pt = df_future.ge(pt, axis=0)
    hit_sl = df_future.le(sl, axis=0)
    
    # Находим день, когда барьер был пробит впервые
    pt_idx = hit_pt.idxmax(axis=1)
    sl_idx = hit_sl.idxmax(axis=1)
    
    # Проверяем, было ли пробитие в принципе на горизонте 20 дней
    any_pt = hit_pt.any(axis=1)
    any_sl = hit_sl.any(axis=1)
    
    # 5. Разметка классов (Labeling)
    # По умолчанию всё = 1 (Флэт / дошли до вертикального барьера)
    labels = pd.Series(1, index=close.index, name='target')
    
    # Условие для Класса 2 (Take Profit):
    # Пробили верх, и при этом (не пробили низ ИЛИ пробили верх раньше низа)
    pt_first = any_pt & (~any_sl | (pt_idx < sl_idx))
    
    # Условие для Класса 0 (Stop Loss):
    # Пробили низ, и при этом (не пробили верх ИЛИ пробили низ раньше или одновременно с верхом)
    sl_first = any_sl & (~any_pt | (sl_idx <= pt_idx))
    
    # Применяем маски
    labels[pt_first] = 2
    labels[sl_first] = 0
    
    # Очистка краевых эффектов
    # В начале нет волатильности, в конце мы не можем заглянуть в будущее
    labels[vol.isna()] = np.nan
    labels.iloc[-horizon:] = np.nan
    
    return labels