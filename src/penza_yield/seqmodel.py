"""Многозадачные модели сезона для внутрисезонного прогноза урожайности: GRU (E02, E03) и 1D-CNN (E04).

Вход за год t:
  * помесячные ряды 16 переменных за месяцы 1..cutoff года t (аномалии относительно окна обучения)
    и канал наличия NDVI; месяцы после отсечки обнуляются и в модель не попадают;
  * статические признаки: log(lag1/ma3), log(lag2/ma3) пяти культур и нормированный год.
Выход: r_t = log(y_t / ma3_t) для пяти культур сразу; прогноз урожайности ŷ_t = ma3_t · exp(r̂_t),
где ma3_t — среднее урожайности трёх предыдущих лет. Модель предсказывает отклонение от недавнего уровня.

Все статистики нормализации считаются только по годам окна обучения. Ансамбль — среднее r̂ по seed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from penza_yield.data import TARGETS, MonthlyData

NDVI = "ndvi"


@dataclass(frozen=True)
class SeqDataset:
    years: np.ndarray          # [n]
    seq: np.ndarray            # [n, 12, v] значения, NaN — пропуск
    seq_mask: np.ndarray       # [n, 12, v] bool
    lag_ratios: np.ndarray     # [n, 2·k] log(lag1/ma3), log(lag2/ma3); NaN в первые годы
    r: np.ndarray              # [n, k] log(y/ma3)
    anchor: np.ndarray         # [n, k] ma3
    y: np.ndarray              # [n, k] урожайность
    variables: list[str]

    def trainable(self) -> np.ndarray:
        """Годы, где определены цель и лаги (с 1993 г.)."""
        return ~(np.isnan(self.r).any(1) | np.isnan(self.lag_ratios).any(1))


def build_dataset(annual: pd.DataFrame, monthly: MonthlyData) -> SeqDataset:
    if list(annual.index) != list(monthly.years):
        raise ValueError("Годы годовой таблицы и помесячных рядов не совпадают")
    y = annual[TARGETS].to_numpy(float)
    yd = pd.DataFrame(y, index=annual.index)
    ma3 = yd.shift(1).rolling(3).mean().to_numpy()
    lag1, lag2 = yd.shift(1).to_numpy(), yd.shift(2).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lag_ratios = np.concatenate([np.log(lag1 / ma3), np.log(lag2 / ma3)], axis=1)
        r = np.log(y / ma3)
    return SeqDataset(np.asarray(annual.index), monthly.values.astype(np.float64), monthly.mask.copy(),
                      lag_ratios, r, ma3, y, list(monthly.variables))


@dataclass(frozen=True)
class GRUConfig:
    hidden: int = 16
    weight_decay: float = 1e-2
    epochs: int = 300
    dropout: float = 0.2
    lr: float = 1e-2
    noise: float = 0.1        # гауссов шум на входной последовательности при обучении (аугментация)
    arch: str = "gru"         # "gru" | "cnn"
    variables: tuple[str, ...] | None = None  # подмножество помесячных переменных; None — все (E02)

    def key(self) -> str:
        k = f"h={self.hidden}|wd={self.weight_decay}|ep={self.epochs}"
        if self.arch != "gru":
            k += f"|arch={self.arch}"
        if self.variables is not None:
            k += f"|vars={len(self.variables)}"
        return k


# Компактный агрономический вход E03/E04 (D10): тепло, влага (осадки и почва), состояние посевов
COMPACT_VARIABLES = ("mean_temp", "rf_sum", "swc_7_28cm", "ndvi")


class SeasonCNN(nn.Module):
    """Свёртка по месяцам (окно 3) → ReLU → среднее и последний месяц → голова со статическими признаками."""

    def __init__(self, n_in: int, n_static: int, n_out: int, cfg: GRUConfig):
        super().__init__()
        self.conv = nn.Conv1d(n_in, cfg.hidden, kernel_size=3, padding=1)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(2 * cfg.hidden + n_static, n_out)

    def forward(self, seq, static):
        h = torch.relu(self.conv(seq.transpose(1, 2)))          # [batch, hidden, months]
        pooled = torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1)
        return self.head(self.drop(torch.cat([pooled, static], dim=1)))


class SeasonGRU(nn.Module):
    def __init__(self, n_in: int, n_static: int, n_out: int, cfg: GRUConfig):
        super().__init__()
        self.gru = nn.GRU(n_in, cfg.hidden, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden + n_static, n_out)

    def forward(self, seq, static):
        _, h = self.gru(seq)
        return self.head(self.drop(torch.cat([h[-1], static], dim=1)))


def _prepare(ds: SeqDataset, train_idx: np.ndarray, idx: np.ndarray, cutoff: int,
             variables: tuple[str, ...] | None = None):
    """Тензоры для строк idx; статистики нормализации — только по train_idx. Месяцы > cutoff отрезаются."""
    names = list(variables) if variables is not None else ds.variables
    cols = [ds.variables.index(n) for n in names]
    v = ds.seq[:, :cutoff, :][:, :, cols]
    m = ds.seq_mask[:, :cutoff, :][:, :, cols]
    tr = np.where(m[train_idx], v[train_idx], np.nan)
    with warnings.catch_warnings():  # месяцы без NDVI в окне обучения дают пустые срезы
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(tr, axis=0)                     # [cutoff, v] — климатическая норма месяца
        sd = np.nanstd(tr, axis=0)
    mu, sd = np.nan_to_num(mu), np.where(np.isnan(sd) | (sd < 1e-8), 1.0, sd)
    z = np.where(m[idx], (v[idx] - mu) / sd, 0.0)       # аномалии; пропуск → 0 (норма)
    if NDVI in names:  # канал наличия NDVI: модель отличает «нет данных» от «норма»
        z = np.concatenate([z, m[idx][:, :, [names.index(NDVI)]].astype(float)], axis=2)
    seq = z

    s_tr = ds.lag_ratios[train_idx]
    s_mu, s_sd = s_tr.mean(0), s_tr.std(0) + 1e-8
    t_tr = ds.years[train_idx].astype(float)
    t = (ds.years[idx] - t_tr.mean()) / (t_tr.std() + 1e-8)
    static = np.concatenate([(ds.lag_ratios[idx] - s_mu) / s_sd, t[:, None]], axis=1)
    return (torch.tensor(seq, dtype=torch.float32), torch.tensor(np.nan_to_num(static), dtype=torch.float32))


def fit_predict(ds: SeqDataset, train_idx, test_idx, cutoff: int, cfg: GRUConfig, seeds) -> np.ndarray:
    """Обучает ансамбль на train_idx, возвращает прогноз урожайности [len(test_idx), k] в т/га."""
    train_idx = np.asarray(train_idx)[ds.trainable()[train_idx]]
    test_idx = np.asarray(test_idx)
    if ds.years[train_idx].max() >= ds.years[test_idx].min():
        raise ValueError("Утечка: в обучении есть годы не раньше прогнозируемого")
    seq_tr, st_tr = _prepare(ds, train_idx, train_idx, cutoff, cfg.variables)
    seq_te, st_te = _prepare(ds, train_idx, test_idx, cutoff, cfg.variables)
    r_mu, r_sd = ds.r[train_idx].mean(0), ds.r[train_idx].std(0) + 1e-8
    target = torch.tensor((ds.r[train_idx] - r_mu) / r_sd, dtype=torch.float32)

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        arch = {"gru": SeasonGRU, "cnn": SeasonCNN}[cfg.arch]
        model = arch(seq_tr.shape[2], st_tr.shape[1], target.shape[1], cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(cfg.epochs):
            noisy = seq_tr + cfg.noise * torch.randn(seq_tr.shape, generator=gen)
            loss = nn.functional.mse_loss(model(noisy, st_tr), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds.append(model(seq_te, st_te).numpy())
    r_hat = np.mean(preds, axis=0) * r_sd + r_mu
    return ds.anchor[test_idx] * np.exp(r_hat)
