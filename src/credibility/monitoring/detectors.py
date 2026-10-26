"""Streaming degradation / drift detectors (small, dependency-free, unit-tested)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import stats as sst


class PageHinkley:
    """Page-Hinkley test for an INCREASE in the mean of a stream (here: relative loss differential).

    m_t = sum(x_i - mean_i - delta);  alarm when m_t - min(m) > threshold.
    """

    def __init__(self, delta: float = 0.1, threshold: float = 4.0) -> None:
        self.delta, self.threshold = delta, threshold
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.threshold


class RollingRatioMonitor:
    """Alarm when the candidate's loss over the last ``k`` realised outcomes exceeds the benchmark's by ``theta``.

    Resumes (alarm off) once the candidate is at least as good as the benchmark over the last ``k`` outcomes.
    Both models keep running in shadow, so recovery can be observed while the benchmark is being served.
    """

    def __init__(self, k: int = 12, theta: float = 0.1) -> None:
        self.k, self.theta = k, theta
        self.buf: deque[tuple[float, float]] = deque(maxlen=k)
        self.alarm = False

    def update(self, loss: float, loss_bm: float) -> bool:
        self.buf.append((loss, loss_bm))
        if len(self.buf) < self.k:
            return self.alarm
        a = sum(x for x, _ in self.buf)
        b = sum(y for _, y in self.buf)
        ratio = a / b if b > 0 else 1.0
        if not self.alarm and ratio > 1.0 + self.theta:
            self.alarm = True
        elif self.alarm and ratio <= 1.0:
            self.alarm = False
        return self.alarm


def relative_differential(loss: float, loss_bm: float) -> float:
    """Bounded symmetric relative loss difference in [-1, 1]; positive when the candidate is worse."""
    denom = 0.5 * (loss + loss_bm)
    if denom <= 0:
        return 0.0
    return float(np.clip((loss - loss_bm) / denom, -1.0, 1.0))


@dataclass
class DriftResult:
    statistic: float
    p_value: float
    drift: bool


def ks_data_drift(reference, recent, alpha: float = 0.01) -> DriftResult:
    """Two-sample Kolmogorov-Smirnov test between a reference window and a recent window of returns."""
    reference, recent = np.asarray(reference, float), np.asarray(recent, float)
    if len(reference) < 30 or len(recent) < 20:
        return DriftResult(float("nan"), float("nan"), False)
    res = sst.ks_2samp(reference, recent)
    return DriftResult(float(res.statistic), float(res.pvalue), bool(res.pvalue < alpha))
