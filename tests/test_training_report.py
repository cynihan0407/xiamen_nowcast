"""training_report 单元测试。"""
from __future__ import annotations

import pandas as pd

import json

import numpy as np

from src.utils.training_report import epoch_summary, evaluate_stage_a_gate, to_json_safe


def test_epoch_summary_and_gate_pass():
    raw = pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1, 2, 2],
            "step": [10, 100, 110, 200, 210, 300],
            "train/loss_epoch": [0.5, 0.45, 0.4, 0.35, 0.32, 0.30],
            "val/loss": [float("nan"), 0.42, float("nan"), 0.38, float("nan"), 0.36],
            "val/csi_b13_240K": [float("nan"), 0.5, float("nan"), 0.8, float("nan"), 0.96],
        }
    )
    ep = epoch_summary(raw)
    assert len(ep) == 3
    gate = evaluate_stage_a_gate(ep, csi_threshold=0.95, min_epochs=3)
    assert gate.checks["csi_reaches_threshold"]
    assert gate.best_val_csi >= 0.95


def test_to_json_safe_numpy_types():
    payload = {"ok": np.bool_(True), "x": np.float64(0.5), "checks": {"a": np.bool_(False)}}
    json.dumps(to_json_safe(payload))
