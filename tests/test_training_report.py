"""training_report 单元测试。"""
from __future__ import annotations

import pandas as pd

import json
from pathlib import Path

import numpy as np

from src.utils.training_report import (
    epoch_summary,
    evaluate_diffusion_summary,
    evaluate_stage_a_gate,
    rank_diffusion_checkpoints,
    to_json_safe,
)


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


def test_evaluate_diffusion_summary_basic():
    raw = pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1, 2, 2, 3, 3],
            "step": [10, 100, 110, 200, 210, 300, 310, 400],
            "train/loss_epoch": [1.2, 1.0, 0.9, 0.85, 0.8, 0.78, 0.77, 0.76],
            "val/edm_loss": [float("nan"), 0.95, float("nan"), 0.80, float("nan"), 0.70, float("nan"), 0.65],
            "val/csi_b13_240K": [float("nan"), 0.02, float("nan"), 0.04, float("nan"), 0.05, float("nan"), 0.06],
        }
    )
    ep = epoch_summary(raw)
    s = evaluate_diffusion_summary(ep)
    assert s.n_epochs == 4
    assert s.best_loss_epoch == 3
    assert abs(s.best_loss_val - 0.65) < 1e-6
    assert s.best_csi_epoch == 3
    assert s.val_loss_decreased
    # 仍偏低 → notes 中应包含 evaluate_nowcast 提示
    assert any("evaluate_nowcast" in n for n in s.notes)
    json.dumps(s.to_dict())


def test_rank_diffusion_checkpoints(tmp_path: Path):
    (tmp_path / "edm-epoch=017-val_edm_loss=0.6172.ckpt").write_text("x")
    (tmp_path / "edm-epoch=120-val_edm_loss=0.5800.ckpt").write_text("y")
    (tmp_path / "last.ckpt").write_text("z")
    rows = rank_diffusion_checkpoints(tmp_path)
    assert [r["epoch"] for r in rows] == [120, 17]
    assert rows[0]["val_edm_loss"] < rows[1]["val_edm_loss"]
