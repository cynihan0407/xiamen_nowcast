"""光流平流模块的单元测试（纯 CPU，小张量）。"""
import torch

from src.models.flow_advect import (
    advect,
    advect_sequence,
    estimate_flow_from_past,
    estimate_flow_pair,
)


def _make_blob(H=64, W=64, cx=32, cy=32, r=6):
    ys = torch.arange(H).view(H, 1).float()
    xs = torch.arange(W).view(1, W).float()
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return torch.exp(-d2 / (2 * r * r))  # [H,W] 高斯团


def test_estimate_flow_recovers_known_shift():
    # ref 在 (cx,cy)，tgt 整体右移 dx、下移 dy → 估计 flow≈(dx,dy)
    dx, dy = 8, 4
    ref = _make_blob(cx=28, cy=28)[None, None]
    tgt = _make_blob(cx=28 + dx, cy=28 + dy)[None, None]
    flow = estimate_flow_pair(ref, tgt, max_disp=4, win=9, scale=4)
    # 在团中心附近取平均，应接近真实位移
    cu = flow[0, 0, 24:36, 24:36].mean().item()
    cv = flow[0, 1, 24:36, 24:36].mean().item()
    assert abs(cu - dx) <= 2.0, f"dx 估计={cu} 期望≈{dx}"
    assert abs(cv - dy) <= 2.0, f"dy 估计={cv} 期望≈{dy}"


def test_advect_moves_pattern():
    # 用已知 flow 平流，团应移动到预期位置
    frame = _make_blob(cx=20, cy=20)[None, None]  # [1,1,64,64]
    flow = torch.zeros(1, 2, 64, 64)
    flow[:, 0] = 3.0  # dx=3 px/step
    flow[:, 1] = 2.0  # dy=2 px/step
    out = advect(frame, flow, steps=4)  # 预期团移动到 (20+12, 20+8)=(32,28)
    # 找 out 的峰值位置
    idx = out[0, 0].flatten().argmax().item()
    py, px = idx // 64, idx % 64
    assert abs(px - 32) <= 2 and abs(py - 28) <= 2, f"峰值=({px},{py}) 期望≈(32,28)"


def test_advect_sequence_shape_and_static():
    last = _make_blob()[None, None].expand(2, 4, 64, 64).contiguous()
    flow = torch.zeros(2, 2, 64, 64)  # 零运动 → 每帧都等于最后一帧
    seq = advect_sequence(last, flow, t_future=12)
    assert seq.shape == (2, 4, 12, 64, 64)
    # 零运动时各时效应≈静态 persistence
    assert torch.allclose(seq[:, :, 0], last, atol=1e-4)
    assert torch.allclose(seq[:, :, -1], last, atol=1e-4)


def test_estimate_flow_from_past_constant_motion():
    # 构造匀速右移序列，估计平均 flow 的 dx 应为正、dy≈0
    frames = []
    for t in range(6):
        frames.append(_make_blob(cx=16 + 3 * t, cy=32))
    past = torch.stack(frames, dim=0)[None]  # [1,6,64,64]
    flow = estimate_flow_from_past(past, max_disp=4, win=9, scale=4)
    cu = flow[0, 0, 28:36, :].mean().item()
    cv = flow[0, 1, 28:36, :].mean().item()
    assert cu > 1.0, f"应估到明显右移 dx>0，得到 {cu}"
    assert abs(cv) < 2.0, f"dy 应接近 0，得到 {cv}"
