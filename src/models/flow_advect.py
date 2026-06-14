"""纯 PyTorch 光流估计 + 平流（Lagrangian persistence 基线）。

用途：把"重复最后一帧"的静态 Persistence 升级为"沿运动外推"的拉格朗日
Persistence。先用块匹配从过去帧估计稠密运动场（像素/步），再用 ``grid_sample``
把最后一帧沿运动平流到每个未来时效，作为残差预报的运动基线。

不依赖 OpenCV，全部张量算子，可在 GPU 上批量运行、可单元测试。

约定：
* 帧张量为归一化域 ``[-1, 1]``，布局 ``[B, C, H, W]`` 或 ``[B, H, W]``。
* 运动场 ``flow`` 形状 ``[B, 2, H, W]``，通道 0 = dx（列方向，向右为正），
  通道 1 = dy（行方向，向下为正），单位为"像素 / 每步（10 min）"。
* 运动定义：``I_ref(x) ≈ I_tgt(x + flow)``，即特征从 ref 帧到 tgt 帧位移 ``flow``。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _box_filter(x: torch.Tensor, win: int) -> torch.Tensor:
    """对 ``[B,1,H,W]`` 做 stride=1 的均值聚合（局部代价窗）。"""
    pad = win // 2
    return F.avg_pool2d(x, kernel_size=win, stride=1, padding=pad)


@torch.no_grad()
def estimate_flow_pair(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    *,
    max_disp: int = 6,
    win: int = 9,
    scale: int = 4,
) -> torch.Tensor:
    """块匹配估计两帧间稠密运动场 ``ref -> tgt``。

    Args:
        ref, tgt: ``[B, 1, H, W]`` 连续两帧（归一化域）。
        max_disp: 在降采样分辨率下的最大搜索位移（候选 ``(2*max_disp+1)**2`` 个）。
        win:      局部 SSD 聚合窗口（奇数）。
        scale:    降采样倍率（加速 + 抗噪）；运动场最终上采样回全分辨率。

    Returns:
        ``[B, 2, H, W]`` 运动场（像素/步，全分辨率坐标系）。
    """
    if ref.ndim != 4 or ref.size(1) != 1:
        raise ValueError(f"ref 需为 [B,1,H,W]，得到 {tuple(ref.shape)}")
    if win % 2 == 0:
        raise ValueError("win 必须为奇数")
    B, _, H, W = ref.shape

    rd = F.avg_pool2d(ref, scale) if scale > 1 else ref
    td = F.avg_pool2d(tgt, scale) if scale > 1 else tgt
    Hd, Wd = rd.shape[-2:]

    best_cost = rd.new_full((B, 1, Hd, Wd), float("inf"))
    best_dx = rd.new_zeros((B, 1, Hd, Wd))
    best_dy = rd.new_zeros((B, 1, Hd, Wd))

    for dy in range(-max_disp, max_disp + 1):
        for dx in range(-max_disp, max_disp + 1):
            # tgt 对齐到 ref：取 tgt(x + (dx,dy)) → roll 负位移
            shifted = torch.roll(td, shifts=(-dy, -dx), dims=(2, 3))
            cost = _box_filter((rd - shifted) ** 2, win)
            better = cost < best_cost
            best_cost = torch.where(better, cost, best_cost)
            best_dx = torch.where(better, rd.new_full(best_dx.shape, float(dx)), best_dx)
            best_dy = torch.where(better, rd.new_full(best_dy.shape, float(dy)), best_dy)

    flow_d = torch.cat([best_dx, best_dy], dim=1)  # [B,2,Hd,Wd]，降采样像素/步
    if scale > 1:
        flow = F.interpolate(flow_d, size=(H, W), mode="bilinear", align_corners=False) * scale
    else:
        flow = flow_d
    return flow


@torch.no_grad()
def estimate_flow_from_past(
    past_b13: torch.Tensor,
    *,
    max_disp: int = 6,
    win: int = 9,
    scale: int = 4,
    smooth: int = 5,
) -> torch.Tensor:
    """从过去若干帧 B13 估计平均运动场（像素/步）。

    Args:
        past_b13: ``[B, T_past, H, W]`` 过去帧（归一化域，单通道 B13）。
        smooth:   对最终运动场做一次均值平滑的窗口（奇数，<=1 关闭）。

    Returns:
        ``[B, 2, H, W]`` 平均运动场（像素/步）。
    """
    if past_b13.ndim != 4:
        raise ValueError(f"past_b13 需为 [B,T,H,W]，得到 {tuple(past_b13.shape)}")
    B, T, H, W = past_b13.shape
    if T < 2:
        return past_b13.new_zeros((B, 2, H, W))

    flows = []
    for t in range(T - 1):
        ref = past_b13[:, t : t + 1]
        tgt = past_b13[:, t + 1 : t + 2]
        flows.append(estimate_flow_pair(ref, tgt, max_disp=max_disp, win=win, scale=scale))
    flow = torch.stack(flows, dim=0).mean(dim=0)  # [B,2,H,W]

    if smooth and smooth > 1:
        pad = smooth // 2
        flow = F.avg_pool2d(flow, kernel_size=smooth, stride=1, padding=pad)
    return flow


def _normalized_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """生成 ``grid_sample`` 用的基准归一化坐标网格 ``[B, H, W, 2]``（x,y∈[-1,1]）。"""
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)  # [H,W,2]
    return grid.unsqueeze(0).expand(B, H, W, 2)


@torch.no_grad()
def advect(frame: torch.Tensor, flow: torch.Tensor, steps: int) -> torch.Tensor:
    """把 ``frame`` 沿 ``flow`` 平流 ``steps`` 步。

    未来帧位置 x 的取值来自 ``frame(x - steps*flow)``（特征沿运动前移）。

    Args:
        frame: ``[B, C, H, W]``。
        flow:  ``[B, 2, H, W]``（像素/步，dx,dy）。
        steps: 外推步数（>=1）。

    Returns:
        平流后的帧 ``[B, C, H, W]``。
    """
    B, C, H, W = frame.shape
    base = _normalized_grid(B, H, W, frame.device, frame.dtype)  # [B,H,W,2]
    # flow 像素 → 归一化位移（grid 坐标范围 2 对应 W-1/H-1 像素）
    dx = flow[:, 0] * steps * (2.0 / max(W - 1, 1))  # [B,H,W]
    dy = flow[:, 1] * steps * (2.0 / max(H - 1, 1))
    samp = torch.stack([base[..., 0] - dx, base[..., 1] - dy], dim=-1)  # [B,H,W,2]
    return F.grid_sample(frame, samp, mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def advect_sequence(last_frame: torch.Tensor, flow: torch.Tensor, t_future: int) -> torch.Tensor:
    """把最后一帧平流到未来 ``t_future`` 个时效，拼成序列。

    Args:
        last_frame: ``[B, C, H, W]`` 最后一个观测帧。
        flow:       ``[B, 2, H, W]`` 运动场（像素/步）。
        t_future:   未来帧数。

    Returns:
        ``[B, C, t_future, H, W]`` 平流基线序列。
    """
    frames = [advect(last_frame, flow, k + 1) for k in range(t_future)]
    return torch.stack(frames, dim=2)
