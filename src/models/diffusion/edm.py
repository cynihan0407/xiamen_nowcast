"""EDM (Karras et al., NeurIPS 2022) 扩散过程：预条件 + 损失 + Heun ODE 采样。

按论文 Eq. (7) 的预条件参数化网络 ``D_theta(x; sigma, cond)``::

    c_skip(sigma) = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out (sigma) = sigma * sigma_data / sqrt(sigma**2 + sigma_data**2)
    c_in  (sigma) = 1 / sqrt(sigma**2 + sigma_data**2)
    c_noise(sigma)= 0.25 * log(sigma)
    D_theta(x; sigma) = c_skip * x + c_out * F_theta(c_in * x; c_noise, cond)

训练时 ``sigma`` 服从 ``log-normal(P_mean, P_std)``。

本模块**不依赖** Lightning，方便单元测试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class EDMConfig:
    """EDM 训练 / 采样超参数。"""

    sigma_data: float = 0.5
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 80.0
    rho: float = 7.0
    # 采样
    num_steps: int = 18
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1.0


def _as_sigma(sigma: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """把标量/形状 [B] 的 sigma 广播为 [B, 1, 1, ...] 与 ``like`` 同 ndim。"""
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=like.dtype, device=like.device)
    sigma = sigma.to(device=like.device, dtype=like.dtype)
    while sigma.ndim < like.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class EDMDiffusion(nn.Module):
    """EDM 扩散包装：把任意 ``denoiser`` 网络变成可训练 / 可采样的扩散模型。

    ``denoiser`` 期望签名::

        denoiser(x_in: Tensor, c_noise: Tensor, cond: Optional[Tensor]) -> Tensor

    其中 ``c_noise.shape == (B,)``，输出与 ``x_in`` 同形状。
    """

    def __init__(
        self,
        denoiser: nn.Module,
        cfg: Optional[EDMConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        kwargs.pop("_target_", None)
        if cfg is None:
            fields = EDMConfig.__dataclass_fields__
            cfg = EDMConfig(**{k: v for k, v in kwargs.items() if k in fields})
        self.cfg = cfg
        self.denoiser = denoiser

    # ------------------------------------------------------------------ utils
    def _precond(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sd = self.cfg.sigma_data
        c_skip = sd**2 / (sigma**2 + sd**2)
        c_out = sigma * sd / torch.sqrt(sigma**2 + sd**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + sd**2)
        c_noise = 0.25 * torch.log(sigma + 1e-12)
        return c_skip, c_out, c_in, c_noise

    def sample_sigma_training(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """log-normal(P_mean, P_std) 采样训练 sigma。返回 shape [B]。"""
        rnd = torch.randn(batch_size, device=device, dtype=dtype)
        return (rnd * self.cfg.P_std + self.cfg.P_mean).exp()

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM 论文式 (8): lambda(sigma) = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2"""
        sd = self.cfg.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    # ------------------------------------------------------------------ core
    def denoise(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向预条件去噪：返回 ``D_theta(x_noisy; sigma, cond)``，与 ``x_noisy`` 同形状。"""
        sigma_b = _as_sigma(sigma, x_noisy)
        c_skip, c_out, c_in, c_noise = self._precond(sigma_b)
        c_noise_flat = c_noise.reshape(x_noisy.size(0))  # [B]
        f = self.denoiser(c_in * x_noisy, c_noise_flat, cond)
        return c_skip * x_noisy + c_out * f

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """EDM 训练损失：MSE 加权，目标为干净 latent ``x_clean``。

        Args:
            x_clean: ``[B, C, ...]`` 干净 latent
            cond:    任意条件张量（通常为过去 latent，形状由网络决定）
            weights: 像素级 / 通道级权重（可选）

        Returns:
            (loss_scalar, log_dict)
        """
        B = x_clean.size(0)
        sigma = self.sample_sigma_training(B, x_clean.device, x_clean.dtype)
        sigma_b = _as_sigma(sigma, x_clean)
        noise = torch.randn_like(x_clean) * sigma_b
        x_noisy = x_clean + noise
        d_pred = self.denoise(x_noisy, sigma, cond)
        lam = self.loss_weight(sigma_b)
        per_elem = (d_pred - x_clean) ** 2
        if weights is not None:
            per_elem = per_elem * weights
        loss = (lam * per_elem).mean()
        logs = {
            "train/edm_mse": per_elem.mean().detach(),
            "train/sigma_mean": sigma.mean().detach(),
            "train/loss_weight_mean": lam.mean().detach(),
        }
        return loss, logs

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def build_sigma_schedule(self, num_steps: Optional[int] = None, device: torch.device | str = "cpu") -> torch.Tensor:
        """EDM 论文式 (5) 的 sigma 时间表，长度 ``num_steps+1``（末位 0）。"""
        N = int(num_steps or self.cfg.num_steps)
        rho = self.cfg.rho
        s_max = self.cfg.sigma_max
        s_min = self.cfg.sigma_min
        i = torch.arange(N, device=device, dtype=torch.float32)
        sigmas = (
            s_max ** (1.0 / rho)
            + i / max(N - 1, 1) * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))
        ) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)  # [N+1]

    @torch.no_grad()
    def heun_sample(
        self,
        shape: tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        *,
        num_steps: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, torch.Tensor, float], None]] = None,
    ) -> torch.Tensor:
        """Heun 二阶 ODE 采样（EDM Algorithm 1）。

        Args:
            shape: 输出张量形状 ``(B, C, ...)``
            cond:  条件张量
            num_steps: 覆盖 cfg.num_steps

        Returns:
            采样得到的干净 latent，形状 == ``shape``
        """
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        sigmas = self.build_sigma_schedule(num_steps=num_steps, device=device).to(dtype)

        x = torch.randn(shape, device=device, dtype=dtype, generator=generator) * sigmas[0]

        N = sigmas.numel() - 1
        s_churn = self.cfg.S_churn
        s_min = self.cfg.S_min
        s_max = self.cfg.S_max
        s_noise = self.cfg.S_noise

        for i in range(N):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            gamma = 0.0
            if s_churn > 0 and s_min <= float(sigma_cur) <= s_max:
                gamma = min(s_churn / N, 2**0.5 - 1.0)
            sigma_hat = sigma_cur * (1.0 + gamma)
            if gamma > 0:
                eps = torch.randn(shape, device=device, dtype=dtype, generator=generator) * s_noise
                x = x + (sigma_hat**2 - sigma_cur**2).sqrt() * eps

            d_cur = self.denoise(x, sigma_hat.expand(shape[0]), cond)
            d_prime = (x - d_cur) / sigma_hat
            x_next = x + (sigma_next - sigma_hat) * d_prime

            if sigma_next > 0:
                d_next_pred = self.denoise(x_next, sigma_next.expand(shape[0]), cond)
                d_double_prime = (x_next - d_next_pred) / sigma_next
                x_next = x + (sigma_next - sigma_hat) * 0.5 * (d_prime + d_double_prime)

            x = x_next
            if callback is not None:
                callback(i, x, float(sigma_cur))

        return x
