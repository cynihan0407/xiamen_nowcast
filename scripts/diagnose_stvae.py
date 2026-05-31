#!/usr/bin/env python3
"""Stage-A STVAE 诊断：按帧 CSI、未来 12 帧 CSI、B13 重建图、冷云偏差统计。

用法（仓库根目录）::

    export XN_VAL_DIR=...
    python scripts/diagnose_stvae.py \\
        --ckpt outputs/stvae/nohup_manual/checkpoints/last.ckpt \\
        --out reports/diagnose/stvae_nohup_manual

可选对比两次训练（如 finetune vs 原 run）::

    python scripts/diagnose_stvae.py \\
        --ckpt outputs/stvae/finetune_b13w5_from73/checkpoints/best.ckpt \\
        --label finetune \\
        --out reports/diagnose/compare \\
        --compare-ckpt outputs/stvae/nohup_manual/checkpoints/last.ckpt \\
        --compare-label baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h8_dataset import H8Dataset, load_blacklist  # noqa: E402
from src.data.normalizers import B13_INDEX, norm_to_kelvin_np  # noqa: E402
from src.data.transforms import CropTransform  # noqa: E402
from src.metrics.csi import csi_at_threshold_k  # noqa: E402
from src.models.vae.stvae import STVAE, STVAEConfig  # noqa: E402

PAST_LEN = 6
FUTURE_LEN = 12
THRESH_K = 240.0


def load_stvae_weights(stvae: STVAE, ckpt_path: str | Path) -> tuple[list[str], list[str]]:
    """从 Lightning STVAE checkpoint 加载权重（内联实现，不依赖 src.utils.checkpoints）。"""
    p = Path(ckpt_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {p}")
    state = torch.load(p, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    model_state: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("model."):
            model_state[k[len("model.") :]] = v
        elif k.startswith("stvae."):
            model_state[k[len("stvae.") :]] = v
        elif not k.startswith(("diffusion.", "ema")):
            model_state[k] = v
    missing, unexpected = stvae.load_state_dict(model_state, strict=False)
    stvae.eval()
    return list(missing), list(unexpected)


def to_json_safe(obj):  # noqa: ANN001
    """JSON 可序列化（避免 numpy 标量）。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STVAE 重建诊断")
    p.add_argument("--ckpt", required=True, help="STVAE Lightning checkpoint 路径")
    p.add_argument("--label", default="model", help="输出子目录名")
    p.add_argument("--out", default="reports/diagnose/stvae", help="报告根目录")
    p.add_argument("--val-dir", default=None, help="默认 $XN_VAL_DIR")
    p.add_argument("--blacklist", default=None, help="problematic_checkpoints.csv")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=200, help="扫描 val 样本数（用于统计）")
    p.add_argument("--n-plot", type=int, default=6, help="保存目视图的个例数（高/中/低 CSI）")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--thresh", type=float, default=THRESH_K, help="CSI 阈值(K)；深对流看 220/210")
    p.add_argument("--compare-ckpt", default=None, help="可选：第二个 ckpt 对比")
    p.add_argument("--compare-label", default="compare")
    # === STVAE 网络结构（必须与训练时一致；默认 = 杠杆2 nd3） ===
    p.add_argument("--in-channels", type=int, default=4)
    p.add_argument("--latent-channels", type=int, default=12)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--num-down", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=18)
    return p.parse_args()


@torch.no_grad()
def reconstruct(stvae: STVAE, x: torch.Tensor) -> torch.Tensor:
    """与训练 validation 一致：forward 含 reparameterize。"""
    recon, _, _ = stvae(x)
    return recon


def per_frame_csi(pred_k: np.ndarray, true_k: np.ndarray, thresh: float) -> list[float]:
    """pred_k/true_k: [T, H, W]"""
    return [csi_at_threshold_k(pred_k[t], true_k[t], thresh)["CSI"] for t in range(pred_k.shape[0])]


def sample_metrics(recon: torch.Tensor, x: torch.Tensor, thresh: float) -> dict:
    """单样本指标。x/recon: [C,T,H,W]"""
    pk = norm_to_kelvin_np(recon[B13_INDEX : B13_INDEX + 1], "B13")[0]  # [T,H,W]
    tk = norm_to_kelvin_np(x[B13_INDEX : B13_INDEX + 1], "B13")[0]
    csis = per_frame_csi(pk, tk, thresh)
    all18 = float(np.mean(csis))
    fut12 = float(np.mean(csis[PAST_LEN:])) if len(csis) > PAST_LEN else all18
    past6 = float(np.mean(csis[:PAST_LEN]))
    cold = tk <= thresh
    bias_cold = float((pk[cold] - tk[cold]).mean()) if cold.any() else float("nan")
    mae_all = float(np.abs(pk - tk).mean())
    return {
        "csi_all18": all18,
        "csi_past6": past6,
        "csi_future12": fut12,
        "mae_b13_K": mae_all,
        "bias_b13_K_cold": bias_cold,
        "cold_fraction": float(cold.mean()),
        "per_frame_csi": csis,
    }


def run_eval(
    stvae: STVAE,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    thresh: float,
) -> tuple[pd.DataFrame, list[dict]]:
    stvae.eval()
    rows: list[dict] = []
    details: list[dict] = []
    n = 0
    for batch in tqdm(loader, desc="diagnose"):
        x = batch["x"].to(device)
        recon = reconstruct(stvae, x)
        for i in range(x.size(0)):
            if n >= max_samples:
                break
            m = sample_metrics(recon[i].cpu(), x[i].cpu(), thresh)
            row = {k: v for k, v in m.items() if k != "per_frame_csi"}
            row["idx"] = n
            if "timestamp" in batch:
                ts = batch["timestamp"]
                row["timestamp"] = ts[i] if isinstance(ts, (list, tuple)) else str(ts)
            rows.append(row)
            details.append(m)
            n += 1
        if n >= max_samples:
            break
    return pd.DataFrame(rows), details


def plot_case(
    recon: torch.Tensor,
    x: torch.Tensor,
    out_path: Path,
    title: str,
    thresh: float,
) -> None:
    pk = norm_to_kelvin_np(recon[B13_INDEX : B13_INDEX + 1], "B13")[0]
    tk = norm_to_kelvin_np(x[B13_INDEX : B13_INDEX + 1], "B13")[0]
    T = pk.shape[0]
    cols = min(6, T)
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for t in range(rows * cols):
        ax = axes[t // cols, t % cols]
        ax.axis("off")
        if t >= T:
            continue
        # RGB: R=true, G=pred, B=diff emphasis on cold mask
        true_m = tk[t] <= thresh
        pred_m = pk[t] <= thresh
        diff = pk[t] - tk[t]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-15, vmax=15)
        ax.contour(true_m, levels=[0.5], colors="lime", linewidths=0.8)
        ax.contour(pred_m, levels=[0.5], colors="red", linewidths=0.8, linestyles="--")
        csi_t = csi_at_threshold_k(pk[t], tk[t], thresh)["CSI"]
        ax.set_title(f"t={t} CSI={csi_t:.2f}", fontsize=8)
    fig.suptitle(f"{title}\n(green=true≤{thresh}K, red dashed=pred)", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="pred-true (K)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summarize_df(df: pd.DataFrame) -> dict:
    cols = ["csi_all18", "csi_past6", "csi_future12", "mae_b13_K", "bias_b13_K_cold"]
    out = {}
    for c in cols:
        if c in df.columns:
            out[f"{c}_mean"] = float(df[c].mean())
            out[f"{c}_std"] = float(df[c].std())
    return out


def per_frame_aggregate(details: list[dict]) -> pd.DataFrame:
    if not details:
        return pd.DataFrame()
    T = len(details[0]["per_frame_csi"])
    arr = np.array([d["per_frame_csi"] for d in details])
    return pd.DataFrame({
        "frame": np.arange(T),
        "csi_mean": arr.mean(axis=0),
        "csi_std": arr.std(axis=0),
        "phase": ["past" if t < PAST_LEN else "future" for t in range(T)],
    })


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    thresh = float(args.thresh)
    stvae_cfg = STVAEConfig(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_down=args.num_down,
        seq_len=args.seq_len,
    )
    print(
        f"[diagnose] STVAE 结构: in={stvae_cfg.in_channels} latent={stvae_cfg.latent_channels} "
        f"base={stvae_cfg.base_channels} num_down={stvae_cfg.num_down} | CSI@{thresh:.0f}K"
    )

    val_dir = args.val_dir or __import__("os").environ.get(
        "XN_VAL_DIR", "/share/home/sera_hujun/val_data_v7_unbiased_501"
    )
    blacklist = load_blacklist(args.blacklist or __import__("os").environ.get("XN_BLACKLIST", "problematic_checkpoints.csv"))
    crop = CropTransform(args.crop, mode="center")

    ds = H8Dataset(val_dir, mode="raw", crop=crop, aug=None, blacklist=blacklist)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def _load(ckpt: str) -> STVAE:
        m = STVAE(stvae_cfg)
        missing, unexpected = load_stvae_weights(m, ckpt)
        if missing or unexpected:
            print(f"[diagnose] load {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
            if unexpected:
                print("  >> 有 unexpected 权重，通常说明 --base/latent/num-down 与 ckpt 不一致！")
        return m.to(device)

    out_root = Path(args.out)
    label_dir = out_root / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    stvae = _load(args.ckpt)
    df, details = run_eval(stvae, loader, device, args.max_samples, thresh)
    df.to_csv(label_dir / "per_sample.csv", index=False)
    pf = per_frame_aggregate(details)
    pf.to_csv(label_dir / "per_frame_csi.csv", index=False)

    # 按帧曲线图
    if len(pf):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pf["frame"], pf["csi_mean"], "o-", label="mean CSI")
        ax.fill_between(pf["frame"], pf["csi_mean"] - pf["csi_std"], pf["csi_mean"] + pf["csi_std"], alpha=0.2)
        ax.axvline(PAST_LEN - 0.5, color="gray", linestyle="--", label="past|future")
        ax.set_xlabel("frame index (0..17)")
        ax.set_ylabel(f"CSI @ {thresh}K")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(label_dir / "per_frame_csi.png", dpi=120)
        plt.close(fig)

    # 高/中/低 CSI 个例图
    order = df.sort_values("csi_all18")
    picks: list[tuple[str, int]] = []
    if len(order):
        picks.append(("worst", int(order.iloc[0]["idx"])))
        picks.append(("median", int(order.iloc[len(order) // 2]["idx"])))
        picks.append(("best", int(order.iloc[-1]["idx"])))
        # 额外挑 3 个均匀分位
        for q, name in [(0.2, "p20"), (0.8, "p80")]:
            picks.append((name, int(order.iloc[int(len(order) * q)]["idx"])))
    picks = picks[: args.n_plot]

    # 重新顺序取样本画图（按 idx）
    idx_to_tensor: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        recon = reconstruct(stvae, x)
        for i in range(x.size(0)):
            if n in [p[1] for p in picks]:
                idx_to_tensor[n] = (recon[i].cpu(), x[i].cpu())
            n += 1
            if n >= args.max_samples:
                break
        if n >= args.max_samples:
            break

    for name, idx in picks:
        if idx not in idx_to_tensor:
            continue
        r, x0 = idx_to_tensor[idx]
        csi = df.loc[df["idx"] == idx, "csi_all18"].iloc[0]
        plot_case(r, x0, label_dir / f"case_{name}_idx{idx}_csi{csi:.3f}.png", f"{args.label} {name} idx={idx}", thresh)

    summary = summarize_df(df)
    summary["ckpt"] = str(Path(args.ckpt).resolve())
    summary["n_samples"] = len(df)
    (label_dir / "summary.json").write_text(
        json.dumps(to_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[{args.label}] n={len(df)}")
    for k, v in summary.items():
        if k.endswith("_mean"):
            print(f"  {k}: {v:.4f}")

    if args.compare_ckpt:
        stvae2 = _load(args.compare_ckpt)
        cmp_dir = out_root / args.compare_label
        cmp_dir.mkdir(parents=True, exist_ok=True)
        df2, det2 = run_eval(stvae2, loader, device, args.max_samples, thresh)
        df2.to_csv(cmp_dir / "per_sample.csv", index=False)
        s2 = summarize_df(df2)
        (cmp_dir / "summary.json").write_text(json.dumps(to_json_safe(s2), indent=2), encoding="utf-8")
        merged = df[["idx", "csi_all18", "csi_future12", "mae_b13_K"]].merge(
            df2[["idx", "csi_all18", "csi_future12", "mae_b13_K"]],
            on="idx",
            suffixes=(f"_{args.label}", f"_{args.compare_label}"),
        )
        merged["delta_csi"] = merged[f"csi_all18_{args.label}"] - merged[f"csi_all18_{args.compare_label}"]
        merged.to_csv(out_root / "compare_per_sample.csv", index=False)
        print(f"\n[compare] mean delta CSI ({args.label}-{args.compare_label}): {merged['delta_csi'].mean():.4f}")

    print(f"\n[done] 报告目录: {label_dir.resolve()}")


if __name__ == "__main__":
    main()
