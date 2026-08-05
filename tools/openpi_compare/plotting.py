"""
Metrics computation and plotting utilities for OpenPI vs dataset comparison.

Implements:
- Horizon-0 comparison: pred[t, 0] vs gt[t]
- Diagonal future-offset comparison: pred[t, k] vs gt[t+k]
- Ensemble absolute-time comparison with exponential/mean/median aggregation
- Per-dimension MAE, RMSE, Pearson correlation
- Comprehensive plotting: overlays, curves, heatmaps, summary
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def get_action_labels(n_dims: int, custom_labels: list[str] | None = None) -> list[str]:
    """Get action dimension labels."""
    if custom_labels is not None:
        labels = [str(label) for label in custom_labels[:n_dims]]
        labels.extend(f"dim_{i}" for i in range(len(labels), n_dims))
        return labels
    return [f"dim_{i}" for i in range(n_dims)]


# --------------------------------------------------------------------------- #
# Ensemble aggregation
# --------------------------------------------------------------------------- #


def aggregate_ensemble_exponential(
    pred_chunks: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Aggregate predictions using exponential weighting.

    For each absolute time tau, collect all predictions covering that time:
        pred_chunk[t, tau - t] for all t where 0 <= tau - t < horizon

    Weight: w_k = exp(-alpha * k) where k = tau - t

    Args:
        pred_chunks: Shape [T, H, D] - T timesteps, H action horizon, D action dim
        alpha: Decay parameter. Higher = more weight on recent predictions.

    Returns:
        pred_ensemble: Shape [T + H - 1, D]
    """
    T, H, D = pred_chunks.shape
    total_times = T + H - 1
    pred_ensemble = np.zeros((total_times, D))
    weight_sums = np.zeros(total_times)

    for t in range(T):
        for k in range(H):
            tau = t + k
            if tau >= total_times:
                break
            w = np.exp(-alpha * k)
            pred_ensemble[tau] += w * pred_chunks[t, k]
            weight_sums[tau] += w

    # Normalize
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
    pred_ensemble /= weight_sums[:, np.newaxis]

    return pred_ensemble


def aggregate_ensemble_mean(
    pred_chunks: np.ndarray,
) -> np.ndarray:
    """Aggregate predictions using mean (unweighted).

    Args:
        pred_chunks: Shape [T, H, D]

    Returns:
        pred_ensemble: Shape [T + H - 1, D]
    """
    T, H, D = pred_chunks.shape
    total_times = T + H - 1
    pred_ensemble = np.zeros((total_times, D))
    count = np.zeros(total_times)

    for t in range(T):
        for k in range(H):
            tau = t + k
            if tau >= total_times:
                break
            pred_ensemble[tau] += pred_chunks[t, k]
            count[tau] += 1

    count = np.where(count > 0, count, 1.0)
    pred_ensemble /= count[:, np.newaxis]

    return pred_ensemble


def aggregate_ensemble_median(
    pred_chunks: np.ndarray,
) -> np.ndarray:
    """Aggregate predictions using median.

    Args:
        pred_chunks: Shape [T, H, D]

    Returns:
        pred_ensemble: Shape [T + H - 1, D]
    """
    T, H, D = pred_chunks.shape
    total_times = T + H - 1
    pred_ensemble = np.zeros((total_times, D))

    for tau in range(total_times):
        values = []
        for t in range(T):
            k = tau - t
            if 0 <= k < H:
                values.append(pred_chunks[t, k])
        if values:
            pred_ensemble[tau] = np.median(values, axis=0)

    return pred_ensemble


def aggregate_ensemble(
    pred_chunks: np.ndarray,
    method: Literal["exp", "mean", "median"] = "exp",
    alpha: float = 0.1,
) -> np.ndarray:
    """Aggregate predictions into ensemble predictions.

    Args:
        pred_chunks: Shape [T, H, D]
        method: Aggregation method
        alpha: Decay parameter (only for "exp" method)

    Returns:
        pred_ensemble: Shape [T + H - 1, D]
    """
    if method == "exp":
        return aggregate_ensemble_exponential(pred_chunks, alpha)
    elif method == "mean":
        return aggregate_ensemble_mean(pred_chunks)
    elif method == "median":
        return aggregate_ensemble_median(pred_chunks)
    else:
        raise ValueError(f"Unknown ensemble method: {method!r}")


# --------------------------------------------------------------------------- #
# Metrics computation
# --------------------------------------------------------------------------- #


def compute_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute per-dimension MAE.

    Args:
        pred: Shape [N, D]
        gt: Shape [N, D]
        mask: Shape [N,] - True where comparison is valid

    Returns:
        MAE per dimension: Shape [D,]
    """
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    return np.mean(np.abs(pred - gt), axis=0)


def compute_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute per-dimension RMSE.

    Args:
        pred: Shape [N, D]
        gt: Shape [N, D]
        mask: Shape [N,] - True where comparison is valid

    Returns:
        RMSE per dimension: Shape [D,]
    """
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    return np.sqrt(np.mean((pred - gt) ** 2, axis=0))


def compute_pearson(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Compute per-dimension Pearson correlation.

    Args:
        pred: Shape [N, D]
        gt: Shape [N, D]
        mask: Shape [N,] - True where comparison is valid

    Returns:
        Pearson r per dimension: Shape [D,] (NaN where variance is 0)
    """
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    if len(pred) < 2:
        return np.full(pred.shape[-1], np.nan)

    r = np.zeros(pred.shape[-1])
    for d in range(pred.shape[-1]):
        p, g = pred[:, d], gt[:, d]
        if np.std(p) < 1e-8 or np.std(g) < 1e-8:
            r[d] = np.nan
        else:
            r[d] = np.corrcoef(p, g)[0, 1]
    return r


def compute_overall_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute overall MAE (all dimensions)."""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    return float(np.mean(np.abs(pred - gt)))


def compute_overall_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute overall RMSE (all dimensions)."""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_offset_metrics(
    pred_chunks: np.ndarray,
    gt_actions: np.ndarray,
    horizon: int,
) -> dict[int, dict[str, float]]:
    """Compute metrics at each horizon offset.

    Args:
        pred_chunks: Shape [T, H, D]
        gt_actions: Shape [T+H-1, D] or [T, D]
        horizon: Action horizon H

    Returns:
        Dict mapping offset k to dict with 'mae', 'rmse' per offset
    """
    T = pred_chunks.shape[0]
    D = pred_chunks.shape[2]

    results: dict[int, dict[str, float]] = {}

    for k in range(horizon):
        # pred_chunk[t, k] vs gt[t + k]
        pred_k_list = []
        gt_k_list = []

        for t in range(T):
            tau = t + k
            if tau < len(gt_actions):
                pred_k_list.append(pred_chunks[t, k])
                gt_k_list.append(gt_actions[tau])

        if pred_k_list:
            pred_k = np.array(pred_k_list)
            gt_k = np.array(gt_k_list)
            results[k] = {
                "mae": float(compute_overall_mae(pred_k, gt_k)),
                "rmse": float(compute_overall_rmse(pred_k, gt_k)),
            }

    return results


# --------------------------------------------------------------------------- #
# Full comparison results
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ComparisonResult:
    """Container for comparison results."""

    frame_indices: np.ndarray
    gt_actions: np.ndarray  # [T, D]
    pred_chunks: np.ndarray  # [T, H, D]
    pred_h0: np.ndarray  # [T, D] - first step of each chunk
    pred_ensemble: np.ndarray  # [T+H-1, D] - ensemble at each absolute time
    valid_mask: np.ndarray  # [T+H-1,] - which times have valid comparison

    n_frames: int
    horizon: int
    action_dim: int
    ensemble_method: str
    alpha: float

    # Metrics
    horizon0_mae: np.ndarray
    horizon0_rmse: np.ndarray
    horizon0_pearson: np.ndarray
    horizon0_overall_mae: float
    horizon0_overall_rmse: float

    ensemble_mae: np.ndarray
    ensemble_rmse: np.ndarray
    ensemble_pearson: np.ndarray
    ensemble_overall_mae: float
    ensemble_overall_rmse: float

    offset_metrics: dict[int, dict[str, float]]



def run_comparison(
    gt_actions: np.ndarray,
    pred_chunks: np.ndarray,
    frame_indices: np.ndarray,
    ensemble_method: Literal["exp", "mean", "median"] = "exp",
    alpha: float = 0.1,
) -> ComparisonResult:
    """Run full comparison between ground truth and predicted actions.

    Args:
        gt_actions: Ground truth actions, shape [T, D]
        pred_chunks: Predicted action chunks, shape [T, H, D]
        frame_indices: Frame indices, shape [T,]
        ensemble_method: Method for ensemble aggregation
        alpha: Decay parameter for exponential ensemble

    Returns:
        ComparisonResult with all metrics and processed predictions
    """
    T, H, D = pred_chunks.shape
    gt_T = gt_actions.shape[0]

    assert T <= gt_T, (
        f"Cannot compare: got {T} predictions but only {gt_T} ground truth frames. "
        f"Each prediction needs a corresponding GT frame."
    )

    # Pred H0 (first action in each chunk)
    pred_h0 = pred_chunks[:, 0, :]  # [T, D]

    # Ensemble predictions
    pred_ensemble = aggregate_ensemble(pred_chunks, method=ensemble_method, alpha=alpha)

    # Valid mask for ensemble comparison
    total_ensemble_times = T + H - 1
    valid_mask = np.zeros(total_ensemble_times, dtype=bool)
    for tau in range(total_ensemble_times):
        if tau < gt_T:
            valid_mask[tau] = True

    # Horizon-0 comparison
    horizon0_mae = compute_mae(pred_h0, gt_actions)
    horizon0_rmse = compute_rmse(pred_h0, gt_actions)
    horizon0_pearson = compute_pearson(pred_h0, gt_actions)
    horizon0_overall_mae = compute_overall_mae(pred_h0, gt_actions)
    horizon0_overall_rmse = compute_overall_rmse(pred_h0, gt_actions)

    # Ensemble comparison - only compare first T timesteps where we have GT
    valid_gt = gt_actions[:T]  # [T, D]
    valid_ensemble = pred_ensemble[:T]  # [T, D]
    ensemble_mae = compute_mae(valid_ensemble, valid_gt)
    ensemble_rmse = compute_rmse(valid_ensemble, valid_gt)
    ensemble_pearson = compute_pearson(valid_ensemble, valid_gt)
    ensemble_overall_mae = compute_overall_mae(valid_ensemble, valid_gt)
    ensemble_overall_rmse = compute_overall_rmse(valid_ensemble, valid_gt)

    # Offset metrics
    offset_metrics = compute_offset_metrics(pred_chunks, gt_actions, H)

    return ComparisonResult(
        frame_indices=frame_indices,
        gt_actions=gt_actions,
        pred_chunks=pred_chunks,
        pred_h0=pred_h0,
        pred_ensemble=pred_ensemble,
        valid_mask=valid_mask,
        n_frames=T,
        horizon=H,
        action_dim=D,
        ensemble_method=ensemble_method,
        alpha=alpha,
        horizon0_mae=horizon0_mae,
        horizon0_rmse=horizon0_rmse,
        horizon0_pearson=horizon0_pearson,
        horizon0_overall_mae=horizon0_overall_mae,
        horizon0_overall_rmse=horizon0_overall_rmse,
        ensemble_mae=ensemble_mae,
        ensemble_rmse=ensemble_rmse,
        ensemble_pearson=ensemble_pearson,
        ensemble_overall_mae=ensemble_overall_mae,
        ensemble_overall_rmse=ensemble_overall_rmse,
        offset_metrics=offset_metrics,
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_action_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    labels: list[str],
    filepath: Path,
    frame_indices: np.ndarray | None = None,
    time_unit: str = "frame",
) -> None:
    """Plot ground truth vs predicted actions as overlaid line plots.

    Args:
        gt: Ground truth, shape [N, D]
        pred: Predictions, shape [N, D]
        title: Plot title
        labels: Dimension labels, length D
        filepath: Where to save the figure
        frame_indices: Optional x-axis values (defaults to range(N))
        time_unit: Label for x-axis
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not available, skipping plot: %s", filepath)
        return

    D = gt.shape[1]
    n_cols = min(3, D)
    n_rows = (D + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    x = frame_indices if frame_indices is not None else np.arange(len(gt))

    for d in range(D):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]
        ax.plot(x, gt[:, d], label="GT", alpha=0.8, linewidth=1.5, color="#2196F3")
        ax.plot(x, pred[:, d], label="Pred", alpha=0.8, linewidth=1.5, color="#FF5722", linestyle="--")
        ax.set_title(labels[d] if d < len(labels) else f"Dim {d}")
        ax.set_xlabel(time_unit)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for d in range(D, n_rows * n_cols):
        row, col = d // n_cols, d % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved overlay plot: %s", filepath)


def plot_offset_curves(
    offset_metrics: dict[int, dict[str, float]],
    horizon: int,
    title: str,
    filepath: Path,
) -> None:
    """Plot MAE/RMSE curves as a function of horizon offset.

    Args:
        offset_metrics: Dict mapping offset k to {mae, rmse}
        horizon: Maximum horizon
        title: Plot title
        filepath: Where to save
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot: %s", filepath)
        return

    offsets = sorted(offset_metrics.keys())
    mae_values = [offset_metrics[k]["mae"] for k in offsets]
    rmse_values = [offset_metrics[k]["rmse"] for k in offsets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax1.plot(offsets, mae_values, marker="o", linewidth=2, color="#2196F3")
    ax1.set_xlabel("Horizon Offset k")
    ax1.set_ylabel("MAE")
    ax1.set_title("MAE vs Horizon Offset")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, horizon - 0.5)

    ax2.plot(offsets, rmse_values, marker="s", linewidth=2, color="#FF5722")
    ax2.set_xlabel("Horizon Offset k")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE vs Horizon Offset")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, horizon - 0.5)

    plt.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved offset curves: %s", filepath)


def plot_error_heatmap(
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    labels: list[str],
    filepath: Path,
    frame_indices: np.ndarray | None = None,
) -> None:
    """Plot error heatmap across time and dimensions.

    Args:
        gt: Ground truth [N, D]
        pred: Predictions [N, D]
        title: Plot title
        labels: Dimension labels
        filepath: Where to save
        frame_indices: X-axis labels
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot: %s", filepath)
        return

    errors = np.abs(pred - gt)  # [N, D]
    N, D = errors.shape

    fig, ax = plt.subplots(figsize=(max(8, D * 1.5), max(4, N * 0.05)))
    im = ax.imshow(errors.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Frame" if frame_indices is None else "Frame Index")
    ax.set_ylabel("Dimension")
    ax.set_yticks(range(D))
    ax.set_yticklabels(labels if len(labels) == D else [f"D{i}" for i in range(D)])

    if frame_indices is not None:
        n_ticks = min(20, N)
        tick_idx = np.linspace(0, N - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(frame_indices[tick_idx], rotation=45)
    else:
        n_ticks = min(20, N)
        tick_idx = np.linspace(0, N - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_idx)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|Error|")
    plt.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved error heatmap: %s", filepath)


def plot_summary(
    result: ComparisonResult,
    labels: list[str],
    save_dir: Path,
    episode: int,
    frame_range: tuple[int, int],
) -> None:
    """Generate summary plot combining key visualizations.

    Args:
        result: ComparisonResult from run_comparison
        labels: Dimension labels
        save_dir: Directory to save plots
        episode: Episode index
        frame_range: (start, end) frame indices
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not available, skipping summary plot")
        return

    n_show = min(7, result.action_dim)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"OpenPI Policy Comparison | Episode {episode} | Frames {frame_range[0]}-{frame_range[1]}",
        fontsize=14,
        fontweight="bold",
    )

    gs = GridSpec(3, n_show, figure=fig, hspace=0.4, wspace=0.3)

    # Row 1: Horizon-0 overlay for first n_show dimensions
    x = result.frame_indices
    for d in range(n_show):
        ax = fig.add_subplot(gs[0, d])
        ax.plot(x, result.gt_actions[:, d], label="GT", alpha=0.8, linewidth=1.2, color="#2196F3")
        ax.plot(x, result.pred_h0[:, d], label="Pred(h0)", alpha=0.8, linewidth=1.2, color="#FF5722", linestyle="--")
        ax.set_title(labels[d] if d < len(labels) else f"D{d}")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Row 2: Ensemble overlay - use first T frames where we have GT
    T = result.pred_chunks.shape[0]
    valid_ens = result.pred_ensemble[:T]
    valid_x = result.frame_indices[:T]
    for d in range(n_show):
        ax = fig.add_subplot(gs[1, d])
        ax.plot(
            valid_x,
            result.gt_actions[:T, d],
            label="GT",
            alpha=0.8,
            linewidth=1.2,
            color="#2196F3",
        )
        ax.plot(
            valid_x,
            valid_ens[:, d],
            label="Ensemble",
            alpha=0.8,
            linewidth=1.2,
            color="#4CAF50",
            linestyle="--",
        )
        ax.set_title(f"{labels[d] if d < len(labels) else f'D{d}'} (Ensemble)")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Row 3: Offset curves
    offsets = sorted(result.offset_metrics.keys())
    mae_vals = [result.offset_metrics[k]["mae"] for k in offsets]
    rmse_vals = [result.offset_metrics[k]["rmse"] for k in offsets]

    ax_mae = fig.add_subplot(gs[2, :3])
    ax_mae.plot(offsets, mae_vals, marker="o", linewidth=2, color="#2196F3")
    ax_mae.set_xlabel("Horizon Offset k")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("MAE vs Horizon Offset")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.set_xlim(-0.5, result.horizon - 0.5)

    ax_rmse = fig.add_subplot(gs[2, 3:6])
    ax_rmse.plot(offsets, rmse_vals, marker="s", linewidth=2, color="#FF5722")
    ax_rmse.set_xlabel("Horizon Offset k")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE vs Horizon Offset")
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.set_xlim(-0.5, result.horizon - 0.5)

    # Metrics annotation
    metrics_text = (
        f"Horizon-0  MAE: {result.horizon0_overall_mae:.4f}  RMSE: {result.horizon0_overall_rmse:.4f}\n"
        f"Ensemble    MAE: {result.ensemble_overall_mae:.4f}  RMSE: {result.ensemble_overall_rmse:.4f}\n"
        f"Method: {result.ensemble_method}"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=9, family="monospace")

    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary plot: %s", save_dir / "summary.png")


def save_metrics_json(result: ComparisonResult, labels: list[str], filepath: Path) -> None:
    """Save metrics as JSON.

    Args:
        result: ComparisonResult
        labels: Dimension labels
        filepath: Where to save
    """
    metrics = {
        "n_frames": result.n_frames,
        "horizon": result.horizon,
        "action_dim": result.action_dim,
        "ensemble_method": result.ensemble_method,
        "alpha": result.alpha,
        "horizon0": {
            "overall_mae": result.horizon0_overall_mae,
            "overall_rmse": result.horizon0_overall_rmse,
            "per_dimension_mae": {labels[i]: float(result.horizon0_mae[i]) for i in range(result.action_dim)},
            "per_dimension_rmse": {labels[i]: float(result.horizon0_rmse[i]) for i in range(result.action_dim)},
            "per_dimension_pearson": {labels[i]: float(result.horizon0_pearson[i]) if not np.isnan(result.horizon0_pearson[i]) else None for i in range(result.action_dim)},
        },
        "ensemble": {
            "overall_mae": result.ensemble_overall_mae,
            "overall_rmse": result.ensemble_overall_rmse,
            "per_dimension_mae": {labels[i]: float(result.ensemble_mae[i]) for i in range(result.action_dim)},
            "per_dimension_rmse": {labels[i]: float(result.ensemble_rmse[i]) for i in range(result.action_dim)},
            "per_dimension_pearson": {labels[i]: float(result.ensemble_pearson[i]) if not np.isnan(result.ensemble_pearson[i]) else None for i in range(result.action_dim)},
        },
        "offset_metrics": {
            str(k): result.offset_metrics[k] for k in sorted(result.offset_metrics.keys())
        },
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", filepath)


# --------------------------------------------------------------------------- #
# Sample preview generation
# --------------------------------------------------------------------------- #


def generate_sample_preview(
    frames: list[dict[str, Any]],
    adapter: Any,
    save_path: Path,
) -> None:
    """Generate a preview image showing first/middle/last frames.

    Args:
        frames: List of sample dicts (at least 3 for useful preview)
        adapter: ObservationAdapter instance for image conversion
        save_path: Where to save the preview image
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping preview")
        return

    if len(frames) == 0:
        logger.warning("No frames provided for preview")
        return

    # Select frames: first, middle, last.
    n_total = len(frames)
    indices = [0]
    if n_total > 2:
        indices.append(n_total // 2)
    if n_total > 1:
        indices.append(n_total - 1)
    indices = sorted(set(indices))

    preview_obs = [adapter.adapt(frames[idx]) for idx in indices]
    camera_keys: list[str] = []
    for obs in preview_obs:
        for key, value in obs.items():
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
                if key not in camera_keys:
                    camera_keys.append(key)
    dataset_camera_keys = [
        key for key in camera_keys
        if key.startswith("observation.images") or key.startswith("observation/images")
    ]
    if dataset_camera_keys:
        camera_keys = dataset_camera_keys
    if not camera_keys:
        logger.warning("No camera images found for preview")
        return

    n_camera_cols = len(camera_keys)
    n_rows = len(indices)
    fig, axes = plt.subplots(
        n_rows, n_camera_cols, figsize=(n_camera_cols * 3, n_rows * 3), squeeze=False
    )

    for plot_idx, (frame_idx, obs) in enumerate(zip(indices, preview_obs, strict=True)):

        for cam_idx, cam_key in enumerate(camera_keys):
            if cam_key not in obs:
                continue
            row = plot_idx
            col = cam_idx
            ax = axes[row, col]
            img = obs[cam_key]
            if img is not None and isinstance(img, np.ndarray):
                ax.imshow(img)
            ax.set_title(f"Frame {frame_idx} | {cam_key}", fontsize=8)
            ax.axis("off")

    # Hide empty subplots
    for row in range(n_rows):
        for col in range(n_camera_cols):
            if row >= len(preview_obs) or col >= len(camera_keys):
                axes[row, col].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample preview: %s", save_path)
