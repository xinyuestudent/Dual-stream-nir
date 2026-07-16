"""Visualization and analysis helpers from the notebook."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def visualize_tsne(features, labels, title: str, save_path: str, num_classes: Optional[int] = None) -> None:
    features = to_numpy(features)
    labels = to_numpy(labels)
    if num_classes is None:
        num_classes = len(np.unique(labels))
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    feat_2d = tsne.fit_transform(features)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 5))
    for cls in range(num_classes):
        idx = labels == cls
        plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], label=f"Class {cls}", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def extract_model_features(model, data_loader, device: torch.device):
    """Extract spectral, image, fused features, logits, and labels."""

    model.eval()
    all_spec, all_img, all_fused, all_logits, all_labels = [], [], [], [], []
    for x, y in data_loader:
        x = x.to(device).float()
        spec_feat = model.spec_branch(x)
        image_result = model.transformer(x)
        image = image_result[0] if isinstance(image_result, tuple) else image_result
        img_feat = model.img_branch(image)
        cross_feat, _ = model.dca(spec_feat, img_feat)
        gate_weights = model.gate(torch.cat([spec_feat, img_feat, cross_feat], dim=1))
        alpha, beta, gamma = gate_weights[:, 0:1], gate_weights[:, 1:2], gate_weights[:, 2:3]
        fused = alpha * spec_feat + beta * img_feat + gamma * cross_feat
        logits = model.fc2(torch.relu(model.fc1(fused)))

        all_spec.append(spec_feat.cpu())
        all_img.append(img_feat.cpu())
        all_fused.append(fused.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    return {
        "spec_feat": torch.cat(all_spec, dim=0),
        "img_feat": torch.cat(all_img, dim=0),
        "fused_feat": torch.cat(all_fused, dim=0),
        "logits": torch.cat(all_logits, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


def structural_entropy(matrix: np.ndarray, bins: int = 64, eps: float = 1e-12) -> float:
    values = np.abs(matrix).ravel()
    if values.max() > 0:
        values = values / (values.max() + eps)
    hist, _ = np.histogram(values, bins=bins, range=(0, 1), density=True)
    p = hist / (hist.sum() + eps)
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def corr_pearson_1d(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) != len(x):
        y = np.interp(np.linspace(0, 1, num=len(x)), np.linspace(0, 1, num=len(y)), y)
    x = (x - x.mean()) / (x.std() + eps)
    y = (y - y.mean()) / (y.std() + eps)
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def evaluate_lsit_dataset(
    model,
    test_loader,
    wavelengths: Optional[Sequence[float]] = None,
    device: str | torch.device = "cuda",
    save_dir: str = "lsit_visuals/panel",
    bins: int = 64,
    format_name: str = "lsit",
    dataname: str = "dataset",
) -> Dict[str, object]:
    """Compute LSIT structure metrics and save mean activation/curve figures."""

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    r0_list, r1_list, rstar_list, entropy_list = [], [], [], []
    mean_map, sum_orig, sum_fp = None, None, None
    seen = 0

    for batch in test_loader:
        spectrum_data = batch[0] if isinstance(batch, (list, tuple)) else batch
        spectrum_data = spectrum_data.to(device).float()
        x_np = spectrum_data.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np.squeeze(1)

        image_result = model.transformer(spectrum_data)
        lsit_out = image_result[0] if isinstance(image_result, tuple) else image_result
        if lsit_out.dim() == 4:
            lsit_out = lsit_out.squeeze(1)
        lsit_np = lsit_out.detach().cpu().numpy()
        batch_size, channels, length = lsit_np.shape

        if mean_map is None:
            mean_map = np.zeros((channels, length), dtype=np.float64)
            sum_orig = np.zeros(length, dtype=np.float64)
            sum_fp = np.zeros(length, dtype=np.float64)

        mean_map += lsit_np.sum(axis=0)
        seen += batch_size
        for b in range(batch_size):
            matrix = lsit_np[b]
            x = x_np[b]
            fingerprint = matrix.mean(axis=0)
            h_norm = structural_entropy(matrix, bins=bins) / np.log2(bins)
            r0 = corr_pearson_1d(x, fingerprint)
            r1 = corr_pearson_1d(np.gradient(x), fingerprint)
            entropy_list.append(h_norm)
            r0_list.append(r0)
            r1_list.append(r1)
            rstar_list.append(max(abs(r0), abs(r1)))
            sum_orig += x
            sum_fp += fingerprint

    mean_map /= max(seen, 1)
    mean_orig = sum_orig / max(seen, 1)
    mean_fp = sum_fp / max(seen, 1)

    def mean_std(values):
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    summary = pd.DataFrame(
        {
            "metric": ["r0", "r1", "r_star", "H_norm"],
            "mean": [mean_std(r0_list)[0], mean_std(r1_list)[0], mean_std(rstar_list)[0], mean_std(entropy_list)[0]],
            "std": [mean_std(r0_list)[1], mean_std(r1_list)[1], mean_std(rstar_list)[1], mean_std(entropy_list)[1]],
        }
    )
    csv_path = os.path.join(save_dir, f"{format_name}_metrics_summary_{dataname}.csv")
    summary.to_csv(csv_path, index=False)

    heat_path = os.path.join(save_dir, f"mean_{format_name}_activation_{dataname}.png")
    plt.figure(figsize=(10, 4))
    plt.imshow(mean_map, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar()
    plt.xlabel("Index" if wavelengths is None else "Wavelength (nm)")
    plt.ylabel(f"{format_name} channel index")
    plt.title(f"Mean {format_name.upper()} Activation")
    if wavelengths is not None and len(wavelengths) == mean_map.shape[1]:
        xticks = np.linspace(0, mean_map.shape[1] - 1, 6, dtype=int)
        plt.xticks(xticks, [f"{wavelengths[i]:.0f}" for i in xticks])
    plt.tight_layout()
    plt.savefig(heat_path, dpi=300)
    plt.close()

    curves_path = os.path.join(save_dir, f"mean_curves_orig_vs_{format_name}_fp_{dataname}.png")
    x_axis = np.arange(mean_map.shape[1]) if wavelengths is None else np.asarray(wavelengths)
    plt.figure(figsize=(8, 4))
    plt.plot(x_axis, _normalize_curve(mean_orig), label="Mean original 1D")
    plt.plot(x_axis, _normalize_curve(mean_fp), label=f"Mean {format_name} fingerprint", linestyle="--")
    plt.xlabel("Index" if wavelengths is None else "Wavelength (nm)")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()

    return {
        "summary_df": summary,
        "paths": {"csv": csv_path, "heatmap": heat_path, "curves": curves_path},
        "distributions": {"r0": r0_list, "r1": r1_list, "r_star": rstar_list, "H_norm": entropy_list},
    }


def _normalize_curve(x: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(x))
    return x / (max_abs if max_abs > 0 else 1.0)


def sample_per_class(features, labels, max_per_class: int = 300, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    features = to_numpy(features)
    labels = to_numpy(labels)
    selected = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(idx.tolist())
    selected = np.asarray(selected)
    return features[selected], labels[selected], selected


def compute_intra_inter_distances(features, labels) -> Tuple[np.ndarray, np.ndarray]:
    features = to_numpy(features)
    labels = to_numpy(labels)
    dist_matrix = pairwise_distances(features, metric="euclidean")
    intra_dist, inter_dist = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_dist.append(dist_matrix[i, j])
            else:
                inter_dist.append(dist_matrix[i, j])
    return np.asarray(intra_dist), np.asarray(inter_dist)


def plot_distance_distribution(intra_dist, inter_dist, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 5), dpi=200)
    x = np.linspace(min(intra_dist.min(), inter_dist.min()), max(intra_dist.max(), inter_dist.max()), 500)
    kde_intra = gaussian_kde(intra_dist)
    kde_inter = gaussian_kde(inter_dist)
    plt.plot(x, kde_intra(x), label="Intra-class Distance", linewidth=2)
    plt.fill_between(x, kde_intra(x), alpha=0.25)
    plt.plot(x, kde_inter(x), label="Inter-class Distance", linewidth=2)
    plt.fill_between(x, kde_inter(x), alpha=0.25)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution of Fused Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_distance_boxplot(intra_dist, inter_dist, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=200)
    plt.boxplot([intra_dist, inter_dist], labels=["Intra-class", "Inter-class"], showfliers=False)
    plt.ylabel("Euclidean Distance")
    plt.title("Distance Comparison of Fused Features")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(conf_correct, conf_wrong, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 5), dpi=200)
    if len(conf_correct) > 1:
        x1 = np.linspace(conf_correct.min(), conf_correct.max(), 500)
        kde_correct = gaussian_kde(conf_correct)
        plt.plot(x1, kde_correct(x1), label="Correct Predictions", linewidth=2)
        plt.fill_between(x1, kde_correct(x1), alpha=0.25)
    if len(conf_wrong) > 1:
        x2 = np.linspace(conf_wrong.min(), conf_wrong.max(), 500)
        kde_wrong = gaussian_kde(conf_wrong)
        plt.plot(x2, kde_wrong(x2), label="Wrong Predictions", linewidth=2)
        plt.fill_between(x2, kde_wrong(x2), alpha=0.25)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Density")
    plt.title("Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confidence_boxplot(conf_correct, conf_wrong, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=200)
    plt.boxplot([conf_correct, conf_wrong], labels=["Correct", "Wrong"], showfliers=False)
    plt.ylabel("Prediction Confidence")
    plt.title("Confidence Comparison")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def summarize_stats(name: str, arr) -> Dict[str, object]:
    arr = np.asarray(arr)
    return {
        "name": name,
        "mean": float(np.mean(arr)) if len(arr) > 0 else None,
        "std": float(np.std(arr)) if len(arr) > 0 else None,
        "min": float(np.min(arr)) if len(arr) > 0 else None,
        "max": float(np.max(arr)) if len(arr) > 0 else None,
        "median": float(np.median(arr)) if len(arr) > 0 else None,
        "num": int(len(arr)),
    }


def reduce_features(features, use_tsne: bool = False, random_state: int = 42):
    reducer = (
        TSNE(n_components=2, perplexity=35, learning_rate=200, init="pca", random_state=random_state)
        if use_tsne
        else PCA(n_components=2, random_state=random_state)
    )
    return reducer.fit_transform(to_numpy(features))


def compute_centroid_stats(emb_2d, labels) -> Dict[str, float]:
    emb_2d = to_numpy(emb_2d)
    labels = to_numpy(labels)
    classes = np.unique(labels)
    global_pairwise = np.linalg.norm(emb_2d[:, None, :] - emb_2d[None, :, :], axis=2)
    global_max_dist = np.max(global_pairwise) + 1e-12
    centroids, intra_dists = {}, []
    for cls in classes:
        pts = emb_2d[labels == cls]
        centroid = pts.mean(axis=0)
        centroids[cls] = centroid
        intra_dists.extend((np.linalg.norm(pts - centroid, axis=1) / global_max_dist).tolist())

    inter_dists = []
    class_list = list(classes)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            inter_dists.append(np.linalg.norm(centroids[class_list[i]] - centroids[class_list[j]]) / global_max_dist)
    return {
        "mean_intra": float(np.mean(intra_dists)),
        "std_intra": float(np.std(intra_dists)),
        "mean_inter": float(np.mean(inter_dists)),
        "std_inter": float(np.std(inter_dists)),
        "ratio_inter_over_intra": float(np.mean(inter_dists) / (np.mean(intra_dists) + 1e-12)),
    }


def plot_centroid_embedding(
    emb_2d,
    labels,
    stats: Dict[str, float],
    save_path: str,
    title: str = "Centroid-aware Visualization of Fused Features",
    method_name: str = "PCA",
    figsize: Tuple[int, int] = (7, 6),
    dpi: int = 300,
) -> None:
    emb_2d = to_numpy(emb_2d)
    labels = to_numpy(labels)
    classes = np.unique(labels)
    cmap = plt.colormaps.get_cmap("tab10")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, cls in enumerate(classes):
        pts = emb_2d[labels == cls]
        color = cmap(i / max(len(classes) - 1, 1))
        ax.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.75, color=color, label=f"Class {cls}", edgecolors="none")
        centroid = pts.mean(axis=0)
        for p in pts:
            ax.plot([centroid[0], p[0]], [centroid[1], p[1]], color=color, alpha=0.18, linewidth=0.8)
        ax.scatter(centroid[0], centroid[1], marker="*", s=220, color=color, edgecolors="k", linewidths=0.8, zorder=5)

    ax.set_title(title, fontsize=15, pad=26)
    subtitle = (
        f"Intra-class: {stats['mean_intra']:.2f} +/- {stats['std_intra']:.2f}    "
        f"Inter-class: {stats['mean_inter']:.2f} +/- {stats['std_inter']:.2f}    "
        f"Inter/Intra: {stats['ratio_inter_over_intra']:.2f}"
    )
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    ax.set_xlabel(f"{method_name} 1", fontsize=12)
    ax.set_ylabel(f"{method_name} 2", fontsize=12)
    ax.legend(fontsize=9, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_distance_and_confidence_report(features, labels, probs, preds, save_dir: str, max_per_class: int = 300) -> Dict[str, object]:
    """Produce the notebook's fused-feature distance and confidence plots."""

    os.makedirs(save_dir, exist_ok=True)
    features_s, labels_s, _ = sample_per_class(features, labels, max_per_class=max_per_class)
    intra_dist, inter_dist = compute_intra_inter_distances(features_s, labels_s)
    plot_distance_distribution(intra_dist, inter_dist, os.path.join(save_dir, "distance_distribution.png"))
    plot_distance_boxplot(intra_dist, inter_dist, os.path.join(save_dir, "distance_boxplot.png"))

    probs = to_numpy(probs)
    labels = to_numpy(labels)
    preds = to_numpy(preds)
    conf = probs.max(axis=1)
    conf_correct = conf[preds == labels]
    conf_wrong = conf[preds != labels]
    plot_confidence_distribution(conf_correct, conf_wrong, os.path.join(save_dir, "confidence_distribution.png"))
    plot_confidence_boxplot(conf_correct, conf_wrong, os.path.join(save_dir, "confidence_boxplot.png"))

    stats = {
        "intra_distance": summarize_stats("intra_distance", intra_dist),
        "inter_distance": summarize_stats("inter_distance", inter_dist),
        "confidence_correct": summarize_stats("confidence_correct", conf_correct),
        "confidence_wrong": summarize_stats("confidence_wrong", conf_wrong),
    }
    with open(os.path.join(save_dir, "summary_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
