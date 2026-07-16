"""Training, testing, and metric utilities."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .models import SpectralAugment


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Return weighted accuracy, unweighted accuracy, and weighted F1."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    total = len(y_true)
    weighted_acc = 0.0
    for cls in classes:
        idx = y_true == cls
        class_acc = (y_pred[idx] == cls).mean() if idx.sum() > 0 else 0.0
        weighted_acc += (idx.sum() / total) * class_acc
    unweighted_acc = balanced_accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return weighted_acc, unweighted_acc, weighted_f1


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def plot_confusion_matrix(cm: np.ndarray, save_path: str, class_names: Optional[Sequence[str]] = None, title: str = "Confusion Matrix") -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    if class_names:
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate_metrics(
    model: torch.nn.Module,
    data_loader,
    criterion,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
    save_dir: str = "./eval_results",
) -> Dict[str, object]:
    """Evaluate classification metrics and save a confusion matrix."""

    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    all_labels, all_preds, all_probs = [], [], []
    total_loss, correct = 0.0, 0

    for x, labels in data_loader:
        x = x.to(device).float()
        labels = labels.to(device).long()
        outputs, _ = model(x)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        correct += (preds == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = 100 * correct / len(data_loader.dataset)
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)
    all_probs = np.asarray(all_probs)
    weighted_acc, unweighted_acc, weighted_f1 = compute_classification_metrics(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"), class_names=class_names)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr") if len(np.unique(all_labels)) > 2 else roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = None

    print("\nEvaluation Summary")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    print(report)
    print(f"Weighted Accuracy: {weighted_acc:.4f} | Unweighted Accuracy: {unweighted_acc:.4f} | Weighted F1: {weighted_f1:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "weighted_accuracy": weighted_acc,
        "unweighted_accuracy": unweighted_acc,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "auc": auc,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def model_summary_info(model: torch.nn.Module, input_shape: Tuple[int, int] = (1, 256), device: str = "cuda") -> None:
    """Print parameter count, optional THOP FLOPs, and average forward time."""

    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024**2)
    print("\n=== Model Complexity Summary ===")
    print(f"Total Parameters: {total_params:,} ({model_size_mb:.2f} MB)")

    try:
        from thop import clever_format, profile

        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
        print(f"Computation: {macs_str} FLOPs | {params_str} Params")
    except Exception as exc:
        print(f"THOP summary skipped: {exc}")

    batch_input = torch.randn(15, input_shape[1]).to(device)
    n_test = 50
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_input)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_test):
            _ = model(batch_input)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    print(f"Average Forward Time: {(t1 - t0) / n_test * 1000:.3f} ms")
    print("=" * 60)


def train(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs: int = 50,
    device: Optional[torch.device] = None,
    vis_interval: int = 50,
    vis_save_dir: str = "lsit_visuals",
    class_names: Optional[Sequence[str]] = None,
    save_best: bool = True,
    save_path: str = "checkpoints/best_model.pt",
    monitor_metric: str = "acc",
    fal_warmup_epochs: int = 5,
    fal_scale: float = 1.0,
    early_stop_patience: int = 20,
    mixup_prob: float = 0.9,
    mixup_alpha: float = 0.01,
    name: str = "run",
    use_augment: bool = False,
) -> List[Dict[str, float]]:
    """Train the dual-stream model and save the best checkpoint."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(vis_save_dir, exist_ok=True)
    checkpoint_dir = os.path.dirname(save_path) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(vis_save_dir, f"train_log_{name}.txt")

    model.to(device)
    model.train()
    best_metric = -np.inf if monitor_metric == "acc" else np.inf
    best_epoch = 0
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []
    augment = SpectralAugment(p_jitter=0.01, p_scale=0.01, p_shift=0.01, p_cutout=0.01, jitter_sigma=0.01, scale_range=(0.2, 0.3))

    for epoch in range(num_epochs):
        running_loss = running_cls = running_lsit = running_fal = 0.0
        correct = 0
        last_tau, last_gate, last_weights = 0.0, (0.0, 0.0, 0.0), torch.zeros(3)

        for spectrum_data, labels in train_loader:
            spectrum_data = spectrum_data.to(device).float()
            labels = labels.to(device).long()
            if use_augment:
                spectrum_data = augment(spectrum_data)

            optimizer.zero_grad()
            use_mixup = np.random.rand() < mixup_prob
            if use_mixup:
                spectrum_data, y_a, y_b, lam = mixup(spectrum_data, labels, alpha=mixup_alpha)

            logits, lsit_loss, fal_loss, tau_mean, gate_means, _, weights = model(spectrum_data, return_loss=True)
            cls_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b) if use_mixup else criterion(logits, labels)
            fal_weight = fal_scale * min(1.0, (epoch + 1) / max(1, fal_warmup_epochs))
            total_loss = cls_loss * 2 + lsit_loss + fal_weight * fal_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_cls += cls_loss.item()
            running_lsit += lsit_loss.item()
            running_fal += (fal_weight * fal_loss).item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            last_tau, last_gate, last_weights = tau_mean, gate_means, weights.detach().cpu()

        train_acc = 100 * correct / len(train_loader.dataset)
        alpha, beta, gamma = last_gate
        msg_train = (
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Total:{running_loss / len(train_loader):.4f} | "
            f"Cls:{running_cls / len(train_loader):.4f} | "
            f"LSIT:{running_lsit / len(train_loader):.4f} | "
            f"FAL:{running_fal / len(train_loader):.4f} | "
            f"TrainAcc:{train_acc:.2f}% | "
            f"tau={last_tau:.3f} | alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f} | "
            f"wpos={last_weights[0]:.3f}, wrbf={last_weights[1]:.3f}, wpoly={last_weights[2]:.3f}"
        )
        print(msg_train)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg_train + "\n")

        if (epoch + 1) % vis_interval == 0 and hasattr(model, "transformer"):
            _save_lsit_samples(model, train_loader, device, vis_save_dir, epoch + 1)

        eval_result = evaluate_metrics(model, test_loader, criterion, device, class_names=class_names, save_dir="eval_results")
        test_acc = float(eval_result["accuracy"])
        avg_test_loss = float(eval_result["loss"])
        metric_value = test_acc if monitor_metric == "acc" else avg_test_loss
        is_better = metric_value > best_metric if monitor_metric == "acc" else metric_value < best_metric
        history.append({"epoch": epoch + 1, "train_acc": train_acc, "test_acc": test_acc, "test_loss": avg_test_loss})

        if save_best and is_better:
            torch.save(model.state_dict(), save_path)
            best_metric = metric_value
            best_epoch = epoch + 1
            no_improve_epochs = 0
            best_cm_path = os.path.join(vis_save_dir, f"best_confusion_matrix_{name}.png")
            plot_confusion_matrix(eval_result["confusion_matrix"], best_cm_path, class_names, title=f"Best Confusion Matrix (Epoch {best_epoch})")
            print(f"Best model saved at epoch {best_epoch}: {save_path}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}; no improvement for {early_stop_patience} epochs.")
            break
        model.train()

    print(f"\nTraining completed. Best epoch: {best_epoch} ({monitor_metric}={best_metric:.4f})")
    return history


@torch.no_grad()
def _save_lsit_samples(model: torch.nn.Module, train_loader, device: torch.device, save_dir: str, epoch: int) -> None:
    model.eval()
    sample_batch = next(iter(train_loader))[0][:4].to(device)
    result = model.transformer(sample_batch)
    lsit_maps = result[0] if isinstance(result, tuple) else result
    for i in range(lsit_maps.size(0)):
        image = lsit_maps[i, 0].cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="viridis")
        plt.title(f"Epoch {epoch} | Sample {i + 1}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch{epoch}_sample{i + 1}.png"), dpi=150)
        plt.close()
