"""Example entry point replacing the notebook's linear execution cells."""

from __future__ import annotations

import argparse

import torch

from .models import DualStreamNIRNet
from .preprocessing import load_spectral_csv
from .train_eval import evaluate_metrics, train
from .visualization import extract_model_features, visualize_tsne


def parse_args():
    parser = argparse.ArgumentParser(description="Train modular LSIT multimodal model.")
    parser.add_argument("--csv", required=True, help="Spectral CSV path.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--input-length", type=int, default=400, help="Length after preprocessing crop.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--transform", default="lsit", choices=["lsit", "gaf", "mtf", "rp"])
    parser.add_argument("--save-path", default="checkpoints/best_model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_spectral_csv(args.csv, label_col=args.label_col, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamNIRNet(input_shape=args.input_length, num_classes=args.num_classes, transform=args.transform)
    model.to(device)
    with torch.no_grad():
        model(torch.randn(1, args.input_length, device=device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(
        model,
        bundle.train_loader,
        bundle.test_loader,
        criterion,
        optimizer,
        num_epochs=args.epochs,
        device=device,
        save_path=args.save_path,
        name=args.transform,
    )
    metrics = evaluate_metrics(model, bundle.test_loader, criterion, device, save_dir="eval_results")
    features = extract_model_features(model, bundle.test_loader, device)
    visualize_tsne(features["fused_feat"], features["labels"], "Fused Features", "lsit_visuals/fused_tsne.png", args.num_classes)
    print(f"Final accuracy: {metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
