import argparse
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader

from algorithms import CoTMix
from kuhar_data import (
    CLASS_NAMES, load_raw_domain, split_target_files, load_selected_files,
    load_subsamples_csv, source_normalization, make_loader, SensorDataset
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                continue
            x = x.to(device, non_blocking=True)
            logits = model.classifier(model.feature_extractor(x))
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    if not ys:
        raise RuntimeError("Target-test loader is empty.")
    return accuracy_score(ys, preds), f1_score(ys, preds, average="macro"), ys, preds


def prepare_raw(args):
    Xs, ys, _ = load_raw_domain(args.source_dir, require_all_classes=True)
    target_train_files, target_test_files = split_target_files(
        args.target_dir, test_fraction=args.target_test_fraction, seed=args.seed
    )
    Xt, yt, _ = load_selected_files(target_train_files)
    Xtest, ytest, _ = load_selected_files(target_test_files)
    # Target labels are retained only for final test evaluation; update() sees Xt only.
    return Xs, ys, Xt, yt, Xtest, ytest


def prepare_subsamples(args):
    X, y = load_subsamples_csv(args.subsamples)
    # Convenience/demo split only. The dataset's described subsample schema has no participant ID.
    # Use per-class split to keep all 18 labels in source and target.
    rng = np.random.RandomState(args.seed)
    src_idx, trg_train_idx, trg_test_idx = [], [], []
    for c in range(len(CLASS_NAMES)):
        ids = np.where(y == c)[0]
        if len(ids) < 5:
            raise RuntimeError(f"Class {CLASS_NAMES[c]} has too few samples: {len(ids)}")
        ids = rng.permutation(ids)
        n_src = int(0.70 * len(ids))
        n_trg_train = int(0.15 * len(ids))
        src_idx.extend(ids[:n_src])
        trg_train_idx.extend(ids[n_src:n_src+n_trg_train])
        trg_test_idx.extend(ids[n_src+n_trg_train:])
    src_idx = np.asarray(src_idx)
    trg_train_idx = np.asarray(trg_train_idx)
    trg_test_idx = np.asarray(trg_test_idx)
    return X[src_idx], y[src_idx], X[trg_train_idx], y[trg_train_idx], X[trg_test_idx], y[trg_test_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "subsamples"], default="raw")
    p.add_argument("--source-dir", default="data/raw_domains/source")
    p.add_argument("--target-dir", default="data/raw_domains/target")
    p.add_argument("--subsamples", default="data/Time_domain_subsamples.csv")
    p.add_argument("--dinov3-repo", default="dinov3")
    p.add_argument("--dinov3-weights", required=True)
    p.add_argument("--output", default="checkpoints/cotmix_dinov3_kuhar.pth")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--target-test-fraction", type=float, default=0.2)
    p.add_argument("--mix-ratio", type=float, default=0.5)
    p.add_argument("--temporal-shift", type=int, default=10)
    p.add_argument("--src-cls-weight", type=float, default=1.0)
    p.add_argument("--trg-entropy-weight", type=float, default=0.1)
    p.add_argument("--src-supcon-weight", type=float, default=0.1)
    p.add_argument("--trg-cont-weight", type=float, default=0.1)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not os.path.isfile(args.dinov3_weights):
        raise FileNotFoundError(
            f"DINOv3 weights not found: {args.dinov3_weights}. "
            "Accept the official DINOv3 model terms and download the ViT-S/16 checkpoint."
        )

    if args.mode == "raw":
        Xs, ys, Xt, yt_train_hidden, Xtest, ytest = prepare_raw(args)
    else:
        Xs, ys, Xt, yt_train_hidden, Xtest, ytest = prepare_subsamples(args)

    # Fit normalization using source data only.
    mean, std = source_normalization(Xs)
    source_ds = SensorDataset(Xs, ys, mean, std)
    target_ds = SensorDataset(Xt, yt_train_hidden, mean, std)
    target_test_ds = SensorDataset(Xtest, ytest, mean, std)

    src_loader = DataLoader(
        source_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    trg_loader = DataLoader(
        target_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        target_test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    if len(src_loader) == 0 or len(trg_loader) == 0:
        raise RuntimeError("Source/target loader has zero batches; lower --batch-size.")

    hparams = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "mix_ratio": args.mix_ratio,
        "temporal_shift": args.temporal_shift,
        "src_cls_weight": args.src_cls_weight,
        "trg_entropy_weight": args.trg_entropy_weight,
        "src_supCon_weight": args.src_supcon_weight,
        "trg_cont_weight": args.trg_cont_weight,
    }
    model = CoTMix(
        configs=None,
        hparams=hparams,
        device=device,
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        pretrained=True,
        image_size=args.image_size,
        num_classes=len(CLASS_NAMES),
    ).to(device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        meters = {}
        steps = 0
        for (src_x, src_y), (trg_x, _target_label_must_not_enter_loss) in zip(src_loader, trg_loader):
            src_x = src_x.to(device, non_blocking=True)
            src_y = src_y.to(device, non_blocking=True)
            trg_x = trg_x.to(device, non_blocking=True)
            losses = model.update(src_x, src_y, trg_x)
            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + v
            steps += 1
        mean_losses = {k: v / max(steps, 1) for k, v in meters.items()}
        target_acc, target_f1, _, _ = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in mean_losses.items())
            + f" | target_acc={target_acc:.4f} | target_macro_f1={target_f1:.4f}"
        )
        if target_f1 > best_f1:
            best_f1 = target_f1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "epoch": epoch,
                "best_target_macro_f1": float(best_f1),
                "target_accuracy_at_best_f1": float(target_acc),
                "class_names": CLASS_NAMES,
                "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist(),
                "image_size": args.image_size,
                "model_name": "dinov3_vits16",
                "dinov3_repo": args.dinov3_repo,
                "hparams": hparams,
                "mode": args.mode,
                "note": (
                    "Target labels are not passed into CoTMix.update; target-test labels "
                    "are used only for metrics. Subsamples mode uses random pseudo-domains."
                ),
            }
            torch.save(checkpoint, args.output)
            print(f"Saved best checkpoint: {args.output}")

    # Final report for the best checkpoint currently on disk.
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    acc, macro_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print("\nFinal target-test report:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0
    ))
    print(f"\nCheckpoint saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
