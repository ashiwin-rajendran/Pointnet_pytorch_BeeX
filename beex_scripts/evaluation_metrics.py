"""
analyze_training_logs.py
------------------------
Parses PointNet++ semantic segmentation training logs and generates:
  - Training/eval loss curves
  - Training/eval accuracy curves
  - Per-class IoU progression over epochs
  - mIoU progression
  - Best epoch summary table
  - Per-class IoU bar chart at best epoch
  - Simulated confusion matrix from final IoU data

Usage:
    python analyze_training_logs.py \
        --log_file /path/to/pointnet2_sem_seg.txt \
        --output_dir ./training_analysis \
        --run_name sonar_seg_normalised
"""

import argparse
import os
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


# ── Colour scheme ─────────────────────────────────────────────────────────────
PALETTE = {
    "train_loss": "#E85D24",
    "eval_loss": "#185FA5",
    "train_acc": "#E85D24",
    "eval_acc": "#185FA5",
    "miou": "#1D9E75",
    "best_marker": "#BA7517",
    "unlabeled": "#888780",
    "seawall": "#1D9E75",
    "anomaly": "#BA7517",
    "seabed": "#8B5E3C",
    "background": "#F8F7F4",
    "grid": "#E0DDD6",
    "text": "#2C2C2A",
    "text_muted": "#5F5E5A",
}

CLASS_COLORS = {
    "unlabeled": PALETTE["unlabeled"],
    "rock_trail": "#3C3489",
    "seawall": PALETTE["seawall"],
    "sheetpile": "#185FA5",
    "ship_hull": "#A32D2D",
    "random_structure": "#993556",
    "anomaly": PALETTE["anomaly"],
    "seabed": PALETTE["seabed"],
}


# ── Log parser ────────────────────────────────────────────────────────────────


def parse_log(log_path: str) -> dict:
    """
    Parse a PointNet++ training log file and return structured data.
    Returns a dict with keys:
        epochs, train_loss, train_acc, eval_loss, eval_acc,
        miou, per_class_iou, per_class_weight, best_miou, best_epoch
    """
    data = defaultdict(list)
    per_class_iou = defaultdict(list)
    per_class_weight = defaultdict(list)

    current_epoch = None
    in_iou_block = False
    epoch_iou_data = {}
    epoch_weight_data = {}

    re_epoch = re.compile(r"\*\*\*\* Epoch (\d+) \((\d+)/(\d+)\)")
    re_train_loss = re.compile(r"Training mean loss:\s+([\d.]+)")
    re_train_acc = re.compile(r"Training accuracy:\s+([\d.]+)")
    re_eval_loss = re.compile(r"Eval mean loss:\s+([\d.]+)")
    re_eval_acc = re.compile(r"Eval accuracy:\s+([\d.]+)")
    re_miou = re.compile(r"eval point avg class IoU:\s+([\d.]+)")
    re_iou_line = re.compile(r"class\s+(\S+)\s+weight:\s+([\d.]+),\s+IoU:\s+([\d.]+)")
    re_iou_header = re.compile(r"------- IoU --------")

    best_miou = 0.0
    best_epoch = 1

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            m = re_epoch.search(line)
            if m:
                if current_epoch is not None and epoch_iou_data:
                    for cls, v in epoch_iou_data.items():
                        per_class_iou[cls].append((current_epoch, v))
                    for cls, v in epoch_weight_data.items():
                        per_class_weight[cls].append((current_epoch, v))
                current_epoch = int(m.group(1))
                epoch_iou_data = {}
                epoch_weight_data = {}
                in_iou_block = False
                continue

            if re_iou_header.search(line):
                in_iou_block = True
                continue

            if in_iou_block:
                m = re_iou_line.search(line)
                if m:
                    cls = m.group(1)
                    weight = float(m.group(2))
                    iou = float(m.group(3))
                    epoch_iou_data[cls] = iou
                    epoch_weight_data[cls] = weight
                else:
                    in_iou_block = False

            m = re_train_loss.search(line)
            if m and current_epoch:
                data["train_loss"].append((current_epoch, float(m.group(1))))

            m = re_train_acc.search(line)
            if m and current_epoch:
                data["train_acc"].append((current_epoch, float(m.group(1))))

            m = re_miou.search(line)
            if m and current_epoch:
                val = float(m.group(1))
                data["miou"].append((current_epoch, val))
                if val > best_miou:
                    best_miou = val
                    best_epoch = current_epoch

            m = re_eval_loss.search(line)
            if m and current_epoch:
                data["eval_loss"].append((current_epoch, float(m.group(1))))

            m = re_eval_acc.search(line)
            if m and current_epoch:
                data["eval_acc"].append((current_epoch, float(m.group(1))))

    # flush last epoch
    if current_epoch is not None and epoch_iou_data:
        for cls, v in epoch_iou_data.items():
            per_class_iou[cls].append((current_epoch, v))

    def to_arrays(pairs):
        if not pairs:
            return np.array([]), np.array([])
        e, v = zip(*pairs)
        return np.array(e), np.array(v)

    result = {}
    for key in ["train_loss", "train_acc", "eval_loss", "eval_acc", "miou"]:
        result[f"{key}_epochs"], result[key] = to_arrays(data[key])

    result["per_class_iou"] = {cls: to_arrays(pairs) for cls, pairs in per_class_iou.items()}
    result["per_class_weight"] = {cls: to_arrays(pairs) for cls, pairs in per_class_weight.items()}
    result["best_miou"] = best_miou
    result["best_epoch"] = best_epoch
    return result


# ── Plot helpers ───────────────────────────────────────────────────────────────


def _style_ax(ax, title="", xlabel="Epoch", ylabel=""):
    ax.set_facecolor(PALETTE["background"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text_muted"])
    ax.yaxis.label.set_color(PALETTE["text_muted"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def _best_vline(ax, best_epoch):
    ax.axvline(
        best_epoch,
        color=PALETTE["best_marker"],
        linewidth=1.2,
        linestyle=":",
        alpha=0.8,
        label=f"Best epoch {best_epoch}",
    )


# ── Individual plots ───────────────────────────────────────────────────────────


def plot_loss(d, out_dir, run_name):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=PALETTE["background"])
    ax.plot(
        d["train_loss_epochs"],
        d["train_loss"],
        color=PALETTE["train_loss"],
        linewidth=1.8,
        label="Train loss",
        alpha=0.9,
    )
    ax.plot(
        d["eval_loss_epochs"], d["eval_loss"], color=PALETTE["eval_loss"], linewidth=1.8, label="Eval loss", alpha=0.9
    )
    _best_vline(ax, d["best_epoch"])
    _style_ax(ax, title="Loss — train vs eval", ylabel="NLL loss")
    ax.legend(fontsize=9, framealpha=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy(d, out_dir, run_name):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=PALETTE["background"])
    ax.plot(
        d["train_acc_epochs"],
        d["train_acc"] * 100,
        color=PALETTE["train_acc"],
        linewidth=1.8,
        label="Train accuracy",
        alpha=0.9,
    )
    ax.plot(
        d["eval_acc_epochs"],
        d["eval_acc"] * 100,
        color=PALETTE["eval_acc"],
        linewidth=1.8,
        label="Eval accuracy",
        alpha=0.9,
    )
    _best_vline(ax, d["best_epoch"])
    _style_ax(ax, title="Point accuracy — train vs eval", ylabel="Accuracy (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, framealpha=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_miou(d, out_dir, run_name):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=PALETTE["background"])
    ax.plot(d["miou_epochs"], d["miou"] * 100, color=PALETTE["miou"], linewidth=2.0, label="mIoU")
    # mark best
    best_val = d["best_miou"] * 100
    ax.scatter(
        [d["best_epoch"]],
        [best_val],
        color=PALETTE["best_marker"],
        zorder=5,
        s=60,
        label=f"Best: {best_val:.1f}% @ ep{d['best_epoch']}",
    )
    _best_vline(ax, d["best_epoch"])
    _style_ax(ax, title="Mean IoU (present classes only)", ylabel="mIoU (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, framealpha=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_miou.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_class_iou(d, out_dir, run_name):
    pci = d["per_class_iou"]
    present = {cls: (ep, iou) for cls, (ep, iou) in pci.items() if len(iou) > 0 and iou.max() > 0.001}
    if not present:
        print("  No per-class IoU data to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=PALETTE["background"])
    for cls, (ep, iou) in present.items():
        color = CLASS_COLORS.get(cls, "#888780")
        ax.plot(ep, iou * 100, linewidth=1.6, label=cls, color=color, alpha=0.9)

    _best_vline(ax, d["best_epoch"])
    _style_ax(ax, title="Per-class IoU over training", ylabel="IoU (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8.5, framealpha=0.6, loc="upper left", ncol=2 if len(present) > 4 else 1)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_per_class_iou.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_epoch_bar(d, out_dir, run_name):
    """Bar chart of per-class IoU at the best epoch."""
    pci = d["per_class_iou"]
    best_ep = d["best_epoch"]
    classes, values, colors = [], [], []

    for cls, (ep_arr, iou_arr) in pci.items():
        if len(ep_arr) == 0:
            continue
        # find value closest to best epoch
        idx = np.argmin(np.abs(ep_arr - best_ep))
        val = iou_arr[idx]
        classes.append(cls)
        values.append(val * 100)
        colors.append(CLASS_COLORS.get(cls, "#888780"))

    if not classes:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=PALETTE["background"])
    x = np.arange(len(classes))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8, alpha=0.92)

    for bar, val in zip(bars, values):
        if val > 1.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=PALETTE["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    _style_ax(ax, title=f"Per-class IoU at best epoch ({best_ep})", ylabel="IoU (%)")
    ax.axhline(
        d["best_miou"] * 100,
        color=PALETTE["miou"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label=f"mIoU={d['best_miou']*100:.1f}%",
    )
    ax.legend(fontsize=9, framealpha=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_best_epoch_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overview_dashboard(d, out_dir, run_name):
    """2×2 dashboard: loss, accuracy, mIoU, per-class bar."""
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE["background"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PALETTE["background"])
    ax1.plot(d["train_loss_epochs"], d["train_loss"], color=PALETTE["train_loss"], linewidth=1.8, label="Train")
    ax1.plot(d["eval_loss_epochs"], d["eval_loss"], color=PALETTE["eval_loss"], linewidth=1.8, label="Eval")
    _best_vline(ax1, d["best_epoch"])
    _style_ax(ax1, "Loss", ylabel="NLL loss")
    ax1.legend(fontsize=8.5, framealpha=0.6)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE["background"])
    ax2.plot(d["train_acc_epochs"], d["train_acc"] * 100, color=PALETTE["train_acc"], linewidth=1.8, label="Train")
    ax2.plot(d["eval_acc_epochs"], d["eval_acc"] * 100, color=PALETTE["eval_acc"], linewidth=1.8, label="Eval")
    _best_vline(ax2, d["best_epoch"])
    _style_ax(ax2, "Point accuracy", ylabel="%")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(fontsize=8.5, framealpha=0.6)

    # ── mIoU ──────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PALETTE["background"])
    ax3.plot(d["miou_epochs"], d["miou"] * 100, color=PALETTE["miou"], linewidth=2.0)
    ax3.scatter(
        [d["best_epoch"]],
        [d["best_miou"] * 100],
        color=PALETTE["best_marker"],
        zorder=5,
        s=60,
        label=f"Best {d['best_miou']*100:.1f}% @ ep{d['best_epoch']}",
    )
    _best_vline(ax3, d["best_epoch"])
    _style_ax(ax3, "mIoU (present classes)", ylabel="%")
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax3.legend(fontsize=8.5, framealpha=0.6)

    # ── Per-class bar at best epoch ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PALETTE["background"])
    pci = d["per_class_iou"]
    best_ep = d["best_epoch"]
    cls_names, iou_vals, bar_cols = [], [], []
    for cls, (ep_arr, iou_arr) in pci.items():
        if len(ep_arr) == 0:
            continue
        idx = np.argmin(np.abs(ep_arr - best_ep))
        cls_names.append(cls)
        iou_vals.append(iou_arr[idx] * 100)
        bar_cols.append(CLASS_COLORS.get(cls, "#888780"))

    if cls_names:
        x = np.arange(len(cls_names))
        bars = ax4.bar(x, iou_vals, color=bar_cols, width=0.6, edgecolor="white", linewidth=0.8, alpha=0.92)
        for bar, val in zip(bars, iou_vals):
            if val > 1.0:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=PALETTE["text"],
                )
        ax4.set_xticks(x)
        ax4.set_xticklabels(cls_names, rotation=30, ha="right", fontsize=8)
        ax4.set_ylim(0, 115)
        ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
        ax4.axhline(d["best_miou"] * 100, color=PALETTE["miou"], linewidth=1.2, linestyle="--", alpha=0.7)
    _style_ax(ax4, f"Per-class IoU @ epoch {best_ep}", ylabel="%")

    fig.suptitle(f"Training summary — {run_name}", color=PALETTE["text"], fontsize=14, fontweight="bold", y=0.98)

    path = os.path.join(out_dir, f"{run_name}_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pseudo_confusion(d, out_dir, run_name):
    """
    Build a pseudo confusion matrix from per-class IoU at the best epoch.
    Diagonal = IoU (proxy for TP rate). Off-diagonal estimated from 1-IoU.
    NOTE: This is an approximation — true confusion needs per-point predictions.
    """
    pci = d["per_class_iou"]
    best_ep = d["best_epoch"]
    present_cls, iou_vals = [], []
    for cls, (ep_arr, iou_arr) in pci.items():
        if len(ep_arr) == 0:
            continue
        iou_arr_max = iou_arr.max()
        idx = np.argmin(np.abs(ep_arr - best_ep))
        val = iou_arr[idx]
        if iou_arr_max > 0.001:
            present_cls.append(cls)
            iou_vals.append(val)

    if len(present_cls) < 2:
        return

    n = len(present_cls)
    iou_vals = np.array(iou_vals)

    # Approximation: diagonal = IoU, off-diagonal = (1-IoU) / (n-1)
    mat = np.zeros((n, n))
    for i, iou in enumerate(iou_vals):
        mat[i, i] = iou
        off = max(0, (1 - iou)) / (n - 1)
        for j in range(n):
            if j != i:
                mat[i, j] = off

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["background"])
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(present_cls, rotation=35, ha="right", fontsize=9, color=PALETTE["text"])
    ax.set_yticklabels(present_cls, fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Predicted", fontsize=9, color=PALETTE["text_muted"])
    ax.set_ylabel("Ground truth", fontsize=9, color=PALETTE["text_muted"])

    for i in range(n):
        for j in range(n):
            txt_color = "white" if mat[i, j] > 0.6 else PALETTE["text"]
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=txt_color)

    ax.set_title(
        f"Approx. confusion matrix @ epoch {best_ep}\n" f"(diagonal = IoU; off-diagonal estimated)",
        color=PALETTE["text"],
        fontsize=10,
        fontweight="bold",
    )
    ax.set_facecolor(PALETTE["background"])
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary JSON + text ────────────────────────────────────────────────────────


def save_summary(d, out_dir, run_name):
    pci = d["per_class_iou"]
    best_ep = d["best_epoch"]

    per_class_at_best = {}
    for cls, (ep_arr, iou_arr) in pci.items():
        if len(ep_arr) == 0:
            continue
        idx = np.argmin(np.abs(ep_arr - best_ep))
        per_class_at_best[cls] = round(float(iou_arr[idx]), 4)

    # find train/eval values at best epoch
    def val_at(epochs, values, ep):
        if len(epochs) == 0:
            return None
        idx = np.argmin(np.abs(epochs - ep))
        return round(float(values[idx]), 4)

    summary = {
        "run_name": run_name,
        "total_epochs": int(d["train_loss_epochs"].max()) if len(d["train_loss_epochs"]) else 0,
        "best_epoch": best_ep,
        "best_miou": round(d["best_miou"], 4),
        "train_loss_at_best": val_at(d["train_loss_epochs"], d["train_loss"], best_ep),
        "eval_loss_at_best": val_at(d["eval_loss_epochs"], d["eval_loss"], best_ep),
        "train_acc_at_best": val_at(d["train_acc_epochs"], d["train_acc"], best_ep),
        "eval_acc_at_best": val_at(d["eval_acc_epochs"], d["eval_acc"], best_ep),
        "per_class_iou_at_best": per_class_at_best,
    }

    json_path = os.path.join(out_dir, f"{run_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # Human-readable text report
    txt_path = os.path.join(out_dir, f"{run_name}_report.txt")
    lines = [
        f"Training Report — {run_name}",
        "=" * 50,
        f"Total epochs     : {summary['total_epochs']}",
        f"Best epoch       : {best_ep}",
        f"Best mIoU        : {summary['best_miou']*100:.2f}%",
        f"Eval accuracy    : {(summary['eval_acc_at_best'] or 0)*100:.2f}%",
        f"Train accuracy   : {(summary['train_acc_at_best'] or 0)*100:.2f}%",
        f"Eval loss        : {summary['eval_loss_at_best']}",
        "",
        "Per-class IoU at best epoch:",
        "-" * 30,
    ]
    for cls, iou in sorted(per_class_at_best.items(), key=lambda x: -x[1]):
        marker = "*" if iou > 0.001 else " "
        lines.append(f"  {marker} {cls:<22} {iou*100:.2f}%")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {txt_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyse PointNet++ training logs")
    parser.add_argument("--log_file", required=True, help="Path to the .txt log file")
    parser.add_argument("--output_dir", default="./training_analysis", help="Output folder for plots and summaries")
    parser.add_argument("--run_name", default=None, help="Run label for file naming (defaults to log stem)")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    run_name = args.run_name or log_path.stem
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nParsing: {log_path}")
    d = parse_log(str(log_path))

    print(f"\nBest mIoU: {d['best_miou']*100:.2f}% at epoch {d['best_epoch']}")
    print(f"\nGenerating plots → {out_dir}/")

    plot_loss(d, out_dir, run_name)
    plot_accuracy(d, out_dir, run_name)
    plot_miou(d, out_dir, run_name)
    plot_per_class_iou(d, out_dir, run_name)
    plot_best_epoch_bar(d, out_dir, run_name)
    plot_overview_dashboard(d, out_dir, run_name)
    plot_pseudo_confusion(d, out_dir, run_name)
    save_summary(d, out_dir, run_name)

    print(f"\nDone. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
