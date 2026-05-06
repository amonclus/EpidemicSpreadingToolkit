"""
DL visualisation — mirrors results/ml plots so figures can be placed side by side.

Naming convention matches ml_analysis.py:
  R1  MAE vs observation window
  R2  Predicted vs True scatter
  R3  Per-model MAE bar chart
  R4  2-D embedding (t-SNE) — DL-specific replacement for feature importance
  C1  Accuracy vs observation window
  C2  Confusion matrix
  C3  Cross-network generalisation
  C4  Per-class F1 bar chart
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, classification_report

try:
    import umap as _umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

# ── Style constants — identical to ml_analysis.py ────────────────────────────
FS_TITLE  = 14
FS_LABEL  = 12
FS_TICK   = 10
FS_LEGEND = 10

MODEL_COLORS = {
    "SIR": "#2196F3", "SIS": "#00BCD4",
    "BP":  "#4CAF50", "WTM": "#FF9800",
    "H1":  "#F44336", "H2":  "#E91E63",
    "H3":  "#9C27B0", "H4":  "#673AB7",
    "H5":  "#3F51B5", "H6":  "#009688",
}
MODEL_NAMES  = ["SIR", "SIS", "BP", "WTM", "H1", "H2", "H3", "H4", "H5", "H6"]
_ID_TO_NAME  = MODEL_NAMES

# Arch line styles (matches the HGB/RF/Ridge palette where comparable)
_ARCH_STYLE = {
    "CNN":           ("^-",  "#2E7D32", 2),
    "LSTM":          ("o-",  "#1565C0", 2),
    "Transformer":   ("s--", "#B71C1C", 2),
    "Ensemble":      ("D-",  "#FF6F00", 2),
    "TwoStage_hard": ("P-",  "#6A1B9A", 2),
}
_HGB_STYLE = ("D:",  "#FF6F00", 1.5)   # HGB baseline reference

NET_COLORS = {
    "ER": "#1565C0", "RGG": "#2E7D32", "Lattice": "#E65100",
    "Facebook": "#6A1B9A", "GitHub": "#00838F",
}


def save_fig(fig, path: str):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── R1: MAE vs observation window ─────────────────────────────────────────────

def plot_R1_mae_vs_window(
    results_reg: dict,          # {arch_name: {t_obs: {"MAE": ..., ...}}}
    t_obs_values: list,
    hgb_maes: dict = None,      # optional {t_obs: mae} for HGB reference line
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))

    for arch, (marker, color, lw) in _ARCH_STYLE.items():
        if arch not in results_reg:
            continue
        maes = [results_reg[arch].get(t, {}).get("MAE", np.nan) for t in t_obs_values]
        ax.plot(t_obs_values, maes, marker, color=color, lw=lw, label=arch)

    if hgb_maes:
        marker, color, lw = _HGB_STYLE
        maes = [hgb_maes.get(t, np.nan) for t in t_obs_values]
        ax.plot(t_obs_values, maes, marker, color=color, lw=lw, label="HGB (ML baseline)")

    ax.set_xlabel("Observation window  t_obs", fontsize=FS_LABEL)
    ax.set_ylabel("MAE",                        fontsize=FS_LABEL)
    ax.set_title("Regression Performance vs Observation Window", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path or "results/dl/R1_mae_vs_window.png")


# ── R2: Predicted vs True scatter ─────────────────────────────────────────────

def plot_R2_pred_vs_true(
    rho_pred: np.ndarray,
    rho_true: np.ndarray,
    model_ids: np.ndarray,
    mae: float,
    r2: float,
    arch_name: str,
    t_obs: int,
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    for mid, name in enumerate(_ID_TO_NAME):
        mask = model_ids == mid
        if mask.any():
            ax.scatter(rho_true[mask], rho_pred[mask],
                       c=MODEL_COLORS[name], label=name,
                       s=8, alpha=0.6, linewidths=0)

    lo = min(rho_true.min(), rho_pred.min()) - 0.01
    hi = max(rho_true.max(), rho_pred.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax.text(0.05, 0.90,
            f"MAE = {mae:.4f}\nR²  = {r2:.4f}",
            transform=ax.transAxes, fontsize=FS_LEGEND,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_xlabel("True ρ_final",      fontsize=FS_LABEL)
    ax.set_ylabel("Predicted ρ_final", fontsize=FS_LABEL)
    ax.set_title(f"Predicted vs True Final Epidemic Size (t_obs={t_obs}, {arch_name})",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, markerscale=2,
              loc="lower right", ncol=2, framealpha=0.85)
    fig.tight_layout()
    save_fig(fig, save_path or f"results/dl/R2_pred_vs_true_{arch_name}_t{t_obs}.png")


# ── R3: Per-model MAE horizontal bar chart ────────────────────────────────────

def plot_R3_per_model_mae(
    rho_pred: np.ndarray,
    rho_true: np.ndarray,
    model_ids: np.ndarray,
    arch_name: str,
    t_obs: int,
    save_path: str = None,
):
    per_model = {
        name: float(np.abs(rho_pred[model_ids == mid] - rho_true[model_ids == mid]).mean())
        for mid, name in enumerate(_ID_TO_NAME)
        if (model_ids == mid).any()
    }
    sorted_items = sorted(per_model.items(), key=lambda x: x[1], reverse=True)
    m_names  = [x[0] for x in sorted_items]
    m_maes   = [x[1] for x in sorted_items]
    m_colors = [MODEL_COLORS[n] for n in m_names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(m_names, m_maes, color=m_colors)
    for bar, val in zip(bars, m_maes):
        ax.text(bar.get_width() + max(m_maes) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=FS_TICK)
    ax.set_xlabel("MAE",   fontsize=FS_LABEL)
    ax.set_title(f"Prediction Difficulty by Model ({arch_name}, t_obs={t_obs})",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(0, max(m_maes) * 1.20)
    fig.tight_layout()
    save_fig(fig, save_path or f"results/dl/R3_per_model_mae_{arch_name}_t{t_obs}.png")


# ── R4: 2-D embedding (t-SNE) — DL-specific ──────────────────────────────────

def plot_R4_embedding_2d(
    model,
    dataloader,
    device: str,
    arch_name: str,
    t_obs: int,
    method: str = "tsne",
    save_path: str = None,
):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for x, _, mid, *_ in dataloader:
            embs.append(model.get_embedding(x.to(device)).cpu().numpy())
            labels.append(mid.numpy())
    embs   = np.concatenate(embs)
    labels = np.concatenate(labels)

    if method == "umap" and _HAS_UMAP:
        coords = _umap.UMAP(n_components=2, random_state=42).fit_transform(embs)
        method_label = "UMAP"
    else:
        perp   = min(30, len(embs) - 1)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                      max_iter=1000).fit_transform(embs)
        method_label = "t-SNE"

    fig, ax = plt.subplots(figsize=(7, 5))
    for mid, name in enumerate(_ID_TO_NAME):
        mask = labels == mid
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=MODEL_COLORS[name], alpha=0.4, s=10, label=name)

    handles = [mpatches.Patch(color=MODEL_COLORS[n], label=n) for n in _ID_TO_NAME]
    ax.legend(handles=handles, fontsize=FS_LEGEND - 1, ncol=2,
              loc="best", framealpha=0.7)
    ax.set_xlabel(f"{method_label}-1", fontsize=FS_LABEL)
    ax.set_ylabel(f"{method_label}-2", fontsize=FS_LABEL)
    ax.set_title(f"Learned Embedding Space — {arch_name} (t_obs={t_obs})",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    save_fig(fig, save_path or f"results/dl/R4_embedding_{arch_name}_t{t_obs}.png")


# ── C1: Accuracy vs observation window ───────────────────────────────────────

def plot_C1_accuracy_vs_window(
    results_cls: dict,          # {arch_name: {t_obs: {"accuracy": ..., ...}}}
    t_obs_values: list,
    hgb_accs: dict = None,      # optional {t_obs: acc} for HGB reference line
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))

    for arch, (marker, color, lw) in _ARCH_STYLE.items():
        if arch not in results_cls:
            continue
        accs = [results_cls[arch].get(t, {}).get("accuracy", np.nan) for t in t_obs_values]
        ax.plot(t_obs_values, accs, marker, color=color, lw=lw, label=arch)

    if hgb_accs:
        marker, color, lw = _HGB_STYLE
        accs = [hgb_accs.get(t, np.nan) for t in t_obs_values]
        ax.plot(t_obs_values, accs, marker, color=color, lw=lw, label="HGB (ML baseline)")

    ax.axhline(0.1, color="gray", linestyle=":", lw=1.5, label="Chance (0.1)")
    ax.set_xlabel("Observation window  t_obs", fontsize=FS_LABEL)
    ax.set_ylabel("Accuracy",                   fontsize=FS_LABEL)
    ax.set_title("Classification Accuracy vs Observation Window", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path or "results/dl/C1_accuracy_vs_window.png")


# ── C2: Confusion matrix ──────────────────────────────────────────────────────

def plot_C2_confusion_matrix(
    cls_pred: np.ndarray,
    cls_true: np.ndarray,
    arch_name: str,
    t_obs: int,
    save_path: str = None,
):
    labels = _ID_TO_NAME
    cm = confusion_matrix(cls_true, cls_pred,
                          labels=list(range(len(labels))), normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.4, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=FS_LABEL)
    ax.set_ylabel("True",      fontsize=FS_LABEL)
    ax.set_title(f"Model Identification Confusion Matrix (t_obs={t_obs}, {arch_name})",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    save_fig(fig, save_path or f"results/dl/C2_confusion_matrix_{arch_name}_t{t_obs}.png")


# ── C3: Cross-network generalisation ─────────────────────────────────────────

def plot_C3_cross_network(
    cls_pred: np.ndarray,
    cls_true: np.ndarray,
    network_types: np.ndarray,   # string array aligned with cls_pred / cls_true
    overall_acc: float,
    arch_name: str,
    t_obs: int,
    save_path: str = None,
):
    networks = [n for n in NET_COLORS if n in network_types]
    net_accs = {
        net: float((cls_pred[network_types == net] == cls_true[network_types == net]).mean())
        for net in networks
        if (network_types == net).any()
    }

    net_labels = list(net_accs.keys())
    net_vals   = [net_accs[n] for n in net_labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(net_labels, net_vals,
                  color=[NET_COLORS[n] for n in net_labels], width=0.5)
    ax.axhline(overall_acc, color="gray", linestyle="--", lw=1.5,
               label=f"t_obs={t_obs} accuracy ({overall_acc:.3f})")
    for bar, val in zip(bars, net_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", fontsize=FS_TICK)
    ax.set_ylabel("Accuracy", fontsize=FS_LABEL)
    ax.set_title(f"Classification Generalisation Across Network Types ({arch_name})",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_ylim(0, min(1.2, max(net_vals) * 1.25) if net_vals else 1.2)
    fig.tight_layout()
    save_fig(fig, save_path or f"results/dl/C3_cross_network_{arch_name}_t{t_obs}.png")


# ── C4: Per-class F1 bar chart ────────────────────────────────────────────────

def plot_C4_per_class_f1(
    cls_pred: np.ndarray,
    cls_true: np.ndarray,
    arch_name: str,
    t_obs: int,
    save_path: str = None,
):
    report = classification_report(cls_true, cls_pred,
                                   labels=list(range(len(_ID_TO_NAME))),
                                   output_dict=True, zero_division=0)
    f1_vals   = [report[str(mid)]["f1-score"] for mid in range(len(_ID_TO_NAME))]
    f1_colors = [MODEL_COLORS[n] for n in _ID_TO_NAME]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(_ID_TO_NAME, f1_vals, color=f1_colors)
    for i, val in enumerate(f1_vals):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=FS_TICK - 1)
    ax.set_ylabel("F1 Score", fontsize=FS_LABEL)
    ax.set_title(f"Per-Model Classification F1 Score (t_obs={t_obs}, {arch_name})",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, 1.18)
    fig.tight_layout()
    save_fig(fig, save_path or f"results/dl/C4_per_class_f1_{arch_name}_t{t_obs}.png")


# ── Attention weights (Transformer-specific) ──────────────────────────────────

def plot_attention_weights(model, sample_series: torch.Tensor, device,
                            save_path: str = None):
    from .models import TransformerEmbedder
    if not isinstance(model, TransformerEmbedder):
        return

    model.eval()
    with torch.no_grad():
        attn = model.get_attention_weights(sample_series.unsqueeze(0).to(device))

    attn_h0   = attn[0, 0].cpu().numpy()
    series_np = sample_series.cpu().numpy()
    t = len(series_np)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                              gridspec_kw={"height_ratios": [1, 3]})
    axes[0].plot(series_np, color="#2196F3")
    axes[0].set_xlim(-0.5, t - 0.5)
    axes[0].set_ylabel("I(t)/N", fontsize=FS_LABEL)
    axes[0].set_title("Input series", fontsize=FS_TITLE)
    axes[0].tick_params(labelsize=FS_TICK)

    im = axes[1].imshow(attn_h0, aspect="auto", origin="upper", cmap="viridis")
    axes[1].set_xlabel("Key timestep",   fontsize=FS_LABEL)
    axes[1].set_ylabel("Query timestep", fontsize=FS_LABEL)
    axes[1].set_title("Attention weights — layer 0, head 0", fontsize=FS_TITLE)
    axes[1].tick_params(labelsize=FS_TICK)
    plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    save_fig(fig, save_path or "results/dl/attention_weights.png")
