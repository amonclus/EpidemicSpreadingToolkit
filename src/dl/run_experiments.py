"""
Master script for the deep-learning epidemic pipeline.
Run from the src/ directory:
    python -m dl.run_experiments
"""
import os
import sys
import random
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# ── Toggle flags ────────────────────────────────────────────────────────────
TRAIN_CNN         = True
TRAIN_LSTM        = True
TRAIN_TRANSFORMER = True
TRAIN_ENSEMBLE    = True
TRAIN_MOE         = False
RUN_VISUALISATIONS = False
FORCE_RETRAIN     = False

T_OBS_VALUES  = [10, 20, 30, 50, 75]
T_OBS_DEFAULT = 30
N_EPOCHS      = 100
BATCH_SIZE    = 64
LR            = 1e-3
ALPHA         = 0.5
PATIENCE      = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dl")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "ml_data")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Seeds ────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ── Imports ──────────────────────────────────────────────────────────────────
# Support both `python -m dl.run_experiments` and `python dl/run_experiments.py`
if __package__ is None or __package__ == "":
    import pathlib
    _pkg_dir = str(pathlib.Path(__file__).parent.parent)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    from dl.dataset import get_dataloaders, EpidemicDataset
    from dl.models import CNNEmbedder, LSTMEmbedder, TransformerEmbedder
    from dl.trainer import Trainer
    from dl.mixture_of_experts import MixtureOfExperts
    from dl.wrappers import DLRegressor, DLClassifier
    from dl.visualisation import (
        plot_R1_mae_vs_window, plot_R2_pred_vs_true, plot_R3_per_model_mae,
        plot_R4_embedding_2d, plot_C1_accuracy_vs_window, plot_C2_confusion_matrix,
        plot_C3_cross_network, plot_C4_per_class_f1, plot_attention_weights,
    )
else:
    from .dataset import get_dataloaders, EpidemicDataset
    from .models import CNNEmbedder, LSTMEmbedder, TransformerEmbedder
    from .trainer import Trainer
    from .mixture_of_experts import MixtureOfExperts
    from .wrappers import DLRegressor, DLClassifier
    from .visualisation import (
        plot_R1_mae_vs_window, plot_R2_pred_vs_true, plot_R3_per_model_mae,
        plot_R4_embedding_2d, plot_C1_accuracy_vs_window, plot_C2_confusion_matrix,
        plot_C3_cross_network, plot_C4_per_class_f1, plot_attention_weights,
    )

# Derived after path setup — feature count depends on which graph columns are in the CSV
N_FEATURES = None  # resolved in main() from the first dataset

_ARCH_REGISTRY = {
    "CNN":         CNNEmbedder,
    "LSTM":        LSTMEmbedder,
    "Transformer": TransformerEmbedder,
}
_TRAIN_FLAGS = {
    "CNN":         lambda: TRAIN_CNN,
    "LSTM":        lambda: TRAIN_LSTM,
    "Transformer": lambda: TRAIN_TRANSFORMER,
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ckpt_path(arch_name: str, t_obs: int) -> str:
    return os.path.join(CKPT_DIR, f"{arch_name}_t{t_obs}.pt")


def _make_bundle(model, arch_name, t_obs, train_loader, n_features):
    """Bundle model weights + normalisation params into a single dict."""
    return {
        "state_dict": model.state_dict(),
        "norm_max":   float(train_loader.dataset.norm_max),
        "feat_mean":  train_loader.dataset.feat_mean,
        "feat_std":   train_loader.dataset.feat_std,
        "n_features": n_features,
        "arch_name":  arch_name,
        "t_obs":      t_obs,
    }


def load_model_bundle(arch_name: str, t_obs: int = T_OBS_DEFAULT):
    """Load a trained model with normalisation params — intended for app inference."""
    ckpt = _ckpt_path(arch_name, t_obs)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint: {ckpt}")
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "state_dict" not in data:
        raise ValueError(f"{ckpt} is in old format — retrain once to upgrade.")
    model = _ARCH_REGISTRY[arch_name](n_features=data["n_features"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    meta = {k: data[k] for k in ("norm_max", "feat_mean", "feat_std", "n_features", "arch_name", "t_obs")}
    return model, meta


def _load_or_train(arch_name: str, t_obs: int, train_loader, val_loader, n_features: int):
    ckpt  = _ckpt_path(arch_name, t_obs)
    model = _ARCH_REGISTRY[arch_name](n_features=n_features)

    if os.path.exists(ckpt) and not FORCE_RETRAIN:
        print(f"  Loading checkpoint: {ckpt}")
        data = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        if isinstance(data, dict) and "state_dict" in data:
            model.load_state_dict(data["state_dict"])
        else:
            # Old raw state_dict — load and upgrade to bundle format
            model.load_state_dict(data)
            torch.save(_make_bundle(model, arch_name, t_obs, train_loader, n_features), ckpt)
            print(f"  Upgraded checkpoint to bundle format.")
        model.to(DEVICE)
        return model, None

    print(f"  Training {arch_name} (t_obs={t_obs}, n_features={n_features}) on {DEVICE} ...")
    trainer = Trainer(
        model, train_loader, val_loader,
        lr=LR, alpha=ALPHA, patience=PATIENCE,
        device=DEVICE, save_dir=CKPT_DIR,
    )
    history = trainer.train(n_epochs=N_EPOCHS, model_name=arch_name, t_obs=t_obs)
    # Overwrite the raw state_dict the trainer saved with a full bundle
    torch.save(_make_bundle(model, arch_name, t_obs, train_loader, n_features), ckpt)
    return model, history


def _rf_baseline_mae(test_loader):
    rf_path   = os.path.join(DATA_DIR, "rf_regressor.pkl")
    feat_path = os.path.join(DATA_DIR, "feature_names.pkl")
    csv_path  = os.path.join(DATA_DIR, "ml_dataset.csv")

    if not os.path.exists(rf_path):
        print("  RF baseline not found, skipping.")
        return np.nan

    try:
        rf = joblib.load(rf_path)
    except Exception as e:
        print(f"  RF baseline could not be loaded ({e}), skipping.")
        return np.nan

    try:
        df = pd.read_csv(csv_path)
        test_indices = test_loader.dataset.indices
        test_df = df.iloc[test_indices]

        if os.path.exists(feat_path):
            feature_names = joblib.load(feat_path)
            available = [c for c in feature_names if c in test_df.columns]
            X_test = test_df[available].values
        else:
            exclude = {"rho_final", "model_id", "is_supercritical"}
            numeric = test_df.select_dtypes(include=[np.number]).columns.tolist()
            X_test = test_df[[c for c in numeric if c not in exclude]].values

        y_test = test_df["rho_final"].values
        preds = rf.predict(X_test)
        return float(np.abs(preds - y_test).mean())
    except Exception as e:
        print(f"  RF baseline prediction failed: {e}")
        return np.nan


def save_dl_models(arch_name: str = "CNN", t_obs: int = T_OBS_DEFAULT):
    """
    Wrap a trained DL model as sklearn-compatible objects and serialise to ml_data/:

      dl_regressor.pkl  — DLRegressor  (predicts rho_final from raw I(t)/N series)
      dl_classifier.pkl — DLClassifier (predicts model_id  from raw I(t)/N series)

    Mirrors ml_analysis.py's save_models() so the app can load both identically.
    Requires a bundle checkpoint (run run_experiments.py at least once first).
    """
    print(f"\n── SAVING DL MODELS ({arch_name}, t_obs={t_obs}) ───────────────────")
    ckpt = _ckpt_path(arch_name, t_obs)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No checkpoint found: {ckpt}\n"
            "Run run_experiments.py first to train the model."
        )

    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "state_dict" not in data:
        raise ValueError(
            f"{ckpt} is in old format (raw state_dict).\n"
            "Load it once via run_experiments.py to auto-upgrade, then retry."
        )

    kwargs = dict(
        state_dict = data["state_dict"],
        arch_name  = data["arch_name"],
        n_features = data["n_features"],
        t_obs      = data["t_obs"],
        norm_max   = data["norm_max"],
        feat_mean  = data["feat_mean"],
        feat_std   = data["feat_std"],
    )

    reg = DLRegressor(**kwargs)
    clf = DLClassifier(**kwargs)

    reg_path = os.path.join(DATA_DIR, "dl_regressor.pkl")
    clf_path = os.path.join(DATA_DIR, "dl_classifier.pkl")
    joblib.dump(reg, reg_path)
    joblib.dump(clf, clf_path)

    print(f"  dl_regressor.pkl  — DLRegressor  ({arch_name}, t_obs={t_obs}, "
          f"n_features={data['n_features']})")
    print(f"  dl_classifier.pkl — DLClassifier ({arch_name}, t_obs={t_obs}, "
          f"n_features={data['n_features']})")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # Verify data files exist
    for path in [
        os.path.join(DATA_DIR, "ml_dataset.csv"),
        os.path.join(DATA_DIR, "ml_I_series.npy"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            print("Run ml_scripts/generate_ml_dataset.py first.")
            sys.exit(1)

    all_results = {}   # {arch_name: {t_obs: metrics_dict}}
    all_histories = {} # {arch_name: {t_obs: history}}
    trained_models = {}  # {arch_name: {t_obs: model}}

    # Resolve feature count from one dataset (graph columns vary by CSV)
    _probe_ds = EpidemicDataset(t_obs=T_OBS_DEFAULT, split="train", seed=42)
    n_features = _probe_ds.features.shape[1]
    print(f"Feature vector size: {n_features} ({_probe_ds.features.shape[1] - 22} graph + 22 time-series)")

    # ── 1. Train / load all architectures at all t_obs values ────────────────
    for arch_name, ArchClass in _ARCH_REGISTRY.items():
        if not _TRAIN_FLAGS[arch_name]():
            continue
        all_results[arch_name] = {}
        all_histories[arch_name] = {}
        trained_models[arch_name] = {}

        for t_obs in T_OBS_VALUES:
            print(f"\n{'='*60}")
            print(f"Architecture: {arch_name} | t_obs={t_obs}")
            print(f"{'='*60}")
            train_loader, val_loader, test_loader = get_dataloaders(
                t_obs=t_obs, batch_size=BATCH_SIZE, seed=42
            )
            model, history = _load_or_train(arch_name, t_obs, train_loader, val_loader,
                                             n_features=n_features)
            trained_models[arch_name][t_obs] = model

            trainer = Trainer(model, train_loader, val_loader, device=DEVICE, save_dir=CKPT_DIR)
            metrics = trainer.evaluate(test_loader)
            all_results[arch_name][t_obs] = metrics
            if history is not None:
                all_histories[arch_name][t_obs] = history

            print(f"  Test MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}  "
                  f"Acc={metrics['accuracy']:.3f}  F1={metrics['macro_f1']:.3f}")

    # ── 1.5 Ensemble evaluation ───────────────────────────────────────────────
    _ENSEMBLE_ARCHS = ["CNN", "LSTM", "Transformer"]

    if TRAIN_ENSEMBLE and len(trained_models) >= 2:
        all_results["Ensemble"] = {}
        for t_obs in T_OBS_VALUES:
            available = [a for a in _ENSEMBLE_ARCHS
                         if a in trained_models and t_obs in trained_models[a]]
            if len(available) < 2:
                continue
            _, _, tl = get_dataloaders(t_obs=t_obs, batch_size=BATCH_SIZE, seed=42)

            per_arch_rho, per_arch_logits = [], []
            rho_true_arr = mid_true_arr = None
            for arch in available:
                m = trained_models[arch][t_obs]
                m.eval()
                rho_list, logit_list, rho_t, mid_t = [], [], [], []
                with torch.no_grad():
                    for x, rho, mid, feat in tl:
                        rp, lg = m(x.to(DEVICE), feat.to(DEVICE))
                        rho_list.append(rp.cpu().numpy())
                        logit_list.append(lg.cpu().numpy())
                        rho_t.append(rho.numpy())
                        mid_t.append(mid.numpy())
                per_arch_rho.append(np.concatenate(rho_list))
                per_arch_logits.append(np.concatenate(logit_list))
                rho_true_arr = np.concatenate(rho_t)
                mid_true_arr = np.concatenate(mid_t)

            ens_rho   = np.mean(per_arch_rho,    axis=0)
            ens_cls   = np.mean(per_arch_logits, axis=0).argmax(axis=1)
            mae  = float(np.abs(ens_rho - rho_true_arr).mean())
            ss_r = float(((rho_true_arr - ens_rho) ** 2).sum())
            ss_t = float(((rho_true_arr - rho_true_arr.mean()) ** 2).sum())
            r2   = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0
            acc  = float((ens_cls == mid_true_arr).mean())
            mf1  = float(f1_score(mid_true_arr, ens_cls, average="macro", zero_division=0))
            all_results["Ensemble"][t_obs] = {"MAE": mae, "R2": r2, "accuracy": acc, "macro_f1": mf1}
            print(f"  Ensemble ({'+'.join(available)}, t_obs={t_obs}) | MAE={mae:.4f}  Acc={acc:.3f}")

    # ── 2. Mixture of Experts at T_OBS_DEFAULT ───────────────────────────────
    moe_results = {}

    if TRAIN_MOE and trained_models:
        train_loader_def, val_loader_def, test_loader_def = get_dataloaders(
            t_obs=T_OBS_DEFAULT, batch_size=BATCH_SIZE, seed=42
        )

        x_tr, y_tr, mid_tr, feat_tr = [], [], [], []
        for x, rho, mid, feat in train_loader_def:
            x_tr.append(x); y_tr.append(rho.numpy())
            mid_tr.append(mid.numpy()); feat_tr.append(feat)
        x_train_tensor    = torch.cat(x_tr)
        feat_train_tensor = torch.cat(feat_tr)
        y_train_rho       = np.concatenate(y_tr)
        y_train_labels    = np.concatenate(mid_tr)

        x_te, y_te, feat_te = [], [], []
        for x, rho, _, feat in test_loader_def:
            x_te.append(x); y_te.append(rho.numpy()); feat_te.append(feat)
        x_test_tensor    = torch.cat(x_te)
        feat_test_tensor = torch.cat(feat_te)
        y_test_rho       = np.concatenate(y_te)

        for arch_name in list(trained_models.keys()):
            if T_OBS_DEFAULT not in trained_models[arch_name]:
                continue
            model = trained_models[arch_name][T_OBS_DEFAULT]
            print(f"\nFitting MoE specialists for {arch_name} ...")
            moe = MixtureOfExperts(gating_model=model)
            moe.fit_specialists(x_train_tensor, feat_train_tensor,
                                 y_train_rho, y_train_labels,
                                 specialist_type="rf", device=DEVICE)

            for mode in ("soft", "hard"):
                res = moe.evaluate(x_test_tensor, feat_test_tensor,
                                   y_test_rho, mode=mode, device=DEVICE)
                key = f"MoE_{mode}_{arch_name}"
                moe_results[key] = res
                print(f"  MoE-{mode} | MAE={res['MAE']:.4f}  R2={res['R2']:.4f}")

    # ── 3. Collect per-sample predictions at T_OBS_DEFAULT ───────────────────
    df_full = pd.read_csv(os.path.join(DATA_DIR, "ml_dataset.csv"))
    detail  = {}   # {arch_name: {rho_pred, rho_true, cls_pred, cls_true, net_types, mae, r2}}

    if trained_models:
        _, _, test_loader_def = get_dataloaders(
            t_obs=T_OBS_DEFAULT, batch_size=BATCH_SIZE, seed=42
        )
        test_indices = test_loader_def.dataset.indices

        for arch_name, models_by_t in trained_models.items():
            if T_OBS_DEFAULT not in models_by_t:
                continue
            model = models_by_t[T_OBS_DEFAULT]
            model.eval()
            rp_list, rt_list, cp_list, ct_list = [], [], [], []
            with torch.no_grad():
                for x, rho, mid, feat in test_loader_def:
                    x    = x.to(DEVICE)
                    feat = feat.to(DEVICE)
                    rho_pred, logits = model(x, feat)
                    rp_list.append(rho_pred.cpu().numpy())
                    rt_list.append(rho.numpy())
                    cp_list.append(logits.argmax(1).cpu().numpy())
                    ct_list.append(mid.numpy())

            rho_pred = np.concatenate(rp_list)
            rho_true = np.concatenate(rt_list)
            cls_pred = np.concatenate(cp_list)
            cls_true = np.concatenate(ct_list)
            net_types = df_full.iloc[test_indices]["network_type"].values

            mae = float(np.abs(rho_pred - rho_true).mean())
            ss_res = float(((rho_true - rho_pred) ** 2).sum())
            ss_tot = float(((rho_true - rho_true.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            detail[arch_name] = dict(
                rho_pred=rho_pred, rho_true=rho_true,
                cls_pred=cls_pred, cls_true=cls_true,
                net_types=net_types, mae=mae, r2=r2,
            )

        # Add ensemble to detail
        if TRAIN_ENSEMBLE and "Ensemble" in all_results and T_OBS_DEFAULT in all_results["Ensemble"]:
            available = [a for a in _ENSEMBLE_ARCHS if a in detail]
            if len(available) >= 2:
                ens_rho_pred = np.mean([detail[a]["rho_pred"] for a in available], axis=0)
                # recompute averaged logits via individual model forward passes
                per_logits = []
                for arch in available:
                    m = trained_models[arch][T_OBS_DEFAULT]; m.eval()
                    lg_list = []
                    with torch.no_grad():
                        for x, rho, mid, feat in test_loader_def:
                            _, lg = m(x.to(DEVICE), feat.to(DEVICE))
                            lg_list.append(lg.cpu().numpy())
                    per_logits.append(np.concatenate(lg_list))
                ens_cls_pred = np.mean(per_logits, axis=0).argmax(axis=1)
                rho_true_ref = detail[available[0]]["rho_true"]
                cls_true_ref = detail[available[0]]["cls_true"]
                net_ref      = detail[available[0]]["net_types"]
                ens_mae = float(np.abs(ens_rho_pred - rho_true_ref).mean())
                ss_r = float(((rho_true_ref - ens_rho_pred) ** 2).sum())
                ss_t = float(((rho_true_ref - rho_true_ref.mean()) ** 2).sum())
                ens_r2 = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0
                detail["Ensemble"] = dict(
                    rho_pred=ens_rho_pred, rho_true=rho_true_ref,
                    cls_pred=ens_cls_pred, cls_true=cls_true_ref,
                    net_types=net_ref, mae=ens_mae, r2=ens_r2,
                )

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    if RUN_VISUALISATIONS:
        print("\nGenerating visualisations ...")

        # R1: MAE vs observation window
        plot_R1_mae_vs_window(
            all_results, T_OBS_VALUES,
            save_path=os.path.join(RESULTS_DIR, "R1_mae_vs_window.png"),
        )

        # C1: Accuracy vs observation window
        plot_C1_accuracy_vs_window(
            all_results, T_OBS_VALUES,
            save_path=os.path.join(RESULTS_DIR, "C1_accuracy_vs_window.png"),
        )

        # Per-architecture plots at T_OBS_DEFAULT
        _, _, tl_default = get_dataloaders(t_obs=T_OBS_DEFAULT, batch_size=BATCH_SIZE, seed=42)
        for arch_name, d in detail.items():
            t = T_OBS_DEFAULT
            results_for_arch = all_results.get(arch_name, {})
            overall_acc = results_for_arch.get(t, {}).get("accuracy",
                          float((d["cls_pred"] == d["cls_true"]).mean()))

            plot_R2_pred_vs_true(
                d["rho_pred"], d["rho_true"], d["cls_true"],
                d["mae"], d["r2"], arch_name, t,
                save_path=os.path.join(RESULTS_DIR, f"R2_pred_vs_true_{arch_name}.png"),
            )
            plot_R3_per_model_mae(
                d["rho_pred"], d["rho_true"], d["cls_true"],
                arch_name, t,
                save_path=os.path.join(RESULTS_DIR, f"R3_per_model_mae_{arch_name}.png"),
            )
            # Embedding plot only for single-model architectures (not ensemble)
            if arch_name in trained_models and t in trained_models[arch_name]:
                plot_R4_embedding_2d(
                    trained_models[arch_name][t], tl_default, DEVICE, arch_name, t,
                    save_path=os.path.join(RESULTS_DIR, f"R4_embedding_{arch_name}.png"),
                )
            plot_C2_confusion_matrix(
                d["cls_pred"], d["cls_true"], arch_name, t,
                save_path=os.path.join(RESULTS_DIR, f"C2_confusion_matrix_{arch_name}.png"),
            )
            plot_C3_cross_network(
                d["cls_pred"], d["cls_true"], d["net_types"],
                overall_acc=overall_acc,
                arch_name=arch_name, t_obs=t,
                save_path=os.path.join(RESULTS_DIR, f"C3_cross_network_{arch_name}.png"),
            )
            plot_C4_per_class_f1(
                d["cls_pred"], d["cls_true"], arch_name, t,
                save_path=os.path.join(RESULTS_DIR, f"C4_per_class_f1_{arch_name}.png"),
            )

        # Attention weights for Transformer
        if "Transformer" in trained_models and T_OBS_DEFAULT in trained_models["Transformer"]:
            _, _, tl1 = get_dataloaders(t_obs=T_OBS_DEFAULT, batch_size=1, seed=42)
            sample_x, *_ = next(iter(tl1))
            plot_attention_weights(
                trained_models["Transformer"][T_OBS_DEFAULT],
                sample_x[0], DEVICE,
                save_path=os.path.join(RESULTS_DIR, "attention_weights.png"),
            )

    # ── 5. Summary table ──────────────────────────────────────────────────────
    rows = []
    for t_obs in T_OBS_VALUES:
        _, _, tl = get_dataloaders(t_obs=t_obs, batch_size=BATCH_SIZE, seed=42)
        rf_mae = _rf_baseline_mae(tl)
        rows.append({"Model": "HGB_baseline", "t_obs": t_obs,
                     "MAE": f"{rf_mae:.4f}" if not np.isnan(rf_mae) else "-",
                     "R2": "-", "Accuracy": "-", "F1": "-"})
        for arch_name in all_results:
            if t_obs not in all_results[arch_name]:
                continue
            m = all_results[arch_name][t_obs]
            rows.append({
                "Model": arch_name, "t_obs": t_obs,
                "MAE": f"{m['MAE']:.4f}", "R2": f"{m['R2']:.4f}",
                "Accuracy": f"{m['accuracy']:.3f}", "F1": f"{m['macro_f1']:.3f}",
            })
        if t_obs == T_OBS_DEFAULT:
            for key, res in moe_results.items():
                rows.append({
                    "Model": key, "t_obs": t_obs,
                    "MAE": f"{res['MAE']:.4f}", "R2": f"{res['R2']:.4f}",
                    "Accuracy": "-", "F1": "-",
                })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    summary_str = summary_df.to_string(index=False)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary_str)
    with open(summary_path, "w") as f:
        f.write(summary_str + "\n")
    print(f"\nSummary saved to {summary_path}")

    # ── 6. Export best model for app use ─────────────────────────────────────
    best_arch = "CNN"  # change to whichever arch performs best after ensemble eval
    if os.path.exists(_ckpt_path(best_arch, T_OBS_DEFAULT)):
        save_dl_models(arch_name=best_arch, t_obs=T_OBS_DEFAULT)


if __name__ == "__main__":
    main()
