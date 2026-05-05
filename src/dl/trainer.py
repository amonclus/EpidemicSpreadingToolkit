import os
import copy
import torch
import numpy as np
from sklearn.metrics import f1_score

from .losses import MultiTaskLoss


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr: float = 1e-3,
        alpha: float = 0.5,
        patience: int = 15,
        device: str = "cpu",
        save_dir: str = "dl/checkpoints/",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = MultiTaskLoss(alpha=alpha)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = total_mae = total_correct = total_n = 0
        all_rho_pred, all_rho_true = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, rho, mid, feat in loader:
                x    = x.to(self.device)
                rho  = rho.to(self.device)
                mid  = mid.to(self.device)
                feat = feat.to(self.device)

                rho_pred, logits = self.model(x, feat)
                loss, _, _, mae = self.criterion(rho_pred, rho, logits, mid)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_mae += mae.item() * bs
                total_correct += (logits.argmax(1) == mid).sum().item()
                total_n += bs
                all_rho_pred.append(rho_pred.detach().cpu())
                all_rho_true.append(rho.detach().cpu())

        avg_loss = total_loss / total_n
        avg_mae = total_mae / total_n
        avg_acc = total_correct / total_n
        return avg_loss, avg_mae, avg_acc

    def train(self, n_epochs: int = 100, model_name: str = "model", t_obs: int = 30):
        history = {k: [] for k in ["train_loss", "val_loss", "train_mae", "val_mae", "train_acc", "val_acc"]}
        best_val_mae = float("inf")
        best_weights = None
        no_improve = 0
        early_stop_epoch = n_epochs

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_mae, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_mae, vl_acc = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(vl_mae)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_mae"].append(tr_mae)
            history["val_mae"].append(vl_mae)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            if vl_mae < best_val_mae:
                best_val_mae = vl_mae
                best_weights = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                ckpt = os.path.join(self.save_dir, f"{model_name}_t{t_obs}.pt")
                torch.save(best_weights, ckpt)
            else:
                no_improve += 1

            print(
                f"Epoch {epoch:3d} | "
                f"loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"MAE {tr_mae:.4f}/{vl_mae:.4f} | "
                f"acc {tr_acc:.3f}/{vl_acc:.3f}"
            )

            if no_improve >= self.patience:
                early_stop_epoch = epoch
                print(f"Early stopping at epoch {epoch}.")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        history["early_stop_epoch"] = early_stop_epoch
        return history

    def evaluate(self, loader):
        self.model.eval()
        all_rho_pred, all_rho_true, all_cls_pred, all_cls_true = [], [], [], []
        total_loss = total_n = 0

        with torch.no_grad():
            for x, rho, mid, feat in loader:
                x    = x.to(self.device)
                rho  = rho.to(self.device)
                mid  = mid.to(self.device)
                feat = feat.to(self.device)
                rho_pred, logits = self.model(x, feat)
                loss, _, _, _ = self.criterion(rho_pred, rho, logits, mid)

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_n += bs
                all_rho_pred.append(rho_pred.cpu().numpy())
                all_rho_true.append(rho.cpu().numpy())
                all_cls_pred.append(logits.argmax(1).cpu().numpy())
                all_cls_true.append(mid.cpu().numpy())

        rho_pred = np.concatenate(all_rho_pred)
        rho_true = np.concatenate(all_rho_true)
        cls_pred = np.concatenate(all_cls_pred)
        cls_true = np.concatenate(all_cls_true)

        mae = float(np.abs(rho_pred - rho_true).mean())
        ss_res = float(((rho_true - rho_pred) ** 2).sum())
        ss_tot = float(((rho_true - rho_true.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        accuracy = float((cls_pred == cls_true).mean())
        macro_f1 = float(f1_score(cls_true, cls_pred, average="macro", zero_division=0))

        return {
            "loss": total_loss / total_n,
            "MAE": mae,
            "R2": r2,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
