import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, rho_pred, rho_true, class_logits, model_ids):
        reg_loss = self.mse(rho_pred, rho_true)
        cls_loss = self.ce(class_logits, model_ids)
        total = self.alpha * reg_loss + (1.0 - self.alpha) * cls_loss
        mae = (rho_pred.detach() - rho_true.detach()).abs().mean()
        return total, reg_loss, cls_loss, mae
