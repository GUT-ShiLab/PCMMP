import torch
import torch.nn as nn

class MSELossNoContrastive(nn.Module):
    def __init__(self):
        super(MSELossNoContrastive, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Compute MSE loss for the regression task.
        """
        return self.mse_loss(predictions, targets)
