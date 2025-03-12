import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0

    for x_g, x_m in train_loader:  # No labels in this case
        x_g, x_m = x_g.to(device), x_m.to(device)

        # Forward pass
        predictions = model(x_g)

        # Compute loss (MSE)
        loss = criterion(predictions, x_m)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def compute_metrics(y_pred, y_true):
    """
    Compute RMSE, MAAPE, and NMSE.
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    abs_percentage_error = np.abs((y_true - y_pred) / y_true)
    maape = np.mean(np.arctan(abs_percentage_error)) * 2 / np.pi
    mse_val = np.mean((y_true - y_pred) ** 2)
    var_y = np.var(y_true)
    nmse = mse_val / var_y if var_y > 0 else np.nan  # Avoid division by zero
    return rmse, maape, nmse


def evaluate(model, dataloader, device):
    """
    Evaluate the MLP model on the test set for a single fold.
    """
    model.eval()
    predictions = []
    targets = []
    metabolite_names = dataloader.dataset.metabolome_data.columns  # Get metabolite names

    with torch.no_grad():
        for x_g, x_m in dataloader:
            x_g, x_m = x_g.to(device), x_m.to(device)
            preds = model(x_g)
            predictions.append(preds.cpu().numpy())
            targets.append(x_m.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate Spearman correlation for each metabolite
    spearman_corrs = []
    rmse_vals = []
    maape_vals = []
    nmse_vals = []
    for i in range(targets.shape[1]):  # Iterate over each metabolite
        corr, _ = spearmanr(predictions[:, i], targets[:, i])
        spearman_corrs.append(corr)

        # Compute additional metrics
        rmse, maape, nmse = compute_metrics(predictions[:, i], targets[:, i])
        rmse_vals.append(rmse)
        maape_vals.append(maape)
        nmse_vals.append(nmse)

    return metabolite_names, spearman_corrs, predictions, targets, rmse_vals, maape_vals, nmse_vals
