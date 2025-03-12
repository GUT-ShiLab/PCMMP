import torch.nn as nn
import torch
import torchdiffeq
from torchdiffeq import odeint

class MLPRegressor(nn.Module):
    """Simple MLP regression model for predicting metabolite abundance."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Dimension of the microbiome data.
            hidden_dim (int): Dimension of the hidden layer in MLP.
            output_dim (int): Number of metabolite features to predict.
        """
        super(MLPRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SimpleConvMLPRegressor(nn.Module):
    """Conv1D + MLP regression model for predicting metabolite abundance."""

    def __init__(self, input_dim, conv_dims, mlp_dims, output_dim, kernel_size=3, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Number of microbiome features (input dimension).
            conv_dims (list[int]): List of output channels for Conv1D layers.
            mlp_dims (list[int]): List of hidden layer sizes for MLP.
            output_dim (int): Number of metabolite features to predict.
            kernel_size (int): Kernel size for Conv1D layers.
            dropout_rate (float): Dropout rate for MLP layers.
        """
        super(SimpleConvMLPRegressor, self).__init__()

        # 1D Convolutional Layers
        conv_layers = []
        in_channels = 1  # Conv1D expects (batch_size, channels, sequence_length)
        for out_channels in conv_dims:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))  # Down-sample with max pooling
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # MLP Layers
        mlp_layers = []
        conv_output_dim = (input_dim // (2 ** len(conv_dims))) * conv_dims[-1]  # Calculate flattened dimension
        in_features = conv_output_dim
        for hidden_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_features, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim

        mlp_layers.append(nn.Linear(in_features, output_dim))  # Output layer
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # Add channel dimension and pass through Conv1D layers
        x = x.unsqueeze(1)  # (batch_size, sequence_length) -> (batch_size, channels=1, sequence_length)
        x = self.conv(x)  # (batch_size, out_channels, reduced_length)

        # Flatten and pass through MLP layers
        x = x.flatten(start_dim=1)  # (batch_size, flattened_features)
        x = self.mlp(x)  # (batch_size, output_dim)
        return x