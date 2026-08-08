from typing import Type

import warnings

import torch
import torch.nn as nn

__all__ = ["Latent"]

class Latent(nn.Module):
    def __init__(self,
                 r_dim: int,
                 hidden_dim: int = 64,
                 z_dim: int = 64,
                 n_hidden: int = 1,
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 is_normalized: bool = True,
                 norm_type: str = 'layer'):
        super().__init__()
        
        # Store parameters as attributes for introspection
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = dropout
        self.is_normalized = is_normalized
        self.norm_type = norm_type
        
        layers = []
        
        # 1. Input Layer
        curr_dim = r_dim
        
        
        for i in range(n_hidden):
            if is_normalized:
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(curr_dim))
                elif norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(curr_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(activation())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            curr_dim = hidden_dim
        
        # 3. Output Layer
        layers.append(nn.Linear(curr_dim, 2 * z_dim))
        
        self.latent = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform representation to latent distribution and sample.
        
        Uses the reparameterization trick to sample from a Gaussian distribution
        parameterized by the network output.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, r_dim).
            
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - z: Sampled latent tensor of shape (batch_size, z_dim)
                - mu: Mean of latent distribution of shape (batch_size, z_dim)
                - log_var: Log-variance of latent distribution of shape (batch_size, z_dim)
        """
        
        z_params = self.latent(x)
        
        # Split into mean and log-variance
        z_output_1, z_output_2 = torch.chunk(z_params, 2, dim=-1)
        
        return z_output_1, z_output_2