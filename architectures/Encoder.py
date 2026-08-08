from typing import Type

import warnings

import torch
import torch.nn as nn


__all__ = ["Encoder"]

class Encoder(nn.Module):
    """
    Encoder network for transforming input features to a latent representation.
    
    A fully connected neural network that encodes input data into a latent space.
    Supports configurable depth, width, activation, normalization, and dropout.
    
    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of encoded output (latent dimension).
        hidden_dim (int, optional): Dimension of hidden layers. Default: 64.
        n_hidden (int, optional): Number of hidden layers. Default: 1.
        activation (Type[nn.Module], optional): Activation function class. Default: nn.ReLU.
        dropout (float, optional): Dropout probability (0 means no dropout). Default: 0.0.
        is_normalized (bool, optional): Whether to apply normalization. Default: True.
        norm_type (str, optional): Type of normalization ('layer' or 'batch'). Default: 'layer'.
        norm_position (str, optional): Position of normalization ('pre' or 'post'). Default: 'pre'.
    
    Example:
        >>> encoder = Encoder(input_dim=10, output_dim=32, hidden_dim=128, n_hidden=2)
        >>> x = torch.randn(16, 10)  # batch_size=16, input_dim=10
        >>> latent = encoder(x)  # shape: (16, 32)
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 n_hidden: int = 1,
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 is_normalized: bool = True,
                 norm_type: str = 'layer',
                 norm_position: str = 'pre'):
        super().__init__()
        
        # Store parameters as attributes for introspection
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = dropout
        self.is_normalized = is_normalized
        self.norm_type = norm_type
        self.norm_position = norm_position
        
        layers = []
        
        # 1. Input Layer
        curr_dim = input_dim
        
        
        for i in range(n_hidden):
            # Pre-normalization
            if is_normalized and norm_position == 'pre':
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(curr_dim))
                elif norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(curr_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(activation())
            
            # Post-normalization
            if is_normalized and norm_position == 'post':
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            curr_dim = hidden_dim
        
        # 3. Output Layer
        layers.append(nn.Linear(curr_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to latent representation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, output_dim).
        """
        return self.encoder(x)