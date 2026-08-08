from typing import Type, Optional, List

import torch
import torch.nn as nn

__all__ = ["MLP"]

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with flexible architecture.
    
    A fully connected neural network with configurable depth, width, activation,
    normalization, dropout, and residual connections.
    
    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output.
        hidden_dim (int, optional): Dimension of hidden layers. Default: 64.
        n_hidden (int, optional): Number of hidden layers. Default: 1.
        activation (Type[nn.Module], optional): Activation function class. Default: nn.ReLU.
        dropout (float, optional): Dropout probability (0 means no dropout). Default: 0.0.
        is_residual (bool, optional): Whether to use residual connections. Default: False.
        is_normalized (bool, optional): Whether to apply layer normalization. Default: True.
        norm_type (str, optional): Type of normalization ('layer' or 'batch'). Default: 'layer'.
        norm_position (str, optional): Position of normalization ('pre' or 'post'). Default: 'pre'.
            'pre': Normalize before linear transformation (standard)
            'post': Normalize after linear+activation (better for raw inputs)
    
    Example:
        >>> mlp = MLP(input_dim=10, output_dim=2, hidden_dim=128, n_hidden=3)
        >>> x = torch.randn(32, 10)  # batch_size=32, input_dim=10
        >>> out = mlp(x)  # shape: (32, 2)
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 n_hidden: int = 1,
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 is_residual: bool = False,
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
        self.is_residual = is_residual
        self.is_normalized = is_normalized
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.layers = nn.ModuleList()
        
        # Proiezioni per il residuo: servono se input_dim != hidden_dim
        # Le memorizziamo in un ModuleList per gestirle nel forward
        self.projections = nn.ModuleList()
        
        curr_dim = input_dim

        for i in range(n_hidden):
            block = []
            
            # 1. Pre-normalization (if norm_position == 'pre')
            if is_normalized and norm_position == 'pre':
                if norm_type == 'layer':
                    block.append(nn.LayerNorm(curr_dim))
                elif norm_type == 'batch':
                    block.append(nn.BatchNorm1d(curr_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            
            # 2. Linear + Activation + Dropout
            self.build_mlp_layer(block, curr_dim, hidden_dim, activation, dropout)
            
            # 3. Post-normalization (if norm_position == 'post')
            if is_normalized and norm_position == 'post':
                if norm_type == 'layer':
                    block.append(nn.LayerNorm(hidden_dim))
                elif norm_type == 'batch':
                    block.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            
            self.layers.append(nn.Sequential(*block))
            
            # 3. Gestione Proiezione Residua
            if is_residual:
                if curr_dim != hidden_dim:
                    # Se le dimensioni non coincidono, creiamo un layer lineare di "aggiustamento"
                    self.projections.append(nn.Linear(curr_dim, hidden_dim))
                else:
                    # Se coincidono, non serve trasformare l'identità
                    self.projections.append(nn.Identity())
            
            curr_dim = hidden_dim

        self.output_layer = nn.Linear(curr_dim, output_dim)

    def build_mlp_layer(self, layers: List[nn.Module], curr_dim: int, 
                        hidden_dim: int, activation: Type[nn.Module], 
                        dropout: float) -> None:
        """Build a single MLP layer block with linear, activation, and optional dropout."""
        layers.append(nn.Linear(curr_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for i, layer_block in enumerate(self.layers):
            if self.is_residual:
                # Applichiamo la proiezione all'identità per far combaciare le dimensioni
                identity = self.projections[i](x)
                out = layer_block(x)
                x = out + identity
            else:
                x = layer_block(x)
        
        return self.output_layer(x)