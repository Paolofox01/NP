import torch
import torch.nn as nn
import numpy as np

__all__ = ["FourierFeatures", "LearnableFourierFeatures"]

class FourierFeatures(nn.Module):
    """
    Random Fourier Features for positional encoding.
    
    Applies random Fourier feature mapping to input coordinates, which helps
    neural networks learn high-frequency functions more effectively.
    
    The transformation is:
        gamma(x) = [sin(2π B x), cos(2π B x)]
    
    where B is a random matrix sampled from N(0, scale^2 I).
    
    References:
        - Rahimi & Recht (2007): "Random Features for Large-Scale Kernel Machines"
        - Tancik et al. (2020): "Fourier Features Let Networks Learn High Frequency Functions"
    
    Args:
        input_dim: Dimension of input coordinates (e.g., 2 for 2D images)
        num_frequencies: Number of random frequencies to use
        scale: Standard deviation of the random Gaussian matrix B.
               Controls the range of frequencies sampled.
               Higher scale -> higher frequencies
        learnable: If True, the frequency matrix B is learnable. Default: False
    
    Input shape: (batch_size, num_points, input_dim)
    Output shape: (batch_size, num_points, 2 * num_frequencies)
    
    Example:
        >>> fourier = FourierFeatures(input_dim=2, num_frequencies=64, scale=10.0)
        >>> x = torch.randn(32, 100, 2)  # batch of 100 2D points
        >>> features = fourier(x)
        >>> print(features.shape)  # (32, 100, 128)
    """
    
    def __init__(self, input_dim: int, num_frequencies: int = 256, 
                 scale: float = 1.0, learnable: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.scale = scale
        
        # Random Gaussian matrix for frequency projection
        B = torch.randn(input_dim, num_frequencies) * scale
        
        if learnable:
            # Make B a learnable parameter
            self.B = nn.Parameter(B)
        else:
            # Register as buffer (part of state_dict but not trained)
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random Fourier feature transformation.
        
        Args:
            x: Input coordinates of shape (batch_size, num_points, input_dim)
        
        Returns:
            Fourier features of shape (batch_size, num_points, 2 * num_frequencies)
        """
        # Project input to frequency space
        # x: (batch, num_points, input_dim)
        # B: (input_dim, num_frequencies)
        # x_proj: (batch, num_points, num_frequencies)
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        
        # Apply sin and cos to get Fourier features
        # Output: (batch, num_points, 2 * num_frequencies)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def extra_repr(self) -> str:
        """String representation for print(model)"""
        return f'input_dim={self.input_dim}, num_frequencies={self.num_frequencies}, scale={self.scale}'


class LearnableFourierFeatures(nn.Module):
    """
    Learnable Fourier Features with trainable frequencies and phases.
    
    Unlike random Fourier features, this version learns the optimal
    frequencies and phases during training.
    
    The transformation is:
        gamma(x) = [sin(2π (W x + b)), cos(2π (W x + b))]
    
    where W and b are learnable parameters.
    
    Args:
        input_dim: Dimension of input coordinates
        num_frequencies: Number of frequencies to learn
        init_scale: Initial scale for weight initialization
    
    Input shape: (batch_size, num_points, input_dim)
    Output shape: (batch_size, num_points, 2 * num_frequencies)
    """
    
    def __init__(self, input_dim: int, num_frequencies: int = 256, 
                 init_scale: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        
        # Learnable frequency weights
        self.W = nn.Parameter(torch.randn(input_dim, num_frequencies) * init_scale)
        
        # Learnable phase shifts
        self.b = nn.Parameter(torch.zeros(num_frequencies))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable Fourier feature transformation.
        
        Args:
            x: Input coordinates of shape (batch_size, num_points, input_dim)
        
        Returns:
            Fourier features of shape (batch_size, num_points, 2 * num_frequencies)
        """
        # x: (batch, num_points, input_dim)
        # W: (input_dim, num_frequencies)
        # x_proj: (batch, num_points, num_frequencies)
        x_proj = 2 * np.pi * (torch.matmul(x, self.W) + self.b)
        
        # Apply sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, num_frequencies={self.num_frequencies}'
