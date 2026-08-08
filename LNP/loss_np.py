import torch
import torch.nn as nn
from typing import Tuple


class GuassianNLL_loss(nn.Module):
    """Gaussian negative log-likelihood loss for conditional models."""

    def __init__(self, min_variance: float = 1e-6):
        super().__init__()
        self.min_variance = min_variance

    def forward(
        self,
        y_pred_mu: torch.Tensor,
        y_pred_var: torch.Tensor,
        y_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Gaussian negative log-likelihood.

        Args:
            y_pred_mu: Predicted mean, shape (batch_size, ..., y_dim)
            y_pred_var: Predicted variance, shape (batch_size, ..., y_dim)
            y_target: Ground truth, same shape as y_pred_mu

        Returns:
            Scalar mean NLL.
        """
        y_pred_var = torch.clamp(y_pred_var, min=self.min_variance)
        nll = 0.5 * (
            torch.log(torch.tensor(2 * torch.pi, device=y_pred_mu.device, dtype=y_pred_mu.dtype))
            + torch.log(y_pred_var)
            + (y_target - y_pred_mu) ** 2 / y_pred_var
        )
        return nll.mean()


class ELBOLossNP(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for Latent Neural Processes.
    
    Computes the negative ELBO which consists of:
    1. Reconstruction term: -E_{q(z|context)}[log p(y_target|z, x_target)]
    2. KL divergence: KL[q(z|context) || p(z)]
    
    The loss is: -ELBO = -E[log p(y|z)] + KL[q(z|target) || q(z|context)]
    
    References:
        Garnelo, Marta, et al. "Neural processes." arXiv preprint arXiv:1807.01622 (2018).
    """
    
    def __init__(self, beta: float = 1.0, lambda_param: float = 1.0):
        """
        Initialize ELBO loss.
        
        Args:
            beta: Weight for KL divergence term (beta-VAE formulation). Default: 1.0
            lambda_param: Weight for parameter loss term. Default: 1.0
        """
        super().__init__()
        self.beta = beta
        self.lambda_param = lambda_param  # <-- missing
    
    def forward(
        self,
        y_pred_mu: torch.Tensor,
        y_pred_var: torch.Tensor,
        y_target: torch.Tensor,
        z_context_mu: torch.Tensor,
        z_context_var: torch.Tensor,
        z_target_mu: torch.Tensor,
        z_target_var: torch.Tensor,
        param_mu=None, param_var=None, theta_true=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute negative ELBO loss for Neural Processes.
        
        Args:
            y_pred_mu: Predicted mean of target outputs, shape (batch_size * num_target, y_dim)
            y_pred_var: Predicted variance of target outputs, shape (batch_size * num_target, y_dim)
            y_target: Ground truth target outputs, shape (batch_size * num_target, y_dim)
            z_context_mu: Mean of q(z|context), shape (batch_size, z_dim)
            z_context_var: Variance of q(z|context), shape (batch_size, z_dim)
            z_target_mu: Mean of q(z|context,target), shape (batch_size, z_dim)
            z_target_var: Variance of q(z|context,target), shape (batch_size, z_dim)
            param_mu: Mean of parameter distribution, shape (S, B, param_dim) or (B, param_dim)
            param_var: Variance of parameter distribution, shape (S, B, param_dim) or (B, param_dim)
            theta_true: True parameter values, shape (B, param_dim)

        Returns:
            Tuple containing:
                - total_loss: Negative ELBO (reconstruction + beta * KL)
                - reconstruction_loss: Negative log-likelihood term
                - kl_loss: KL divergence term
                - param_loss: Parameter loss term
        """
        reconstruction_loss = self.gaussian_nll(y_target, y_pred_mu, y_pred_var)
        kl_loss = self.kl_divergence_gaussians(
            z_target_mu, z_target_var, z_context_mu, z_context_var
        )

        param_loss = torch.tensor(0.0, device=y_pred_mu.device)
        if param_mu is not None and theta_true is not None:
            # theta_true: (B, param_dim), param_mu: (S, B, param_dim) or (B, param_dim)
            if param_mu.dim() == 3:
                theta_true = theta_true.unsqueeze(0)   # (1, B, param_dim)
            param_var_clamped = param_var.clamp(min=1e-6)
            nll_param = 0.5 * (
                torch.log(param_var_clamped)
                + (theta_true - param_mu) ** 2 / param_var_clamped
            )
            param_loss = nll_param.mean()

        total_loss = reconstruction_loss + self.beta * kl_loss + self.lambda_param * param_loss
        return total_loss, reconstruction_loss, kl_loss, param_loss
    
    def gaussian_nll(
        self,
        y_true: torch.Tensor,
        y_mu: torch.Tensor,
        y_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood.
        
        Args:
            y_true: Ground truth, shape (batch_size, y_dim)
            y_mu: Predicted mean, shape (batch_size, y_dim)
            y_var: Predicted variance, shape (batch_size, y_dim)
            
        Returns:
            Mean negative log-likelihood across batch
        """
        # -log p(y | mu, var) = 0.5 * [log(2π) + log(var) + (y - mu)^2 / var]
        
        # Clamp variance for numerical stability
        y_var = torch.clamp(y_var, min=1e-6)
        
        mse = (y_true - y_mu) ** 2
        nll = 0.5 * (torch.log(torch.tensor(2 * torch.pi)) + torch.log(y_var) + mse / y_var)
        
        # Sum over dimensions, mean over batch
        return nll.mean()
    
    def kl_divergence_gaussians(
        self,
        mu_q: torch.Tensor,
        var_q: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussian distributions.
        
        KL[q(z) || p(z)] where both are Gaussian with diagonal covariance.
        For Neural Processes: KL[q(z|context,target) || q(z|context)]
        
        Args:
            mu_q: Mean of q(z|context,target), shape (batch_size, z_dim)
            var_q: Variance of q(z|context,target), shape (batch_size, z_dim)
            mu_p: Mean of q(z|context), shape (batch_size, z_dim)
            var_p: Variance of q(z|context), shape (batch_size, z_dim)
            
        Returns:
            Mean KL divergence across batch
        """
        # KL[N(mu_q, var_q) || N(mu_p, var_p)] = 
        # 0.5 * sum[log(var_p/var_q) + (var_q + (mu_q - mu_p)^2) / var_p - 1]
        # = 0.5 * sum[log(var_p/var_q) + var_q/var_p + (mu_q - mu_p)^2/var_p - 1]
        
        # Convert log_var to var
        log_var_q = torch.log(var_q)
        log_var_p = torch.log(var_p)
        
        kl = 0.5 * (
            log_var_p - log_var_q  # log(var_p / var_q)
            + var_q / var_p  # var_q / var_p
            + (mu_q - mu_p) ** 2 / var_p  # (mu_q - mu_p)^2 / var_p
            - 1.0
        )
        
        # Sum over dimensions, mean over batch
        return kl.mean()


class MSELoss(nn.Module):
    """Simple MSE loss for baseline comparison."""
    
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared error.
        
        Args:
            y_pred: Predicted outputs, shape (batch_size, y_dim)
            y_target: Ground truth outputs, shape (batch_size, y_dim)
            
        Returns:
            Mean squared error
        """
        return torch.mean((y_pred - y_target) ** 2) 