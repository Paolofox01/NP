from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvCNP"]


class ConvCNP(nn.Module):
    """
    Convolutional Neural Process (ConvCNP).

    This variant maps irregular context points to a regular grid using an
    RBF kernel, applies a convolutional stack on the grid, and interpolates
    the outputs back to target locations. It supports 1D, 2D, or 3D grids.
    Time can be handled either via basis functions or by using a 3D grid over
    (t, x, y) with Conv3d.

    Args:
        x_dim: Dimensionality of input coordinates.
            If time_basis_size > 0, expects [t, x] or [t, x, y].
            If use_time_grid is True, expects [t, x, y].
        y_dim: Dimensionality of outputs.
        grid_size: Grid resolution. For 1D, an int. For 2D, (H, W) or int.
            For 3D, (D, H, W) or int (applied to all dims).
        conv_channels: Number of channels in convolution layers, or a list.
        n_conv_layers: Number of convolution layers if conv_channels is int.
        kernel_size: Convolution kernel size.
        activation: Activation function class.
        dropout: Dropout probability after activations.
        is_normalized: Whether to apply normalization in conv blocks.
        norm_type: Normalization type: "layer" or "batch".
        grid_bounds: Optional bounds for the grid. For 1D: (min, max).
            For 2D: ((x_min, x_max), (y_min, y_max)). For 3D:
            ((t_min, t_max), (x_min, x_max), (y_min, y_max)). If None,
            bounds are inferred per batch from context and target points.
        density_sigma: RBF kernel length-scale for gridding. Can be float or
            per-dimension sequence.
        time_basis_size: Number of temporal basis functions. If > 0, time is
            read from the first coordinate and mixed into the context signal.
        use_time_grid: If True, treat time as a grid dimension and use Conv3d.
        time_basis_scale: RBF scale for temporal basis in normalized time.
        min_variance: Added to predicted variance for stability.
        eps: Small value to avoid division by zero.
        align_corners: Passed to grid_sample for coordinate alignment.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        grid_size: Union[int, Tuple[int, int]] = 128,
        conv_channels: Union[int, Sequence[int]] = 64,
        n_conv_layers: int = 4,
        kernel_size: int = 5,
        activation: type = nn.ReLU,
        dropout: float = 0.0,
        is_normalized: bool = True,
        norm_type: str = "layer",
        grid_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        density_sigma: Union[float, Sequence[float]] = 0.1,
        time_basis_size: int = 0,
        time_basis_scale: float = 0.2,
        use_time_grid: bool = False,
        min_variance: float = 1e-4,
        eps: float = 1e-6,
        align_corners: bool = True,
    ) -> None:
        super().__init__()

        self.use_time_grid = bool(use_time_grid)
        if self.use_time_grid:
            if time_basis_size != 0:
                raise ValueError("time_basis_size must be 0 when use_time_grid is True")
            if x_dim != 3:
                raise ValueError("use_time_grid=True expects x_dim=3 with [t, x, y]")
            self.time_basis_size = 0
            self.spatial_dim = 3
        else:
            self.time_basis_size = int(time_basis_size)
            if self.time_basis_size < 0:
                raise ValueError("time_basis_size must be >= 0")

            if self.time_basis_size > 0:
                self.spatial_dim = x_dim - 1
                if self.spatial_dim not in (1, 2):
                    raise ValueError(
                        f"ConvCNP supports 1D/2D spatial grids with time basis, got spatial_dim={self.spatial_dim}"
                    )
            else:
                self.spatial_dim = x_dim
                if self.spatial_dim not in (1, 2, 3):
                    raise ValueError(f"ConvCNP supports x_dim=1, 2, or 3, got {x_dim}")

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.grid_size = self._normalize_grid_size(grid_size)
        self.grid_bounds = self._normalize_grid_bounds(grid_bounds)
        self.density_sigma = self._normalize_density_sigma(density_sigma)
        self.time_basis_scale = float(time_basis_scale)
        self.min_variance = min_variance
        self.eps = eps
        self.align_corners = align_corners

        if self.time_basis_size > 0:
            time_centers = torch.linspace(0.0, 1.0, self.time_basis_size)
            self.register_buffer("time_centers", time_centers)
        else:
            self.register_buffer("time_centers", torch.empty(0))

        channels = self._normalize_conv_channels(conv_channels, n_conv_layers)
        if self.time_basis_size > 0:
            in_channels = 1 + y_dim * self.time_basis_size
            head_out_dim = 2 * y_dim * self.time_basis_size
        else:
            in_channels = 1 + y_dim
            head_out_dim = 2 * y_dim

        conv_layers = []
        if self.spatial_dim == 1:
            conv_cls = nn.Conv1d
            dropout_cls = nn.Dropout
        elif self.spatial_dim == 2:
            conv_cls = nn.Conv2d
            dropout_cls = nn.Dropout2d
        else:
            conv_cls = nn.Conv3d
            dropout_cls = nn.Dropout3d

        current_channels = in_channels
        for out_channels in channels:
            conv_layers.append(
                conv_cls(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                )
            )
            if is_normalized:
                conv_layers.append(self._make_norm_layer(out_channels, norm_type))
            conv_layers.append(activation())
            if dropout > 0:
                conv_layers.append(dropout_cls(dropout))
            current_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.head = conv_cls(current_channels, head_out_dim, kernel_size=1)

        self._initialize_weights()

    def _normalize_grid_size(self, grid_size: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if self.spatial_dim == 1:
            if isinstance(grid_size, (tuple, list)):
                if len(grid_size) != 1:
                    raise ValueError("grid_size must be int for spatial_dim=1")
                return (1, int(grid_size[0]))
            return (1, int(grid_size))

        if self.spatial_dim == 2:
            if isinstance(grid_size, (tuple, list)):
                if len(grid_size) != 2:
                    raise ValueError("grid_size must be (H, W) for x_dim=2")
                return (int(grid_size[0]), int(grid_size[1]))
            return (int(grid_size), int(grid_size))

        if self.spatial_dim == 3:
            if isinstance(grid_size, (tuple, list)):
                if len(grid_size) != 3:
                    raise ValueError("grid_size must be (D, H, W) for x_dim=3")
                return (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
            return (int(grid_size), int(grid_size), int(grid_size))

        return (int(grid_size), int(grid_size))

    def _normalize_grid_bounds(
        self,
        grid_bounds: Optional[Sequence[Tuple[float, float]]],
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        if grid_bounds is None:
            return None

        if self.spatial_dim == 1:
            if len(grid_bounds) == 2 and isinstance(grid_bounds[0], (int, float)):
                return ((float(grid_bounds[0]), float(grid_bounds[1])),)
            if (
                len(grid_bounds) == 1
                and isinstance(grid_bounds[0], (tuple, list))
                and len(grid_bounds[0]) == 2
            ):
                return ((float(grid_bounds[0][0]), float(grid_bounds[0][1])),)
            raise ValueError("grid_bounds must be (min, max) for spatial_dim=1")

        if self.spatial_dim == 2:
            if len(grid_bounds) != 2:
                raise ValueError("grid_bounds must be ((x_min,x_max),(y_min,y_max)) for x_dim=2")
            return tuple((float(a), float(b)) for a, b in grid_bounds)

        if len(grid_bounds) != 3:
            raise ValueError(
                "grid_bounds must be ((t_min,t_max),(x_min,x_max),(y_min,y_max)) for x_dim=3"
            )

        return tuple((float(a), float(b)) for a, b in grid_bounds)

    def _normalize_density_sigma(self, density_sigma: Union[float, Sequence[float]]) -> torch.Tensor:
        if isinstance(density_sigma, (tuple, list)):
            if len(density_sigma) != self.spatial_dim:
                raise ValueError("density_sigma must match spatial_dim")
            return torch.tensor(density_sigma, dtype=torch.float32)

        return torch.tensor([float(density_sigma)] * self.spatial_dim, dtype=torch.float32)

    def _normalize_conv_channels(
        self,
        conv_channels: Union[int, Sequence[int]],
        n_conv_layers: int,
    ) -> Tuple[int, ...]:
        if isinstance(conv_channels, (tuple, list)):
            if len(conv_channels) == 0:
                raise ValueError("conv_channels cannot be empty")
            return tuple(int(ch) for ch in conv_channels)

        if n_conv_layers < 1:
            raise ValueError("n_conv_layers must be >= 1")

        return tuple(int(conv_channels) for _ in range(n_conv_layers))

    def _make_norm_layer(self, channels: int, norm_type: str) -> nn.Module:
        if norm_type == "batch":
            if self.spatial_dim == 1:
                return nn.BatchNorm1d(channels)
            if self.spatial_dim == 2:
                return nn.BatchNorm2d(channels)
            return nn.BatchNorm3d(channels)
        if norm_type == "layer":
            return nn.GroupNorm(1, channels)
        raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        if isinstance(self.head, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            out_channels = self.head.out_channels // 2
            with torch.no_grad():
                self.head.weight[out_channels:, ...] *= 0.1
                if self.head.bias is not None:
                    self.head.bias[out_channels:] = -2.0

    def _build_grid(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, ...], torch.Tensor, torch.Tensor]:
        device = x_context.device
        dtype = x_context.dtype

        if self.grid_bounds is None:
            x_all = torch.cat([x_context, x_target], dim=1)
            x_min = x_all.amin(dim=(0, 1))
            x_max = x_all.amax(dim=(0, 1))
        else:
            bounds = torch.tensor(self.grid_bounds, device=device, dtype=dtype)
            x_min = bounds[:, 0]
            x_max = bounds[:, 1]

        span = torch.clamp(x_max - x_min, min=1e-6)
        x_max = x_min + span

        if self.spatial_dim == 1:
            width = self.grid_size[1]
            x_lin = torch.linspace(x_min[0], x_max[0], width, device=device, dtype=dtype)
            grid_coords = x_lin[:, None]
            return grid_coords, (1, width), x_min, x_max

        if self.spatial_dim == 2:
            height, width = self.grid_size
            x_lin = torch.linspace(x_min[0], x_max[0], width, device=device, dtype=dtype)
            y_lin = torch.linspace(x_min[1], x_max[1], height, device=device, dtype=dtype)
            y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing="ij")
            grid_coords = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
            return grid_coords, (height, width), x_min, x_max

        depth, height, width = self.grid_size
        t_lin = torch.linspace(x_min[0], x_max[0], depth, device=device, dtype=dtype)
        x_lin = torch.linspace(x_min[1], x_max[1], width, device=device, dtype=dtype)
        y_lin = torch.linspace(x_min[2], x_max[2], height, device=device, dtype=dtype)
        t_grid, y_grid, x_grid = torch.meshgrid(t_lin, y_lin, x_lin, indexing="ij")
        grid_coords = torch.stack([t_grid, x_grid, y_grid], dim=-1).reshape(-1, 3)
        return grid_coords, (depth, height, width), x_min, x_max

    def _context_to_grid(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        grid_coords: torch.Tensor,
        grid_shape: Tuple[int, ...],
        t_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_context, _ = x_context.shape
        sigma = self.density_sigma.to(device=x_context.device, dtype=x_context.dtype)
        diff = x_context.unsqueeze(2) - grid_coords.unsqueeze(0)
        dist2 = (diff / sigma).pow(2).sum(dim=-1)
        weights = torch.exp(-0.5 * dist2)

        density = weights.sum(dim=1)

        if self.time_basis_size > 0:
            if t_context is None:
                raise ValueError("t_context is required when time_basis_size > 0")
            basis = self._time_basis(t_context)
            basis_weights = weights.unsqueeze(-1) * basis.unsqueeze(2)
            basis_density = basis_weights.sum(dim=1)
            weighted_y = basis_weights.unsqueeze(-1) * y_context.unsqueeze(2).unsqueeze(3)
            signal = weighted_y.sum(dim=1) / (basis_density.unsqueeze(-1) + self.eps)
            signal = signal.reshape(batch_size, grid_coords.size(0), self.time_basis_size * self.y_dim)
        else:
            signal = (weights.unsqueeze(-1) * y_context.unsqueeze(2)).sum(dim=1)
            signal = signal / (density.unsqueeze(-1) + self.eps)

        grid_features = torch.cat([density.unsqueeze(-1), signal], dim=-1)

        if self.spatial_dim == 1:
            width = grid_shape[1]
            grid_features = grid_features.reshape(batch_size, width, -1)
            return grid_features.permute(0, 2, 1)

        if self.spatial_dim == 2:
            height, width = grid_shape
            grid_features = grid_features.reshape(batch_size, height, width, -1)
            return grid_features.permute(0, 3, 1, 2)

        depth, height, width = grid_shape
        grid_features = grid_features.reshape(batch_size, depth, height, width, -1)
        return grid_features.permute(0, 4, 1, 2, 3)

    def _normalize_coords(
        self,
        x: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.clamp(x_max - x_min, min=1e-6)
        return 2.0 * (x - x_min) / scale - 1.0

    def _make_sampling_grid(
        self,
        x_target: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self._normalize_coords(x_target, x_min, x_max)

        if self.spatial_dim == 1:
            x_norm = x_norm[..., 0]
            y_norm = torch.zeros_like(x_norm)
            grid = torch.stack([x_norm, y_norm], dim=-1)
            return grid.unsqueeze(2)

        if self.spatial_dim == 2:
            grid = torch.stack([x_norm[..., 0], x_norm[..., 1]], dim=-1)
            return grid.unsqueeze(2)

        grid = torch.stack([x_norm[..., 1], x_norm[..., 2], x_norm[..., 0]], dim=-1)
        return grid.unsqueeze(2).unsqueeze(3)

    def _split_time_space(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.time_basis_size > 0:
            return x[..., :1], x[..., -self.spatial_dim:]
        return None, x

    def _time_basis(self, t: torch.Tensor) -> torch.Tensor:
        t = t.squeeze(-1)
        centers = self.time_centers.to(device=t.device, dtype=t.dtype)
        scale = max(self.time_basis_scale, self.eps)
        return torch.exp(-0.5 * ((t[..., None] - centers) / scale) ** 2)

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass through the ConvCNP.

        Args:
            x_context: Context inputs of shape (batch_size, num_context, x_dim)
            y_context: Context outputs of shape (batch_size, num_context, y_dim)
            x_target: Target inputs of shape (batch_size, num_target, x_dim)
            y_target: Unused, kept for interface compatibility.

        Returns:
            Tuple[Tensor, Tensor]: predicted mean and predicted variance.
        """
        t_context, x_context_spatial = self._split_time_space(x_context)
        t_target, x_target_spatial = self._split_time_space(x_target)

        grid_coords, grid_shape, x_min, x_max = self._build_grid(x_context_spatial, x_target_spatial)
        grid_tensor = self._context_to_grid(
            x_context_spatial,
            y_context,
            grid_coords,
            grid_shape,
            t_context=t_context,
        )

        features = self.conv(grid_tensor)
        grid_out = self.head(features)

        if self.spatial_dim == 1:
            grid_out = grid_out.unsqueeze(2)

        grid_query = self._make_sampling_grid(x_target_spatial, x_min, x_max)
        sampled = F.grid_sample(
            grid_out,
            grid_query,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )

        # F.grid_sample outputs:
        # 1D/2D -> (B, C, num_target, 1)
        # 3D    -> (B, C, num_target, 1, 1)

        # Flatten the trailing spatial dimensions dynamically:
        sampled = sampled.view(batch_size, grid_out.size(1), num_target)
        sampled = sampled.permute(0, 2, 1)
        batch_size, num_target, _ = sampled.shape

        if self.time_basis_size > 0:
            if t_target is None:
                raise ValueError("t_target is required when time_basis_size > 0")
            basis = self._time_basis(t_target)
            basis = basis / (basis.sum(dim=-1, keepdim=True) + self.eps)
            sampled = sampled.reshape(batch_size, num_target, self.time_basis_size, 2 * self.y_dim)
            mu_k, raw_k = torch.chunk(sampled, 2, dim=-1)
            y_pred_mu = (mu_k * basis.unsqueeze(-1)).sum(dim=2)
            y_pred_raw = (raw_k * basis.unsqueeze(-1)).sum(dim=2)
        else:
            y_pred_mu, y_pred_raw = torch.chunk(sampled, 2, dim=-1)
        y_pred_var = self.min_variance + F.softplus(y_pred_raw)

        return y_pred_mu, y_pred_var
