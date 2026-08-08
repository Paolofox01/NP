from typing import Optional, Type

import warnings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Added DeepONetDecoder to the imports
from architectures import Decoder, DeepONetDecoder, Encoder, Latent, MLP, FourierFeatures, LearnableFourierFeatures

import torch
import torch.nn as nn


class LatNP_simple_2(nn.Module):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 r_dim: int,
                 hidden_dim: int = 64,
                 z_dim: int = 64,
                 n_hidden: int = 1,
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 is_normalized: bool = True,
                 norm_type: str = 'layer',
                 fourier_vars: Optional[int] = 0,
                 include_raw_x_with_fourier: bool = True,
                 num_frequencies: int = 64,
                 fourier_scale: float = 10.0,
                 learnable_fourier: bool = False,
                 use_deeponet_decoder: bool = False,  # <--- Added toggle
                 p: int = 128,                        # <--- Added DeepONet projection dimension
                 t_dim: int = 1,                       # <--- Fixed number of context timesteps
                 coeff_per_channel: bool = True):       # <--- coeff_net output granularity
        super().__init__()

        # t_dim must be known at construction time because coeff_net (below) is
        # applied per-timestep over a fixed number of timesteps; it's an
        # architecture hyperparameter, not something derivable purely from a
        # single forward() call's tensor shapes.
        self.t_dim = t_dim

        self.use_deeponet_decoder = use_deeponet_decoder
        self.coeff_per_channel = coeff_per_channel

        # === Fourier Features for positional encoding ===
        self.use_fourier = fourier_vars is not None
        if self.use_fourier:
            if fourier_vars == 0:
                fourier_vars = x_dim  # Default to using all dimensions for Fourier features
            if fourier_vars < 0 or fourier_vars > x_dim:
                raise ValueError(f"fourier_vars must be in [0, {x_dim}], got {fourier_vars}")
            if learnable_fourier:
                self.fourier = LearnableFourierFeatures(input_dim=fourier_vars,
                                                        num_frequencies=num_frequencies,
                                                        init_scale=fourier_scale)
            else:
                self.fourier = FourierFeatures(input_dim=fourier_vars,
                                          num_frequencies=num_frequencies,
                                          scale=fourier_scale,
                                          learnable=False)
            if include_raw_x_with_fourier:
                fourier_dim = x_dim + 2 * num_frequencies
            else:
                fourier_dim = (x_dim - fourier_vars) + 2 * num_frequencies
        else:
            self.fourier = None
            fourier_dim = x_dim

        self.fourier_y = None
        fourier_y_dim = y_dim

        # === IMPROVED: Direct encoding of (x,y) pairs ===
        self.context_encoder = MLP(input_dim=fourier_dim + fourier_y_dim, output_dim=r_dim,
                                   hidden_dim=hidden_dim, n_hidden=n_hidden,
                                   activation=activation, dropout=0.0,
                                   is_normalized=is_normalized, norm_type=norm_type,
                                   norm_position='post')

        # coeff_net: produces an unconstrained coefficient for each timestep,
        # purely as a function of that timestep's own raw time value. Applied
        # independently per timestep: (batch, t_dim, 1) -> (batch, t_dim, C)
        # where C = r_dim if coeff_per_channel else 1 (broadcasts over r_dim).
        # This is deliberately decoupled from r_timestep_mean (the "basis"
        # being weighted) so the coefficient pathway can't be circular with
        # the value pathway it's meant to combine.
        coeff_out_dim = r_dim if self.coeff_per_channel else 1
        self.coeff_net = MLP(input_dim=1, output_dim=coeff_out_dim,
                             hidden_dim=hidden_dim, n_hidden=n_hidden,
                             activation=activation, dropout=0.0,
                             is_normalized=is_normalized, norm_type=norm_type,
                             norm_position='post')

        # === IMPROVED: Dual path - deterministic anchor ===
        self.deterministic_agg = nn.Sequential(
            nn.Linear(r_dim, r_dim),
            nn.LayerNorm(r_dim) if is_normalized else nn.Identity(),
            activation(),
            nn.Dropout(dropout)
        )

        # Stochastic path: latent variable
        self.latent = Latent(r_dim=r_dim, hidden_dim=hidden_dim, z_dim=z_dim,
                             n_hidden=n_hidden, activation=activation, dropout=0.0,
                             is_normalized=is_normalized, norm_type=norm_type)

        # === IMPROVED: Decoder Selection Routing ===
        if self.use_deeponet_decoder:
            self.decoder = DeepONetDecoder(branch_dim=z_dim + r_dim,  # z + deterministic anchor
                                           trunk_dim=fourier_dim,     # purely spatial coordinates
                                           output_dim=y_dim,
                                           p=p,
                                           hidden_dim=hidden_dim,
                                           n_hidden=n_hidden + 1,
                                           activation=activation,
                                           dropout=dropout,
                                           is_normalized=is_normalized,
                                           norm_type=norm_type)
        else:
            self.decoder = Decoder(input_dim=z_dim + r_dim + fourier_dim,
                                   output_dim=y_dim, hidden_dim=hidden_dim,
                                   n_hidden=n_hidden + 1, activation=activation, dropout=dropout,
                                   is_normalized=is_normalized, norm_type=norm_type)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.fourier_dim = fourier_dim
        self.fourier_vars = fourier_vars
        self.include_raw_x_with_fourier = include_raw_x_with_fourier
        self.fourier_y_dim = fourier_y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim

        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights for stable Neural Process training.
        Supports both Standard MLP Decoders and DeepONet Decoders.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Decoder Initialization
        if hasattr(self.decoder, 'branch_var_head'):
            for head in [self.decoder.branch_var_head, self.decoder.trunk_var_head]:
                final_layer = None
                for layer in reversed(list(head.modules())):
                    if isinstance(layer, nn.Linear):
                        final_layer = layer
                        break

                if final_layer is not None:
                    with torch.no_grad():
                        final_layer.weight *= 0.1

        elif hasattr(self.decoder, 'decoder') and len(self.decoder.decoder) > 0:
            final_layer = None
            for layer in reversed(list(self.decoder.decoder.modules())):
                if isinstance(layer, nn.Linear):
                    final_layer = layer
                    break

            if final_layer is not None:
                output_dim = final_layer.out_features // 2
                with torch.no_grad():
                    final_layer.weight[output_dim:, :] *= 0.1
                    if final_layer.bias is not None:
                        final_layer.bias[output_dim:] = -2.0

        if hasattr(self.latent, 'latent') and len(self.latent.latent) > 0:
            latent_final_layer = None
            for layer in reversed(list(self.latent.latent.modules())):
                if isinstance(layer, nn.Linear):
                    latent_final_layer = layer
                    break

            if latent_final_layer is not None:
                z_dim = latent_final_layer.out_features // 2
                with torch.no_grad():
                    latent_final_layer.weight[z_dim:, :] *= 0.1
                    if latent_final_layer.bias is not None:
                        latent_final_layer.bias[z_dim:] = -2.0

        # coeff_net final-layer scaling: keep initial coefficients small so the
        # unconstrained weighted sum over timesteps doesn't produce large
        # activations before the network has learned anything useful. This
        # mirrors the down-scaling already applied to the variance heads above.
        if hasattr(self, 'coeff_net'):
            coeff_final_layer = None
            for layer in reversed(list(self.coeff_net.modules())):
                if isinstance(layer, nn.Linear):
                    coeff_final_layer = layer
                    break
            if coeff_final_layer is not None:
                with torch.no_grad():
                    coeff_final_layer.weight *= 0.1

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor,
                t_context: torch.Tensor, context_mask: torch.Tensor,
                x_target: torch.Tensor, y_target: torch.Tensor = None,
                t_target: torch.Tensor = None,
                num_samples: int = 1) -> tuple:
        """
        Shapes:
            x_context:    (batch, t_dim, num_context, x_dim)
            y_context:    (batch, t_dim, num_context, y_dim)
            t_context:    (batch, t_dim) — raw time coordinate for each of the
                          model's fixed t_dim timesteps.
            context_mask: (batch, t_dim, num_context) — 1 for a valid/observed
                          context point, 0 for a padded one.
            x_target:     (batch, num_target, x_dim)
            y_target:     (batch, num_target, y_dim), optional
            t_target:     (batch, num_target) — raw time coordinate for each target point.
        """

        batch_size, t_num, num_context, x_dim = x_context.shape

        # x_target has no time dimension (it's queried independently of the
        # context timesteps), so this unpacks as 3-D, not 4-D.
        _, num_target, _ = x_target.shape
        _, _, _, y_dim = y_context.shape

        # === STEP 0: Apply Fourier features to x coordinates ===
        if self.use_fourier:
            # x_context is 4-D (batch, t_dim, num_context, x_dim); slicing needs
            # a third leading colon to hit the feature axis instead of num_context.
            x_context_raw_prefix = x_context[:, :, :, :(self.x_dim - self.fourier_vars)]
            x_context_raw_fourier = x_context[:, :, :, -self.fourier_vars:]

            # x_target is 3-D (batch, num_target, x_dim); two leading colons is correct here.
            x_target_raw_prefix = x_target[:, :, :(self.x_dim - self.fourier_vars)]
            x_target_raw_fourier = x_target[:, :, -self.fourier_vars:]

            x_context_fourier = self.fourier(x_context_raw_fourier)
            x_target_fourier = self.fourier(x_target_raw_fourier)

            if self.include_raw_x_with_fourier:
                x_context_encoded = torch.cat([x_context_raw_prefix, x_context_raw_fourier, x_context_fourier], dim=-1)
                x_target_encoded = torch.cat([x_target_raw_prefix, x_target_raw_fourier, x_target_fourier], dim=-1)
            else:
                x_context_encoded = torch.cat([x_context_raw_prefix, x_context_fourier], dim=-1)
                x_target_encoded = torch.cat([x_target_raw_prefix, x_target_fourier], dim=-1)
        else:
            x_context_encoded = x_context
            x_target_encoded = x_target

        # === STEP 1: Encode context (x,y) pairs ===
        y_context_encoded = y_context
        context_input = torch.cat([x_context_encoded, y_context_encoded], dim=-1)
        context_encoded = self.context_encoder(context_input.reshape(-1, self.fourier_dim + self.fourier_y_dim))
        r_context = context_encoded.reshape(batch_size, t_num, num_context, self.r_dim)

        # context_mask: (batch, t_dim, num_context) -> (batch, t_dim, num_context, 1)
        mask = context_mask.to(r_context.dtype).unsqueeze(-1)
        counts_per_timestep = mask.sum(dim=2).clamp(min=1e-8)  # (batch, t_dim, 1)

        # Masked mean pooling over context points, per timestep. This is the
        # "basis" -- the actual observed per-timestep content.
        r_timestep_mean = (r_context * mask).sum(dim=2) / counts_per_timestep  # (batch, t_dim, r_dim)

        # === STEP 2: Aggregate timesteps into a single global representation ===
        # coeff_time: per-timestep coefficient, computed ONLY from that
        # timestep's own raw time value. This is deliberately decoupled from
        # r_timestep_mean (the basis being weighted) -- no circularity between
        # "what we're combining" and "what decides how to combine it."
        # t_context: (batch, t_dim) -> (batch * t_dim, 1) -> coeff_net applied
        # per-timestep -> (batch, t_dim, C), C = r_dim or 1 depending on
        # self.coeff_per_channel.
        coeff_time = self.coeff_net(t_context.reshape(-1, 1))
        coeff_out_dim = self.r_dim if self.coeff_per_channel else 1
        coeff_time = coeff_time.reshape(batch_size, t_num, coeff_out_dim)

        # Unconstrained basis expansion: coefficients can be any real value
        # (including negative), so timesteps can be combined contrastively,
        # not just convexly blended the way softmax attention would force.
        # Normalized by t_num so the aggregate's scale doesn't grow with the
        # number of timesteps.
        r_context_aggr = (coeff_time * r_timestep_mean).sum(dim=1) / t_num  # (batch, r_dim)

        # Deterministic anchor
        r_det = self.deterministic_agg(r_context_aggr)

        # Stochastic path
        z_context_mu, z_context_coef = self.latent(r_context_aggr)
        z_context_var = 0.0001 + torch.nn.functional.softplus(z_context_coef)

        std_context = torch.sqrt(z_context_var)
        epsilon_context = torch.randn(num_samples, batch_size, self.z_dim, device=x_context.device)
        z_context = z_context_mu.unsqueeze(0) + std_context.unsqueeze(0) * epsilon_context

        # =========================================================
        # PREPARE DECODER INPUTS
        # =========================================================
        # For DeepONet Trunk: Purely geometry in this simple model
        trunk_input = x_target_encoded

        # Expansions for standard concatenated decoders / Branch inputs
        x_expanded = x_target_encoded.unsqueeze(0).expand(num_samples, -1, -1, -1)
        r_det_branch = r_det.unsqueeze(0).expand(num_samples, -1, -1)
        r_expanded = r_det.unsqueeze(0).unsqueeze(2).expand(num_samples, -1, num_target, -1)

        if y_target is not None:
            # === Training Path (Posterior) ===
            # === Training Path (Posterior) ===
            y_target_encoded = y_target
            target_input = torch.cat([x_target_encoded, y_target_encoded], dim=-1)
            target_encoded = self.context_encoder(target_input.reshape(-1, self.fourier_dim + self.fourier_y_dim))
            r_target = target_encoded.reshape(batch_size, num_target, self.r_dim)

            # Apply the same coefficient-based aggregation to the target too
            if t_target is not None:
                coeff_target = self.coeff_net(t_target.reshape(-1, 1))
                coeff_out_dim = self.r_dim if self.coeff_per_channel else 1
                coeff_target = coeff_target.reshape(batch_size, num_target, coeff_out_dim)
                r_target_aggr = (coeff_target * r_target).sum(dim=1) / num_target
            else:
                r_target_aggr = r_target.mean(dim=1)


            z_target_mu, z_target_coef = self.latent(r_target_aggr)
            z_target_var = 0.0001 + torch.nn.functional.softplus(z_target_coef)

            std_target = torch.sqrt(z_target_var)
            epsilon_target = torch.randn(num_samples, batch_size, self.z_dim, device=x_target.device)
            z_target = z_target_mu.unsqueeze(0) + std_target.unsqueeze(0) * epsilon_target

            # --- ROUTING TO DECODER ---
            if self.use_deeponet_decoder:
                branch_input_target = torch.cat([z_target, r_det_branch], dim=-1)
                y_pred_mu, y_pred_raw = self.decoder(branch_x=branch_input_target, trunk_x=trunk_input)
            else:
                z_expanded = z_target.unsqueeze(2).expand(-1, -1, num_target, -1)
                decoder_input = torch.cat([z_expanded, r_expanded, x_expanded], dim=-1)
                decoder_input_flat = decoder_input.reshape(-1, self.z_dim + self.r_dim + self.fourier_dim)

                y_pred_mu, y_pred_raw = self.decoder(decoder_input_flat)
                y_pred_mu = y_pred_mu.reshape(num_samples, batch_size, num_target, self.y_dim)
                y_pred_raw = y_pred_raw.reshape(num_samples, batch_size, num_target, self.y_dim)

            y_pred_var = 1e-6 + nn.functional.softplus(y_pred_raw)

            if num_samples == 1:
                y_pred_mu = y_pred_mu.squeeze(0)
                y_pred_var = y_pred_var.squeeze(0)

            return y_pred_mu, y_pred_var, z_context_mu, z_context_var, z_target_mu, z_target_var

        # === Testing Path (Prior) ===
        if self.use_deeponet_decoder:
            branch_input_context = torch.cat([z_context, r_det_branch], dim=-1)
            y_pred_mu, y_pred_raw = self.decoder(branch_x=branch_input_context, trunk_x=trunk_input)
        else:
            z_expanded = z_context.unsqueeze(2).expand(-1, -1, num_target, -1)
            decoder_input = torch.cat([z_expanded, r_expanded, x_expanded], dim=-1)
            decoder_input_flat = decoder_input.reshape(-1, self.z_dim + self.r_dim + self.fourier_dim)

            y_pred_mu, y_pred_raw = self.decoder(decoder_input_flat)
            y_pred_mu = y_pred_mu.reshape(num_samples, batch_size, num_target, self.y_dim)
            y_pred_raw = y_pred_raw.reshape(num_samples, batch_size, num_target, self.y_dim)

        y_pred_var = 1e-6 + nn.functional.softplus(y_pred_raw)

        if num_samples == 1:
            y_pred_mu = y_pred_mu.squeeze(0)
            y_pred_var = y_pred_var.squeeze(0)

        return y_pred_mu, y_pred_var, z_context_mu, z_context_var