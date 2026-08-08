from typing import Type

import warnings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Ensure both decoders are imported
from architectures import Decoder, DeepONetDecoder, Encoder, Latent, MLP, Self_Attn, Cross_Attn, FourierFeatures, LearnableFourierFeatures

import torch
import torch.nn as nn

class LatNP(nn.Module):
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
                 fourier_vars: int = 0,
                 include_raw_x_with_fourier: bool = True,
                 num_frequencies: int = 64,
                 fourier_scale: float = 10.0,
                 num_heads: int = 8,
                 use_skip: bool = True,
                 learnable_fourier: bool = False,
                 use_deeponet_decoder: bool = False,  # <--- ADDED TOGGLE
                 p: int = 128):                       # <--- ADDED DEEPONET PROJECTION DIM
        super().__init__()
        
        self.use_deeponet_decoder = use_deeponet_decoder
        
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
        
        # === Encoders and Attention ===
        self.context_encoder = MLP(input_dim=fourier_dim + fourier_y_dim, output_dim=r_dim,
                                   hidden_dim=hidden_dim, n_hidden=n_hidden,
                                   activation=activation, dropout=0.0,
                                   is_normalized=is_normalized, norm_type=norm_type,
                                   norm_position='post')
        
        self.x_encoder = MLP(input_dim=fourier_dim, output_dim=r_dim,
                             hidden_dim=hidden_dim, n_hidden=n_hidden,
                             activation=activation, dropout=0.0,
                             is_normalized=is_normalized, norm_type=norm_type,
                             norm_position='post')
        
        self.self_attn = Self_Attn(input_dim=r_dim, output_dim=r_dim,
                                   num_heads=num_heads, dropout=dropout)
        self.norm_self_attn = nn.LayerNorm(r_dim) if is_normalized and use_skip else nn.Identity()
        
        self.deterministic_agg = nn.Sequential(
            nn.Linear(r_dim, r_dim),
            nn.LayerNorm(r_dim) if is_normalized else nn.Identity(),
            activation(),
            nn.Dropout(dropout)
        )
        
        self.latent = Latent(r_dim=r_dim, hidden_dim=hidden_dim, z_dim=z_dim,
                             n_hidden=n_hidden, activation=activation, dropout=0.0,
                             is_normalized=is_normalized, norm_type=norm_type)
        
        self.cross_attn = Cross_Attn(query_dim=r_dim, context_dim=r_dim, value_dim=r_dim, 
                                     output_dim=r_dim, num_heads=num_heads, dropout=dropout)
        self.norm_cross_attn = nn.LayerNorm(r_dim) if is_normalized and use_skip else nn.Identity()
        
        # === DECODER SELECTION ROUTING ===
        if self.use_deeponet_decoder:
            self.decoder = DeepONetDecoder(branch_dim=z_dim + r_dim,
                                           trunk_dim=fourier_dim + r_dim,
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
        
        # Latent Initialization
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
        
    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor, 
                x_target: torch.Tensor, y_target: torch.Tensor = None,
                num_samples: int = 1) -> tuple:
        
        batch_size, num_context, x_dim = x_context.shape
        _, num_target, _ = x_target.shape
        _, _, y_dim = y_context.shape
        
        # === STEP 0: Apply Fourier features to x coordinates ===
        if self.use_fourier:
            x_context_raw_prefix = x_context[:, :, :(self.x_dim - self.fourier_vars)]
            x_target_raw_prefix = x_target[:, :, :(self.x_dim - self.fourier_vars)]
            x_context_raw_fourier = x_context[:, :, -self.fourier_vars:]
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
        
        # === STEP 1 & 2: Encode context and Self-attention ===
        y_context_encoded = y_context
        context_input = torch.cat([x_context_encoded, y_context_encoded], dim=-1)
        context_encoded = self.context_encoder(context_input.reshape(-1, self.fourier_dim + self.fourier_y_dim))
        context_encoded = context_encoded.reshape(batch_size, num_context, self.r_dim)
        
        attn_out = self.self_attn(context_encoded)
        r_context = self.norm_self_attn(context_encoded + attn_out) 
        
        # === STEP 3: Dual path aggregation (Context) ===
        r_det = r_context.mean(dim=1)
        r_det = self.deterministic_agg(r_det)
        
        r_context_mean = r_context.mean(dim=1)
        z_context_mu, z_context_coef = self.latent(r_context_mean)
        z_context_var = 0.0001 + torch.nn.functional.softplus(z_context_coef)
        
        std_context = torch.sqrt(z_context_var)
        epsilon_context = torch.randn(num_samples, batch_size, self.z_dim, device=x_context.device)
        z_context = z_context_mu.unsqueeze(0) + std_context.unsqueeze(0) * epsilon_context
        
        # === Cross-attention ===
        x_context_emb = self.x_encoder(x_context_encoded.reshape(-1, self.fourier_dim))
        x_context_emb = x_context_emb.reshape(batch_size, num_context, self.r_dim)
        
        x_target_emb = self.x_encoder(x_target_encoded.reshape(-1, self.fourier_dim))
        x_target_emb = x_target_emb.reshape(batch_size, num_target, self.r_dim)
        
        cross_attn_out = self.cross_attn(query=x_target_emb, context=x_context_emb, value=r_context)
        r_cross_attn = self.norm_cross_attn(x_target_emb + cross_attn_out)
        
        # =========================================================
        # PREPARE DECODER INPUTS
        # =========================================================
        # For DeepONet: Trunk only needs to be evaluated once per target location
        trunk_input = torch.cat([x_target_encoded, r_cross_attn], dim=-1)
        
        # For Standard Decoder: We need fully expanded tensors
        r_expanded = r_cross_attn.unsqueeze(0).expand(num_samples, -1, -1, -1)
        x_expanded = x_target_encoded.unsqueeze(0).expand(num_samples, -1, -1, -1)
        
        # Expand r_det for the Branch
        r_det_branch = r_det.unsqueeze(0).expand(num_samples, -1, -1)
        
        if y_target is not None:
            # === Training Path (Posterior) ===
            y_target_encoded = y_target
            target_input = torch.cat([x_target_encoded, y_target_encoded], dim=-1)
            target_encoded = self.context_encoder(target_input.reshape(-1, self.fourier_dim + self.fourier_y_dim))
            target_encoded = target_encoded.reshape(batch_size, num_target, self.r_dim)
            
            target_attn_out = self.self_attn(target_encoded)
            r_target = self.norm_self_attn(target_encoded + target_attn_out)
            r_target_mean = r_target.mean(dim=1)
            
            z_target_mu, z_target_coef = self.latent(r_target_mean)
            z_target_var = 0.0001 + torch.nn.functional.softplus(z_target_coef)
            
            std_target = torch.sqrt(z_target_var)
            epsilon_target = torch.randn(num_samples, batch_size, self.z_dim, device=x_target.device)
            z_target = z_target_mu.unsqueeze(0) + std_target.unsqueeze(0) * epsilon_target
            
            
            # --- ROUTING TO DECODER ---
            if self.use_deeponet_decoder:
                # Passes unexpanded local features, lets the decoder broadcast
                branch_input_target = torch.cat([z_target, r_det_branch], dim=-1)
                y_pred_mu, y_pred_raw = self.decoder(branch_x=branch_input_target, trunk_x=trunk_input)
            else:
                # Fully expands everything for standard MLP concatenation
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
            # Passes unexpanded local features, lets the decoder broadcast
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