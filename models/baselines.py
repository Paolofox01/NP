"""Baselines for the sensor-to-field problem: SHRED, DeepONet (deterministic and probabilistic)
and the Senseiver.

SHRED and DeepONet read the sensors' recent history with a sequence encoder, on the same tasks the
Neural Processes see (pinball_data.py). At each time step the encoder gets, for every sensor slot, the
reading, a 0/1 flag (0 = no reading: a dropped sensor, or a step before the task's lag; its value
is set to 0) and the sensor's (x, y) position, plus the relative time of the step. The positions only
carry information when the sensors vary between tasks; with fixed sensors they are constant.
Histories of different lags are padded at the start, so the last step is always t0.

Two encoders, chosen with encoder="lstm" | "transformer":
- SensorLSTM: the original choice; last hidden state.
- SensorTransformer: self-attention over the time steps (pre-norm), padded steps masked out,
  then the mean over the real steps.

- SHRED (Williams et al., 2023): encoder -> shallow decoder that outputs the whole field at once.
  Deterministic, trained on the MSE.
- DeepONet: encoder as branch, coordinate trunk (Fourier features + MLP), dot product. MSE.
  decoder="attention" (attention DeepONet): instead of compressing the sensor history into one
  vector and taking a dot product, each target's trunk features cross-attend to the encoder's
  per-time-step tokens, then a linear head gives the prediction. Targets never see each other.
- Probabilistic DeepONet: the same with a variance output (a second branch/trunk pair for the dot
  product, a second output of the head for attention), trained on the Gaussian NLL (with an MSE
  warm-up at the start).

- Senseiver (Santos et al., 2023): a Perceiver IO over the sensor readings as a set (value +
  sine-cosine encoding of time and position, like the NPs' context), with learned output queries.
  Probabilistic by default (mean and variance, Gaussian NLL); probabilistic=False is the original
  deterministic model (MSE).

Every model maps a batch (the dictionaries of pinball_data.py) to (mean, var), each (B, T) (or (B, T, V)
for V output variables, in DeepONet and Senseiver), with var None for the deterministic ones. Their loss
and training loop are in
training/train_baselines.py.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers import FourierFeatures, MultiheadAttention, TransformerBlock, inverse_softplus
from models.base import InputScaler

__all__ = ["SensorMLP", "SensorLSTM", "SensorTransformer", "make_sensor_encoder", "SHRED", "DeepONet",
           "SineCosineEncoding", "Senseiver", "ExponentialGP"]

Batch = Dict[str, Tensor]
Prediction = Tuple[Tensor, Optional[Tensor]]


# ----------------------------------------------------------------------------- sensor encoders

def _sensor_steps(batch: Batch) -> Tensor:
    """Per time step: for each sensor slot (reading(s), flag, position), and the relative time:
    (B, L, S * (V + 1 + P) + 1), with V variables per reading and P coordinates per position."""
    history, on = batch["sensor_history"], batch["sensor_on"]           # (B, L, S[, V]), (B, L, S)
    if history.ndim == 3:
        history = history.unsqueeze(-1)
    B, L, S, V = history.shape
    position = batch["sensor_position"].reshape(B, 1, -1).expand(B, L, -1)     # (B, L, S * P)
    return torch.cat([history.reshape(B, L, S * V), on, position, batch["sensor_time"]], dim=-1)

class SensorMLP(nn.Module):
    """Classic MLP branch over the current sensor readings at t0 (the last step, (B, hidden))."""

    def __init__(self, n_sensors: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1,
                 y_dim: int = 1, pos_dim: int = 2):
        super().__init__()
        self.out_dim = hidden_size
        in_dim = n_sensors * (y_dim + 1 + pos_dim) + 1
        self.mlp = _mlp([in_dim] + [hidden_size] * num_layers, dropout=dropout)
        self.act = nn.ReLU()

    def forward(self, batch: Batch) -> Tensor:
        return self.act(self.mlp(_sensor_steps(batch)[:, -1]))

    def tokens(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """Per-step MLP embeddings (B, L, hidden) and padding mask (B, L)."""
        out = self.act(self.mlp(_sensor_steps(batch)))
        return out, batch["sensor_on"].sum(-1) == 0

class SensorLSTM(nn.Module):
    """LSTM over the sensor history; returns the last hidden state of the top layer (B, hidden)."""

    def __init__(self, n_sensors: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1,
                 y_dim: int = 1, pos_dim: int = 2):
        super().__init__()
        self.out_dim = hidden_size
        self.lstm = nn.LSTM(n_sensors * (y_dim + 1 + pos_dim) + 1, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, batch: Batch) -> Tensor:
        _, (h, _) = self.lstm(_sensor_steps(batch))
        return h[-1]

    def tokens(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """The output at every time step (B, L, hidden), and which steps have no reading (B, L)."""
        out, _ = self.lstm(_sensor_steps(batch))
        return out, batch["sensor_on"].sum(-1) == 0


class SensorTransformer(nn.Module):
    """Transformer encoder over the time steps of the sensor history -> (B, hidden).

    Each step is embedded with a linear layer, together with Fourier features of its relative time
    (the attention itself has no notion of order). Steps with no reading at all (before the task's
    lag) are masked out of the attention and of the final mean, so the output does not depend on
    the padding.
    """

    def __init__(self, n_sensors: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1,
                 num_heads: int = 4, time_frequencies: int = 8, y_dim: int = 1, pos_dim: int = 2):
        super().__init__()
        self.out_dim = hidden_size
        self.time_features = FourierFeatures(1, time_frequencies, scale=1.0, include_input=False)
        self.embed = nn.Linear(n_sensors * (y_dim + 1 + pos_dim) + 1 + self.time_features.out_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=4 * hidden_size,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)

    def tokens(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """One token per time step (B, L, hidden), and which steps have no reading (B, L)."""
        steps = torch.cat([_sensor_steps(batch), self.time_features(batch["sensor_time"])], dim=-1)
        padding = batch["sensor_on"].sum(-1) == 0                          # (B, L): no reading at that step
        return self.norm(self.layers(self.embed(steps), src_key_padding_mask=padding)), padding

    def forward(self, batch: Batch) -> Tensor:
        h, padding = self.tokens(batch)
        keep = (~padding).unsqueeze(-1).float()
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


def make_sensor_encoder(kind: str, n_sensors: int, hidden_size: int, num_layers: int,
                        dropout: float, num_heads: int, y_dim: int = 1, pos_dim: int = 2) -> nn.Module:
    if kind == "mlp":
        return SensorMLP(n_sensors, hidden_size, num_layers, dropout, y_dim, pos_dim)
    if kind == "lstm":
        return SensorLSTM(n_sensors, hidden_size, num_layers, dropout, y_dim, pos_dim)
    if kind == "transformer":
        return SensorTransformer(n_sensors, hidden_size, num_layers, dropout, num_heads,
                                 y_dim=y_dim, pos_dim=pos_dim)
    raise ValueError(f"encoder must be 'mlp', 'lstm' or 'transformer', got {kind!r}")


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    """Linear layers with ReLU (and dropout) between them, none after the last."""
    layers: List[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- models

class SHRED(nn.Module):
    """SHallow REcurrent Decoder: sensor history -> encoder -> shallow MLP -> field (full or via POD modes)."""

    probabilistic = False

    def __init__(self, n_sensors: int, n_nodes: int, n_params: int = 0, encoder: str = "lstm",
                 hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4,
                 decoder_sizes: Sequence[int] = (350, 400), dropout: float = 0.1,
                 y_dim: int = 1, pos_dim: int = 2,
                 pod_basis: Optional[Tensor] = None, pod_mean: Optional[Tensor] = None):
        super().__init__()
        n_modes = pod_basis.shape[-1] if pod_basis is not None else None
        self.kwargs = dict(n_sensors=n_sensors, n_nodes=n_nodes, n_params=n_params, encoder=encoder,
                           hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                           decoder_sizes=list(decoder_sizes), dropout=dropout,
                           y_dim=y_dim, pos_dim=pos_dim, n_modes=n_modes)
        self.encoder = make_sensor_encoder(encoder, n_sensors, hidden_size, num_layers, dropout, num_heads,
                                           y_dim=y_dim, pos_dim=pos_dim)
        out_dim = n_modes if n_modes is not None else n_nodes * y_dim
        self.decoder = _mlp([hidden_size + n_params, *decoder_sizes, out_dim], dropout)
        self.n_params, self.n_nodes, self.y_dim = n_params, n_nodes, y_dim

        # Registra la base POD (n_nodes, y_dim, K) e la media (n_nodes, y_dim) come buffer non addestrabili
        self.register_buffer("pod_basis", pod_basis)
        self.register_buffer("pod_mean", pod_mean)

    def forward(self, batch: Batch) -> Prediction:
        h = self.encoder(batch)
        if self.n_params:
            h = torch.cat([h, batch["mu"]], dim=-1)
        coeffs = self.decoder(h)                                           # (B, K) oppure (B, n_nodes * y_dim)

        if self.pod_basis is not None:
            idx = batch["target_index"]                                    # (B, T)
            # Estrae la base POD solo nei punti target del batch: (B, T, y_dim, K)
            basis_t = self.pod_basis[idx]
            pred = torch.einsum("btyk,bk->bty", basis_t, coeffs)
            if self.pod_mean is not None:
                pred = pred + self.pod_mean[idx]
            return (pred.squeeze(-1) if self.y_dim == 1 else pred), None

        if self.y_dim == 1:
            return coeffs.gather(1, batch["target_index"]), None
        field = coeffs.view(-1, self.n_nodes, self.y_dim)
        idx = batch["target_index"].unsqueeze(-1).expand(-1, -1, self.y_dim)
        return field.gather(1, idx), None

class DeepONet(nn.Module):
    """DeepONet with a sequence-encoder branch.

    Branch: an LSTM (encoder="lstm", as in the original scripts) or a SensorTransformer.
    Trunk input: the spatial coordinates, their Fourier features and (with mu) the flow parameters.

    decoder="dot" (the DeepONet): u(x) = <branch(sensors), trunk(x)> / sqrt(p) + bias, with the
        branch compressed to one vector. probabilistic=True adds a variance branch/trunk pair:
        var = min_var + softplus(<., .> + bias), the bias starting low so the variance starts small.
    decoder="attention": the trunk features of each target (projected to encoder_hidden) are the
        query of attention_layers cross-attention blocks over the branch's per-time-step tokens
        (padded steps masked); a linear head gives the mean (and the variance, if probabilistic).
    With y_dim > 1 variables, every variable has its own heads (mean, and variance if probabilistic),
    and the prediction is (B, T, y_dim) instead of (B, T).
    """

    def __init__(self, n_sensors: int, n_params: int = 0, probabilistic: bool = False, p: int = 128,
                 encoder: str = "lstm", encoder_hidden: int = 256, encoder_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.1, decoder: str = "dot", attention_layers: int = 1,
                 trunk_hidden: int = 256, head_hidden: int = 128, coord_dim: int = 2,
                 num_frequencies: int = 32, fourier_scale: float = 1.0, min_var: float = 1e-4,
                 init_var: float = 0.05, y_dim: int = 1, pos_dim: int = 2):
        super().__init__()
        if decoder not in ("dot", "attention"):
            raise ValueError(f"decoder must be 'dot' or 'attention', got {decoder!r}")
        self.kwargs = dict(n_sensors=n_sensors, n_params=n_params, probabilistic=probabilistic, p=p,
                           encoder=encoder, encoder_hidden=encoder_hidden, encoder_layers=encoder_layers,
                           num_heads=num_heads, dropout=dropout, decoder=decoder,
                           attention_layers=attention_layers,
                           trunk_hidden=trunk_hidden, head_hidden=head_hidden, coord_dim=coord_dim,
                           num_frequencies=num_frequencies, fourier_scale=fourier_scale,
                           min_var=min_var, init_var=init_var, y_dim=y_dim, pos_dim=pos_dim)
        self.probabilistic, self.p, self.n_params, self.coord_dim = probabilistic, p, n_params, coord_dim
        self.min_var, self.decoder, self.y_dim = min_var, decoder, y_dim

        self.branch = make_sensor_encoder(encoder, n_sensors, encoder_hidden, encoder_layers, dropout, num_heads,
                                          y_dim, pos_dim)
        self.fourier = FourierFeatures(coord_dim, num_frequencies, fourier_scale,
                                       learnable=True, include_input=True)
        self.trunk = _mlp([self.fourier.out_dim + n_params, trunk_hidden, trunk_hidden])
        self.trunk_act = nn.ReLU()

        n_outputs = y_dim * (2 if probabilistic else 1)                    # the means, then the variances
        if decoder == "dot":
            self.branch_heads = nn.ModuleList(_mlp([encoder_hidden, head_hidden, p]) for _ in range(n_outputs))
            self.trunk_heads = nn.ModuleList(_mlp([trunk_hidden, head_hidden, p]) for _ in range(n_outputs))
            self.bias = nn.Parameter(torch.zeros(n_outputs))
            if probabilistic:
                with torch.no_grad():
                    self.bias[y_dim:] = inverse_softplus(init_var - min_var)
        else:
            self.query = nn.Linear(trunk_hidden, encoder_hidden)
            self.cross_attention = nn.ModuleList(
                TransformerBlock(encoder_hidden, num_heads, dropout) for _ in range(attention_layers))
            self.output_norm = nn.LayerNorm(encoder_hidden)
            self.head = nn.Linear(encoder_hidden, n_outputs)
            if probabilistic:
                with torch.no_grad():
                    self.head.weight[y_dim:].mul_(0.1)
                    self.head.bias[y_dim:] = inverse_softplus(init_var - min_var)

    def _split(self, out: Tensor) -> Prediction:
        """(B, T, n_outputs) -> mean and variance, each (B, T) for one variable or (B, T, y_dim)."""
        mean = out[..., :self.y_dim]
        var = self.min_var + F.softplus(out[..., self.y_dim:]) if self.probabilistic else None
        if self.y_dim == 1:
            mean, var = mean.squeeze(-1), (var.squeeze(-1) if var is not None else None)
        return mean, var

    def forward(self, batch: Batch) -> Prediction:
        coords = batch["x_target"][..., -self.coord_dim:]                  # (B, T, coord_dim): spatial only
        trunk_in = self.fourier(coords)
        if self.n_params:
            mu = batch["mu"].unsqueeze(1).expand(-1, coords.shape[1], -1)
            trunk_in = torch.cat([trunk_in, mu], dim=-1)
        t = self.trunk_act(self.trunk(trunk_in))                           # (B, T, trunk_hidden)

        if self.decoder == "attention":
            tokens, padding = self.branch.tokens(batch)                    # (B, L, encoder_hidden)
            q = self.query(t)
            for block in self.cross_attention:
                q = block(q, tokens, tokens, key_padding_mask=padding)
            return self._split(self.head(self.output_norm(q)))             # (B, T, n_outputs)

        h = self.branch(batch)                                             # (B, encoder_hidden)
        outputs = [
            (b_head(h).unsqueeze(1) * t_head(t)).sum(-1) / math.sqrt(self.p) + bias
            for b_head, t_head, bias in zip(self.branch_heads, self.trunk_heads, self.bias)
        ]
        return self._split(torch.stack(outputs, dim=-1))


# ----------------------------------------------------------------------------- Senseiver

class SineCosineEncoding(nn.Module):
    """Fixed sine-cosine encoding of coordinates in [-1, 1] (as in the Perceiver / Senseiver):
    for every dimension, [x, sin(pi f_k x), cos(pi f_k x)] with n_frequencies frequencies f_k
    spaced linearly from 1 to max_frequency / 2. (..., dim) -> (..., dim * (2 n_frequencies + 1))."""

    def __init__(self, dim: int, n_frequencies: int = 16, max_frequency: float = 32.0):
        super().__init__()
        self.register_buffer("frequencies", torch.linspace(1.0, max_frequency / 2, n_frequencies))
        self.out_dim = dim * (2 * n_frequencies + 1)

    def forward(self, x: Tensor) -> Tensor:
        angles = math.pi * x.unsqueeze(-1) * self.frequencies                  # (..., dim, F)
        return torch.cat([x.unsqueeze(-1), angles.sin(), angles.cos()], dim=-1).flatten(-2)


class Senseiver(nn.Module):
    """Senseiver (Santos et al., 2023): Perceiver IO for field reconstruction from sparse sensors.

    Encoder: each reading is a token Linear([y, enc(x)]), with x = ([mu,] relative time, position)
    scaled to [-1, 1] and enc the sine-cosine encoding. n_latents learned latents (Q_in) read the
    tokens with an attention block (cross-attention + MLP, self-attention + MLP); a second block,
    with weights shared across n_recurrent repetitions, lets the latents re-read the tokens and
    process them again. Padded readings are masked out.
    Decoder: for every target and every output variable the query is Linear([enc(x_q), Q_out_v]),
    Q_out_v a learned vector per variable; one cross-attention to the latents and a linear head
    give the mean (and the variance: min_var + softplus, if probabilistic).

    Targets never see each other, and the readings form a set (no slots), so any number of
    sensors at any positions can be used, as in the Neural Processes.
    """

    def __init__(self, x_dim: int, y_dim: int = 1, x_low=-1.0, x_high=1.0, probabilistic: bool = True,
                 n_latents: int = 32, width: int = 128, num_heads: int = 4, n_recurrent: int = 2,
                 n_frequencies: int = 16, max_frequency: float = 32.0, query_dim: int = 16,
                 dropout: float = 0.0, min_var: float = 1e-4, init_var: float = 0.05):
        super().__init__()
        x_low = torch.as_tensor(x_low, dtype=torch.float32).expand(x_dim).tolist()
        x_high = torch.as_tensor(x_high, dtype=torch.float32).expand(x_dim).tolist()
        self.kwargs = dict(x_dim=x_dim, y_dim=y_dim, x_low=x_low, x_high=x_high, probabilistic=probabilistic,
                           n_latents=n_latents, width=width, num_heads=num_heads, n_recurrent=n_recurrent,
                           n_frequencies=n_frequencies, max_frequency=max_frequency, query_dim=query_dim,
                           dropout=dropout, min_var=min_var, init_var=init_var)
        self.probabilistic, self.y_dim, self.n_recurrent, self.min_var = probabilistic, y_dim, n_recurrent, min_var

        self.scaler = InputScaler(x_dim, x_low, x_high)
        self.encoding = SineCosineEncoding(x_dim, n_frequencies, max_frequency)
        self.embed = nn.Linear(y_dim + self.encoding.out_dim, width)
        block = lambda: TransformerBlock(width, num_heads, dropout)
        self.latents = nn.Parameter(torch.randn(n_latents, width) * 0.02)               # Q_in
        self.read_1, self.process_1 = block(), block()                                   # attention block 1
        self.read_2, self.process_2 = block(), block()                                   # attention block 2 (recurrent)
        self.latent_norm = nn.LayerNorm(width)

        self.output_queries = nn.Parameter(torch.randn(y_dim, query_dim) * 0.02)         # Q_out, one per variable
        self.query = nn.Linear(self.encoding.out_dim + query_dim, width)
        self.cross_attention = MultiheadAttention(width, num_heads, dropout)
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2 if probabilistic else 1)
        if probabilistic:
            with torch.no_grad():
                self.head.weight[1].mul_(0.1)
                self.head.bias[1] = inverse_softplus(init_var - min_var)

    def encode(self, x_context: Tensor, y_context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Readings (B, N, x_dim), (B, N, y_dim) -> latents (B, n_latents, width)."""
        tokens = self.embed(torch.cat([y_context, self.encoding(self.scaler(x_context, mask))], dim=-1))
        latents = self.latents.expand(tokens.shape[0], -1, -1)
        latents = self.read_1(latents, tokens, tokens, key_padding_mask=mask)
        latents = self.process_1(latents, latents, latents)
        for _ in range(self.n_recurrent):
            latents = self.read_2(latents, tokens, tokens, key_padding_mask=mask)
            latents = self.process_2(latents, latents, latents)
        return self.latent_norm(latents)

    def forward(self, batch: Batch) -> Prediction:
        latents = self.encode(batch["x_context"], batch["y_context"], batch.get("context_mask"))
        a_q = self.encoding(self.scaler(batch["x_target"], batch.get("target_mask")))   # (B, T, E)
        B, T, E = a_q.shape
        V = self.y_dim
        queries = torch.cat([a_q.unsqueeze(2).expand(B, T, V, E),
                             self.output_queries.expand(B, T, V, -1)], dim=-1)          # (B, T, V, E + q)
        h = self.cross_attention(self.query(queries).reshape(B, T * V, -1), latents, latents)
        out = self.head(self.output_norm(h)).reshape(B, T, V, -1)
        mean = out[..., 0]
        var = self.min_var + F.softplus(out[..., 1]) if self.probabilistic else None
        if V == 1:                                                                        # (B, T), like the other baselines
            mean = mean.squeeze(-1)
            var = var.squeeze(-1) if var is not None else None
        return mean, var

# ----------------------------------------------------------------------------- Gaussian Process

class ExponentialGP(nn.Module):
    """Exact Gaussian Process Regression with an anisotropic exponential (Laplace / Matern 1/2) kernel:
        k(x, x') = sigma_f^2 * exp( - || (x - x') / l ||_2 )
    Learns per-channel lengthscales (one per input coordinate in x_dim), signal variance, and noise
    variance via gradient descent in train_baseline, and handles padded batches via context_mask.
    """

    probabilistic = True

    def __init__(self, x_dim: int, y_dim: int = 1, init_lengthscale: float = 0.5,
                 init_var: float = 1.0, init_noise: float = 0.05, min_var: float = 1e-4):
        super().__init__()
        self.kwargs = dict(x_dim=x_dim, y_dim=y_dim, init_lengthscale=init_lengthscale,
                           init_var=init_var, init_noise=init_noise, min_var=min_var)
        self.x_dim, self.y_dim, self.min_var = x_dim, y_dim, min_var

        self.raw_lengthscale = nn.Parameter(
            torch.full((y_dim, x_dim), float(inverse_softplus(init_lengthscale)))
        )
        self.raw_var = nn.Parameter(
            torch.full((y_dim,), float(inverse_softplus(init_var)))
        )
        self.raw_noise = nn.Parameter(
            torch.full((y_dim,), float(inverse_softplus(max(init_noise - min_var, 1e-4))))
        )

    def _hyperparams(self) -> Tuple[Tensor, Tensor, Tensor]:
        lengthscale = F.softplus(self.raw_lengthscale).clamp(min=1e-3)     # (V, D)
        var = F.softplus(self.raw_var).clamp(min=1e-4)                     # (V,)
        noise = self.min_var + F.softplus(self.raw_noise)                  # (V,)
        return lengthscale, var, noise

    def _kernel(self, x1: Tensor, x2: Tensor, lengthscale: Tensor, var: Tensor) -> Tensor:
        """x1: (B, N1, D), x2: (B, N2, D) -> (B, V, N1, N2)."""
        scale = lengthscale.unsqueeze(0).unsqueeze(2)                      # (1, V, 1, D)
        x1_s = x1.unsqueeze(1) / scale                                     # (B, V, N1, D)
        x2_s = x2.unsqueeze(1) / scale                                     # (B, V, N2, D)
        dist = torch.cdist(x1_s, x2_s, p=2.0)                              # (B, V, N1, N2)
        return var.view(1, -1, 1, 1) * torch.exp(-dist)

    def forward(self, batch: Batch) -> Prediction:
        x_c, y_c = batch["x_context"], batch["y_context"]                  # (B, N_c, D), (B, N_c, V)
        x_t = batch["x_target"]                                            # (B, T, D)
        if y_c.ndim == 2:
            y_c = y_c.unsqueeze(-1)
        B, N_c, _ = x_c.shape

        lengthscale, var, noise = self._hyperparams()
        K_cc = self._kernel(x_c, x_c, lengthscale, var)                    # (B, V, N_c, N_c)
        K_ct = self._kernel(x_c, x_t, lengthscale, var)                    # (B, V, N_c, T)

        eye = torch.eye(N_c, device=x_c.device, dtype=x_c.dtype).view(1, 1, N_c, N_c)
        K_y = K_cc + noise.view(1, -1, 1, 1) * eye

        y_v = y_c.permute(0, 2, 1).unsqueeze(-1)                           # (B, V, N_c, 1)

        # Mask out padded context points so they have zero influence on Cholesky and predictions
        mask = batch.get("context_mask")                                   # (B, N_c), True = padding
        if mask is not None and mask.any():
            valid = ~mask                                                  # (B, N_c)
            valid_2d = (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(1)  # (B, 1, N_c, N_c)
            K_y = torch.where(valid_2d, K_y, eye.expand_as(K_y))
            K_ct = K_ct * valid.view(B, 1, N_c, 1)
            y_v = y_v * valid.view(B, 1, N_c, 1)

        L = torch.linalg.cholesky(K_y)                                     # (B, V, N_c, N_c)
        alpha = torch.cholesky_solve(y_v, L)                               # (B, V, N_c, 1)
        mean = torch.matmul(K_ct.transpose(-2, -1), alpha).squeeze(-1).permute(0, 2, 1)  # (B, T, V)

        v = torch.linalg.solve_triangular(L, K_ct, upper=False)            # (B, V, N_c, T)
        pred_var = (var.view(1, -1, 1) - (v ** 2).sum(dim=2) + noise.view(1, -1, 1)).permute(0, 2, 1)
        pred_var = pred_var.clamp(min=self.min_var)                        # (B, T, V)

        if self.y_dim == 1:
            return mean.squeeze(-1), pred_var.squeeze(-1)
        return mean, pred_var