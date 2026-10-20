"""Baselines for the sensor-to-field problem: SHRED, DeepONet and probabilistic DeepONet.

All three read the sensors' recent history with a sequence encoder, on the same tasks the Neural
Processes see (pinball_data.py). At each time step the encoder gets, for every sensor slot, the
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

Every model maps a batch (the dictionaries of pinball_data.py) to (mean, var), each (B, T),
with var None for the deterministic ones. Their loss and training loop are in
training/train_baselines.py.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers import FourierFeatures, TransformerBlock, inverse_softplus

__all__ = ["SensorLSTM", "SensorTransformer", "make_sensor_encoder", "SHRED", "DeepONet"]

Batch = Dict[str, Tensor]
Prediction = Tuple[Tensor, Optional[Tensor]]


# ----------------------------------------------------------------------------- sensor encoders

def _sensor_steps(batch: Batch) -> Tensor:
    """Per time step: for each sensor slot (reading, flag, x, y), and the relative time: (B, L, 4S + 1)."""
    history, on = batch["sensor_history"], batch["sensor_on"]           # (B, L, S) each
    B, L, S = history.shape
    position = batch["sensor_position"].reshape(B, 1, 2 * S).expand(B, L, -1)  # (B, L, 2S)
    return torch.cat([history, on, position, batch["sensor_time"]], dim=-1)


class SensorLSTM(nn.Module):
    """LSTM over the sensor history; returns the last hidden state of the top layer (B, hidden)."""

    def __init__(self, n_sensors: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim = hidden_size
        self.lstm = nn.LSTM(4 * n_sensors + 1, hidden_size, num_layers=num_layers,
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
                 num_heads: int = 4, time_frequencies: int = 8):
        super().__init__()
        self.out_dim = hidden_size
        self.time_features = FourierFeatures(1, time_frequencies, scale=1.0, include_input=False)
        self.embed = nn.Linear(4 * n_sensors + 1 + self.time_features.out_dim, hidden_size)
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
                        dropout: float, num_heads: int) -> nn.Module:
    if kind == "lstm":
        return SensorLSTM(n_sensors, hidden_size, num_layers, dropout)
    if kind == "transformer":
        return SensorTransformer(n_sensors, hidden_size, num_layers, dropout, num_heads)
    raise ValueError(f"encoder must be 'lstm' or 'transformer', got {kind!r}")


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
    """SHallow REcurrent Decoder: sensor history -> encoder -> shallow MLP -> the whole field.

    Original sizes: LSTM 64 x 2 layers, decoder 350 -> 400, dropout 0.1. With
    encoder="transformer" the LSTM is replaced by a SensorTransformer of the same width and depth.
    The field is predicted at every mesh node; the loss only uses the nodes of the task's targets.
    """

    probabilistic = False

    def __init__(self, n_sensors: int, n_nodes: int, n_params: int = 0, encoder: str = "lstm",
                 hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4,
                 decoder_sizes: Sequence[int] = (350, 400), dropout: float = 0.1):
        super().__init__()
        self.kwargs = dict(n_sensors=n_sensors, n_nodes=n_nodes, n_params=n_params, encoder=encoder,
                           hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                           decoder_sizes=list(decoder_sizes), dropout=dropout)
        self.encoder = make_sensor_encoder(encoder, n_sensors, hidden_size, num_layers, dropout, num_heads)
        self.decoder = _mlp([hidden_size + n_params, *decoder_sizes, n_nodes], dropout)
        self.n_params = n_params

    def forward(self, batch: Batch) -> Prediction:
        h = self.encoder(batch)
        if self.n_params:
            h = torch.cat([h, batch["mu"]], dim=-1)
        field = self.decoder(h)                                            # (B, n_nodes)
        return field.gather(1, batch["target_index"]), None


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
    """

    def __init__(self, n_sensors: int, n_params: int = 0, probabilistic: bool = False, p: int = 128,
                 encoder: str = "lstm", encoder_hidden: int = 256, encoder_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.1, decoder: str = "dot", attention_layers: int = 1,
                 trunk_hidden: int = 256, head_hidden: int = 128, coord_dim: int = 2,
                 num_frequencies: int = 32, fourier_scale: float = 1.0, min_var: float = 1e-4,
                 init_var: float = 0.05):
        super().__init__()
        if decoder not in ("dot", "attention"):
            raise ValueError(f"decoder must be 'dot' or 'attention', got {decoder!r}")
        self.kwargs = dict(n_sensors=n_sensors, n_params=n_params, probabilistic=probabilistic, p=p,
                           encoder=encoder, encoder_hidden=encoder_hidden, encoder_layers=encoder_layers,
                           num_heads=num_heads, dropout=dropout, decoder=decoder,
                           attention_layers=attention_layers,
                           trunk_hidden=trunk_hidden, head_hidden=head_hidden, coord_dim=coord_dim,
                           num_frequencies=num_frequencies, fourier_scale=fourier_scale,
                           min_var=min_var, init_var=init_var)
        self.probabilistic, self.p, self.n_params, self.coord_dim = probabilistic, p, n_params, coord_dim
        self.min_var, self.decoder = min_var, decoder

        self.branch = make_sensor_encoder(encoder, n_sensors, encoder_hidden, encoder_layers, dropout, num_heads)
        self.fourier = FourierFeatures(coord_dim, num_frequencies, fourier_scale,
                                       learnable=True, include_input=True)
        self.trunk = _mlp([self.fourier.out_dim + n_params, trunk_hidden, trunk_hidden])
        self.trunk_act = nn.ReLU()

        n_outputs = 2 if probabilistic else 1
        if decoder == "dot":
            self.branch_heads = nn.ModuleList(_mlp([encoder_hidden, head_hidden, p]) for _ in range(n_outputs))
            self.trunk_heads = nn.ModuleList(_mlp([trunk_hidden, head_hidden, p]) for _ in range(n_outputs))
            self.bias = nn.Parameter(torch.zeros(n_outputs))
            if probabilistic:
                with torch.no_grad():
                    self.bias[1] = inverse_softplus(init_var - min_var)
        else:
            self.query = nn.Linear(trunk_hidden, encoder_hidden)
            self.cross_attention = nn.ModuleList(
                TransformerBlock(encoder_hidden, num_heads, dropout) for _ in range(attention_layers))
            self.output_norm = nn.LayerNorm(encoder_hidden)
            self.head = nn.Linear(encoder_hidden, n_outputs)
            if probabilistic:
                with torch.no_grad():
                    self.head.weight[1].mul_(0.1)
                    self.head.bias[1] = inverse_softplus(init_var - min_var)

    def forward(self, batch: Batch) -> Prediction:
        coords = batch["x_target"][..., -self.coord_dim:]                  # (B, T, 2): spatial only
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
            out = self.head(self.output_norm(q))                           # (B, T, 1 or 2)
            mean = out[..., 0]
            var = self.min_var + F.softplus(out[..., 1]) if self.probabilistic else None
            return mean, var

        h = self.branch(batch)                                             # (B, encoder_hidden)
        outputs = [
            (b_head(h).unsqueeze(1) * t_head(t)).sum(-1) / math.sqrt(self.p) + bias
            for b_head, t_head, bias in zip(self.branch_heads, self.trunk_heads, self.bias)
        ]
        mean = outputs[0]
        var = self.min_var + F.softplus(outputs[1]) if self.probabilistic else None
        return mean, var