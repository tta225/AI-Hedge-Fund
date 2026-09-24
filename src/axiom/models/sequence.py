"""Sequence architectures: causal transformer and recurrent ensemble.

The architectures are the uploaded ones. ``PriceTransformer`` keeps its causal
mask, learned position embedding, and three heads; the recurrent model keeps
its bidirectional stack with attention pooling — with one correction noted at
:class:`RecurrentEnsemble`, because bidirectionality on a forecasting task is a
lookahead leak wearing an architecture's clothes.

Everything lives behind :func:`~axiom.models.base.require_torch` so importing
this module without the extra installed gives an actionable message instead of
a traceback from three frames down.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from axiom.models.base import DeepModel, require_torch


def _build_price_transformer(torch: Any, n_features: int, config: dict[str, Any]) -> Any:
    nn = torch.nn

    d_model = int(config.get("d_model", 128))
    n_heads = int(config.get("n_heads", 4))
    n_layers = int(config.get("n_layers", 3))
    d_ff = int(config.get("d_ff", 4 * d_model))
    dropout = float(config.get("dropout", 0.1))
    max_len = int(config.get("max_seq_length", 512))

    if d_model % n_heads:
        raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:
            batch, seq, _ = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            shape = (batch, seq, self.n_heads, self.d_k)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
            # The causal mask is the model's entire claim to being tradable:
            # without it every position attends to its own future and the
            # network learns to read the answer off the input.
            mask = torch.triu(
                torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attention = self.dropout(torch.softmax(scores, dim=-1))
            context = (attention @ v).transpose(1, 2).reshape(batch, seq, d_model)
            return self.proj(context)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = CausalSelfAttention()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: Any) -> Any:
            # Pre-norm rather than the original's post-norm: post-norm needs a
            # learning-rate warmup to train stably at depth, and the uploaded
            # training loop had none.
            x = x + self.attention(self.norm1(x))
            return x + self.ff(self.norm2(x))

    class PriceTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(n_features, d_model)
            self.position = nn.Embedding(max_len, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.return_head = nn.Linear(d_model, 1)
            self.volatility_head = nn.Linear(d_model, 1)

        def forward(self, x: Any) -> Any:
            batch, seq, _ = x.shape
            if seq > max_len:
                raise ValueError(
                    f"sequence of {seq} exceeds max_seq_length={max_len}"
                )
            positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, -1)
            h = self.input_projection(x) + self.position(positions)
            for block in self.blocks:
                h = block(h)
            last = self.final_norm(h)[:, -1, :]
            return self.return_head(last).squeeze(-1)

        def volatility(self, x: Any) -> Any:
            batch, seq, _ = x.shape
            positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, -1)
            h = self.input_projection(x) + self.position(positions)
            for block in self.blocks:
                h = block(h)
            last = self.final_norm(h)[:, -1, :]
            return torch.nn.functional.softplus(self.volatility_head(last)).squeeze(-1)

    return PriceTransformer()


def _build_recurrent(torch: Any, n_features: int, config: dict[str, Any]) -> Any:
    nn = torch.nn
    hidden = int(config.get("hidden_size", 96))
    layers = int(config.get("n_layers", 2))
    dropout = float(config.get("dropout", 0.1))

    class RecurrentNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Unidirectional, unlike the uploaded bidirectional stack. A
            # backward pass over the input window lets every timestep see the
            # end of its own window; the network then reports the last hidden
            # state, which by construction encodes bars after the ones it is
            # supposed to be forecasting from. It trains beautifully and
            # forecasts nothing.
            self.lstm = nn.LSTM(
                n_features, hidden, layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.gru = nn.GRU(
                n_features, hidden, layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.attention = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )

        def forward(self, x: Any) -> Any:
            lstm_out, _ = self.lstm(x)
            gru_out, _ = self.gru(x)
            joint = torch.cat([lstm_out, gru_out], dim=-1)
            weights = torch.softmax(self.attention(joint), dim=1)
            pooled = (joint * weights).sum(dim=1)
            return self.head(pooled).squeeze(-1)

    return RecurrentNet()


class _TorchModel(DeepModel):
    """Shared training loop for the torch-backed models."""

    def _build(self, n_features: int) -> Any:  # pragma: no cover - subclassed
        raise NotImplementedError

    def _fit_epochs(
        self, model: Any, x_train: np.ndarray, y_train: np.ndarray, epochs: int
    ) -> None:
        torch = require_torch()
        model.train()
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-2)),
        )
        loss_fn = torch.nn.SmoothL1Loss()
        x = torch.tensor(x_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.float32)
        batch_size = int(self.config.get("batch_size", 64))

        for _ in range(epochs):
            permutation = torch.randperm(len(x))
            for start in range(0, len(x), batch_size):
                index = permutation[start : start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(model(x[index]), y[index])
                loss.backward()
                # Financial targets are heavy-tailed; one outlier batch without
                # clipping can move every weight far enough to undo the epoch.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

    def _forward(self, model: Any, x: np.ndarray) -> np.ndarray:
        torch = require_torch()
        model.eval()
        with torch.no_grad():
            output = model(torch.tensor(x, dtype=torch.float32))
        return output.cpu().numpy().astype(float)


class PriceTransformerModel(_TorchModel):
    """Causal transformer over a window of features."""

    name = "price_transformer"

    def _build(self, n_features: int) -> Any:
        return _build_price_transformer(require_torch(), n_features, self.config)


class RecurrentEnsemble(_TorchModel):
    """Parallel LSTM and GRU stacks pooled by learned attention."""

    name = "recurrent_ensemble"

    def _build(self, n_features: int) -> Any:
        return _build_recurrent(require_torch(), n_features, self.config)
