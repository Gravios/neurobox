"""
neurobox.analysis.classifiers.torch_backends
=============================================
PyTorch implementations of behavioural-state classifiers.

Provides four backends:

* :class:`PatternNetMLP`   — single hidden layer, sigmoid activation,
                             softmax output.  Faithful reproduction of
                             MATLAB's ``patternnet``.
* :class:`MLPClassifierTorch` — multi-layer perceptron with ReLU
                                activations and dropout.
* :class:`Conv1DClassifier` — 1-D convolutional network operating on
                              a temporal context window.
* :class:`BiLSTMClassifier` — bidirectional LSTM for sequence
                              labelling.

All four register themselves on import via :mod:`._registry`.

Importing this module pulls in :mod:`torch`; do not import it
unconditionally from ``neurobox/__init__.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import _registry
from .base import Classifier, FitInfo


# ──────────────────────────────────────────────────────────────────── #
# Shared training utilities                                             #
# ──────────────────────────────────────────────────────────────────── #

def _select_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool,
    device:     torch.device,
) -> DataLoader:
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(Xt, yt),
        batch_size = batch_size,
        shuffle    = shuffle,
        num_workers= 0,                 # in-process; fine for CPU/GPU
    )


def _train_loop(
    model:      nn.Module,
    X:          np.ndarray,
    y:          np.ndarray,
    *,
    n_classes:  int,
    epochs:     int,
    batch_size: int,
    lr:         float,
    weight_decay: float,
    val_split:  float,
    patience:   int,
    device:     torch.device,
    verbose:    bool = False,
) -> tuple[int, float, float]:
    """Generic train loop with optional held-out validation + early stopping.

    Returns ``(epochs_used, final_train_loss, final_val_loss)``.
    """
    model = model.to(device)
    n = X.shape[0]

    # Optional validation split
    if val_split and val_split > 0:
        n_val = int(round(n * val_split))
        if n_val < 8:
            n_val = 0
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]
    else:
        Xtr, ytr, Xva, yva = X, y, None, None

    train_loader = _make_loader(Xtr, ytr, batch_size, True,  device)
    val_loader   = (
        _make_loader(Xva, yva, batch_size, False, device)
        if Xva is not None else None
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val   = float("inf")
    best_state = None
    bad_epochs = 0
    train_loss_final = float("nan")
    val_loss_final   = float("nan")
    epochs_used = 0

    for ep in range(1, epochs + 1):
        model.train()
        loss_acc = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()
            loss_acc += float(loss.item())
            n_batches += 1
        train_loss_final = loss_acc / max(1, n_batches)
        epochs_used = ep

        if val_loader is None:
            if verbose:
                print(f"  epoch {ep:3d}  train_loss={train_loss_final:.4f}")
            continue

        model.eval()
        val_acc = 0.0
        nb = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                val_acc += float(F.cross_entropy(model(xb), yb).item())
                nb += 1
        val_loss = val_acc / max(1, nb)
        val_loss_final = val_loss
        if verbose:
            print(
                f"  epoch {ep:3d}  train_loss={train_loss_final:.4f}  "
                f"val_loss={val_loss:.4f}"
            )
        # Early stopping
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return epochs_used, train_loss_final, val_loss_final


# ──────────────────────────────────────────────────────────────────── #
# PatternNet (MATLAB patternnet equivalent)                             #
# ──────────────────────────────────────────────────────────────────── #

class _PatternNetModule(nn.Module):
    """Single hidden layer + sigmoid + softmax output.

    Matches MATLAB ``patternnet(nNeurons)``: one hidden layer of
    size *nNeurons* with tan-sigmoid activation, output layer with
    softmax.  The activation is ``tanh`` here (numerically equivalent
    to MATLAB's ``tansig``).
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden,   n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)              # logits — softmax applied in CE loss


class PatternNetMLP(Classifier):
    """MATLAB ``patternnet``-equivalent classifier.

    Parameters
    ----------
    n_neurons:
        Width of the single hidden layer.  Default 25 matches MATLAB's
        ``bhv_nn_multi_session_patternnet`` default.
    epochs, batch_size, lr, weight_decay:
        Standard training hyperparameters.
    val_split, patience:
        Held-out fraction and early-stopping patience.  Default
        ``val_split=0.0`` (= train on the entire bootstrap matrix,
        matching MATLAB).  Set ``val_split=0.1`` to enable.
    device:
        ``'cuda'``, ``'cpu'``, or ``None`` (auto).
    """

    backend = "patternnet"

    def __init__(
        self,
        n_neurons:    int   = 25,
        epochs:       int   = 200,
        batch_size:   int   = 256,
        lr:           float = 1e-3,
        weight_decay: float = 0.0,
        val_split:    float = 0.0,
        patience:     int   = 20,
        device:       str | None = None,
        verbose:      bool  = False,
    ) -> None:
        super().__init__()
        self.n_neurons    = int(n_neurons)
        self.epochs       = int(epochs)
        self.batch_size   = int(batch_size)
        self.lr           = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_split    = float(val_split)
        self.patience     = int(patience)
        self.device       = device
        self.verbose      = bool(verbose)
        self._model: _PatternNetModule | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "PatternNetMLP":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = int(y.max() + 1) if y.size else 0
        device = _select_device(self.device)
        self._model = _PatternNetModule(X.shape[1], self.n_neurons, n_classes)
        ep_used, tl, vl = _train_loop(
            self._model, X, y,
            n_classes  = n_classes,
            epochs     = self.epochs,
            batch_size = self.batch_size,
            lr         = self.lr,
            weight_decay = self.weight_decay,
            val_split  = self.val_split,
            patience   = self.patience,
            device     = device,
            verbose    = self.verbose,
        )
        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train    = X.shape[0],
            n_features = X.shape[1],
            n_classes  = n_classes,
            backend    = self.backend,
            epochs     = ep_used,
            train_loss = tl,
            val_loss   = vl,
            extra      = {"n_neurons": self.n_neurons},
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PatternNetMLP not fitted.")
        device = _select_device(self.device)
        self._model.to(device).eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self._model(torch.as_tensor(X).to(device))
            probs  = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def _save_state(self, dirpath: Path) -> None:
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "n_neurons":  self.n_neurons,
            },
            dirpath / "model.pt",
        )

    @classmethod
    def _load_state(cls, dirpath: Path) -> "PatternNetMLP":
        bundle = torch.load(dirpath / "model.pt", map_location="cpu",
                            weights_only=False)
        meta_path = dirpath / "meta.json"
        import json
        meta = json.loads(meta_path.read_text())
        n_features = meta["n_features_in_"]
        n_classes  = meta["n_classes_"]
        clf = cls(n_neurons=bundle["n_neurons"])
        clf._model = _PatternNetModule(n_features, bundle["n_neurons"], n_classes)
        clf._model.load_state_dict(bundle["state_dict"])
        clf._model.eval()
        return clf


# ──────────────────────────────────────────────────────────────────── #
# Multi-layer perceptron                                                #
# ──────────────────────────────────────────────────────────────────── #

class _MLPModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...],
        n_classes:  int,
        dropout:    float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifierTorch(Classifier):
    """Multi-layer perceptron with ReLU + dropout."""

    backend = "mlp"

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64, 64),
        dropout:      float = 0.2,
        epochs:       int   = 200,
        batch_size:   int   = 256,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        val_split:    float = 0.1,
        patience:     int   = 20,
        device:       str | None = None,
        verbose:      bool  = False,
    ) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout      = float(dropout)
        self.epochs       = int(epochs)
        self.batch_size   = int(batch_size)
        self.lr           = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_split    = float(val_split)
        self.patience     = int(patience)
        self.device       = device
        self.verbose      = bool(verbose)
        self._model: _MLPModule | None = None

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = int(y.max() + 1) if y.size else 0
        self._model = _MLPModule(X.shape[1], self.hidden_sizes, n_classes,
                                 self.dropout)
        ep_used, tl, vl = _train_loop(
            self._model, X, y,
            n_classes=n_classes, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, val_split=self.val_split,
            patience=self.patience, device=_select_device(self.device),
            verbose=self.verbose,
        )
        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train=X.shape[0], n_features=X.shape[1], n_classes=n_classes,
            backend=self.backend, epochs=ep_used,
            train_loss=tl, val_loss=vl,
            extra={"hidden_sizes": list(self.hidden_sizes),
                   "dropout": self.dropout},
        )
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("MLPClassifierTorch not fitted.")
        device = _select_device(self.device)
        self._model.to(device).eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            return F.softmax(
                self._model(torch.as_tensor(X).to(device)), dim=1
            ).cpu().numpy()

    def _save_state(self, dirpath):
        torch.save({
            "state_dict":   self._model.state_dict(),
            "hidden_sizes": list(self.hidden_sizes),
            "dropout":      self.dropout,
        }, dirpath / "model.pt")

    @classmethod
    def _load_state(cls, dirpath):
        import json
        meta = json.loads((dirpath / "meta.json").read_text())
        bundle = torch.load(dirpath / "model.pt", map_location="cpu",
                            weights_only=False)
        clf = cls(hidden_sizes=tuple(bundle["hidden_sizes"]),
                  dropout=bundle["dropout"])
        clf._model = _MLPModule(meta["n_features_in_"],
                                tuple(bundle["hidden_sizes"]),
                                meta["n_classes_"],
                                bundle["dropout"])
        clf._model.load_state_dict(bundle["state_dict"])
        clf._model.eval()
        return clf


# ──────────────────────────────────────────────────────────────────── #
# 1-D CNN with temporal context window                                  #
# ──────────────────────────────────────────────────────────────────── #

class _Conv1DModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes:  int,
        n_filters:  int,
        kernel:     int,
        depth:      int,
        dropout:    float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for d in range(depth):
            layers += [
                nn.Conv1d(in_ch, n_filters, kernel_size=kernel,
                          padding=kernel // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = n_filters
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                   nn.Linear(n_filters, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, context, n_features) → permute to (batch, n_features, context)
        return self.net(x.permute(0, 2, 1))


def _build_context(X: np.ndarray, context_half: int) -> np.ndarray:
    """Pad X with edge values and roll into context windows.

    Returns ``(T, 2*context_half+1, n_features)``.
    """
    pad = np.repeat(X[:1], context_half, axis=0)
    pad_end = np.repeat(X[-1:], context_half, axis=0)
    Xp = np.concatenate([pad, X, pad_end], axis=0)
    n = X.shape[0]
    w = 2 * context_half + 1
    out = np.empty((n, w, X.shape[1]), dtype=X.dtype)
    for k in range(w):
        out[:, k, :] = Xp[k:k + n, :]
    return out


class Conv1DClassifier(Classifier):
    """1-D CNN that sees a ±context_half-sample window around each time step."""

    backend = "cnn"

    def __init__(
        self,
        context_half: int = 8,
        n_filters:    int = 32,
        kernel:       int = 5,
        depth:        int = 2,
        dropout:      float = 0.2,
        epochs:       int   = 100,
        batch_size:   int   = 128,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        val_split:    float = 0.1,
        patience:     int   = 15,
        device:       str | None = None,
        verbose:      bool  = False,
    ) -> None:
        super().__init__()
        self.context_half = int(context_half)
        self.n_filters    = int(n_filters)
        self.kernel       = int(kernel)
        self.depth        = int(depth)
        self.dropout      = float(dropout)
        self.epochs       = int(epochs)
        self.batch_size   = int(batch_size)
        self.lr           = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_split    = float(val_split)
        self.patience     = int(patience)
        self.device       = device
        self.verbose      = bool(verbose)
        self._model: _Conv1DModule | None = None

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        Xc = _build_context(X, self.context_half)
        n_classes = int(y.max() + 1) if y.size else 0
        self._model = _Conv1DModule(
            X.shape[1], n_classes, self.n_filters, self.kernel,
            self.depth, self.dropout,
        )
        ep_used, tl, vl = _train_loop(
            self._model, Xc, y,
            n_classes=n_classes, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, val_split=self.val_split,
            patience=self.patience, device=_select_device(self.device),
            verbose=self.verbose,
        )
        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train=X.shape[0], n_features=X.shape[1], n_classes=n_classes,
            backend=self.backend, epochs=ep_used, train_loss=tl, val_loss=vl,
            extra={"context_half": self.context_half,
                   "n_filters": self.n_filters,
                   "kernel": self.kernel, "depth": self.depth,
                   "dropout": self.dropout},
        )
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("Conv1DClassifier not fitted.")
        device = _select_device(self.device)
        self._model.to(device).eval()
        X = np.asarray(X, dtype=np.float32)
        Xc = _build_context(X, self.context_half)
        out = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, Xc.shape[0], bs):
                xb = torch.as_tensor(Xc[i:i+bs]).to(device)
                out.append(F.softmax(self._model(xb), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def _save_state(self, dirpath):
        torch.save({
            "state_dict": self._model.state_dict(),
            "context_half": self.context_half,
            "n_filters":  self.n_filters,
            "kernel":     self.kernel,
            "depth":      self.depth,
            "dropout":    self.dropout,
        }, dirpath / "model.pt")

    @classmethod
    def _load_state(cls, dirpath):
        import json
        meta = json.loads((dirpath / "meta.json").read_text())
        b = torch.load(dirpath / "model.pt", map_location="cpu",
                       weights_only=False)
        clf = cls(
            context_half=b["context_half"], n_filters=b["n_filters"],
            kernel=b["kernel"], depth=b["depth"], dropout=b["dropout"],
        )
        clf._model = _Conv1DModule(meta["n_features_in_"], meta["n_classes_"],
                                    b["n_filters"], b["kernel"],
                                    b["depth"], b["dropout"])
        clf._model.load_state_dict(b["state_dict"])
        clf._model.eval()
        return clf


# ──────────────────────────────────────────────────────────────────── #
# Bidirectional LSTM                                                    #
# ──────────────────────────────────────────────────────────────────── #

class _BiLSTMModule(nn.Module):
    def __init__(self, n_features, hidden, n_layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: (batch, seq, n_features)
        h, _ = self.lstm(x)
        return self.head(h)              # (batch, seq, n_classes)


class BiLSTMClassifier(Classifier):
    """Bidirectional LSTM for sequence labelling.

    Trains on chunks of length *seq_len* drawn from the bootstrap
    feature matrix (which has no inherent sequence structure since
    rows are sampled from many separate bouts; we treat consecutive
    rows as a synthetic sequence anyway, which works because the
    bootstrap groups same-state rows together in blocks).
    """

    backend = "lstm"

    def __init__(
        self,
        hidden:       int   = 64,
        n_layers:     int   = 1,
        dropout:      float = 0.0,
        seq_len:      int   = 64,
        epochs:       int   = 50,
        batch_size:   int   = 32,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        val_split:    float = 0.1,
        patience:     int   = 10,
        device:       str | None = None,
        verbose:      bool  = False,
    ) -> None:
        super().__init__()
        self.hidden       = int(hidden)
        self.n_layers     = int(n_layers)
        self.dropout      = float(dropout)
        self.seq_len      = int(seq_len)
        self.epochs       = int(epochs)
        self.batch_size   = int(batch_size)
        self.lr           = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_split    = float(val_split)
        self.patience     = int(patience)
        self.device       = device
        self.verbose      = bool(verbose)
        self._model: _BiLSTMModule | None = None

    def _to_seq(self, X, y=None):
        n = X.shape[0]
        # Pad to multiple of seq_len, then reshape
        pad = (-n) % self.seq_len
        if pad:
            X = np.concatenate([X, np.repeat(X[-1:], pad, axis=0)], axis=0)
            if y is not None:
                y = np.concatenate([y, np.repeat(y[-1:], pad, axis=0)], axis=0)
        Xs = X.reshape(-1, self.seq_len, X.shape[1])
        ys = None if y is None else y.reshape(-1, self.seq_len)
        return Xs, ys, pad

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        Xs, ys, _ = self._to_seq(X, y)
        n_classes = int(y.max() + 1) if y.size else 0
        self._model = _BiLSTMModule(X.shape[1], self.hidden, self.n_layers,
                                    n_classes, self.dropout)
        device = _select_device(self.device)
        self._model.to(device)

        # Bespoke loop: per-step CE over (batch, seq, classes)
        n = Xs.shape[0]
        n_val = int(round(n * self.val_split)) if self.val_split > 0 else 0
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr_t = torch.as_tensor(Xs[tr_idx])
        ytr_t = torch.as_tensor(ys[tr_idx])
        Xva_t = torch.as_tensor(Xs[val_idx]) if n_val else None
        yva_t = torch.as_tensor(ys[val_idx]) if n_val else None

        optim = torch.optim.Adam(self._model.parameters(),
                                 lr=self.lr, weight_decay=self.weight_decay)
        best_val   = float("inf")
        best_state = None
        bad = 0
        ep_used = 0
        train_loss_final = float("nan")
        val_loss_final   = float("nan")

        for ep in range(1, self.epochs + 1):
            self._model.train()
            order = torch.randperm(Xtr_t.shape[0])
            loss_acc = 0.0
            n_b = 0
            for i in range(0, Xtr_t.shape[0], self.batch_size):
                sub = order[i:i + self.batch_size]
                xb = Xtr_t[sub].to(device)
                yb = ytr_t[sub].to(device)
                optim.zero_grad()
                logits = self._model(xb)
                loss = F.cross_entropy(
                    logits.reshape(-1, n_classes), yb.reshape(-1),
                )
                loss.backward()
                optim.step()
                loss_acc += float(loss.item())
                n_b += 1
            train_loss_final = loss_acc / max(1, n_b)
            ep_used = ep

            if Xva_t is None:
                if self.verbose:
                    print(f"  epoch {ep:3d}  train_loss={train_loss_final:.4f}")
                continue
            self._model.eval()
            with torch.no_grad():
                xb = Xva_t.to(device); yb = yva_t.to(device)
                logits = self._model(xb)
                vl = float(F.cross_entropy(
                    logits.reshape(-1, n_classes), yb.reshape(-1),
                ).item())
            val_loss_final = vl
            if self.verbose:
                print(f"  epoch {ep:3d}  train_loss={train_loss_final:.4f}  "
                      f"val_loss={vl:.4f}")
            if vl < best_val - 1e-5:
                best_val = vl
                best_state = {k: v.detach().clone()
                              for k, v in self._model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  early stop at epoch {ep}")
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train=X.shape[0], n_features=X.shape[1], n_classes=n_classes,
            backend=self.backend, epochs=ep_used,
            train_loss=train_loss_final, val_loss=val_loss_final,
            extra={"hidden": self.hidden, "n_layers": self.n_layers,
                   "seq_len": self.seq_len, "dropout": self.dropout},
        )
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("BiLSTMClassifier not fitted.")
        device = _select_device(self.device)
        self._model.to(device).eval()
        X = np.asarray(X, dtype=np.float32)
        n_orig = X.shape[0]
        Xs, _, pad = self._to_seq(X)
        outs = []
        with torch.no_grad():
            bs = self.batch_size
            for i in range(0, Xs.shape[0], bs):
                xb = torch.as_tensor(Xs[i:i+bs]).to(device)
                out = F.softmax(self._model(xb), dim=2).cpu().numpy()
                outs.append(out)
        flat = np.concatenate(outs, axis=0).reshape(-1, self.n_classes_)
        if pad:
            flat = flat[:n_orig]
        return flat

    def _save_state(self, dirpath):
        torch.save({
            "state_dict": self._model.state_dict(),
            "hidden":     self.hidden,
            "n_layers":   self.n_layers,
            "dropout":    self.dropout,
            "seq_len":    self.seq_len,
        }, dirpath / "model.pt")

    @classmethod
    def _load_state(cls, dirpath):
        import json
        meta = json.loads((dirpath / "meta.json").read_text())
        b = torch.load(dirpath / "model.pt", map_location="cpu",
                       weights_only=False)
        clf = cls(hidden=b["hidden"], n_layers=b["n_layers"],
                  dropout=b["dropout"], seq_len=b["seq_len"])
        clf._model = _BiLSTMModule(meta["n_features_in_"], b["hidden"],
                                    b["n_layers"], meta["n_classes_"],
                                    b["dropout"])
        clf._model.load_state_dict(b["state_dict"])
        clf._model.eval()
        return clf


# ──────────────────────────────────────────────────────────────────── #
# Registration                                                          #
# ──────────────────────────────────────────────────────────────────── #

_registry.register("patternnet", PatternNetMLP)
_registry.register("mlp",        MLPClassifierTorch)
_registry.register("cnn",        Conv1DClassifier)
_registry.register("lstm",       BiLSTMClassifier)
