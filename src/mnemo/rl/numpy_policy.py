"""Torch-free inference for the trained MaskablePPO policy.

The deployed policy is a tiny tanh MLP with a masked-categorical head, so serving only needs numpy.
`export_policy` converts an sb3 checkpoint to an `.npz`; `NumpyPolicy` runs it and is verified to
reproduce `MaskablePPO.predict(deterministic=True)` (see tests/test_numpy_policy.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain import Action
from ..engine import Decision
from ..features import feature_mask
from ..operations import Overflow


def export_policy(model_zip: str | Path, out_path: str | Path) -> Path:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_zip), device="cpu")
    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if hasattr(m, "weight")] + [pol.action_net]
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float64)
    out = Path(out_path)
    np.savez(out, n_layers=np.array(len(layers)), **arrays)
    return out


class NumpyPolicy:
    overflow: Overflow = "reject"

    def __init__(self, npz_path: str | Path, *, name: str = "mnemo-ppo", ablate: tuple[str, ...] = ()) -> None:
        data = np.load(str(npz_path))
        n = int(data["n_layers"])
        self.weights = [(data[f"w{i}"], data[f"b{i}"]) for i in range(n)]
        self.name = name
        self.fmask = feature_mask(ablate)

    def reset(self) -> None:
        return None

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float64)
        for w, b in self.weights[:-1]:
            x = np.tanh(x @ w.T + b)
        w, b = self.weights[-1]
        return x @ w.T + b

    def act(self, d: Decision) -> Action:
        logits = self.logits(d.obs * self.fmask)
        logits = np.where(d.mask, logits, -np.inf)
        return Action(int(np.argmax(logits)))
