"""
Model architectures — imported by train.py, quantize.py, benchmarks/ and
federated-learning/ so there is exactly one definition of each network.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.schema import N_CLASSES, N_FEATURES


class EdgeShieldNet(nn.Module):
    """Edge-appropriate MLP (~55k params — still ~8x fewer than the baseline
    and a few hundred KB on disk). This is THE EdgeShield model: trained
    centrally in train.py, federated in run_federated.py, quantized in
    quantize.py, served by ml-inference-service."""

    def __init__(self, in_dim: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class BaselineNet(nn.Module):
    """'Centralized IEEE-style' baseline: a much larger float32 MLP (~700k
    params), representative of published IoT-IDS models that assume server-
    class hardware. Used ONLY by benchmarks/compare_baseline.py to produce the
    measured baseline-vs-EdgeShield table."""

    def __init__(self, in_dim: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_onnx(model: nn.Module, path, in_dim: int = N_FEATURES):
    model.eval()
    dummy = torch.zeros(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
