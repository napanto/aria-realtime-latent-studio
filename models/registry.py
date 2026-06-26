"""Static registry of the two real-time Aria model backends.

Each :class:`ModelSpec` describes where a model's weights/config live on the
Hugging Face Hub, where they land under the local ``weights/`` directory, and
which runtime backend drives it. This module is pure data + path helpers — it
imports no torch/mlx, so it is cheap to import from the GUI, the download
script, and tests alike.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from studio import WEIGHTS_DIR, ASSETS_DIR


class Backend(str, enum.Enum):
    """How a model is executed at run time."""

    MLX = "mlx"           # plain Aria, real-time MLX demo engine


@dataclass(frozen=True)
class HFFile:
    """One file to fetch from the Hub."""

    repo_id: str
    repo_type: str  # "model" | "dataset"
    path_in_repo: str
    local_name: str  # filename under the model's weights subdir


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    backend: Backend
    files: tuple[HFFile, ...]
    # MLX runtime knobs.
    aria_model_config: str | None = None     # e.g. "medium-emb"
    tokenizer_config_local: str | None = None  # relative to assets/ or weights/
    # Free-form notes surfaced in the GUI tooltip / README.
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def weights_subdir(self) -> Path:
        return WEIGHTS_DIR / self.key

    def local_path(self, hf_file: HFFile) -> Path:
        return self.weights_subdir / hf_file.local_name

    @property
    def primary_weight(self) -> Path:
        """The main checkpoint file (first listed)."""
        return self.local_path(self.files[0])

    def is_downloaded(self) -> bool:
        return all(self.local_path(f).exists() for f in self.files)


# ---------------------------------------------------------------------------
# HF asset coordinates (confirmed against the live repos 2026-06-24).
# ---------------------------------------------------------------------------
_JAZZ_MODEL_REPO = "napanto/jazz-piano-performance-modeling"           # model repo
_ARIA_BASE_REPO = "loubb/aria-medium-base"                       # model repo


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # (a) original Aria — real-time MLX demo checkpoint.
    "aria_base": ModelSpec(
        key="aria_base",
        display_name="Aria (original)",
        backend=Backend.MLX,
        aria_model_config="medium-emb",
        # The original demo ships its own tokenizer config; we vendor it.
        tokenizer_config_local="assets/demo-tokenizer-config.json",
        files=(
            HFFile(_ARIA_BASE_REPO, "model", "model-demo.safetensors",
                   "model-demo.safetensors"),
        ),
        notes="Upstream EleutherAI/aria real-time demo weights (medium-emb).",
    ),
    # (b) our jazz fine-tuned Aria — ALREADY MLX-converted on the Hub.
    "aria_jazz": ModelSpec(
        key="aria_jazz",
        display_name="Aria (jazz fine-tuned)",
        backend=Backend.MLX,
        aria_model_config="medium-emb",
        # This deployment ships its own tokenizer config alongside the weights.
        tokenizer_config_local="weights/aria_jazz/tokenizer-config.json",
        files=(
            HFFile(_JAZZ_MODEL_REPO, "model",
                   "aria-real-time/mlx-deployed/model.safetensors",
                   "model.safetensors"),
            HFFile(_JAZZ_MODEL_REPO, "model",
                   "aria-real-time/mlx-deployed/config.json", "config.json"),
            HFFile(_JAZZ_MODEL_REPO, "model",
                   "aria-real-time/mlx-deployed/tokenizer-config.json",
                   "tokenizer-config.json"),
        ),
        notes="Drop-in jazz replacement for the MLX demo checkpoint.",
    ),
}


# Per-model sampling defaults (the GUI applies these when a model is selected).
# Both Aria models sample with temperature + min_p (the real-time demo sampler).
# Each model lists ONLY the sampling knobs its sampler actually uses, with the
# value it was tuned/swept with, so the GUI renders exactly these and the knob
# count adapts per model.
MODEL_SAMPLING: dict[str, dict] = {
    "aria_base":       {"temperature": 1.2, "min_p": 0.035},
    "aria_jazz":       {"temperature": 1.1, "min_p": 0.02},
}


def get_sampling(key: str) -> dict:
    return MODEL_SAMPLING.get(key, {"temperature": 1.0, "min_p": 0.03})


def get_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model key {key!r}; valid: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]


def resolve_asset(rel_or_local: str) -> Path:
    """Resolve a ``tokenizer_config_local`` string to an absolute path.

    Accepts paths relative to the repo root (``assets/...`` or ``weights/...``).
    ``weights/...`` paths honour the ``LATENT_STUDIO_WEIGHTS`` override so the
    weights dir can live outside the repo (e.g. on a bigger volume).
    """
    p = Path(rel_or_local)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "weights":
        return (WEIGHTS_DIR.joinpath(*parts[1:])).resolve()
    return (ASSETS_DIR.parent / rel_or_local).resolve()
