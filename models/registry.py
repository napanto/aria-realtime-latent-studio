"""Static registry of the four model backends.

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
    TORCH_VAE = "torch"   # AriaVAE / Cadenza, PyTorch (MPS/CPU) latent engine
    MLX_VAE = "mlx_vae"   # AriaVAE / Cadenza, parity-checked MLX latent engine


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
    # MLX-only knobs (None for VAEs)
    aria_model_config: str | None = None     # e.g. "medium-emb"
    tokenizer_config_local: str | None = None  # relative to assets/ or weights/
    # VAE-only knobs (None for plain Aria)
    z_dim: int | None = None
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
_JAZZ_MODEL_REPO = "napaalm/jazz-piano-ispr-2025-2026"           # model repo
_ARIA_BASE_REPO = "loubb/aria-medium-base"                       # model repo
_VAE_DATA_REPO = "napaalm/jazz-piano-performance-generation"     # dataset repo


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
    # (c) AriaVAE — frozen Aria real-time decoder + 31.4M latent add-on (torch).
    "aria_vae": ModelSpec(
        key="aria_vae",
        display_name="AriaVAE (latent)",
        backend=Backend.TORCH_VAE,
        z_dim=128,
        # The AriaVAE decoder is the same real-time Aria, so it reuses the
        # deployed jazz tokenizer config (downloaded with aria_jazz). We also
        # vendor a fallback copy of the demo tokenizer in assets/.
        tokenizer_config_local="weights/aria_jazz/tokenizer-config.json",
        files=(
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/B05_pipeline/stageB/last.pt", "last.pt"),
        ),
        notes=(
            "z=128. Attributes: velocity_mean, velocity_std, note_density, "
            "ioi_entropy, pitch_mean, pitch_std, pedal_fraction. "
            "Prefix-injected (8 soft tokens) into a FROZEN real-time Aria "
            "decoder — same backbone as aria_jazz."
        ),
    ),
    # (d) Cadenza VAE — from-scratch Composer+Performer (torch).
    "cadenza_vae": ModelSpec(
        key="cadenza_vae",
        display_name="Cadenza VAE (latent)",
        backend=Backend.TORCH_VAE,
        z_dim=128,
        files=(
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/A05_kongFT/stageB/best.pt", "composer_best.pt"),
            # The Performer fill model is pretrained alongside the Composer.
            # Path filled in by download script once confirmed; see notes.
        ),
        notes=(
            "z=128, in-attention (additive W_pre) z-injection. Two-stage: "
            "Composer (composition tokens) -> Performer fill -> MIDI. "
            "Needs a Performer checkpoint too (see STATUS.md)."
        ),
        extra={
            "performer_ckpt_hint": (
                "vae_campaign/A05_kongFT/performer_recreated/best.pt"
            ),
        },
    ),
    # (e) AriaVAE — real-time MLX latent engine (parity-checked on M1).
    # Same latent as aria_vae, but the z-prefix + per-layer z-residual ride the
    # KV-cached MLX decoder => ~52 tok/s instead of torch full-reeval (TODO #6).
    "aria_vae_mlx": ModelSpec(
        key="aria_vae_mlx",
        display_name="AriaVAE (latent, real-time MLX)",
        backend=Backend.MLX_VAE,
        z_dim=128,
        tokenizer_config_local="weights/aria_jazz/tokenizer-config.json",
        files=(
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/ariavae_mlx/aria_vae_decoder.safetensors",
                   "aria_vae_decoder.safetensors"),
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/ariavae_mlx/aria_vae_latent.safetensors",
                   "aria_vae_latent.safetensors"),
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/ariavae_mlx/aria_vae_config.json",
                   "aria_vae_config.json"),
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/ariavae_mlx/latent_directions.npz",
                   "latent_directions.npz"),
        ),
        notes=(
            "Parity-checked MLX AriaVAE: 8 soft z-prefix tokens prefilled into "
            "the KV cache + per-layer z-residual on the frozen jazz Aria decoder. "
            "~52 tok/s / 19 ms/token (int8, M1). Probe fit on 320 PiJAMA windows."
        ),
    ),
    # (f) Cadenza — MLX two-stage (Composer + recreated Performer).
    "cadenza_vae_mlx": ModelSpec(
        key="cadenza_vae_mlx",
        display_name="Cadenza VAE (latent, MLX 2-stage)",
        backend=Backend.MLX_VAE,
        z_dim=128,
        files=(
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/cadenza_mlx/cadenza_composer.safetensors",
                   "cadenza_composer.safetensors"),
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/cadenza_mlx/cadenza_performer.safetensors",
                   "cadenza_performer.safetensors"),
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/cadenza_mlx/cadenza_config.json",
                   "cadenza_config.json"),
            HFFile(_VAE_DATA_REPO, "dataset",
                   "vae_campaign/cadenza_mlx/latent_directions_cadenza.npz",
                   "latent_directions_cadenza.npz"),
        ),
        notes=(
            "MLX Composer + the RECREATED Performer "
            "(vae_campaign/A05_kongFT/performer_recreated, val ppl 37.1) baked "
            "into cadenza_performer.safetensors. Two-stage render unblocked "
            "(TODO #5); latent control note_density +0.94."
        ),
    ),
}


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
