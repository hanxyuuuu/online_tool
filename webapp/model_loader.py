from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import ModuleType

import torch


MODEL_SCRIPT_PATH = Path(r"D:\project\NT\Motif_logo\cnn_multimodal_mstc_crossattn_v2_pro_2.py")
MODEL_WEIGHTS_PATH = Path(r"D:\project\NT\Motif_logo\best_model_v2_pro.pth")


@dataclass
class LoadedModel:
    module: ModuleType
    model: torch.nn.Module
    device: torch.device


_MODEL_CACHE: LoadedModel | None = None
_MODEL_LOCK = Lock()


def _import_model_module() -> ModuleType:
    if not MODEL_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Model script not found: {MODEL_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("tf_dna_model_module", str(MODEL_SCRIPT_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import model module from {MODEL_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _instantiate_model(module: ModuleType) -> torch.nn.Module:
    return module.MultiModalMSTC_CrossAttn(
        dna_channels=4,
        prot_channels=module.AA_VOCAB_SIZE,
        dna_branch_channels=64,
        prot_branch_channels=64,
        dna_kernels=(5, 9, 13),
        prot_kernels=(9, 15, 21),
        attn_heads=4,
    )


def get_loaded_model() -> LoadedModel:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    with _MODEL_LOCK:
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE

        if not MODEL_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")

        device = torch.device("cpu")
        module = _import_model_module()
        model = _instantiate_model(module)
        state_dict = torch.load(str(MODEL_WEIGHTS_PATH), map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Model weights are incompatible with the model definition. "
                f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
            )

        model.to(device)
        model.eval()
        _MODEL_CACHE = LoadedModel(module=module, model=model, device=device)
        return _MODEL_CACHE
