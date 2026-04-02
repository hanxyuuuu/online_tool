from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from fastapi import HTTPException

from .model_loader import get_loaded_model
from .schemas import InputSummary, PredictionResponse


DNA_EXPECTED_LENGTH = 101
PROTEIN_MAX_LENGTH = 800
VALID_DNA_CHARS = set("ACGTN")
VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class NormalizedInput:
    dna_sequence: str
    protein_sequence: str
    summary: InputSummary


def _clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def normalize_inputs(dna_sequence: str, protein_sequence: str) -> NormalizedInput:
    dna_clean = _clean_sequence(dna_sequence)
    protein_clean = _clean_sequence(protein_sequence)

    if not dna_clean:
        raise HTTPException(status_code=422, detail="DNA sequence is required.")
    if not protein_clean:
        raise HTTPException(status_code=422, detail="Protein sequence is required.")

    invalid_dna = sorted(set(dna_clean) - VALID_DNA_CHARS)
    if invalid_dna:
        raise HTTPException(
            status_code=422,
            detail=f"DNA sequence contains invalid characters: {', '.join(invalid_dna)}",
        )

    invalid_protein = sorted({ch for ch in protein_clean if not ch.isalpha()})
    if invalid_protein:
        raise HTTPException(
            status_code=422,
            detail=f"Protein sequence contains invalid non-letter characters: {', '.join(invalid_protein)}",
        )

    messages: List[str] = []
    original_dna_length = len(dna_clean)
    dna_was_padded = False
    dna_was_truncated = False

    if len(dna_clean) > DNA_EXPECTED_LENGTH:
        dna_clean = dna_clean[:DNA_EXPECTED_LENGTH]
        dna_was_truncated = True
        messages.append(
            f"DNA input was truncated from {original_dna_length} bp to {DNA_EXPECTED_LENGTH} bp."
        )
    elif len(dna_clean) < DNA_EXPECTED_LENGTH:
        dna_clean = dna_clean + ("N" * (DNA_EXPECTED_LENGTH - len(dna_clean)))
        dna_was_padded = True
        messages.append(
            f"DNA input was padded with N from {original_dna_length} bp to {DNA_EXPECTED_LENGTH} bp."
        )

    original_protein_length = len(protein_clean)
    protein_was_truncated = False
    protein_had_unknown_residues = False

    normalized_protein_chars: List[str] = []
    unknown_count = 0
    for ch in protein_clean:
        if ch in VALID_PROTEIN_CHARS:
            normalized_protein_chars.append(ch)
        elif ch.isalpha():
            normalized_protein_chars.append("X")
            protein_had_unknown_residues = True
            unknown_count += 1
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Protein sequence contains invalid character: {ch}",
            )

    protein_clean = "".join(normalized_protein_chars)
    if unknown_count:
        messages.append(
            f"Protein input contained {unknown_count} non-standard residue(s); they were mapped to X/UNK."
        )
    if len(protein_clean) > PROTEIN_MAX_LENGTH:
        protein_clean = protein_clean[:PROTEIN_MAX_LENGTH]
        protein_was_truncated = True
        messages.append(
            f"Protein input was truncated from {original_protein_length} aa to {PROTEIN_MAX_LENGTH} aa."
        )

    summary = InputSummary(
        original_dna_length=original_dna_length,
        normalized_dna_length=len(dna_clean),
        original_protein_length=original_protein_length,
        normalized_protein_length=len(protein_clean),
        dna_was_padded=dna_was_padded,
        dna_was_truncated=dna_was_truncated,
        protein_was_truncated=protein_was_truncated,
        protein_had_unknown_residues=protein_had_unknown_residues,
        messages=messages,
    )
    return NormalizedInput(dna_sequence=dna_clean, protein_sequence=protein_clean, summary=summary)


def predict_sequences(dna_sequence: str, protein_sequence: str) -> PredictionResponse:
    normalized = normalize_inputs(dna_sequence, protein_sequence)
    loaded = get_loaded_model()
    module = loaded.module
    model = loaded.model
    device = loaded.device

    dna_np = module.one_hot_encode_dna(normalized.dna_sequence, max_len=DNA_EXPECTED_LENGTH)
    protein_np = module.one_hot_encode_protein(normalized.protein_sequence, max_len=PROTEIN_MAX_LENGTH)

    dna_tensor = torch.from_numpy(dna_np).unsqueeze(0).to(device)
    protein_tensor = torch.from_numpy(protein_np).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    protein_tensor = protein_tensor.clone().detach().requires_grad_(True)

    logit, attn_weights = model(dna_tensor, protein_tensor, return_attn=True)
    positive_logit = logit.squeeze(0)
    probability = torch.sigmoid(positive_logit)
    positive_logit.backward()

    dna_scores_raw = attn_weights.detach().cpu().numpy().reshape(-1).astype(np.float32)
    dna_scores_norm = _min_max_normalize(dna_scores_raw)

    protein_scores_full = (
        (protein_tensor.grad * protein_tensor)
        .sum(dim=1)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float32)
    )
    protein_length = len(normalized.protein_sequence)
    protein_scores_raw = protein_scores_full[:protein_length]
    protein_scores_norm = _min_max_normalize(np.abs(protein_scores_raw))

    predicted_label = int(probability.item() >= 0.5)
    predicted_class_text = "TFBS-positive" if predicted_label == 1 else "TFBS-negative"

    return PredictionResponse(
        normalized_dna_sequence=normalized.dna_sequence,
        normalized_protein_sequence=normalized.protein_sequence,
        input_summary=normalized.summary,
        logit=float(positive_logit.detach().cpu().item()),
        probability=float(probability.detach().cpu().item()),
        predicted_label=predicted_label,
        predicted_class_text=predicted_class_text,
        dna_importance_raw=dna_scores_raw.tolist(),
        dna_importance_norm=dna_scores_norm.tolist(),
        protein_importance_raw=protein_scores_raw.tolist(),
        protein_importance_norm=protein_scores_norm.tolist(),
    )
