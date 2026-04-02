from __future__ import annotations

from typing import List

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    dna_sequence: str
    protein_sequence: str


class InputSummary(BaseModel):
    original_dna_length: int
    normalized_dna_length: int
    original_protein_length: int
    normalized_protein_length: int
    dna_was_padded: bool
    dna_was_truncated: bool
    protein_was_truncated: bool
    protein_had_unknown_residues: bool
    messages: List[str]


class PredictionResponse(BaseModel):
    normalized_dna_sequence: str
    normalized_protein_sequence: str
    input_summary: InputSummary
    logit: float
    probability: float
    predicted_label: int
    predicted_class_text: str
    dna_importance_raw: List[float]
    dna_importance_norm: List[float]
    protein_importance_raw: List[float]
    protein_importance_norm: List[float]
