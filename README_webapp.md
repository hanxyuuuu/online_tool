# TF-DNA Inference Web App

This project provides a local FastAPI demo for the trained TF-DNA binding model. It uses the original model definition and weights directly from:

- `D:\project\NT\Motif_logo\cnn_multimodal_mstc_crossattn_v2_pro_2.py`
- `D:\project\NT\Motif_logo\best_model_v2_pro.pth`

The app runs on CPU and exposes a simple browser UI for:

- protein sequence input
- DNA sequence input
- predicted probability and class
- DNA importance visualization
- protein importance visualization

## Runtime

The validated Python environment for this demo is:

- `D:\ANACONDA\envs\vit_env\python.exe`

That environment already contains a working PyTorch installation that can load the provided weights.

## Install backend web dependencies

If `fastapi`, `uvicorn`, or `jinja2` are not installed in `vit_env`, install them there:

```powershell
& "D:\ANACONDA\envs\vit_env\python.exe" -m pip install -r requirements.txt
```

## Run the app

From this repository root:

```powershell
& "D:\ANACONDA\envs\vit_env\python.exe" -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Then open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Input normalization

### DNA

- DNA is uppercased and whitespace is removed.
- Only `A/C/G/T/N` are accepted.
- The model expects exactly `101` bp.
- If the input is shorter than `101`, it is padded with `N`.
- If the input is longer than `101`, it is truncated to `101`.

The normalized DNA sequence is returned by the API and shown in the UI.

### Protein

- Protein is uppercased and whitespace is removed.
- Non-letter characters are rejected.
- Standard amino acids (`ACDEFGHIKLMNPQRSTVWY`) are preserved.
- Other alphabetic residues are converted to `X`, which is then encoded through the model's UNK channel.
- If the input is longer than `800` aa, it is truncated to `800`.

The UI only displays the normalized real protein sequence length. It does not display padded positions from the model tensor.

## Attribution methods

### DNA attribution

DNA importance uses the model's returned cross-attention weights from:

- `forward(dna_x, prot_x, return_attn=True)`

These attention weights are returned as raw values and normalized to `[0, 1]` for rendering.

### Protein attribution

Protein importance uses `gradient * input` on the protein one-hot tensor:

1. enable gradients on the one-hot protein tensor
2. backpropagate from the positive-class logit
3. compute `(grad * input).sum(channel)`
4. keep only the true normalized sequence length
5. normalize the per-position magnitudes for the UI

This is not a protein-side attention map. It is a gradient-based attribution signal.

## API

### `GET /`

Serves the browser UI.

### `POST /api/predict`

Request body:

```json
{
  "dna_sequence": "ACGT...",
  "protein_sequence": "MSTN..."
}
```

Response includes:

- normalized sequences
- input normalization summary
- `logit`
- `probability`
- `predicted_label`
- `predicted_class_text`
- DNA raw and normalized importance arrays
- protein raw and normalized importance arrays

## Current limitations

- CPU inference only in this demo.
- The app runs single-sequence inference only.
- DNA attribution is attention-based, not gradient-based.
- Protein attribution may be noisy because it is gradient-based.
- Model assets are read from their original external path and are not bundled into this repo.
