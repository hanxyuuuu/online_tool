# TF-DNA Inference Web App

This project provides a local and deployable FastAPI demo for the trained TF-DNA binding model. The deployable app bundles the inference-ready model definition and weights inside the repository under:

- `webapp/model_assets/multimodal_model.py`
- `webapp/model_assets/best_model_v2_pro.pth`

The bundled model definition preserves the original one-hot encoding behavior and the deployed `MultiModalMSTC_CrossAttn` architecture used to load the provided state dict.

The app runs on CPU and exposes a simple browser UI for:

- protein sequence input
- DNA sequence input
- optional email input for result delivery
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

## Bundled model assets

By default, the app loads model assets from the repository itself, which makes it suitable for GitHub push and Render deployment.

Optional environment variable overrides are supported:

```powershell
$env:TFDNA_MODEL_SCRIPT_PATH = "C:\path\to\multimodal_model.py"
$env:TFDNA_MODEL_WEIGHTS_PATH = "C:\path\to\best_model_v2_pro.pth"
```

If these variables are unset, the app uses the bundled repository assets.

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
- `email_requested`
- `email_delivery_status`
- `email_delivery_message`
- DNA raw and normalized importance arrays
- protein raw and normalized importance arrays

## Email delivery

If the user provides an email address in the UI, the backend will try to send the prediction summary after inference.

Email delivery uses SMTP from environment variables:

```powershell
$env:TFDNA_SMTP_HOST = "smtp.example.com"
$env:TFDNA_SMTP_PORT = "587"
$env:TFDNA_SMTP_USERNAME = "sender@example.com"
$env:TFDNA_SMTP_PASSWORD = "your-password"
$env:TFDNA_SMTP_FROM = "sender@example.com"
$env:TFDNA_SMTP_STARTTLS = "true"
$env:TFDNA_SMTP_SSL = "false"
```

Notes:

- If no email address is provided, prediction still runs and delivery is skipped.
- If SMTP is not configured, prediction still runs and the UI/API reports `not_configured`.
- If SMTP delivery fails, prediction still runs and the UI/API reports `failed`.

## GitHub and Render deployment

### Push to GitHub

After you commit the repository, push it to a GitHub repository as usual.

### Render

This repository includes a ready-to-use `render.yaml`.

Render setup:

1. Create a new Render Web Service from this GitHub repository.
2. Let Render detect the included `render.yaml`.
3. In the Render dashboard, fill in the SMTP environment variables:
   - `TFDNA_SMTP_HOST`
   - `TFDNA_SMTP_PORT`
   - `TFDNA_SMTP_USERNAME`
   - `TFDNA_SMTP_PASSWORD`
   - `TFDNA_SMTP_FROM`
   - `TFDNA_SMTP_STARTTLS`
   - `TFDNA_SMTP_SSL`
4. Deploy.

The generated Render URL will be publicly accessible. You can then bind a custom domain in Render if needed.

## Hugging Face Spaces deployment

This repository now includes the files required for a Docker Space:

- `README.md` with `sdk: docker`
- `Dockerfile`
- bundled model assets inside `webapp/model_assets/`

To deploy on Hugging Face Spaces:

1. Create a new Space and choose the `Docker` SDK.
2. Push this repository content to the Space repository.
3. In the Space settings, add runtime secrets or variables for:
   - `TFDNA_SMTP_HOST`
   - `TFDNA_SMTP_PORT`
   - `TFDNA_SMTP_USERNAME`
   - `TFDNA_SMTP_PASSWORD`
   - `TFDNA_SMTP_FROM`
   - `TFDNA_SMTP_STARTTLS`
   - `TFDNA_SMTP_SSL`
4. Let the Space build and start.

Hugging Face reads the YAML block from the root `README.md` and the container definition from the `Dockerfile`.

## Current limitations

- CPU inference only in this demo.
- The app runs single-sequence inference only.
- DNA attribution is attention-based, not gradient-based.
- Protein attribution may be noisy because it is gradient-based.
- Model assets are read from their original external path and are not bundled into this repo.
