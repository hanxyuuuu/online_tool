---
title: TF-DNA Binding Atlas
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
fullWidth: true
short_description: Predict TF-DNA binding, visualize sequence importance, and optionally email results.
---

# TF-DNA Binding Atlas

This Space serves a FastAPI app for transcription factor binding site prediction from one protein sequence and one 101 bp DNA sequence.

Features:

- probability and class prediction
- DNA attention visualization
- protein gradient-times-input saliency
- optional email delivery of the prediction summary

The deployment-ready model assets are bundled inside:

- `webapp/model_assets/multimodal_model.py`
- `webapp/model_assets/best_model_v2_pro.pth`

For local development and deployment details, see:

- `README_webapp.md`
