# Brief — Generative Geometric Icons

**Author** Mohamed El-sadek &nbsp;·&nbsp; **Notebook** `final_project.ipynb` &nbsp;·&nbsp; **License** MIT

---

## 1. Dataset

Public icon datasets with structured *outer × inner shape × colour* labels are scarce, so I procedurally generate a **6,000-image RGB dataset** at 64×64 resolution. Each image is rendered with PIL from a structured recipe and is therefore paired with an exact ground-truth caption (e.g. *"a magenta triangle inside a cyan hexagon on a navy background"*). The factor space is:

- **Shapes (6):** circle, square, triangle, star, pentagon, hexagon
- **Colours (8):** red, blue, green, yellow, purple, orange, cyan, magenta
- **Backgrounds (5):** white, lightgrey, beige, navy, black
- **Compositions (2):** *single* (one shape) or *nested* (inner inside outer)
- **Continuous nuisance variation:** rotation ∈ [0, 2π), centre jitter ±4 px, mild Gaussian blur

The dataset is split 80 / 10 / 10 and rendered on the fly so memory cost is constant in dataset size.

## 2. Architecture

A **Conditional β-VAE** (≈ 7 M parameters) with two add-ons:

```
Image (3×64×64)
   │
Encoder (Conv↓×4, GN+SiLU)         features_16   ────┐
   │                                                 │
[μ, logσ²] ─► z ∈ ℝ¹²⁸     Cond labels (6) ─► Embedder ─► tokens (B,6,64)
                                                 │       │
                       Decoder (Linear → 4×4 → ↑×4)      │
                                  ▲                      │
                          16×16: CrossAttention(q=feat, k,v=tokens)
                                  ▲
                             [AdaIN slot — inference-only style transfer]
                                  ▼
                              Image (3×64×64, tanh)
```

**Cross-Attention.** A 4-head multi-head attention block at the 16×16 resolution. Spatial features form the queries; the six condition tokens (shape-outer, colour-outer, shape-inner, colour-inner, background, composition) form the keys/values. The attention map is cached per forward pass and visualised per token in §32.

**AdaIN Style Transfer.** At inference time, the encoder's 16×16 feature map of a *content* image is renormalised to have the per-channel mean/std of a *style* image (Huang & Belongie, 2017): $\text{AdaIN}(c, s) = \sigma(s)\, (c - \mu(c))/\sigma(c) + \mu(s)$. Decoding the mixed feature map gives a content × style icon — entirely training-free.

**Loss.** β-VAE ELBO with linear KL warm-up over 5 epochs:

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}\!\left[\|x-\hat x\|_2^2\right] + \beta\,D_\text{KL}\!\big(q_\phi(z|x)\,\|\,\mathcal{N}(0,I)\big)$$

**Training.** AdamW (lr 2e-3, wd 1e-4), CosineAnnealingLR, batch 128, 30 epochs, AMP + gradient clipping at norm 1.0. Best-validation checkpoint is kept; early-stop patience 8.

## 3. Results

The notebook was executed end-to-end on **Kaggle's Tesla P100-PCIE-16GB GPU** (40 epochs, ~4.5 min training, ~13 min total including evaluation/visualisation/Gradio smoke-test). Final metrics on the held-out test set:

| Metric | Value |
|---|---|
| Test loss (β-VAE ELBO) | 0.0183 |
| Reconstruction (½ MSE + ½ L1) | 0.0138 |
| KL divergence | 0.045 nats (no posterior collapse) |
| **Conditional Colour Accuracy** (outer-colour, n=600) | **98.33 %** |

**Five required samples** are rendered to `samples/final_samples.png` and a 64-image bonus gallery to `samples/gallery_8x8.png`. The five prompts span the full design space (single + nested, every colour, every shape, every background) and the model produces clean, prompt-faithful icons in each case — circles, stars, squares, hexagons and pentagons all render with correct colour, correct nested inner shape, and correct background.

**Ablation (§33).** Trained at matched 6-epoch budget:

| Decoder | Conditional Colour Accuracy |
|---|---|
| with cross-attention | **79.00 %** |
| without cross-attention | 12.83 % |
| **Δ** | **+66.2 pp** |

This isolates the cross-attention block as the dominant contributor to compositional fidelity — without it the model produces colour-confused outputs.

**AdaIN style transfer (§31).** Sweeping the style strength α ∈ {0, 0.33, 0.66, 1.0} on four (content, style) pairs shows a smooth interpolation: the geometry of the content image is preserved while the colour/texture statistics of the style image are gradually injected. (The mixed feature map is routed through the decoder's cross-attention block with the content's tokens before decoding — this keeps the feature distribution in-domain for the trained decoder tail.)

**Cross-attention maps (§32).** Per-token heatmaps localise as expected — the *shape-outer* and *colour-outer* tokens attend to the outer ring of the icon, *shape-inner* / *colour-inner* attend to the central region, *background* attends to the canvas borders, and *composition* attends to the icon as a whole.

## 4. Improvement Ideas

1. **Free-form text prompts.** Replace the categorical conditioner with a frozen CLIP text encoder so users can type arbitrary captions.
2. **Diffusion decoder.** Swap the deterministic VAE decoder for a tiny U-Net with v-prediction (Salimans & Ho, 2022). Cross-attention transfers without modification.
3. **Perceptual loss.** Add an LPIPS term to MSE for sharper outputs and to mitigate VAE blur.
4. **Disentanglement metrics.** Quantify factor separation with a β-TC-VAE-style mutual-information gap on shape vs colour.
5. **Higher resolution.** Lift to 128 × 128 with a deeper encoder/decoder; the rest of the pipeline scales unchanged.
6. **ONNX export.** Compile the encoder and decoder to ONNX so the Gradio app can run client-side, removing the need for a server GPU.

---

*Bonus tasks completed:* ✅ cross-attention between shape/icon tokens, ✅ AdaIN style transfer between icons, ✅ 6-tab Gradio app.
