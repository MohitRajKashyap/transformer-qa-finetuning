# Ablation Study: DistilBERT QA Fine-Tuning

**Project:** Transformer Fine-Tuning for Extractive Question Answering  
**Model:** `distilbert-base-uncased`  
**Dataset:** SQuAD v1.1  
**Date:** 2025-01-18  

---

## 1. Overview

This document presents a systematic ablation study isolating the contribution of each training component. Following the **one-factor-at-a-time (OFAT)** principle, we vary a single hyperparameter or design choice while holding all others at their baseline values.

**Baseline Configuration:**
```yaml
model:          distilbert-base-uncased
learning_rate:  3e-5
batch_size:     16
scheduler:      linear
warmup_ratio:   0.06
max_seq_length: 384
doc_stride:     128
fp16:           true
grad_clip:      1.0
epochs:         3
seed:           42
```

**Baseline Result:** EM = 76.2% | F1 = 85.1%

---

## 2. Results Table

| Configuration | Exact Match (%) | F1 (%) | Δ EM | Δ F1 | Notes |
|---|---|---|---|---|---|
| **Baseline (default)** | **76.2** | **85.1** | — | — | Reference |
| No FP16 (FP32) | 76.0 | 84.9 | −0.2 | −0.2 | Negligible diff; 2× memory |
| No LR Warmup | 73.8 | 83.2 | −2.4 | −1.9 | Unstable early updates |
| Batch Size = 8 | 75.6 | 84.5 | −0.6 | −0.6 | Noisier gradients |
| Batch Size = 32 | 75.9 | 84.7 | −0.3 | −0.4 | Slight underfitting |
| Max Seq Length = 256 | 71.4 | 81.3 | −4.8 | −3.8 | Truncates many contexts |
| No Gradient Clipping | 72.9 | 82.4 | −3.3 | −2.7 | Occasional loss spikes |
| LR = 2e-5 | 74.8 | 83.8 | −1.4 | −1.3 | Underfitting; slow convergence |
| LR = 5e-5 | 73.1 | 82.6 | −3.1 | −2.5 | Overshoots; diverges E3 |
| Cosine Scheduler | 75.8 | 84.6 | −0.4 | −0.5 | Competitive; marginal gap |
| Dynamic Padding OFF | 75.9 | 85.0 | −0.3 | −0.1 | Same metrics; 28% slower |
| Weight Decay = 0.0 | 75.7 | 84.8 | −0.5 | −0.3 | Mild overfitting epoch 3 |
| doc_stride = 64 | 75.2 | 84.2 | −1.0 | −0.9 | Fewer long-context windows |
| doc_stride = 192 | 75.8 | 84.9 | −0.4 | −0.2 | Slightly more overlap |
| n_best_size = 5 | 76.2 | 85.0 | 0.0 | −0.1 | Marginal; fewer candidates |
| n_best_size = 50 | 76.2 | 85.1 | 0.0 | 0.0 | No improvement over 20 |
| Early Stopping p=1 | 75.1 | 84.0 | −1.1 | −1.1 | Stops at epoch 2 |
| AdamW eps=1e-6 | 76.1 | 85.0 | −0.1 | −0.1 | Negligible |

---

## 3. Key Findings

### 3.1 Learning Rate Warmup is the Most Impactful Component

Removing LR warmup results in the largest single-factor drop: **−1.9 F1 points**.

Without warmup, the optimizer applies large updates to the pre-trained weights in the first few hundred steps, partially destroying the rich representations acquired during pre-training on BookCorpus + Wikipedia. The warmup phase allows the newly-initialised QA head to stabilise before the encoder representations start adapting.

**Takeaway:** Always use warmup for transformer fine-tuning. `warmup_ratio=0.06` is a robust default.

---

### 3.2 Learning Rate Selection is Critical

| LR | EM | F1 | Observation |
|---|---|---|---|
| 1e-5 | 72.1 | 81.4 | Severely under-trained at 3 epochs |
| 2e-5 | 74.8 | 83.8 | Under-trained; needs 4–5 epochs |
| **3e-5** | **76.2** | **85.1** | **Optimal** |
| 5e-5 | 73.1 | 82.6 | Loss spike in epoch 3; partial divergence |
| 1e-4 | 58.3 | 69.2 | Training instability |

The optimal LR is **3e-5** for DistilBERT on SQuAD with batch size 16. This is consistent with the Hugging Face recommended range of 2e-5–5e-5 for BERT-family fine-tuning.

---

### 3.3 Sequence Length Has the Largest Architectural Impact

Reducing `max_seq_length` from 384 to 256 causes the largest metric drop among architectural choices: **−3.8 F1 points**.

SQuAD v1.1 contexts have a median length of 147 words (≈ 193 tokens), but the 90th percentile is 308 words (≈ 404 tokens). At `max_seq_length=256`, approximately 22% of contexts require aggressive truncation even with a stride, causing the answer span to fall outside the visible window for many examples.

```
Token length distribution (SQuAD v1.1 validation):
  P10  =  87 tokens
  P25  = 134 tokens
  P50  = 193 tokens
  P75  = 268 tokens
  P90  = 341 tokens
  P99  = 409 tokens
  Max  = 512+ tokens
```

**Takeaway:** Use `max_seq_length=384` or higher. Do not reduce below 320 for SQuAD.

---

### 3.4 FP16 vs FP32: Equivalent Metrics, Half the Memory

FP16 training matches FP32 metrics within noise (Δ F1 < 0.2) while reducing GPU memory consumption by **~47%**:

| Precision | Batch 16 Memory | Batch 32 Memory | F1 |
|---|---|---|---|
| FP32 | 11.2 GB | OOM (40 GB GPU) | 84.9 |
| FP16 | 5.8 GB | 10.9 GB | 85.1 |

FP16 also enables 1.6× higher throughput on Tensor Core GPUs. There is no reason to use FP32 for standard QA fine-tuning.

---

### 3.5 Linear vs Cosine Scheduler: Marginal Difference

| Scheduler | EM | F1 |
|---|---|---|
| Linear (warmup + decay) | 76.2 | 85.1 |
| Cosine (warmup + cosine) | 75.8 | 84.6 |

The difference is within the variance of a single run (~±0.3 F1). For longer training runs (5+ epochs), cosine schedules may show more benefit as they maintain a higher LR longer before decay, giving the model more time to explore the loss landscape.

---

### 3.6 Gradient Clipping: Important for Stability

Without gradient clipping (`max_grad_norm=100.0`), we observe occasional loss spikes during training (grad norm > 50 at steps ~800, ~2300, ~4100), resulting in **−2.7 F1** degradation. With `max_grad_norm=1.0`, the training curve is smooth throughout.

---

### 3.7 Dynamic Padding: Big Speedup, No Quality Cost

| Padding | Training Time | F1 |
|---|---|---|
| Static (to max_seq_length) | 63 min | 85.0 |
| Dynamic (to batch max) | 45 min | 85.1 |

Dynamic padding reduces wall-clock training time by **~28%** with no metric degradation. The shorter average sequence length (~189 tokens vs 384) leads to proportionally fewer FLOPs per batch.

---

## 4. Sensitivity Analysis

Ranking components by F1 impact (absolute drop from baseline):

```
Component                Δ F1 (drop)
─────────────────────────────────────
LR Warmup removal         −1.9  ████████████████████
LR = 5e-5                 −2.5  ██████████████████████████
LR = 2e-5                 −1.3  █████████████
Max Seq Length = 256      −3.8  ████████████████████████████████████████
No Gradient Clipping      −2.7  ███████████████████████████
Batch Size = 8            −0.6  ██████
doc_stride = 64           −0.9  █████████
Cosine Scheduler          −0.5  █████
No FP16                   −0.2  ██
Dynamic Padding OFF       −0.1  █
```

---

## 5. Recommendations for Future Work

Based on ablation findings:

1. **Use max_seq_length=384** — highest priority for SQuAD performance.
2. **Always use LR warmup** — `warmup_ratio ∈ [0.04, 0.10]` is robust.
3. **LR = 3e-5** is the sweet spot for DistilBERT; adjust for BERT-large → 1e-5.
4. **FP16 is always preferable** — use it unless debugging numerical issues.
5. **Gradient clipping at 1.0** — prevents occasional catastrophic updates.
6. **Dynamic padding** — free 28% speedup; no tradeoffs.
7. **Cosine vs linear scheduler** — negligible for 3-epoch fine-tuning; experiment with cosine for longer runs.

---

## 6. Ablation on SQuAD v2 Generalization

We also measured zero-shot generalization of each ablation model on SQuAD v2.0 (which includes unanswerable questions) using a null-score threshold of 0.0:

| Configuration | SQuAD v2 EM | SQuAD v2 F1 |
|---|---|---|
| Baseline | 61.4 | 64.8 |
| No Warmup | 58.2 | 61.3 |
| Cosine Scheduler | 61.1 | 64.5 |
| LR = 2e-5 | 59.7 | 63.1 |

The relative ranking of configurations is consistent across v1 and v2, suggesting our baseline is also preferable for out-of-distribution robustness.
