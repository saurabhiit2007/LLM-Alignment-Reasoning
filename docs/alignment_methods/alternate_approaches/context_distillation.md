# Context Distillation

## 1. Overview

Context distillation is a technique that internalizes the behavior induced by a system prompt directly into a model's weights, eliminating the need to include that prompt at inference time. It was introduced by Bai et al. (2022) as a component of the Constitutional AI pipeline.

**Core idea:** A model conditioned on an alignment prompt (e.g., "Be helpful, harmless, and honest") produces better outputs than the same model without it. Context distillation fine-tunes the model on those better outputs — but *without* the prompt as input — so the model learns to behave as if the alignment prompt is always present.

This is distinct from **knowledge distillation** (compressing a larger model into a smaller one). In context distillation, the teacher and student are the same base model; the only difference is whether the alignment prompt is in the input.

---

## 2. How It Works

**Step 1 — Generate aligned responses:**

Sample responses from the model conditioned on an alignment context $c$ (a system prompt encoding desired behavior):

$$y \sim p_\theta(y \mid x, c)$$

**Step 2 — Fine-tune without the context:**

Train the model to produce those same responses given only the raw input $x$:

$$\mathcal{L} = -\mathbb{E}_{(x,y)}\left[\log p_\theta(y \mid x)\right]$$

The objective is to minimize the KL divergence between the prompted and unprompted distributions:

$$\min_\theta\; \mathrm{KL}\!\left[p(y \mid x, c)\;\|\; p_\theta(y \mid x)\right]$$

**Step 3 — Deploy without the context prompt:**

The fine-tuned model produces aligned outputs without needing $c$ in every request.

---

## 3. Role in Constitutional AI

In Bai et al. (2022), context distillation is used in the **SL-CAI** (Supervised Learning — Constitutional AI) stage:

1. Generate responses using a helpful-only model prompted with a set of principles (the "constitution")
2. Ask the model to critique and revise those responses (also prompted)
3. Fine-tune the model on the final revised responses *without* the constitution in the input

The result is a model that follows constitutional principles in its weights, not just its context window. This supervised stage precedes the RLHF stage in the full CAI pipeline.

---

## 4. Advantages and Limitations

**Advantages:**

- **Inference efficiency**: No prompt overhead on every request — reduces latency and token cost
- **Consistency**: Behavioral constraints are in weights, not in a prompt that could be overridden or injected against
- **Reduced prompt injection risk**: Attackers cannot override safety behaviors by manipulating the context if those behaviors are weight-encoded

**Limitations:**

- **Generalization bounds**: The student model only learns what the teacher's prompted distribution covers — rare or out-of-distribution queries may not generalize
- **Static alignment**: Changes to the desired behavior require retraining; prompt-based systems can be updated instantly
- **Capability risk**: SFT on a narrow distribution can degrade the model's broader capabilities if the dataset isn't diverse enough
- **No explicit reward signal**: Unlike RLHF, there is no mechanism to push responses beyond the quality ceiling of the prompted teacher outputs

---

## 5. Comparison with Related Techniques

| Technique | Alignment signal | Prompt at inference | Iterative improvement |
|-----------|-----------------|--------------------|-----------------------|
| Prompt engineering | System prompt | Yes | No |
| Context distillation | System prompt → SFT | No | No |
| RLHF | Human preferences → RM | Optional | Yes (RL loop) |
| Constitutional AI | AI self-critique + SFT/RLHF | No | Yes |

Context distillation sits between pure prompting (fragile, reversible) and full RLHF (expensive, iterative). It is most useful as a preprocessing step that creates a well-initialized policy before RLHF, or as a standalone technique when RLHF is too expensive.

---

*Source: Bai et al. (2022) — Constitutional AI: Harmlessness from AI Feedback [[arXiv:2212.08073]](https://arxiv.org/abs/2212.08073)*
