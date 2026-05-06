# Direct Preference Optimization (DPO)

## 1. Overview

DPO eliminates the separate reward model and RL loop of RLHF. It optimizes the policy directly from preference pairs — producing the same alignment effect as PPO but through a supervised-style objective. The key insight: the optimal policy under the RLHF objective has a closed-form relationship with the reward function, allowing the reward to be reparameterized as a ratio of policy probabilities.

| Stage | PPO-RLHF | DPO |
|-------|----------|-----|
| SFT | Same | Same |
| Reward model training | Required | Not needed |
| RL loop (PPO) | Required | Replaced by DPO loss |

---

## 2. Training Data

Each example is a triplet $(x, y_w, y_l)$:
- $x$: prompt
- $y_w$: preferred (winner) response
- $y_l$: dispreferred (loser) response

The model learns to assign higher probability to $y_w$ relative to $y_l$, while staying close to the reference model.

---

## 3. The DPO Objective

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\!\left( \beta \left[ \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right] \right)\right]$$

Where:
- $\pi_\text{ref}$: frozen SFT model (reference)
- $\beta$: inverse temperature — controls how aggressively to deviate from the reference
- $\sigma$: sigmoid function

**Intuition:** Maximize the margin between the log-likelihood ratio of the winner and the loser, relative to the reference model. If the model already assigns higher probability to $y_w$ than the reference does, and lower to $y_l$, the loss is small.

---

## 4. The Implicit Reward

The DPO derivation shows that the optimal RLHF policy implies an implicit reward function:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$$

This means DPO implicitly learns the same reward as RLHF — without training a separate reward model. The log-ratio of policy to reference *is* the reward.

> "Your language model is secretly a reward model."

---

## 5. The $\beta$ Parameter

- **Low $\beta$ (e.g., 0.1):** stays close to the reference — conservative alignment
- **High $\beta$ (e.g., 0.5):** deviates more aggressively — stronger alignment, higher risk of mode collapse
- Typical range: **0.1–0.5**

---

## 6. DPO vs PPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| Reward model | Explicit (trained separately) | Implicit (reparameterized into policy) |
| RL loop | Yes | No |
| KL penalty | Explicit term in objective | Handled via reference model ratio |
| Training stability | Sensitive to many hyperparameters | More stable |
| Models in memory | 4 (policy, old policy, reward, value) | 2 (policy, reference) |
| Data format | Scalar reward per response | Preference pairs |
| When to prefer | Verifiable scalar rewards, complex multi-objective | Preference data, simpler setup |

---

## 7. Variants

**IPO (Identity Preference Optimization):** Replaces sigmoid with a squared loss, improving stability when preferences are noisy.

**KTO (Kahneman-Tversky Optimization):** Uses binary thumbs-up/thumbs-down labels instead of pairwise comparisons. Useful when you only have pointwise feedback, not head-to-head comparisons.

**Iterative DPO:** Periodically updates the reference model with the current policy, allowing the model to improve beyond the initial SFT baseline over multiple rounds.

**Online DPO:** Generates new preference pairs on-the-fly during training using the current policy, avoiding distribution mismatch between offline preference data and the model's current outputs.

---

## 8. Limitations

**Offline data distribution mismatch.** DPO trains on a fixed preference dataset collected from the SFT model. As the policy improves, the training data becomes off-distribution. Iterative or Online DPO mitigates this.

**No explicit reward signal.** DPO cannot directly express "this response is worth +3 reward, that one +7." It only learns from relative preferences, limiting fine-grained control.

**Mode collapse at high $\beta$.** Aggressive $\beta$ values cause the model to collapse to a narrow output distribution that always produces the "winning" style regardless of prompt.

**Preference data cost.** High-quality pairwise annotations are expensive. Synthetic preference data (from a stronger model) is increasingly used as a workaround.

---

*Sources: Rafailov et al. (2023) — Direct Preference Optimization: Your Language Model is Secretly a Reward Model [[arXiv:2305.18290]](https://arxiv.org/abs/2305.18290)*
