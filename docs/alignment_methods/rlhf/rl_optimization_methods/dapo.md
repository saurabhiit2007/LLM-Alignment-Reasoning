# DAPO

## 1. Overview

**DAPO** (Decoupled Clip and Dynamic Sampling Policy Optimization) is an RL fine-tuning algorithm for LLMs developed by ByteDance Seed and Tsinghua AIR (Yu et al., 2025). It extends GRPO with four targeted fixes that address failure modes observed when scaling RL to long chain-of-thought (CoT) reasoning tasks.

**Key result:** DAPO achieves **50 points on AIME 2024** using Qwen2.5-32B (vs. 47 for DeepSeek-R1-Zero-32B) using 50% fewer training steps, with fully open-sourced training code.

---

## 2. The Big Picture: Why GRPO Needs Fixing at Scale

GRPO works well in small-scale settings but exhibits three failure modes when training long-CoT reasoning models at scale:

| Problem | Root cause | DAPO fix |
|---------|-----------|----------|
| **Entropy collapse** | Symmetric clip penalizes exploration | Clip-Higher (asymmetric clip) |
| **Wasted compute on trivial prompts** | Groups with all-correct/all-wrong get zero gradient | Dynamic Sampling |
| **Short-response bias** | Sequence-level loss weights short/long equally by count | Token-level Policy Gradient Loss |
| **Reward noise from truncation** | Truncated responses get harsh penalties | Overlong Reward Shaping |

---

## 3. DAPO Formulation

DAPO starts from the GRPO objective but modifies each component:

$$
J_{\text{DAPO}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[
\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min \Big(
  \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_i,\;
  \text{clip}\!\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}} \right) \hat{A}_i
\Big)
\right]
$$

where $\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$ (decoupled clipping) and the sum is over tokens rather than sequences.

The KL penalty term is **removed** in DAPO — entropy is controlled by the asymmetric clip instead.

---

## 4. The Four Key Innovations

### 4.1 Clip-Higher (Decoupled Clip Ratio)

**Problem:** GRPO uses a symmetric clip $[1-\varepsilon, 1+\varepsilon]$. The upper clip constrains how much the model can increase the probability of good responses, suppressing exploration and causing entropy collapse — the model converges prematurely to a narrow output distribution.

**Fix:** Use separate thresholds for the lower and upper clip bounds:

$$
\text{clip}\!\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\right), \quad \varepsilon_{\text{high}} > \varepsilon_{\text{low}}
$$

Typical values: $\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$.

**Effect:** The model can increase probability of correct responses more aggressively (via the looser upper bound), while still being constrained from catastrophically reducing probability of any response (via the tighter lower bound). This maintains entropy and promotes diversity.

---

### 4.2 Dynamic Sampling

**Problem:** When all $G$ samples for a prompt have the same reward (all correct or all incorrect), the group advantage is zero — no gradient signal is generated. These prompts waste compute.

**Fix:** During each training batch, filter out prompt groups where all samples produce the same outcome:

```
Keep group i only if:  0 < accuracy_i < 1
```

Sampling continues until the batch has the required number of *effective* (non-degenerate) groups.

**Effect:** Every gradient step uses only prompts that provide a learning signal. This improves both training efficiency and stability.

**Hyperparameter:** Minimum and maximum group accuracy thresholds. Typical: filter groups with accuracy = 0 or accuracy = 1.

---

### 4.3 Token-Level Policy Gradient Loss

**Problem:** GRPO and most RL methods use a *sequence-level* loss — the advantage for the entire sequence is computed once and applied equally to all tokens:

$$
\mathcal{L}_{\text{seq}} = \frac{1}{G} \sum_{i=1}^G \left(\sum_{t=1}^{|o_i|} \text{PG}(o_{i,t})\right)
$$

This means a group of $G$ short sequences contributes the same total gradient magnitude as a group of $G$ long sequences. In long-CoT tasks, short responses dominate training because they appear more often in batches.

**Fix:** Normalize loss by total token count rather than by sequence count:

$$
\mathcal{L}_{\text{token}} = \frac{\sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{PG}(o_{i,t})}{\sum_{i=1}^G |o_i|}
$$

**Effect:** Long reasoning chains contribute proportionally more to the gradient, preventing the model from learning to give short non-reasoning answers.

---

### 4.4 Overlong Reward Shaping

**Problem:** When a generated response is truncated at the maximum length limit, it typically receives a large negative reward (incomplete answer = wrong). This creates noisy, unstable gradient signals and discourages the model from generating longer reasoning chains even when they are beneficial.

**Fix:** Apply a soft length penalty instead of hard truncation reward:

$$
r_{\text{shaped}}(o) = \begin{cases}
r(o) & |o| \leq L_{\max} \\
r_{\text{min}} \cdot \frac{|o| - L_{\max}}{L_{\text{cache}} - L_{\max}} & L_{\max} < |o| \leq L_{\text{cache}}
\end{cases}
$$

Responses that exceed $L_{\max}$ receive a gradually increasing penalty rather than an abrupt large negative reward.

**Effect:** Smoother reward landscape; more stable gradients for long-CoT training.

---

## 5. DAPO vs GRPO vs PPO

| Aspect | DAPO | GRPO | PPO |
|--------|------|------|-----|
| **Critic needed** | No | No | Yes |
| **Clip ratio** | Asymmetric (ε_low ≠ ε_high) | Symmetric | Symmetric |
| **KL penalty** | No (entropy via clip) | Yes | Yes |
| **Loss level** | Token | Sequence | Token |
| **Trivial group filtering** | Yes (dynamic sampling) | No | No |
| **Long-CoT stability** | High | Moderate | Moderate |
| **Entropy collapse prevention** | Explicit (Clip-Higher) | Partial (KL) | Partial (entropy bonus) |
| **AIME 2024 (32B)** | 50 pts | 47 pts (DeepSeek R1-Zero) | — |

---

## 6. Limitations

**More hyperparameters:** $\varepsilon_{\text{low}}$, $\varepsilon_{\text{high}}$, dynamic sampling thresholds, overlong shaping parameters — each adds a tuning dimension. The original paper provides defaults but these may not transfer across model families.

**Dynamic sampling changes batch composition:** Filtering trivial groups means the effective batch size varies. If a difficulty-calibrated dataset is used and most prompts are trivial (or all hard), sampling can stall. Careful dataset curation is required.

**No formal convergence guarantee:** Removing the KL penalty and using asymmetric clipping breaks the theoretical bounds from PPO's trust-region analysis. Empirical stability is well-demonstrated but not theoretically proven.

**Requires verifiable rewards:** Like GRPO and RLVR, DAPO depends on a binary or graded correctness signal. It is not directly applicable to open-ended generation tasks without a reward model.
