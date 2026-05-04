## 1. Overview

**REINFORCE Leave-One-Out (RLOO)** is a multi-sample variant of REINFORCE that uses the mean reward of the *other* samples in a group as the baseline for each individual sample. It was applied to LLM fine-tuning by Ahmadian et al. (2024) in *"Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs"* (ACL 2024).

**Key result:** RLOO consistently outperforms PPO, DPO, and RAFT across all tested models and datasets while using 50–70% less GPU memory and running 2–3× faster than PPO.

---

## 2. The Big Picture: From REINFORCE to RLOO

Vanilla REINFORCE estimates the gradient from a single sample per prompt, leading to high variance. The solution: sample $k$ responses per prompt and use the other $k-1$ responses as a baseline.

| Method | Samples per prompt | Baseline | Variance | Critic needed |
|--------|-------------------|----------|----------|--------------|
| REINFORCE | 1 | Running mean (batched) | High | No |
| RLOO | k | Leave-one-out mean | Low | No |
| PPO | 1 | Trained value function | Lowest | Yes |
| GRPO | k | Group-normalized | Low | No |

---

## 3. Formulation

### 3.1 RLOO Gradient Estimator

For a prompt $x$, sample $k$ responses $\{y_1, \ldots, y_k\}$ from the current policy. The gradient for the $i$-th sample uses the mean reward of the remaining $k-1$ samples as baseline:

$$
\nabla_\theta J(\theta) = \frac{1}{k} \sum_{i=1}^k \nabla_\theta \log \pi_\theta(y_i | x) \cdot \underbrace{\left( R_i - \frac{1}{k-1} \sum_{j \neq i} R_j \right)}_{\text{leave-one-out advantage } \hat{A}_i}
$$

where $R_i = r_\phi(x, y_i) - \beta \log \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)}$ is the KL-penalized reward.

### 3.2 Why This Baseline Is Effective

The leave-one-out baseline is:

- **Unbiased:** $\mathbb{E}[\hat{A}_i] = \mathbb{E}[R_i - b_i] = \mathbb{E}[R_i] - \mathbb{E}[b_i]$, and since $y_i \perp \{y_j\}_{j \neq i}$, the baseline does not correlate with the gradient term.
- **Low variance:** The baseline is computed on-the-fly for each sample at each step — no moving average lag or critic bias.
- **Free:** No extra model or training required — just multiple forward passes.

### 3.3 Relationship to GRPO

RLOO and GRPO both sample $k$ responses per prompt and use group-based baselines. The key difference is in the clipping:

| | RLOO | GRPO |
|-|------|------|
| **Base algorithm** | REINFORCE (no clipping) | PPO (with clipping) |
| **Advantage** | Leave-one-out mean | Group mean (z-score normalized) |
| **KL constraint** | Explicit additive penalty | Explicit additive penalty |
| **Bias** | Unbiased | Group normalization introduces slight bias |

RLOO's gradient is theoretically unbiased; GRPO's z-score normalization changes the effective gradient scale.

---

## 4. Implementation

```python
def rloo_loss(policy, ref_policy, reward_model, batch, k=4, beta=0.05):
    """
    batch: dict with 'prompts' (B,) and 'responses' (B, k) —
           k responses already sampled per prompt.
    """
    prompts = batch['prompts']          # (B,)
    responses = batch['responses']      # (B, k)

    B = len(prompts)
    rewards = torch.zeros(B, k)
    logp = torch.zeros(B, k)
    logp_ref = torch.zeros(B, k)

    for i in range(k):
        rewards[:, i] = reward_model(prompts, responses[:, i])
        logp[:, i] = policy.log_prob(prompts, responses[:, i])
        logp_ref[:, i] = ref_policy.log_prob(prompts, responses[:, i])

    # KL-penalized returns: (B, k)
    returns = rewards - beta * (logp - logp_ref).detach()

    # Leave-one-out baseline for sample i: mean of the other k-1 returns
    total = returns.sum(dim=1, keepdim=True)          # (B, 1)
    loo_baseline = (total - returns) / (k - 1)        # (B, k)

    advantages = (returns - loo_baseline).detach()

    # Policy gradient loss
    loss = -(logp * advantages).mean()
    return loss
```

**Key details:**

- `logp_ref` must be computed with `torch.no_grad()` — reference policy is frozen
- `advantages` must be detached before multiplying by `logp` — baseline must not receive gradients
- Use gradient clipping (max norm 1.0) for stability

---

## 5. Why RLOO Instead of PPO?

Ahmadian et al. (2024) argue that most of PPO's complexity was designed for RL from scratch, which does not apply to LLM fine-tuning:

**PPO's value function is unnecessary:** A pre-trained LLM already generates high-quality text. The variance of single-sample REINFORCE is low enough that the bias introduced by a learned value function outweighs the variance reduction benefit.

**Empirical results (Ahmadian et al., 2024):**

- RLOO outperforms PPO on helpfulness and harmlessness benchmarks
- RLOO uses 50–70% less vRAM than PPO
- RLOO runs 2× faster (1B models) to 3× faster (6.9B models) than PPO
- RLOO outperforms DPO and RAFT on all tested datasets

---

## 6. Comparison with Other Methods

| Aspect | RLOO | PPO | GRPO | DPO |
|--------|------|-----|------|-----|
| **Critic needed** | No | Yes | No | No |
| **Samples per prompt** | k | 1 | k (G) | N/A (offline) |
| **Baseline** | Leave-one-out | Value function | Group z-score | Implicit |
| **Unbiased gradient** | Yes | No | No | N/A |
| **Memory vs PPO** | −50–70% | baseline | −50% | −50% |
| **Speed vs PPO** | 2–3× faster | baseline | ~2× faster | ~2× faster |
| **Clipping** | None | Yes | Yes | N/A |
| **Best for** | General RLHF | Stable RL | Verifiable rewards | Preference data |

---

## 7. Limitations

**No clipping:** Without a trust-region constraint, RLOO can take large gradient steps on high-variance batches. Gradient clipping is the primary safeguard, but it is less principled than PPO's clip ratio.

**Memory scales with k:** Storing $k$ responses and their log-probabilities per prompt increases memory linearly with $k$. Typical $k = 4$–8 is manageable.

**Online sampling required:** Like all on-policy methods, RLOO must generate new samples at each training step. This is more expensive than DPO, which trains on a static offline dataset.

**No token-level credit assignment:** The same advantage is applied to every token in a sequence. For long chain-of-thought responses, this means early tokens receive incorrect credit signals. See [DAPO](dapo.md) for token-level loss.
