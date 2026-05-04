## 1. Overview

**REINFORCE** is the foundational policy gradient algorithm (Williams, 1992) and the conceptual basis for all modern RL-based LLM fine-tuning. It directly optimizes the expected reward by following the gradient of the log-probability of generated sequences, weighted by their reward.

While PPO and GRPO dominate current RLHF pipelines, recent work (Ahmadian et al., 2024) showed that REINFORCE and its multi-sample variant RLOO consistently outperform PPO when starting from a strong pre-trained LLM — challenging the assumption that PPO's complexity is necessary.

---

## 2. The Big Picture: REINFORCE in RLHF

| Stage | Role |
|-------|------|
| 1. SFT | Initialize policy π_θ from supervised fine-tuning |
| 2. Reward model | Score completions: r(x, y) |
| 3. REINFORCE | Update π_θ to maximize E[r(x, y)] − β·KL |

No critic/value network is trained. The reward signal flows directly back to the policy through the log-probability gradient.

---

## 3. Formulation

### 3.1 Policy Gradient Theorem

For a language model generating sequence $y = (y_1, \ldots, y_T)$ given prompt $x$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot R(x, y) \right]
$$

where $R(x, y)$ is the total return — typically the reward model score minus a KL penalty:

$$
R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

### 3.2 Variance Reduction with a Baseline

The gradient is unbiased for any constant baseline $b$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot \big(R(x, y) - b\big) \right]
$$

Common baseline choices:

| Baseline | Computation | Notes |
|----------|------------|-------|
| Zero (no baseline) | None | Highest variance |
| Running mean | Mean over recent rewards | Simple, slight bias |
| Per-prompt mean | Mean over multiple samples from same prompt | On-the-fly, unbiased |
| Value function (PPO) | Trained critic network | Lowest variance, adds bias and complexity |

### 3.3 Token-Level vs Sequence-Level

**Sequence-level:** One gradient signal per sequence.

$$
\nabla_\theta J = \nabla_\theta \log \pi_\theta(y|x) \cdot R(x,y) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t | x, y_{<t}) \cdot R(x,y)
$$

**Token-level (per-step):** Each token receives an individual credit assignment signal. Used in DAPO; important for long chain-of-thought responses where early tokens are far from the final reward.

---

## 4. Why REINFORCE Works Well for LLMs

PPO was designed for RL from scratch with unstable, non-stationary policies. LLMs violate these assumptions:

- **Strong initialization:** A pre-trained LLM already generates fluent, reasonable text — the policy doesn't need a critic to stay stable.
- **Concentrated distributions:** At each step, the LLM assigns high probability mass to a few tokens, so the variance from single-sample estimation is lower than in tabular RL.
- **Short optimization horizon:** RLHF fine-tunes for relatively few steps on top of SFT, so the complex value-function bootstrapping of PPO adds little benefit.

---

## 5. Implementation

```python
def reinforce_loss(policy, ref_policy, reward_model, batch, beta=0.05):
    prompts, responses = batch['prompts'], batch['responses']

    # Reward model scores
    rewards = reward_model(prompts, responses)

    # KL penalty
    logp = policy.log_prob(prompts, responses)
    logp_ref = ref_policy.log_prob(prompts, responses)
    kl = logp - logp_ref

    # Total return with baseline (running mean over batch)
    returns = rewards - beta * kl
    baseline = returns.mean().detach()
    advantages = returns - baseline

    # Policy gradient loss
    loss = -(logp * advantages).mean()
    return loss
```

**Key implementation details:**

- The reference policy is frozen — no gradient flows through `logp_ref`
- Gradient clipping (max norm 1.0) is essential to prevent instability
- Using `detach()` on the baseline is mandatory — the baseline must not affect the gradient

---

## 6. REINFORCE vs PPO

| Aspect | REINFORCE | PPO |
|--------|-----------|-----|
| **Critic required** | No | Yes |
| **Memory** | ~1× model size | ~2× (policy + critic) |
| **Variance** | Higher (baseline helps) | Lower (value estimates) |
| **Stability** | Good with LLM initialization | Sensitive to critic quality |
| **Implementation** | Simple | Complex |
| **Compute** | Fast | Slow |
| **Bias** | Unbiased gradient | Biased (value function) |

---

## 7. Limitations

**High variance:** Without a learned value function, gradient estimates can be noisy — especially on tasks where reward varies widely across samples. Mitigated by baselines and multi-sample averaging (see [RLOO](rloo.md)).

**No clipping:** Vanilla REINFORCE has no trust-region constraint. A single unlucky batch can cause large, destabilizing updates. Mitigated by gradient clipping and small learning rates.

**Sequence-level credit assignment:** The same gradient signal is applied to every token in the sequence, regardless of which tokens actually contributed to the reward. Mitigated by token-level loss (see [DAPO](dapo.md)).
