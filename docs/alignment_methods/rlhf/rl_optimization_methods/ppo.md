# Proximal Policy Optimization (PPO)

## 1. Overview

PPO is the RL algorithm used in Stage 3 of the RLHF pipeline to fine-tune the policy model using scalar rewards from the reward model. It maintains training stability by constraining how much the policy can change per update, preventing the model from collapsing into degenerate reward-hacking behavior.

See [RLHF Pipeline](../rlhf_pipeline.md) for the full three-stage context (SFT → Reward Model → PPO).

---

## 2. The Four Models in PPO-RLHF

| Component | Role | Frozen? |
|-----------|------|---------|
| **Policy model** ($\pi_\theta$) | The LLM being trained | No |
| **Reference model** ($\pi_\text{ref}$) | Frozen copy of the policy at training start | Yes |
| **Reward model** ($r_\phi$) | Scores each (prompt, response) pair | Yes |
| **Value head** ($V_\theta$) | Estimates expected reward from a state | No (trained alongside policy) |

The value head is the key cost of PPO — it requires an extra model of similar size to the policy. [GRPO](grpo.md) and [RLOO](rloo.md) eliminate this.

---

## 3. The PPO Objective

### 3.1 Probability Ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{old}(a_t \mid s_t)}$$

Measures how much the new policy changes the probability of each action relative to when that action was sampled.

### 3.2 Clipped Surrogate Loss

$$L^\text{PPO}(\theta) = \mathbb{E}_t \left[ \min\!\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A_t \right) \right]$$

- $\epsilon$ is typically 0.1–0.2
- If $A_t > 0$: encourages increasing probability of the action, but capped at $1+\epsilon$
- If $A_t < 0$: encourages decreasing probability, but floored at $1-\epsilon$

### 3.3 KL Penalty Against Reference

In RLHF, a KL penalty is added to prevent cumulative drift from the original model:

$$L_\text{total} = -L^\text{PPO} + c_1 L^\text{value} - c_2 H[\pi_\theta] + \beta\, D_{KL}(\pi_\theta \| \pi_\text{ref})$$

**Clip vs KL — why both?** Clipping is a *per-step* guard — it prevents any single update from being too large. KL against the reference is a *global* guard — it prevents cumulative drift over many small steps from straying far from the original pretrained model. A model can take many clipped steps that individually look safe but cumulatively collapse the policy. The KL term catches this.

---

## 4. Advantage and Value Estimation

The **advantage** $A_t = R_t - V_\theta(s_t)$ measures how much better an action was than the expected baseline.

In RLHF, reward is typically a single scalar per sequence (not per token), so:

$$A = r_\phi(x, y) - V_\theta(x)$$

For longer sequences, **Generalized Advantage Estimation (GAE)** smooths the signal across tokens:

$$A_t^\text{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)$$

$\lambda \in [0.9, 0.97]$ controls the bias-variance tradeoff.

---

## 5. Training Loop

1. Sample response from current policy for each prompt
2. Score with reward model → $r$
3. Compute log-probs from both policy and reference model
4. Estimate value → $V_\theta(s)$
5. Compute advantage: $A = r - V_\theta(s)$
6. Compute probability ratio $r_t = \pi_\theta / \pi_\text{old}$
7. Update policy with clipped surrogate loss
8. Update value head to predict returns more accurately
9. Apply entropy bonus + KL penalty

---

## 6. Limitations

**Computational cost.** Four models in memory simultaneously (policy, reference, reward, value head). Requires distributed training for 7B+ models.

**Reward hacking.** The policy learns to exploit weaknesses in the reward model. KL penalty limits this but does not eliminate it. See [KL Penalty & Reward Hacking](../kl_penalty_reward_hacking.md).

**Sensitivity to hyperparameters.** $\epsilon$ (clip), $\beta$ (KL coefficient), $c_1$ (value weight), $c_2$ (entropy weight) all interact — unstable if misconfigured.

**Credit assignment.** One reward per sequence, applied to all tokens equally. Token-level credit is ambiguous. See [DAPO](dapo.md) for the token-level loss fix.

---

## 7. PPO vs Alternatives

| Aspect | PPO | GRPO | DPO | RLOO |
|--------|-----|------|-----|------|
| Critic required | Yes | No | No | No |
| Memory | ~2× policy | ~1.5× policy | ~2× policy | ~1.5× policy |
| Reward type | Scalar | Scalar | Preference pairs | Scalar |
| Clipping | Yes | Yes | No | No |
| Best for | General RLHF | Reasoning (verifiable rewards) | Preference alignment | General RLHF (cheaper PPO) |

---

*Sources: Schulman et al. (2017) — Proximal Policy Optimization Algorithms [[arXiv:1707.06347]](https://arxiv.org/abs/1707.06347) · Ouyang et al. (2022) — InstructGPT [[arXiv:2203.02155]](https://arxiv.org/abs/2203.02155)*
