## 1. Overview

**Grouped Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm introduced in the **DeepSeek** series (DeepSeekMath, DeepSeek-R1) to fine-tune **Large Language Models (LLMs)** efficiently on reasoning-intensive tasks.  

Unlike traditional PPO, which requires a **critic (value network)**, GRPO eliminates the critic and computes **relative advantages within groups** of sampled outputs.  

This approach reduces computational cost and stabilizes training, making it well-suited for large-scale language model alignment.

---

---

## 2. The Big Picture: From PPO to GRPO

Traditional **RLHF** pipelines (using PPO) require a policy model, a reward model, and a value function. GRPO simplifies this process by using **group-wise relative advantages** instead of an explicit value estimator.

| Stage | PPO-Based RLHF | GRPO-Based Alignment |
|-------|----------------|----------------------|
| 1️⃣ SFT | Train base LLM on human demonstrations | ✅ Same |
| 2️⃣ RM  | Train reward or value model | ❌ Removed (uses reward function directly) |
| 3️⃣ RL  | Fine-tune using PPO updates | ✅ Fine-tune using group-based GRPO objective |

This design significantly reduces training instability and memory usage while preserving the benefits of policy-gradient fine-tuning.

---

---

## 3. Intuitive Understanding

For each prompt, GRPO samples **G** candidate responses from the old policy, evaluates each response using a reward function, and compares them within the group.  

The model then updates its policy to favor responses that outperform others in the same group — a *relative* rather than *absolute* improvement process.

**Intuitive comparison:**

* **PPO** optimizes each response using absolute advantages from a critic.
* **GRPO** optimizes by ranking multiple sampled responses and pushing the policy toward higher-ranked ones.

This allows GRPO to focus on *comparative improvement* while maintaining diversity and avoiding overfitting to noisy rewards.

---

---

## 4. Training Data and Setup

Each GRPO training example includes:

* **Prompt**: \( q \)
* **Group of outputs**: \( \{o_1, o_2, \dots, o_G\} \) sampled from the old policy \( \pi_{\text{old}} \)
* **Reward values**: \( r_i = r(q, o_i) \) from a scoring or reward function

The policy model \( \pi_\theta \) is optimized to assign higher probabilities to outputs with higher *relative* rewards, regularized by a KL penalty with respect to a frozen **reference policy** \( \pi_{\text{ref}} \).

---

---

## 5. GRPO Formulation

### 5.1. Objective Function

GRPO generalizes the PPO objective using group-wise normalization:

$$
J_{\mathrm{GRPO}}(\theta)
= \mathbb{E}_{q, \{o_i\}} \left[
\frac{1}{G} \sum_{i=1}^G
\min \Big(
  \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} A_i,\,
  \text{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i
\Big)

- \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \| \pi_{\text{ref}}\big)
\right]
$$

where:

* $\pi_{\text{old}}$: policy before update (often the policy from the previous iteration)
* $A_i$: normalized advantage within the group  
* $\epsilon$: PPO clipping coefficient (typically 0.1-0.2)
* $\beta$: KL regularization coefficient (typically 0.001-0.01)
* $\pi_{\text{ref}}$: frozen reference model (typically the SFT model)

---

### 5.2. Grouped Advantage

The *relative* advantage \(A_i\) is computed within each group:

$$
A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G}) + \epsilon_{\text{small}}}
$$

where:

* $r_i$ is the reward for output $o_i$
* $\epsilon_{\text{small}}$ is a small constant (e.g., 1e-8) to prevent division by zero

This ensures that updates depend on *relative* performance rather than absolute reward magnitude.

**Key insight:** By normalizing advantages within each group, GRPO automatically adapts to different reward scales and focuses on **relative ranking** rather than absolute values.

---

### 5.3. KL Regularization

The KL term ensures that the updated policy remains close to the reference model:

$$
D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}}) =
\mathbb{E}_{o \sim \pi_\theta} \left[
\log \frac{\pi_\theta(o|q)}{\pi_{\text{ref}}(o|q)}
\right]
$$

In practice, this is often computed as:

$$
D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}}) =
\log \frac{\pi_\theta(o_i|q)}{\pi_{\text{ref}}(o_i|q)}
$$

for each output $o_i$ in the group.

---

### 5.4. Intuition

* **Group-normalized advantages** remove the need for a critic by comparing samples against each other.
* **KL regularization** prevents the model from drifting too far from the reference policy, maintaining stability.
* **Clipping** prevents large, unstable policy updates that could degrade performance.
* **Efficiency**: GRPO avoids computing value baselines, making it highly scalable for LLMs.

---

### 5.5. Implementation Details

* **Group size (G)** — Typically 4–16 samples per prompt (8 is common).
* **β (beta)** — 0.001–0.01 to control KL regularization strength.
* **ε (epsilon)** — Clipping coefficient, often 0.1–0.2.
* **Reference policy** — Frozen SFT model to anchor learning.
* **Reward function** — Task-specific (e.g., correctness, coherence, reasoning completeness).
* **Advantage normalization** — Essential for stable updates; normalize per group with small epsilon.
* **Temperature** — Sampling temperature for generating diverse outputs (typically 0.6-1.0).
* **Learning rate** — Typically smaller than SFT (e.g., 1e-6 to 1e-5).

---

---

## 6. Implementation Example (Pseudocode)

```python
import torch
import numpy as np

# Hyperparameters
G = 8  # Group size
beta = 0.01  # KL coefficient
epsilon = 0.2  # Clipping coefficient
eps_small = 1e-8  # For numerical stability

for prompt in dataset:
    # Step 1: Sample G outputs from old policy
    outputs = [policy_old.generate(prompt) for _ in range(G)]
    
    # Step 2: Compute rewards for each output
    rewards = [reward_fn(prompt, o) for o in outputs]
    
    # Step 3: Normalize advantages within the group
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + eps_small
    advantages = [(r - mean_r) / std_r for r in rewards]
    
    # Step 4: Compute log probabilities
    logp_old = [policy_old.logprob(prompt, o) for o in outputs]
    logp_new = [policy.logprob(prompt, o) for o in outputs]
    
    # Step 5: Compute probability ratios
    ratios = [torch.exp(lp_new - lp_old) 
              for lp_new, lp_old in zip(logp_new, logp_old)]
    
    # Step 6: Compute clipped surrogate objective
    surr1 = [r * A for r, A in zip(ratios, advantages)]
    surr2 = [torch.clamp(r, 1-epsilon, 1+epsilon) * A 
             for r, A in zip(ratios, advantages)]
    surr = [torch.min(s1, s2) for s1, s2 in zip(surr1, surr2)]
    
    # Step 7: Compute policy loss (negative because we maximize)
    loss_policy = -torch.mean(torch.stack(surr))
    
    # Step 8: Compute KL divergence with reference policy
    logp_ref = [ref_policy.logprob(prompt, o) for o in outputs]
    kl_div = [lp_new - lp_ref 
              for lp_new, lp_ref in zip(logp_new, logp_ref)]
    kl_loss = beta * torch.mean(torch.stack(kl_div))
    
    # Step 9: Total loss
    loss = loss_policy + kl_loss
    
    # Step 10: Update policy
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
```

---

---

## 7. Why GRPO Instead of PPO?

| Aspect                 | PPO                                    | GRPO                                   |
|------------------------|----------------------------------------|----------------------------------------|
| **Critic / Value Net** | Required                               | ❌ Removed                             |
| **Advantage Computation** | From value estimates (GAE)           | Group-normalized rewards               |
| **KL Regularization**  | Explicit or adaptive penalty           | Included via reference policy          |
| **Training Stability** | Sensitive to critic/value bias         | More stable and memory-efficient       |
| **Data Efficiency**    | Uses single rollout per update         | Leverages multiple outputs per prompt  |
| **Compute Cost**       | High (policy + value models)           | Low (policy-only)                      |
| **Memory Usage**       | 2x model parameters (policy + critic)  | 1x model parameters (policy only)      |
| **Suitability**        | General RL tasks                       | LLM fine-tuning with verifiable rewards|
| **Variance**           | Lower (value baseline reduces variance)| Higher (no baseline, group normalization)|

---

---

## 8. Key Advantages of GRPO

### ✅ 1. No Critic Network Required
Eliminates the need to train and maintain a separate value network, reducing memory and computational costs.

### ✅ 2. Memory Efficiency
Only requires storing the policy model and reference model (frozen), roughly 50% memory savings compared to PPO.

### ✅ 3. Training Stability
Group-based normalization is less sensitive to reward scale and distribution shifts compared to critic-based methods.

### ✅ 4. Simplicity
Fewer hyperparameters and components to tune compared to PPO with GAE.

### ✅ 5. Better for Sparse Rewards
Works well when rewards are binary or sparse, as group comparison remains meaningful.

---

---

## 9. Limitations and Challenges

### 📉 1. Group Reward Homogeneity
If all responses in a group have similar rewards, normalized advantages approach zero, yielding weak gradients.

**Solution:** Increase group size or use temperature sampling to generate more diverse outputs.

### 🔄 2. Reward Function Quality
GRPO still relies on reward signal design; noisy or biased rewards can misguide optimization.

**Solution:** Use multiple reward models or ensemble approaches; validate rewards on held-out data.

### ⚖️ 3. KL Coefficient Sensitivity
If β is too small, the model may drift from the reference policy; too large, and updates stall.

**Solution:** Use adaptive KL coefficient scheduling or monitor KL divergence during training.

### 💡 4. Group Size Tradeoff
Larger groups improve ranking precision but increase compute cost linearly.

**Solution:** Start with G=8 and adjust based on compute budget and reward variance.

### 🎭 5. Limited Exploration
As with PPO, GRPO may struggle to explore novel or diverse outputs if rewards are narrow.

**Solution:** Use entropy bonuses or diverse sampling strategies during generation.

### 📊 6. Higher Variance than PPO
Without a value baseline, GRPO can have higher gradient variance, potentially requiring more samples.

**Solution:** Increase group size or batch size to reduce variance.

---

---

### 10. Practical Tips for Using GRPO

1. **Start with a strong SFT model** — GRPO works best when initialized from a well-supervised model.

2. **Use temperature sampling** — Generate diverse outputs (temperature 0.7-1.0) to ensure meaningful group comparisons.

3. **Monitor KL divergence** — Track KL with reference policy; if it grows too large, increase β.

4. **Validate reward function** — Manually inspect high and low reward samples to ensure reward alignment.

5. **Gradual RL fine-tuning** — Start with small learning rates and short training runs to avoid instability.

6. **Use best-of-N as baseline** — Compare GRPO results against simple best-of-N sampling from the SFT model.

7. **Track multiple metrics** — Monitor reward, KL divergence, policy entropy, and task-specific metrics.

---

---

### 11. GRPO vs. Other Methods

| Method | Critic Required | Sample Efficiency | Memory Cost | Best For |
|--------|----------------|-------------------|-------------|----------|
| **PPO** | Yes | Moderate | High | General RL, continuous control |
| **GRPO** | No | High (group-based) | Low | LLM alignment, reasoning tasks |
| **DPO** | No | Very High | Low | Preference learning |
| **RRHF** | No | Moderate | Low | Simple ranking-based tasks |
| **RLHF (PPO)** | Yes | Moderate | High | Conversational AI, general alignment |

**When to use GRPO:**

- You have a reliable reward function (not just preferences)
- You need memory efficiency (no critic)
- You're working on reasoning or math tasks
- You can sample multiple outputs per prompt efficiently

**When to use alternatives:**

- **DPO:** You only have preference data, no absolute rewards
- **PPO:** You need lower variance or are in non-LLM RL domains
- **RRHF:** You want even simpler ranking without clipping

---

---
