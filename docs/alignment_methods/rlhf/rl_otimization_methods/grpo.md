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

## 📝 Commonly Asked Interview Questions on GRPO

### Conceptual Questions

#### Q1: What is GRPO and how does it differ from PPO?

**Answer:**
GRPO (Grouped Relative Policy Optimization) is an RL algorithm for LLM alignment that eliminates the need for a critic/value network by using group-wise relative advantages.

**Key differences:**
- **PPO** uses a critic to estimate value baselines → GRPO uses group normalization
- **PPO** computes absolute advantages → GRPO computes relative advantages within groups
- **PPO** requires 2x memory (policy + critic) → GRPO requires 1x memory (policy only)
- **PPO** is general-purpose → GRPO is optimized for LLMs with verifiable rewards

---

#### Q2: Why does GRPO normalize advantages within groups?

**Answer:**
Group normalization serves three purposes:

1. **Eliminates the critic:** By comparing samples against each other, we don't need a value baseline
2. **Scale invariance:** The algorithm works regardless of absolute reward magnitude
3. **Focuses on ranking:** Updates are driven by which outputs are better relative to others, not absolute values

This makes GRPO robust to reward scale changes and reduces sensitivity to reward function design.

---

#### Q3: What happens if all outputs in a group have the same reward?

**Answer:**
If all rewards are identical, the standard deviation becomes zero (or very small with epsilon), making all normalized advantages approximately zero. This means:

- **No gradient signal** — the policy won't update for this prompt
- **This is intentional** — if all outputs are equally good/bad, there's nothing to learn

**Solutions:**
- Increase group size G to get more diversity
- Use temperature sampling instead of greedy decoding
- Check if the reward function is too coarse-grained

---

### Technical Questions

#### Q4: Walk me through the GRPO update equation step by step.

**Answer:**

1. **Sample G outputs** from current policy for prompt q
2. **Compute rewards** r₁, r₂, ..., rG using reward function
3. **Normalize advantages**: Aᵢ = (rᵢ - mean(r)) / std(r)
4. **Compute probability ratio**: ratio = π_new(o|q) / π_old(o|q)
5. **Apply clipping**: clip(ratio, 1-ε, 1+ε)
6. **Take minimum**: min(ratio × A, clip(ratio) × A)
7. **Add KL penalty**: -β × KL(π_new || π_ref)
8. **Average over group** and update policy

The clipping prevents large updates, and KL regularization keeps the policy close to the reference.

---

#### Q5: How do you choose the group size G? What are the tradeoffs?

**Answer:**

**Tradeoffs:**

| Small G (4-6) | Large G (12-16) |
|---------------|-----------------|
| ✅ Lower compute cost | ❌ Higher compute cost |
| ✅ Faster iteration | ❌ Slower iteration |
| ❌ Higher variance | ✅ Lower variance |
| ❌ Less reliable ranking | ✅ More reliable ranking |

**Practical guidance:**
- Start with G=8 as a reasonable default
- Increase G if training is unstable or high variance
- Decrease G if compute-constrained
- Monitor the standard deviation of rewards within groups — if consistently low, increase G

---

#### Q6: What is the role of the reference policy π_ref?

**Answer:**
The reference policy (typically the frozen SFT model) serves as an anchor point:

1. **Prevents mode collapse:** KL penalty prevents the policy from assigning near-zero probability to most outputs
2. **Maintains language quality:** Keeps outputs fluent and coherent like the original SFT model
3. **Stable optimization:** Provides a fixed comparison point throughout training
4. **Controls drift:** The β coefficient controls how much the policy can deviate

Without π_ref, the policy could exploit the reward function in unintended ways (e.g., generating nonsensical text that happens to score well).

---

#### Q7: How does GRPO handle exploration vs. exploitation?

**Answer:**
GRPO's exploration comes from:

1. **Sampling diversity:** Using temperature > 0 during generation creates diverse outputs
2. **Group comparison:** Even with the same prompt, different samples explore different strategies
3. **KL regularization:** Prevents premature convergence by maintaining entropy

**However:** GRPO has limited exploration compared to traditional RL because:
- It's on-policy (uses current policy samples)
- No explicit exploration bonus
- Relies on sampling temperature for diversity

**Solutions for better exploration:**
- Add entropy bonus to objective
- Use diverse prompts in training data
- Sample with higher temperature early in training

---

### Practical Implementation Questions

#### Q8: How do you debug GRPO training that isn't improving?

**Answer:**
Systematic debugging checklist:

1. **Check reward signal:**
   - Are rewards varying within groups? (If std ≈ 0, no learning signal)
   - Manually inspect high vs. low reward samples
   - Verify reward function alignment with goal

2. **Check KL divergence:**
   - Is KL too high? (Policy drifting too far → increase β)
   - Is KL near zero? (Policy not updating → decrease β or increase LR)

3. **Check advantage distribution:**
   - Plot advantage values — should be roughly centered at 0
   - Check for outliers or constant values

4. **Check sampling:**
   - Are outputs diverse? (If not, increase temperature)
   - Are all outputs trivial? (SFT model may be too weak)

5. **Check hyperparameters:**
   - Learning rate too high/low
   - Epsilon clipping too restrictive
   - Group size too small

---

#### Q9: What are the memory requirements for GRPO training?

**Answer:**

**During training, you need:**
1. **Policy model** (trainable parameters)
2. **Reference model** (frozen, can share most weights)
3. **Old policy logits** (for ratio computation)
4. **Activations** (for backward pass)

**Memory breakdown:**
- Policy parameters: ~P (model size)
- Reference parameters: ~P (but can be offloaded or shared)
- Activations: ~B × L × H (batch size × sequence length × hidden size)
- Gradient memory: ~P

**Compared to PPO:**
- PPO needs policy + critic ≈ 2P parameters
- GRPO needs policy + reference ≈ 2P parameters (but reference is frozen)
- **Net savings:** ~50% trainable parameters, easier to scale

**Optimization tricks:**
- Share embeddings between policy and reference
- Offload reference to CPU
- Use gradient checkpointing
- Reduce group size or sequence length

---

#### Q10: How would you implement GRPO from scratch? What are the key components?

**Answer:**

**Key components:**

```python
class GRPOTrainer:
    def __init__(self, policy, ref_policy, reward_fn, 
                 beta=0.01, epsilon=0.2, group_size=8):
        self.policy = policy  # Trainable
        self.ref_policy = ref_policy  # Frozen
        self.reward_fn = reward_fn
        self.beta = beta
        self.epsilon = epsilon
        self.group_size = group_size
        
    def compute_advantages(self, rewards):
        """Normalize rewards within group"""
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8
        return [(r - mean) / std for r in rewards]
    
    def compute_loss(self, prompt, outputs):
        """Main GRPO loss computation"""
        # 1. Get rewards
        rewards = [self.reward_fn(prompt, o) for o in outputs]
        advantages = self.compute_advantages(rewards)
        
        # 2. Compute log probs
        logp_old = self.get_logprobs(self.policy_old, prompt, outputs)
        logp_new = self.get_logprobs(self.policy, prompt, outputs)
        logp_ref = self.get_logprobs(self.ref_policy, prompt, outputs)
        
        # 3. Policy loss with clipping
        ratios = torch.exp(logp_new - logp_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        # 4. KL penalty
        kl_loss = self.beta * torch.mean(logp_new - logp_ref)
        
        return policy_loss + kl_loss
```

**Critical implementation details:**
1. Keep reference policy frozen throughout training
2. Update policy_old periodically (every K steps)
3. Use numerical stability (eps=1e-8 in std computation)
4. Gradient clipping for stability
5. Monitor all components (reward, KL, advantages) separately

---

### Advanced Questions

#### Q11: Can GRPO work with preference data instead of absolute rewards?

**Answer:**
**Theoretically:** Yes, you could convert preferences to pseudo-rewards, but this is not ideal.

**Why GRPO isn't designed for preferences:**
- Preferences give you "A > B" comparisons, not absolute scores
- Group normalization requires absolute reward values
- You'd need to assign arbitrary reward scales

**Better alternatives for preference data:**
- **DPO (Direct Preference Optimization):** Directly optimizes from preferences without rewards
- **RRHF:** Uses ranking loss on preferences
- **Preference-based reward modeling:** Train a reward model first, then use GRPO

**If you must use preferences with GRPO:**
1. Convert preferences to Bradley-Terry rewards
2. Use reward modeling to get absolute scores
3. Apply standard GRPO with these scores

---

#### Q12: How does GRPO compare to DPO (Direct Preference Optimization)?

**Answer:**

| Aspect | GRPO | DPO |
|--------|------|-----|
| **Input data** | Absolute rewards | Pairwise preferences |
| **Optimization** | On-policy RL | Offline supervised |
| **Sampling** | Requires generation | Uses static dataset |
| **Critic** | No | No |
| **KL constraint** | Explicit penalty | Implicit in loss |
| **Compute** | High (sampling) | Low (static data) |
| **When to use** | Verifiable rewards (math, code) | Human preferences |

**Key insight:** 
- **DPO** is simpler and more data-efficient when you have preferences
- **GRPO** is better when you have a reliable reward function and can afford sampling

---

#### Q13: What modifications would you make to GRPO for a multi-turn dialogue task?

**Answer:**

**Challenges for multi-turn:**
1. Credit assignment across turns
2. Longer sequences (memory constraints)
3. Context dependency

**Modifications:**

1. **Reward shaping:**
   - Per-turn rewards + final outcome reward
   - Discounted cumulative reward: R = Σ γᵗ rₜ

2. **Group sampling:**
   - Sample entire dialogues, not single turns
   - Use trajectory-level normalization

3. **Context handling:**
   - Include conversation history in prompt
   - Use sliding window for long contexts

4. **Efficiency:**
   - Share computation across turns
   - Cache intermediate states
   - Use smaller group sizes due to longer sequences

**Example modification:**
```python
# Instead of single response
outputs = [policy.generate(prompt) for _ in range(G)]

# Multi-turn trajectory
trajectories = [policy.generate_dialogue(context, num_turns=5) 
                for _ in range(G)]
rewards = [dialogue_reward(traj) for traj in trajectories]
```

---

#### Q14: How would you adapt GRPO for continual learning or lifelong learning scenarios?

**Answer:**

**Key challenges:**
1. **Catastrophic forgetting:** Policy might forget earlier tasks
2. **Distribution shift:** New tasks may have different reward distributions
3. **Reference drift:** Reference policy becomes outdated

**Adaptation strategies:**

1. **Elastic reference policy:**
   ```python
   # Periodically update reference to blend old and new
   ref_policy = α * ref_policy_old + (1-α) * policy_current
   ```

2. **Task-specific KL coefficients:**
   - Higher β for older tasks (preserve knowledge)
   - Lower β for new tasks (allow adaptation)

3. **Replay buffer:**
   - Store prompts from earlier tasks
   - Sample mixed batches across tasks

4. **Multi-task reward:**
   ```python
   reward_total = Σ_tasks λₜ * reward_t(output)
   ```

5. **Periodic reference reset:**
   - After learning new tasks, freeze new policy as reference
   - Maintains quality on recent tasks

---

#### Q15: What are the failure modes of GRPO and how do you detect them?

**Answer:**

**Common failure modes:**

1. **Reward hacking:**
   - **Symptom:** High rewards but poor actual quality
   - **Detection:** Manual evaluation of high-reward samples
   - **Fix:** Improve reward function, add diversity penalties

2. **Mode collapse:**
   - **Symptom:** All outputs become very similar
   - **Detection:** Low diversity metrics, low entropy
   - **Fix:** Increase temperature, add entropy bonus, increase β

3. **KL explosion:**
   - **Symptom:** KL divergence grows unbounded
   - **Detection:** Monitor KL > threshold (e.g., 10)
   - **Fix:** Increase β, decrease learning rate, reset to reference

4. **Gradient instability:**
   - **Symptom:** Loss spikes, NaN gradients
   - **Detection:** Track gradient norms, loss variance
   - **Fix:** Gradient clipping, smaller learning rate, larger group size

5. **Weak learning signal:**
   - **Symptom:** No improvement over iterations
   - **Detection:** Flat reward curve, KL near zero
   - **Fix:** Check reward function, increase group size, check sampling diversity

**Monitoring dashboard should include:**
- Reward distribution (mean, std, min, max)
- KL divergence with reference
- Advantage distribution
- Policy entropy
- Gradient norms
- Sample diversity metrics

---

### Comparison Questions

#### Q16: When would you choose GRPO over REINFORCE?

**Answer:**

**REINFORCE** is the basic policy gradient algorithm with advantages:
```
∇J = E[∇log π(a|s) × (R - baseline)]
```

**GRPO advantages over REINFORCE:**

1. **Variance reduction:**
   - REINFORCE uses single sample → high variance
   - GRPO uses group comparison → lower variance through normalization

2. **No baseline needed:**
   - REINFORCE needs learned baseline (or uses rewards directly)
   - GRPO uses group mean as implicit baseline

3. **Stability:**
   - REINFORCE can have large gradient variance
   - GRPO has clipping and KL constraints

4. **Sample efficiency:**
   - REINFORCE uses one sample per update
   - GRPO leverages G samples per prompt

**When to use REINFORCE instead:**
- Extremely simple tasks
- When you can't afford multiple samples
- When you have a very good learned baseline already

**In practice:** GRPO is almost always better for LLM tasks due to variance reduction and stability.

---
