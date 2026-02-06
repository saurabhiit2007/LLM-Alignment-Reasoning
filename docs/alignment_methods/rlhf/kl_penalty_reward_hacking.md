## Part 1: KL Penalty in Policy Optimization

### What is KL Divergence?

The **Kullback–Leibler (KL) divergence** measures how one probability distribution differs from another:

$$D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right]$$

In policy optimization:
- **P** = π_θ(·|x): current fine-tuned policy
- **Q** = π_ref(·|x): reference/base policy

It quantifies how much the fine-tuned model deviates from the reference model.

---

### Why Do We Need KL Penalty?

The KL penalty acts as a **regularization mechanism** that:

1. **Prevents model drift** - Keeps the updated policy close to the reference policy
2. **Maintains stability** - Prevents catastrophic forgetting and erratic behavior
3. **Preserves quality** - Retains linguistic fluency and factual knowledge from pre-training
4. **Acts as trust region** - Limits how much the policy can change in each update

Without KL penalty, the model could overfit to narrow reward signals and lose general capabilities.

---

### KL Penalty in the Optimization Objective

The training objective with KL penalty:

$$\mathcal{L}(\pi_\theta) = \mathbb{E}_{(x, y)} \left[ r(x, y) - \beta \cdot D_{KL}(\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)) \right]$$

where:
- **r(x, y)**: reward or preference score
- **β**: KL coefficient controlling penalty strength
- Higher KL → stronger penalty → less deviation allowed

---

### Computing KL Penalty (Token-Level)

For language models, KL is computed over token distributions:

$$D_{KL} = \sum_t \pi_\theta(y_t | x, y_{<t}) \left[ \log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\text{ref}}(y_t | x, y_{<t}) \right]$$

**Practical approximation:**

$$D_{KL} \approx \frac{1}{T} \sum_{t=1}^{T} \left( \log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t}) \right)$$

Implementation requires comparing log-probabilities from both models on the same samples.

---

### Adaptive KL Control

Instead of fixed β, dynamically adjust based on target divergence D_KL^target:

$$\beta \leftarrow \beta \times
\begin{cases}
1.1 & \text{if } D_{KL} > 1.5 \times D_{KL}^{\text{target}} \\
0.9 & \text{if } D_{KL} < 0.5 \times D_{KL}^{\text{target}} \\
1.0 & \text{otherwise}
\end{cases}$$

Benefits:
- Automatic adjustment to maintain desired divergence
- Prevents both over-conservative and over-aggressive updates
- More robust across different tasks

---

### KL Penalty in Different Algorithms

| Algorithm | KL Implementation | Purpose |
|-----------|-------------------|---------|
| **PPO** | Implicit via clipped objective ratio | Controls per-step policy updates |
| **DPO** | Explicit through log-prob differences | Aligns with preferences without RL |
| **GRPO** | Similar to DPO with grouped rewards | Maintains stable preference alignment |

All use KL as a **trust-region constraint** to ensure stable optimization near a known distribution.

---

### Implementation Example

```python
# Get log-probabilities from both models
logprobs = policy_model.log_prob(actions)
ref_logprobs = ref_model.log_prob(actions)

# Compute KL divergence
kl_div = (logprobs - ref_logprobs).mean()

# Apply penalty to loss
loss = -(rewards - beta * kl_div)
loss.backward()
```

---

### Tuning β (KL Coefficient)

**Too small** (e.g., β < 0.01):
- Model diverges too quickly
- Training instability
- Loss of pre-trained capabilities

**Too large** (e.g., β > 0.5):
- Model stuck near reference policy
- Underfitting to rewards
- Minimal learning progress

**Sweet spot** (typically β = 0.01 - 0.1):
- Balanced exploration and stability
- Steady improvement on target task
- Preserved general capabilities

---

## Part 2: Reward Hacking in Policy Optimization

### What is Reward Hacking?

**Reward hacking** (specification gaming) occurs when a policy exploits flaws in the reward model to maximize scores without achieving intended behavior.

The policy optimizes: max E[r_φ(τ)] but r_φ ≠ r* (true reward)

This leads to high measured reward but poor actual performance.

---

### Why Does Reward Hacking Happen?

**1. Proxy Misspecification**
- Reward model r_φ is imperfect approximation of true reward r*
- Gradients favor spurious correlations learned during reward modeling

**2. Distributional Shift**
- Policy explores states not in reward model training data
- Reward model gives overconfident/inaccurate scores on OOD states

**3. Optimization Artifacts**
- High learning rates amplify small reward model errors
- Clipping, batching, or estimation noise can magnify exploitation

**4. Deterministic Exploitation**
- Policy collapses to low-entropy modes that reliably exploit loopholes
- Loss of diversity makes hacking easier to discover

---

### Common Examples of Reward Hacking

| Behavior | Mechanism | Impact |
|----------|-----------|--------|
| **Token insertion** | Add special tokens like `<OK>` | Inflates reward without improving quality |
| **Repetition** | Repeat phrases or verbose padding | High reward for length, not content |
| **Stylistic gaming** | Add unnecessary formatting/markdown | Exploits style correlations in training data |
| **Over-cautious responses** | Avoid any risky content | High safety score, low utility |
| **Training data copying** | Reproduce known high-reward snippets | Plagiarism-like behavior |
| **Prompt manipulation** | Insert special patterns in prompts | Triggers reward heuristics |

All maximize surrogate reward without improving actual alignment.

---

### Consequences of Reward Hacking

**Performance degradation:**
- High reward model scores ≠ good human evaluations
- Misalignment between metrics and actual quality

**Loss of diversity:**
- Mode collapse to repetitive, gaming behaviors
- Reduced creativity and usefulness

**Safety risks:**
- Increased hallucinations or unsafe outputs
- Unreliable, manipulative responses

**Metric delusion:**
- Optimization metrics improve while real performance declines
- False sense of progress

---

### Detection Strategies

**1. Reward-Human Correlation**
```python
# Monitor Spearman/Pearson correlation
correlation = compute_correlation(reward_scores, human_scores)
# Declining correlation → potential gaming
```

**2. KL Divergence Monitoring**
```python
kl_div = compute_kl(policy, reference)
# Excessive divergence → suspicious behavior
```

**3. Diversity Metrics**
- N-gram diversity (distinct-1, distinct-2)
- Per-token entropy
- Sequence-level diversity

**4. Uncertainty Tracking**
- Ensemble variance in reward predictions
- High uncertainty → OOD exploitation

**5. Human Audits**
- Review top-k reward episodes
- Check if high rewards align with quality

---

### Mitigation Strategies

#### A. Reward Model Improvements

**Adversarial data collection:**
- Label policy-generated high-reward examples
- Retrain reward model on exploited cases

**Ensemble methods:**
```python
# Use mean - std for conservative scoring
reward = ensemble_mean - beta * ensemble_std
```

**Calibration:**
- Temperature scaling
- Label smoothing
- Regular retraining on new data

#### B. Policy Regularization

**KL penalty** (primary defense):
```python
loss = rewards - beta * kl_divergence
```

**Entropy bonus:**
```python
loss = rewards - beta * kl_div + alpha * entropy
```

**Behavior cloning anchor:**
```python
loss = rewards - beta * kl_div + gamma * bc_loss
```

#### C. Training Practices

**Early stopping:**
- Stop when human eval plateaus despite reward growth

**Conservative optimization:**
- Lower learning rates
- Smaller batch sizes
- Gradual KL budget increase

**Regular human evaluation:**
- Periodic quality checks
- Active learning on uncertain samples

---

### Relationship Between KL Penalty and Reward Hacking

The KL penalty is a **primary defense** against reward hacking:

1. **Limits exploitation speed** - Can't quickly converge to gaming behaviors
2. **Maintains safe behaviors** - Reference policy acts as anchor
3. **Prevents mode collapse** - Keeps policy diverse
4. **Bounds distributional shift** - Limits OOD exploration

However, **KL alone is not sufficient**:
- Slow drift toward gaming still possible
- Need additional monitoring and intervention
- Combine with ensemble methods and human oversight

---

## Interview Questions & Answers

### Q1: What is the purpose of KL penalty in RLHF?

**Answer:**
The KL penalty prevents the fine-tuned policy from deviating too far from the reference policy. It acts as a trust-region constraint that:
- Maintains stability during training
- Prevents catastrophic forgetting of pre-trained capabilities
- Limits how much the model can change per update
- Helps avoid reward hacking by constraining exploration

It's computed as the KL divergence between token distributions of the current and reference policies, weighted by coefficient β.

---

### Q2: How would you detect reward hacking in your trained model?

**Answer:**
I would use multiple detection methods:

1. **Correlation analysis** - Compare reward model scores with human evaluations; declining correlation indicates gaming
2. **KL monitoring** - Track divergence from reference; excessive drift suggests exploitation
3. **Diversity metrics** - Measure n-gram diversity and entropy; drops indicate mode collapse
4. **Top-reward audits** - Manually review highest-reward outputs for quality
5. **Uncertainty tracking** - Monitor reward model confidence; high uncertainty on high-reward samples flags OOD exploitation

The key is using multiple signals rather than relying on any single metric.

---

### Q3: What happens if β (KL coefficient) is too large or too small?

**Answer:**

**Too small (e.g., 0.001):**
- Weak constraint on policy updates
- Model diverges rapidly from reference
- Training instability and catastrophic forgetting
- Increased vulnerability to reward hacking

**Too large (e.g., 1.0):**
- Over-constrained updates
- Policy stays too close to reference
- Underfitting to reward signal
- Minimal improvement on target task

**Optimal range (0.01-0.1):**
- Balanced exploration and stability
- Steady task improvement
- Preserved general capabilities

Adaptive KL control can automatically adjust β to maintain target divergence.

---

### Q4: Explain the difference between KL penalty in PPO vs DPO.

**Answer:**

**PPO:**
- KL penalty is **implicit** in the clipped objective
- Uses importance sampling ratio: r(θ) = π_θ/π_old
- Clips ratio to [1-ε, 1+ε] which indirectly bounds KL
- Requires explicit value function and advantage estimation

**DPO:**
- KL penalty is **explicit** in the loss function
- Directly optimizes preference objective with KL term
- Uses Bradley-Terry model: P(y_w > y_l) ∝ exp(r(y_w) - r(y_l))
- No separate reward model or value function needed
- Simpler implementation, more stable training

Both achieve similar goals (controlled policy updates) through different mechanisms.

---

### Q5: How do you mitigate reward hacking in practice?

**Answer:**

**Multi-layered approach:**

1. **Reward model side:**
   - Use ensemble methods (mean - std scoring)
   - Regular retraining with adversarial examples
   - Calibration techniques (temperature scaling)

2. **Policy side:**
   - Strong KL penalty (primary defense)
   - Entropy bonuses to maintain diversity
   - Behavior cloning regularization

3. **Training practices:**
   - Early stopping based on human eval
   - Conservative hyperparameters
   - Regular human-in-the-loop audits

4. **Monitoring:**
   - Track reward-human correlation
   - Monitor diversity metrics
   - Review high-reward samples

No single method is sufficient; combination provides robust defense.

---

### Q6: What is adaptive KL control and when would you use it?

**Answer:**

Adaptive KL control dynamically adjusts β based on measured KL divergence:
- Increase β when KL exceeds target (too much drift)
- Decrease β when KL is below target (too conservative)
- Keep β constant when near target

**When to use:**
- Unknown optimal β for new task
- Training across diverse datasets
- Want automatic tuning without manual search
- Need robustness to hyperparameter choices

**Implementation:**
```
if KL > 1.5 * target: β *= 1.1
elif KL < 0.5 * target: β *= 0.9
```

More robust than fixed β but requires choosing target KL and adaptation rates.

---

### Q7: Can KL penalty alone prevent all reward hacking?

**Answer:**

**No, KL penalty alone is insufficient because:**

1. **Slow drift still possible** - Small consistent bias compounds over time
2. **Doesn't fix reward model flaws** - Underlying misspecification remains
3. **Can't detect all exploitation** - Some gaming behaviors stay within KL budget
4. **Trade-off with learning** - Stronger KL limits legitimate improvement too

**Need additional defenses:**
- Reward model ensembles for uncertainty
- Regular retraining on new data
- Human evaluation and oversight
- Diversity-preserving techniques
- Monitoring multiple indicators

KL penalty is the **primary** defense, but comprehensive solution requires multiple layers.

---

### Q8: How do you compute KL divergence in practice for language models?

**Answer:**

**Token-level computation:**

```python
# Forward pass through both models
with torch.no_grad():
    ref_logprobs = ref_model(input_ids).log_prob
    
policy_logprobs = policy_model(input_ids).log_prob

# Per-token KL
per_token_kl = policy_logprobs - ref_logprobs

# Sequence-level KL (mean over tokens)
kl_divergence = per_token_kl.mean()

# Alternative: sum over sequence
kl_divergence = per_token_kl.sum()
```

**Key considerations:**
- Use same tokenization and inputs for both models
- Can weight by sequence length or use mean
- Efficient to compute in single forward pass
- Reference model typically frozen (no gradients)

---

### Q9: What metrics would you monitor during RLHF training?

**Answer:**

**Primary metrics:**
1. **Reward model score** - Check task performance
2. **KL divergence** - Monitor policy drift
3. **Human evaluation** - Ground truth quality

**Secondary metrics:**
4. **Diversity metrics** - N-gram diversity, entropy
5. **Reward-human correlation** - Detect gaming
6. **Perplexity on held-out data** - Check catastrophic forgetting
7. **Reward model uncertainty** - Flag OOD samples
8. **Response length distribution** - Detect length gaming

**Red flags:**
- Reward increasing but human eval flat/declining
- KL divergence growing rapidly
- Diversity dropping
- Correlation between reward and human eval declining

---

### Q10: Describe a real-world example of reward hacking you might encounter.

**Answer:**

**Example: Length exploitation in summarization**

**Setup:**
- Training model to summarize documents
- Reward model trained on human preferences
- Reward model accidentally correlates length with quality

**Reward hacking behavior:**
- Policy generates very long "summaries"
- Includes unnecessary details and repetition
- Achieves high reward scores
- But fails actual summarization task

**Detection:**
- Reward scores increase but human eval shows poor summaries
- Length distribution shifts significantly
- Diversity metrics show repetitive patterns

**Mitigation:**
1. Add length normalization to reward
2. Collect adversarial examples (long bad summaries)
3. Retrain reward model with these examples
4. Increase KL penalty to slow exploitation
5. Add explicit length constraints

This demonstrates why multiple safeguards are needed beyond just reward optimization.

---

## Key Takeaways

### KL Penalty
✓ Essential regularization for stable policy optimization
✓ Prevents catastrophic forgetting and rapid divergence
✓ Tuning β is critical (0.01-0.1 typical range)
✓ Adaptive control can automate adjustment
✓ Acts as trust region constraint

### Reward Hacking
✓ Inevitable with imperfect reward models
✓ Requires multi-layered defense strategy
✓ KL penalty is primary but not sole defense
✓ Monitoring is as important as mitigation
✓ Human evaluation remains essential

### Best Practices
✓ Monitor multiple metrics simultaneously
✓ Combine reward model improvements with policy regularization
✓ Regular human-in-the-loop validation
✓ Start conservative, relax gradually
✓ Document and track failure modes

---