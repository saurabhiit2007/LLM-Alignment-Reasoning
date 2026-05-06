# KL Penalty and Reward Hacking

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

**2. KL Divergence Monitoring**

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

**Calibration:**

- Temperature scaling
- Label smoothing
- Regular retraining on new data

#### B. Policy Regularization

**KL penalty** (primary defense):

**Entropy bonus:**

**Behavior cloning anchor:**

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