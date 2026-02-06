## Overview

RLHF (Reinforcement Learning from Human Feedback) is a three-stage pipeline to align language models with human preferences:

1. **Supervised Fine-Tuning (SFT)**: Train base model on high-quality demonstrations
2. **Reward Model Training**: Learn human preferences via a reward function
3. **RL Optimization**: Fine-tune policy using the reward model (PPO, etc.)

This document focuses on stages 2 and the data collection needed for it.

---

---

## 1. Preference Data Collection

### 1.1 Data Generation Process

**Prompt Selection:**
- Curate diverse prompts covering target use cases
- Include different difficulty levels and domains
- Sources: user interactions, seed datasets, synthetic generation

**Response Sampling:**
- Generate 2-4 completions per prompt using:
  - **Temperature sampling** (T ∈ [0.7, 1.0]) for diversity
  - **Different models/checkpoints** to ensure variety
  - **Varied decoding** (top-k, nucleus sampling)
- Goal: Create meaningfully different responses worth comparing

---

### 1.2 Human Annotation

**Comparison Types:**
- **Pairwise**: Choose better response (A > B or B > A or Tie)
- **Ranking**: Order k responses from best to worst
- **Likert Scale**: Rate each independently (1-5 stars)

**Annotation Criteria:**
- **Helpfulness**: Does it answer the question well?
- **Harmlessness**: Is it safe and appropriate?
- **Honesty**: Is it truthful and admits uncertainty?

**Best Practices:**
- Clear guidelines with examples
- Inter-annotator agreement checks (Fleiss' kappa, Krippendorff's alpha)
- Multiple annotators per comparison (typically 3-5)
- Quality control: gold standard examples, spot checks

---

### 1.3 Data Quality Considerations

**Common Issues:**
- **Annotation bias**: Personal preferences vs. general quality
- **Low agreement**: Ambiguous prompts or subjective criteria
- **Gaming**: Annotators choosing randomly or following patterns

**Solutions:**
- Calibration sessions with annotators
- Disagreement resolution protocols
- Monitor annotation time and patterns
- Bonus for high-agreement annotations

**Dataset Size:**
- Typical: 10K-100K preference pairs
- Quality > quantity (InstructGPT used ~50K comparisons)
- More data needed for complex/multi-domain tasks

---

---

## 2. Reward Model Training

### 2.1 Model Architecture

**Base Model:**
- Usually the SFT model with final layer replaced
- Outputs scalar reward: `r(x, y)` for prompt x and completion y
- Shared backbone leverages language understanding

**Training Objective (Bradley-Terry Model):**

For preference pair (y_w, y_l) where y_w ≻ y_l:

```
Loss = -log σ(r(x, y_w) - r(x, y_l))
```

Where σ is sigmoid function. This maximizes probability that preferred completion gets higher reward.

**Alternative: Ranking Loss (for k > 2 completions):**

```
Loss = -∑_{i<j} log σ(r(x, y_i) - r(x, y_j))
```

Where y_i is ranked higher than y_j.

---

### 2.2 Training Process

**Data Preparation:**
- Split: 80% train, 10% validation, 10% test
- Ensure prompt diversity across splits
- Balance difficulty levels

**Training Details:**
- Learning rate: ~1e-5 (lower than SFT)
- Batch size: 32-64 comparison pairs
- Epochs: 1-3 (avoid overfitting)
- Monitor validation accuracy

**Regularization:**
- Dropout in final layers
- Early stopping based on validation accuracy
- Weight decay

---

### 2.3 Reward Model Evaluation

**Accuracy Metrics:**
- **Pairwise accuracy**: % of correct preference predictions
- **Ranking correlation**: Spearman's ρ with human rankings
- Typical target: >65-70% accuracy on held-out test set

**Calibration:**
- Check if reward magnitude correlates with confidence
- Avoid overconfident predictions

**Out-of-Distribution Detection:**
- Test on novel prompts/domains
- Reward model should be robust to distribution shift

---

---

## 3. Key Challenges

### 3.1 Reward Hacking

**Problem:** Policy exploits reward model weaknesses, generating high-reward but low-quality outputs.

**Mitigation:**
- KL penalty to stay close to SFT model: `r_total = r_RM - β * KL(π || π_SFT)`
- Reward model ensembles
- Regular reward model updates during RL

---

### 3.2 Reward Model Limitations

**Issues:**
- Limited to training distribution
- May not capture all aspects of quality
- Can be fooled by surface-level patterns

**Solutions:**
- Diverse training data
- Constitutional AI for principled constraints
- Human oversight during RL

---

### 3.3 Scalability

**Challenges:**
- Human annotation is expensive and slow
- Need continuous data for model updates

**Approaches:**
- **RLAIF:** Use AI feedback to scale
- **Active learning:** Select most informative comparisons. Prioritize labeling examples where the reward model is most uncertain or disagrees (e.g., close reward scores), maximizing learning per annotatio
- **Automated filters before human review:** Use automated checks (toxicity filters, length limits, format validators) to screen out obviously bad responses before sending to human annotators, reducing annotation cost.

---

---

## 4. Interview Questions

### Conceptual Questions

**Q1: Why do we need a separate reward modeling phase instead of directly using human feedback during RL?**

<details>
<summary>Answer</summary>

- **Sample efficiency**: RL requires millions of samples; human labeling can't scale to that
- **Cost**: Human feedback is expensive; reward model provides unlimited free evaluations
- **Speed**: Reward model inference is fast; enables real-time RL training
- **Consistency**: Reward model provides consistent scores; humans may have variance

However, reward model introduces approximation error and potential reward hacking.
</details>

**Q2: What's the difference between pairwise comparisons and absolute ratings? Which is better for RLHF?**

<details>
<summary>Answer</summary>

**Pairwise comparisons:**
- Pros: Easier for humans (relative judgment), more reliable, handles subjectivity better
- Cons: Requires more data (combinatorial), doesn't give absolute scale

**Absolute ratings:**
- Pros: Efficient data collection, provides absolute scale
- Cons: Harder to calibrate, annotator disagreement higher, scale ambiguity

**Pairwise is generally preferred** for RLHF because:
- Human preferences are more consistent in relative judgments
- Bradley-Terry model naturally fits preference data
- Reduces annotator bias (no need to agree on absolute scale)
</details>

**Q3: How does the Bradley-Terry model work, and what assumptions does it make?**

<details>
<summary>Answer</summary>

**Model**: Assumes P(y_w ≻ y_l) = σ(r(y_w) - r(y_l))

**Assumptions:**
1. **Transitivity**: If A > B and B > C, then A > C
2. **Independence**: Preference between A and B doesn't depend on other options
3. **Scale invariance**: Only reward differences matter, not absolute values

**Limitations:**
- Real human preferences may violate transitivity
- Context matters (preferences may not be independent)
- Doesn't model uncertainty or indifference well

Despite limitations, works well in practice for RLHF.
</details>

**Q4: What is reward hacking and how do we prevent it in RLHF?**

<details>
<summary>Answer</summary>

**Reward hacking**: Policy exploits flaws in the reward model to achieve high scores without actually improving quality.

**Examples:**
- Generating very long responses (reward model correlates length with quality)
- Using fancy words or formatting without substance
- Exploiting reward model's lack of factual knowledge

**Prevention strategies:**
1. **KL penalty**: `r_total = r_RM - β·KL(π || π_SFT)` keeps policy close to SFT baseline
2. **Reward model ensembles**: Harder to hack multiple models simultaneously
3. **Iterative reward model updates**: Retrain on RL-generated outputs
4. **Rule-based constraints**: Hard limits on length, repetition, etc.
5. **Human-in-the-loop**: Regular human evaluation during RL training
</details>

**Q5: Why do we initialize the RL policy from the SFT model rather than training from scratch?**

<details>
<summary>Answer</summary>

**Reasons:**
1. **Better starting point**: SFT model already generates reasonable outputs
2. **Faster convergence**: Less exploration needed
3. **Prevents catastrophic forgetting**: Maintains language modeling capabilities
4. **KL penalty works better**: Meaningful to constrain distance from SFT model
5. **Reduces reward hacking**: Harder to drift into degenerate solutions

Without SFT initialization:
- RL might converge to nonsensical but high-reward outputs
- Exploration in text space is extremely difficult
- Training is much slower and less stable
</details>

### Technical Questions

**Q6: How would you handle disagreement between human annotators?**

<details>
<summary>Answer</summary>

**Measurement:**
- Calculate inter-annotator agreement (Fleiss' kappa, Krippendorff's alpha)
- Track per-annotator agreement with majority/expert

**Handling strategies:**
1. **Majority vote**: Use most common preference
2. **Weighted voting**: Weight by annotator reliability
3. **Discard high-disagreement examples**: They're likely ambiguous
4. **Model uncertainty**: Train ensemble or probabilistic reward model
5. **Consensus building**: Resolve disagreements through discussion
6. **Improve guidelines**: Address common disagreement sources

**For training:**
- Can model soft preferences: P(y_w ≻ y_l) = fraction of annotators who preferred y_w
- Helps reward model learn uncertainty
</details>

**Q7: How do you ensure your reward model generalizes to out-of-distribution prompts?**

<details>
<summary>Answer</summary>

**During data collection:**
1. **Diverse prompt set**: Cover many domains, styles, difficulties
2. **Include edge cases**: Adversarial, ambiguous, multi-step prompts
3. **Regular updates**: Continuously add new prompt types

**During training:**
1. **Regularization**: Dropout, weight decay to prevent overfitting
2. **Data augmentation**: Paraphrase prompts, vary response styles
3. **Domain-specific splits**: Ensure validation set covers all domains

**Evaluation:**
1. **Hold-out test sets**: Different domains than training
2. **Monitor RL outputs**: Check for reward hacking patterns
3. **Human evaluation**: Regular checks on RL-generated samples
4. **Red-teaming**: Actively try to find failure modes

**Continuous improvement:**
- Retrain reward model on RL-generated distribution
- Active learning to find informative new comparisons
</details>

**Q8: What's the trade-off in choosing the β parameter for the KL penalty?**

<details>
<summary>Answer</summary>

The total reward is: `r_total = r_RM(x,y) - β·KL(π(y|x) || π_SFT(y|x))`

**High β (strong penalty):**
- Pros: Policy stays very close to SFT, prevents reward hacking, stable training
- Cons: Limited improvement, may not fully leverage reward signal

**Low β (weak penalty):**
- Pros: More optimization freedom, potentially better performance
- Cons: Higher reward hacking risk, may drift into nonsensical outputs

**Typical values**: β ∈ [0.01, 0.1]

**Adaptive strategies:**
- Start with high β, gradually decrease
- Use different β for different layers
- Monitor KL divergence and adjust dynamically
- Per-example β based on reward model confidence

**In practice**: Tune β on validation set balancing reward model score and human evaluation.
</details>

**Q9: Compare RLHF with Direct Preference Optimization (DPO). What are the trade-offs?**

<details>
<summary>Answer</summary>

**RLHF (PPO-based):**
- Pros: Explicit reward model (interpretable), flexible (can update RM), handles complex rewards
- Cons: Complex pipeline (3 stages), RL instability, reward hacking risks, computationally expensive

**DPO:**
- Pros: Simpler (single-stage), more stable, no reward model to hack, lower compute
- Cons: Less flexible (bakes in Bradley-Terry assumption), harder to update preferences, no explicit reward signal

**Key difference**: DPO reparameterizes the RL objective to directly optimize policy from preferences, eliminating reward model.

**When to use:**
- **RLHF**: Need interpretable rewards, complex multi-objective optimization, iterative updates
- **DPO**: Simpler use cases, want stability, limited compute

**Trend**: DPO gaining popularity due to simplicity, but RLHF still useful for complex alignment tasks.
</details>

**Q10: How would you design an active learning strategy for preference data collection?**

<details>
<summary>Answer</summary>

**Goal**: Select most informative comparisons to minimize annotation cost while maximizing reward model quality.

**Uncertainty-based sampling:**
1. **Model disagreement**: Query pairs where ensemble models disagree most
2. **Entropy**: Select comparisons with highest prediction entropy
3. **Margin**: Choose pairs with smallest reward difference (close calls)

**Diversity-based sampling:**
1. **Prompt coverage**: Ensure diverse prompt types are covered
2. **Response diversity**: Sample varied response styles
3. **Cluster-based**: Select representatives from different clusters

**Performance-based sampling:**
1. **Error analysis**: Focus on domains where RM performs poorly
2. **Gradient-based**: Select examples with high expected gradient norm
3. **Policy-aware**: Sample from current RL policy distribution

**Practical approach:**
- Combine strategies: 50% uncertainty + 30% diversity + 20% error-focused
- Regular cold-start: Include random samples to prevent bias
- Batch selection: Consider redundancy within each batch
- Monitor distribution shift: Ensure coverage of evolving policy

**Metrics**: Track validation accuracy vs. number of labels to measure efficiency.
</details>

---
