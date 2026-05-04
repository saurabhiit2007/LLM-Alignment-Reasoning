# RLHF Pipeline

## 1. Overview

RLHF (Reinforcement Learning from Human Feedback) is a three-stage pipeline to align language models with human preferences, introduced in Christiano et al. (2017) and scaled to modern LLMs by Ouyang et al. (2022) via InstructGPT:

1. **Supervised Fine-Tuning (SFT)** — Train a base model on high-quality human demonstrations
2. **Reward Model Training** — Learn a scalar reward function from human preference comparisons
3. **RL Optimization** — Fine-tune the SFT policy to maximize the learned reward (via PPO or alternatives)

Each stage builds on the previous: SFT gives a competent starting policy, the reward model encodes human judgment, and RL shifts the policy toward higher-reward outputs while a KL penalty prevents it from drifting too far.

---

## 2. Supervised Fine-Tuning (SFT)

The base pretrained model is fine-tuned on a curated set of prompt–response pairs written or selected by human labelers. The goal is to teach the model to follow instructions before any preference signal is introduced.

**Key choices:**

- Dataset: 10K–100K high-quality demonstrations (InstructGPT used ~13K)
- Labelers write ideal responses; lower-quality responses are filtered out
- Standard cross-entropy training on the response tokens only
- Learning rate ~1e-5, 1–3 epochs to avoid overfitting the small dataset

The SFT model becomes both the starting point for RL optimization and the reference policy for the KL penalty.

---

## 3. Preference Data Collection

### 3.1 Data Generation

**Prompt selection:**

- Curate diverse prompts covering target use cases
- Include different difficulty levels and domains
- Sources: user interactions, seed datasets, synthetic generation

**Response sampling:**

- Generate 2–4 completions per prompt using temperature sampling (T ∈ [0.7, 1.0]) and varied decoding (top-k, nucleus) to ensure meaningful differences worth comparing

---

### 3.2 Human Annotation

**Comparison types:**

- **Pairwise**: Choose the better response (A > B, B > A, or Tie)
- **Ranking**: Order k responses from best to worst
- **Likert Scale**: Rate each independently (1–5 stars)

Pairwise comparison is generally preferred — relative judgments are easier for humans, more reliable, and fit naturally into the Bradley-Terry model used for reward training.

**Annotation criteria (InstructGPT):**

- **Helpfulness**: Does it answer the question well?
- **Harmlessness**: Is it safe and appropriate?
- **Honesty**: Is it truthful and admits uncertainty?

**Best practices:**

- Clear guidelines with worked examples
- Inter-annotator agreement checks (Fleiss' κ, Krippendorff's α)
- Multiple annotators per comparison (typically 3–5)
- Gold-standard examples for quality control

---

### 3.3 Data Quality

**Common issues:**

- **Annotation bias**: Personal preferences vs. general quality
- **Low agreement**: Ambiguous prompts or subjective criteria
- **Gaming**: Annotators choosing randomly or following patterns

**Mitigations:**

- Calibration sessions before annotation begins
- Disagreement resolution protocols
- Monitor annotation time and patterns
- Discard high-disagreement pairs (likely ambiguous)

**Dataset size:** 10K–100K preference pairs; InstructGPT used ~50K comparisons. Quality dominates quantity.

---

## 4. Reward Model Training

### 4.1 Architecture

The reward model is typically initialized from the SFT model with the final language-modeling head replaced by a linear layer that outputs a scalar `r(x, y)` for prompt `x` and completion `y`. The shared backbone retains language understanding.

### 4.2 Training Objective

The Bradley-Terry model treats preferences probabilistically. For a preference pair $(y_w, y_l)$ where $y_w \succ y_l$:

$$\mathcal{L} = -\log \sigma\bigl(r(x, y_w) - r(x, y_l)\bigr)$$

This maximizes the probability that the preferred completion receives a higher reward. For ranking over $k > 2$ completions:

$$\mathcal{L} = -\sum_{i < j} \log \sigma\bigl(r(x, y_i) - r(x, y_j)\bigr)$$

where $y_i$ is ranked above $y_j$.

**Bradley-Terry assumptions:**

- *Transitivity*: if A > B and B > C, then A > C
- *Independence*: the A vs. B preference doesn't depend on other options
- Real human preferences can violate both; the model works well in practice despite this

### 4.3 Training Details

- Learning rate: ~1e-5 (lower than SFT)
- Batch size: 32–64 comparison pairs
- Epochs: 1–3 (avoid overfitting)
- Regularization: dropout, weight decay, early stopping on validation accuracy
- Split: 80% train / 10% validation / 10% test

### 4.4 Evaluation

- **Pairwise accuracy**: % of held-out preferences predicted correctly (target: >65–70%)
- **Ranking correlation**: Spearman's ρ with human rankings
- Test on novel prompts to check out-of-distribution robustness

---

## 5. RL Optimization

The SFT policy $\pi_\theta$ is fine-tuned to maximize expected reward while staying close to the SFT reference via a KL penalty:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\bigl[r(x, y) - \beta \cdot \mathrm{KL}(\pi_\theta(\cdot|x) \,\|\, \pi_\mathrm{SFT}(\cdot|x))\bigr]$$

**KL penalty role:** High $\beta$ keeps the policy close to SFT (stable, lower reward hacking risk); low $\beta$ allows more optimization freedom but risks generating high-reward but incoherent outputs. Typical values: $\beta \in [0.01, 0.1]$.

**Optimization algorithm:** InstructGPT used PPO. Alternatives include REINFORCE, RLOO, GRPO, and DPO (which eliminates the reward model entirely). See the [RL Optimization Methods](rl_optimization_methods/ppo.md) pages for details.

---

## 6. Key Challenges

### 6.1 Reward Hacking

The policy exploits reward model weaknesses — generating long, verbose, or surface-level impressive outputs that score well but lack quality. Mitigations:

- KL penalty (primary defense)
- Reward model ensembles (harder to fool multiple models simultaneously)
- Iterative reward model updates trained on RL-generated outputs
- Rule-based hard constraints (length limits, repetition penalties)

### 6.2 Reward Model Limitations

The reward model is a fixed approximation of human preferences — it generalizes imperfectly, can be fooled by surface-level patterns, and doesn't update during RL. Solutions: diverse training data, Constitutional AI for principled constraints, and periodic human evaluation on RL outputs.

### 6.3 Scalability

Human annotation is expensive and slow. Approaches to scale:

- **RLAIF**: Use AI feedback to replace or augment human labels
- **Active learning**: Prioritize labeling examples where the reward model is most uncertain (close reward scores), maximizing learning per annotation budget
- **Automated pre-filtering**: Use toxicity filters, length limits, and format validators to screen out obvious failures before sending to human annotators

---

*Sources: Christiano et al. (2017) [[arXiv:1706.03741]](https://arxiv.org/abs/1706.03741) · Ouyang et al. (2022) — InstructGPT [[arXiv:2203.02155]](https://arxiv.org/abs/2203.02155)*
