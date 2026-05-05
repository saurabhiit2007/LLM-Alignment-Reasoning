# Constitutional AI

## 1. Overview

Constitutional AI (CAI) is a training methodology introduced by Anthropic (Bai et al., 2022) to produce helpful, harmless, and honest AI systems using AI-generated feedback guided by a set of explicit principles — the "constitution" — rather than relying exclusively on human preference labels for safety training.

The technique has two stages:

1. **SL-CAI** — Supervised learning: the model critiques and revises its own outputs using constitutional principles, then is fine-tuned on the revisions
2. **RL-CAI** — RLHF with AI feedback: AI-generated preference labels (not human labels) train a reward model, which is then used for PPO-based RL fine-tuning

---

## 2. The Constitution

The constitution is a list of natural-language principles that encode desired behavior. Examples from the paper:

- *"Choose the response that is least likely to contain harmful or unethical content."*
- *"Choose the response that is most helpful, accurate, and non-deceptive."*
- *"Choose the response that a thoughtful, senior Anthropic employee would consider optimal."*

Principles are sampled randomly during training so no single principle dominates. This makes the behavioral constraints explicit and auditable — unlike RLHF where values are implicitly encoded in annotator preferences.

---

## 3. Stage 1 — SL-CAI (Supervised Learning)

**Starting point:** A helpful-only model — trained to be useful but with no harmlessness training, so it will often comply with harmful requests.

**The critique-revision loop** (repeated 1–4 times per example):

1. Generate an initial response to a potentially harmful prompt using the helpful-only model
2. Sample a constitutional principle at random
3. Prompt the model to critique its own response against that principle
4. Prompt the model to revise the response to address the critique

**Fine-tuning:** Collect the final revised responses and fine-tune the model on them *without* the constitutional principles in the input (context distillation). The model learns to produce aligned outputs without needing the constitution in every prompt.

**Why start with a harmful-only model?** The paper deliberately uses a model that complies with harmful requests so that the critique-revision loop has meaningful work to do — starting from a safe model produces little signal for the supervised stage.

---

## 4. Stage 2 — RL-CAI (RLHF with AI Feedback)

**Generating preference pairs:** Use the SL-CAI model to generate pairs of responses to a set of prompts.

**AI labeling:** Present each pair to a feedback model (the SL-CAI model itself) along with a constitutional principle. The model reasons step-by-step (chain-of-thought) and then selects which response better follows the principle. This produces a large preference dataset at low cost.

**Reward model training:** Train a preference model (PM) on the AI-generated preference labels using the standard Bradley-Terry objective.

**RL optimization:** Fine-tune the SL-CAI policy using PPO against the PM reward, with a KL penalty to prevent excessive drift from the supervised baseline:

$$r_\text{total} = r_\text{PM}(x, y) - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_\text{SL-CAI})$$

---

## 5. Results

- CAI models reduce harmful outputs substantially while **maintaining or improving helpfulness** relative to RLHF-only harmless models
- Crowdworkers rated RL-CAI models as both more helpful *and* more harmless than prior RLHF baselines
- Chain-of-thought AI labeling (reasoning before the preference label) improves label quality over direct preference elicitation
- The explicit constitution makes the trade-offs transparent: researchers can inspect and modify what principles are being optimized

---

## 6. Advantages and Limitations

**Advantages:**

- **Scales harmlessness training without human labels** — critique and revision is automated
- **Explicit values**: the constitution is readable and modifiable; RLHF values are implicit in annotator behavior
- **Helpfulness preserved**: starting from a helpful-only model and applying principles separately avoids the over-refusal failure mode common in RLHF-only safety training
- **Reduces annotator burden**: humans still label helpfulness data, but harmlessness data is AI-generated

**Limitations:**

- Requires a capable base model to perform meaningful self-critique — weak models produce low-quality critiques
- Constitution design requires careful value trade-off decisions; conflicting principles need prioritization
- The AI feedback model inherits the base model's biases — there is no external ground truth for whether a critique is correct
- Still relies on human-written principles; the choice of constitution is a normative decision, not a technical one

---

## 7. Relation to Other Techniques

| Technique | Feedback source | Explicit values | Iterative refinement |
|-----------|----------------|-----------------|---------------------|
| RLHF | Human annotators | No | Yes (RL loop) |
| RLAIF | AI judge | No | Yes (RL loop) |
| Constitutional AI | AI judge + principles | Yes | Yes (critique-revise + RL) |
| Context distillation | Prompted model outputs | Implicit in prompt | No |

CAI is effectively RLAIF with the addition of explicit constitutional principles and the supervised critique-revision stage. Context distillation is used within SL-CAI as the fine-tuning mechanism.

---

*Source: Bai et al. (2022) — Constitutional AI: Harmlessness from AI Feedback [[arXiv:2212.08073]](https://arxiv.org/abs/2212.08073)*
