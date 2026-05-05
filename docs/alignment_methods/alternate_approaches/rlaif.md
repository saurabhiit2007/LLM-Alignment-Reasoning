# RLAIF

## 1. Overview

**RLAIF (Reinforcement Learning from AI Feedback)** replaces human annotators in the RLHF pipeline with a capable AI judge (e.g., GPT-4, Claude) that generates preference labels at scale. The pipeline structure is identical to RLHF — preference data → reward model → RL optimization — but the labeling step is automated.

The technique was systematically studied by Lee et al. (2023) at Google, who showed that RLAIF achieves performance **comparable to RLHF** on summarization and helpfulness tasks, with dramatically lower annotation cost.

**Key result (Lee et al.):** On a summarization task, RLAIF achieved a 71% win rate over the SFT baseline vs. 73% for RLHF — within noise, and without any human labels.

---

## 2. RLAIF vs RLHF

| Dimension | RLHF | RLAIF |
|-----------|------|-------|
| Label source | Human annotators | AI judge model |
| Cost | High ($/label) | Low (API cost) |
| Speed | Slow (days–weeks) | Fast (hours) |
| Scale | Limited by human bandwidth | Arbitrarily scalable |
| Consistency | Variable (inter-annotator variance) | High |
| Bias risk | Human biases | Inherits judge model biases |
| Nuanced judgment | Strong | Weaker on subtle or cultural distinctions |

**When to prefer RLAIF:** large-scale alignment, rapid iteration, lower-stakes domains, or as a complement to a small human-labeled seed dataset.

---

## 3. AI Judge Design

### 3.1 Judge Model Selection

- Use a model stronger than the student being trained (e.g., GPT-4 as judge for a Llama-70B student)
- Same-model self-judging creates feedback loops and is generally avoided
- Frontier API models are common judges; open-weight >70B models are the minimum effective threshold in practice

### 3.2 Prompting the Judge

The prompting strategy is the most critical design choice:

- Present two responses A and B for a given prompt
- Ask the judge to reason before deciding (chain-of-thought before preference label)
- Request structured output: preferred response + explanation
- Include explicit evaluation criteria: helpfulness, accuracy, safety

Chain-of-thought prompting of the judge (forcing a written rationale before the label) measurably improves preference quality vs. asking for the label directly.

### 3.3 Preference Quality Control

**Agreement filtering:** Generate 3–5 judgments per pair; keep only pairs where the judge agrees ≥80% of the time. Reduces noise from judge inconsistency.

**Confidence thresholding:** Filter out comparisons where the judge expresses low confidence or produces near-identical reasoning for both sides.

**Human spot-checking:** Validate AI preferences against human labels on 5–10% of data. If agreement drops below ~75%, revisit judge prompting or model choice.

---

## 4. Preference Data Pipeline

**Response generation:**

- Sample 4–16 responses per prompt using varied decoding (temperature, top-p) and different model checkpoints to create diverse candidates
- Pairing strategies: best-vs-worst, adjacent ranking, or random pairs from the candidate set

**Preference pair volume:** RLAIF typically requires 2–3× more preference pairs than RLHF to achieve comparable performance due to higher label noise.

**Reward model training:** Identical to RLHF — Bradley-Terry loss on the AI-generated preference pairs. See [RLHF Pipeline](../rlhf/rlhf_pipeline.md) for architecture and training details.

---

## 5. Variants

### Constitutional RLAIF

The AI judge evaluates responses against a set of explicit principles ("Choose the response that is more helpful and less harmful"). Closely related to [Constitutional AI](constitutional_ai.md). Provides interpretable, auditable preference criteria.

### Self-Rewarding Language Models (Yuan et al., 2024)

The model judges its own outputs in an iterative loop: generate responses → self-evaluate → train on self-generated preferences → repeat. Requires careful initialization to avoid degenerate reward hacking. Meta showed this can improve instruction-following iteratively across rounds.

### RLAIF-V (Zhang et al., 2024)

Extends RLAIF to vision-language models, using a multimodal AI judge for image-grounded preference comparisons.

---

## 6. Limitations

- **Bias amplification:** The judge's biases (verbosity preference, format sensitivity, sycophancy) are baked into the reward model and compounded through RL optimization
- **No ground truth:** Unlike human feedback, there is no external reference for whether the AI judge is actually correct on nuanced or contested questions
- **Distribution ceiling:** The student model is bounded by the judge's own capability — it cannot learn preferences the judge cannot recognize
- **Self-referential loops:** If the student and judge share pretraining data or architecture, the feedback may reinforce shared failure modes

---

*Sources: Lee et al. (2023) — RLAIF [[arXiv:2309.00267]](https://arxiv.org/abs/2309.00267) · Yuan et al. (2024) — Self-Rewarding LMs [[arXiv:2401.10020]](https://arxiv.org/abs/2401.10020)*
