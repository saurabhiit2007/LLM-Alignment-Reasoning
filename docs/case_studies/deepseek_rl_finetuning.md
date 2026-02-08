## 1. Overview

This document provides a comprehensive overview of the DeepSeek-R1 strategy for fine-tuning and preference-tuning large language models (LLMs). It covers the RL methods, distinctions from traditional approaches, the GRPO optimization algorithm, multi-stage training pipeline, reward design, model distillation, and additional technical details.

DeepSeek-R1 introduces a novel approach to improving reasoning capabilities and general instruction-following in large language models using reinforcement learning (RL). Rather than relying solely on large human-annotated supervised fine-tuning datasets and learned reward models, DeepSeek emphasizes verifiable tasks (particularly reasoning, mathematics, and code), multi-stage pipelines, and knowledge distillation to smaller models.

### 1.1 Key Variants

- **DeepSeek-R1-Zero**: RL-only variant without initial supervised fine-tuning (SFT). Uses verifiable reasoning tasks (e.g., math, code, logic) with automatically computable reward signals.
- **DeepSeek-R1**: Multi-stage pipeline starting with a "cold-start" SFT, followed by reasoning-oriented RL, generation of an SFT dataset from high-quality RL outputs, further SFT fine-tuning, and then a second RL stage for broader instruction-following.

---

---

## 2. The GRPO Algorithm: Overview

This section provides a summary of the core optimization algorithm used in DeepSeek-R1: Group Relative Policy Optimization (GRPO).

**For full mathematical details and derivation, see the dedicated [GRPO Algorithm](../methods/grpo.md) document.**

### 2.1 Intuition

Instead of using a value network (critic) as in PPO, GRPO operates by sampling multiple candidate outputs for each prompt and comparing their performance **within the group** through ranking. This approach encourages responses that outperform peers in the same group, emphasizing **relative** improvement rather than absolute reward magnitudes. GRPO offers more stable training for LLMs at scale by avoiding the complexity and instability of critic/value training.

---

### 2.2 Key Features

- **Group-based candidate sampling**: Group size $G$ typically ranges from 8–16
- **Advantage computation**:
  $$A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}$$
  where $r_i$ is the reward of candidate $o_i$
- **PPO-style clipped ratio**: Applied to new policy versus old policy for each candidate
- **KL regularization**: Prevents drift from a reference policy
- **No explicit value function**: Critical for large-scale LLM fine-tuning efficiency

---

---

## 3. Reward Design

DeepSeek divides reward design into two main domains: reasoning-oriented tasks and general instruction-following tasks.

### 3.1 Reasoning-Oriented Tasks

The reward function for reasoning tasks includes:

- **Correctness**: Automatically verified through solvers for math answers or compilers/tests for code solutions
- **Chain-of-Thought (CoT) / Format Enforcement**: Encourages structured reasoning via tags or designated reasoning segments
- **Language Consistency / Style**: Penalizes language mixing (e.g., mixing English and Mandarin) or incoherent formatting
- **Weighted Sum**: The overall reward combines correctness with readability and style metrics

---

### 3.2 General Instruction-Following Tasks

For broader tasks, DeepSeek employs:

- Preference models or mixtures of rule-based checks for helpfulness, harmlessness, and style
- Learned reward models for tasks beyond verifiable reasoning domains
- Integration after the reasoning-oriented RL stage to develop comprehensive instruction-following capabilities

---

---

## 4. Multi-Stage Training Pipeline

The DeepSeek-R1 training strategy follows a systematic multi-stage approach:

| Stage | Description |
|-------|-------------|
| **Stage 1: Cold-Start SFT** | A small curated dataset of chain-of-thought reasoning examples bootstraps the model, stabilizing the initial policy before intensive RL. |
| **Stage 2: Reasoning-Oriented RL** | GRPO applied to verifiable reasoning tasks (math, code, logic) drives emergent reasoning capability—either via RL only (DeepSeek-R1-Zero) or RL after SFT (DeepSeek-R1). |
| **Stage 3: Rejection Sampling → SFT Dataset** | RL-generated outputs are filtered by quality and readability to create a high-quality SFT dataset, addressing issues like language mixing or readability observed in R1-Zero. |
| **Stage 4: Second RL Stage (General Instruction-Following)** | Expands prompt coverage to include broad instructions and incorporates general reward signals (helpfulness, style, harmlessness) to generalize beyond reasoning tasks. |
| **Stage 5: Distillation to Smaller Models** | Uses the high-capability RL-trained model as a teacher to generate reasoning-rich data, then fine-tunes smaller student models via SFT on that data (rather than performing full RL on smaller models). |

### 4.1 Pipeline Highlights

- The reasoning-only RL variant (R1-Zero) demonstrates that emergent reasoning can arise via RL alone (without SFT) but suffers from readability and language consistency issues
- For R1 proper, the cold-start SFT "kick-starts" the policy, improving readability and general language handling before RL
- Distilled models are available in multiple sizes: 1.5B, 7B, 8B, 14B, 32B, and 70B parameters, based on Qwen2.5 and Llama3 series
- According to public sources, R1 achieved reasoning performance comparable to OpenAI's o1-1217 model on reasoning and multitask benchmarks

---

---


## 5. Distinctive Features Compared to Traditional Methods

| Feature | Conventional RLHF / SFT + RL | DeepSeek-R1 Strategy |
|---------|------------------------------|----------------------|
| **Initial SFT** | Often uses large human-annotated datasets | R1-Zero: none; R1: small cold-start SFT |
| **Reward Source** | Learned reward model (often from human preferences) | Reasoning tasks: rule-based correctness + ranking; General tasks: mixture |
| **Policy Optimization** | PPO (with value network/critic, learned rewards) | GRPO (group ranking + clipped ratio + KL penalty) |
| **Domain Focus** | Broad instruction-following from the start | Emphasis on reasoning first → then general instructions |
| **Post-RL Dataset Generation** | Sometimes limited | RL outputs → filtered → SFT dataset → distillation |
| **Distillation to Smaller Models** | Optional / less emphasized | Explicit large → dataset → smaller models path |
| **Emergence of Reasoning** | Often via SFT + RL; may require large annotated data | Demonstrated via RL alone (R1-Zero), then refined by SFT + RL |

---

---

## 6. Technical & Training Details

### 6.1 Model Sizes and Releases

- **Base models**: R1-Zero and R1 are built on a 37B-activated parameter MoE architecture with a total of 671B parameters
- **Distilled variants**: Available in 1.5B, 7B, 8B, 14B, 32B, and 70B parameter configurations based on Qwen2.5 and Llama3 series

---

### 6.2 Reward Function & Sampling Details

- **Reasoning tasks**: The reward function is largely rule-based, checking final answers for correctness, format tags for reasoning sections, and penalizing language mixing
- **Process vs. outcome rewards**: Outcome rewards (correct answer) proved more effective than weaker signals like process rewards (number of reasoning steps)
- **Distillation sampling**: For R1-Distill models, generation settings included temperature 0.6, top-p 0.95, with 64 responses per query used for pass@1 estimation

---

### 6.3 Training and Distillation Strategy

The distillation process leverages the high-capability teacher model to generate large "reasoning-rich" datasets. Student models are then fine-tuned via SFT (not full RL) to inherit reasoning patterns. While smaller models may underperform the teacher, they offer substantial cost and efficiency improvements.

---

### 6.4 Observed Strengths & Weaknesses

**Strengths:**
- Emergent reasoning capability via RL
- High performance on reasoning benchmarks
- Efficient multi-stage training approach

**Weaknesses:**
- R1-Zero exhibits readability and language mixing issues due to skipping SFT
- Distilled models experience some performance degradation compared to the full model
- General instruction-following may lag in smaller or early-stage variants

---

---

## 7. Summary Table

| Component | Role | Example / Notes |
|-----------|------|-----------------|
| **Policy Model (LLM)** | Learns improved policy via RL | DeepSeek-R1, DeepSeek-R1-Zero |
| **Reference Model** | Provides KL regularization baseline | Frozen SFT model (in GRPO) |
| **Reward Function** | Scores responses | Correctness, readability, chain-of-thought format |
| **Group Size (G)** | Sampling granularity for GRPO | 8–16 outputs per prompt (typical) |
| **Advantage ($A_i$)** | Relative performance metric within group | Normalization: $(r_i - \text{mean}) / \text{std}$ |
| **Objective (GRPO)** | PPO-style surrogate + KL penalty | See GRPO doc for full derivation |
| **Training Pipeline** | Multi-stage (Cold-SFT → RL → SFT → RL → Distill) | Reasoning first, then broad instruction |
| **Distillation** | Transfer reasoning to smaller models | Student models 1.5B–70B params |
| **Goal** | Efficient reasoning/instruction fine-tuning | Stable RL fine-tuning for large LLMs |

---

---

## 8. Advantages & Limitations

### Advantages

- **Emergent reasoning**: R1-Zero demonstrates reasoning capability via RL without relying solely on large human-annotated SFT datasets
- **Efficient training**: Multi-stage strategy combining SFT, RL, filtering, and distillation
- **Verifiable rewards**: Correctness and format-based signals reduce noise and training instability
- **Scalable deployment**: Distillation enables smaller, deployable models with reasoning capability for cost-effective production use

---

### Limitations

- **Limited transparency**: Full details on datasets, hyperparameters, and training costs are not publicly available
- **Instruction-following gaps**: General instruction-following beyond reasoning may lag, especially in R1-Zero and smaller distilled variants
- **Distillation trade-offs**: SFT-only post-distillation may not fully retain RL-derived benefits in smaller models
- **Filtering dependency**: Effective reward design and RL output filtering remain critical; low-quality RL outputs create bottlenecks

---
