## Overview

Verification metrics measure whether a model's output can be checked for correctness without human judgment. They are the backbone of scalable evaluation for coding, mathematics, and formal reasoning — and increasingly, the reward signal in RL training pipelines.

---

## 1. pass@k

**Origin:** Chen et al. (2021), introduced with the HumanEval benchmark for code generation.

**Definition:** The probability that *at least one* of k independently sampled completions passes all test cases for a given problem.

**Unbiased estimator** (from the original paper):

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

Where:

- $n$ = total samples generated per problem (typically $n \geq k$)
- $c$ = number of samples that pass all tests
- $k$ = number of samples shown to the user

**Why not just compute fraction correct?** A naïve fraction is biased when $c/n$ is used as a point estimate. The formula above gives an unbiased estimate by averaging over all possible subsets of size $k$.

### pass@1 vs pass@k

| Metric | What it measures | When to use |
|--------|-----------------|-------------|
| **pass@1** | Single-shot correctness | Production: user gets one answer |
| **pass@k (k>1)** | Coverage / diversity of solutions | Evaluation or Best-of-N settings |

**Typical values on HumanEval:**

- Early GPT-3: pass@1 ~28%
- GPT-4: pass@1 ~85%+
- Frontier models (2025): pass@1 90–95%+

---

## 2. Majority Voting (maj@k / Self-Consistency)

Sample $k$ independent completions; take the most common final answer as the prediction.

**Formal definition:**

$$\text{maj@}k = \mathbf{1}\!\left[\text{mode}\{y_1, \ldots, y_k\} = y^*\right]$$

**Key insight (Wang et al., 2023):** For reasoning tasks with a fixed final answer (math problems, multiple-choice), majority voting over chain-of-thought samples is significantly more accurate than greedy decoding alone.

**When it works:**

- Tasks with discrete, verifiable final answers (GSM8K, MATH, AIME)
- When errors in individual samples are uncorrelated

**When it fails:**

- Open-ended generation (no natural "mode")
- When the model is systematically wrong (correlated errors dominate)

---

## 3. Best-of-N (BoN) with a Verifier

Generate $N$ candidates; a verifier selects the best one. The metric is the accuracy of the selected candidate.

**Variants:**

| Verifier type | Selection criterion | Use case |
|--------------|--------------------|---------| 
| **Oracle** | Ground truth answer | Upper-bound measurement |
| **ORM** (Outcome Reward Model) | Reward assigned to complete solution | Math, code |
| **PRM** (Process Reward Model) | Aggregated step-level scores | Multi-step reasoning |
| **LLM judge** | Preference score | Open-ended tasks |
| **Test suite** | pass@1 on held-out tests | Code generation |

**Scaling behavior:** Best-of-N accuracy follows a log-linear relationship with $N$ — doubling $N$ gives a consistent additive improvement in accuracy. This is the empirical basis for test-time compute scaling.

See [ORMs & PRMs](../reasoning_techniques/test_time_compute/orm_prm.md) for detailed coverage of reward-model verifiers.

---

## 4. Functional Correctness

Code evaluation uses *functional correctness*: does the generated code produce the right output on a test suite?

### HumanEval

- 164 hand-written Python problems with docstrings and unit tests
- Metric: pass@k (k = 1, 10, 100)
- **Limitation:** Small dataset; now saturated by frontier models

### SWE-bench Verified

- Real GitHub issues requiring understanding an existing codebase, identifying the root cause, and producing a patch that passes the repo's test suite
- Harder than HumanEval by an order of magnitude
- **Status (2025):** Claude Opus 4.6 leads at ~80.8% resolve rate; considered the current gold standard for coding ability

### LiveCodeBench

- Continuously updated with new competitive programming problems post-training cutoff
- Addresses contamination by using only problems published after model training
- More reliable for frontier model comparisons than static HumanEval

---

## 5. Reinforcement Learning with Verifiable Rewards (RLVR)

**Core idea:** Use a deterministic verifier as the reward signal in RL training, eliminating the need for a trained reward model.

**Training loop:**

1. Model generates a chain-of-thought and final answer
2. Verifier checks the answer against ground truth (or test cases)
3. Binary or graded reward is returned
4. Policy is updated (typically via GRPO or PPO)

**Verifiable domains:**

| Domain | Verification method |
|--------|---------------------|
| Mathematics | Symbolic answer matching (e.g., `sympy.simplify(a - b) == 0`) |
| Code | Test suite execution |
| Formal proofs | Proof assistant (Lean 4, Coq) |
| Multiple-choice | Exact string match |

**Why RLVR outperforms learned reward models for reasoning:**

- No reward hacking: the verifier is ground truth, not a learned approximation
- No distribution shift: verification is always accurate regardless of what the policy generates
- Cheap to scale: verification is computationally trivial compared to reward model inference

**Used in:** DeepSeek-R1, Qwen QwQ, recent o-series reasoning models.

---

## 6. Process-Level Verification

Standard pass@k and functional correctness only check *answers*. Process-level metrics check whether the *reasoning steps* are correct.

### CoT-Pass@k

Extends pass@k by requiring both correct intermediate steps and a correct final answer. A model that reaches the right answer through invalid reasoning is marked incorrect.

**Why it matters:** Models can "shortcut" to correct answers via pattern matching without genuine reasoning. CoT-Pass@k penalizes this.

### Step-Level Reward (PRM Signal)

A PRM assigns a score $r_t \in [0,1]$ to each reasoning step $t$. The aggregate score is:

$$R = \prod_{t=1}^{T} r_t \quad \text{(product aggregation)}$$

or

$$R = \min_{t} r_t \quad \text{(min aggregation)}$$

Min aggregation penalizes any single incorrect step; product aggregation penalizes chains with many weak steps. Both have been used in practice; min is more conservative.

---

## 7. Math Benchmarks as Verification Targets

| Benchmark | Difficulty | Notes |
|-----------|------------|-------|
| GSM8K | Elementary/middle school | Saturated (~99% for frontier models) |
| MATH-500 | Competition math (AMC/AIME) | Symbolic + multi-step; still discriminative |
| AIME 2025/2026 | High-school olympiad | Current frontier; top models 85–94% |
| GPQA Diamond | PhD-level (bio/phys/chem) | Google-proof; human experts ~65% |
| HLE (Humanity's Last Exam) | Expert-level, cross-domain | Designed to be unsolvable by current LLMs |

**Saturation trajectory:** Each generation of frontier models forces the community to a harder benchmark. GPQA Diamond and AIME are the current frontier; HLE is positioned as the next.

---

## 8. Failure Modes and Limitations

### Metric Gaming

- Models fine-tuned on RLVR can learn to produce answers matching the format the verifier expects without correct reasoning (e.g., guessing answer format for math)
- Mitigation: require full solution trace, use PRM to verify steps

### Test Suite Incompleteness

- A code solution can pass all provided test cases but fail on edge cases
- Mitigation: generate diverse tests (property-based testing, fuzzing)

### Contamination

- If AIME 2024 problems appear in training data, pass@k on AIME 2024 is inflated
- Mitigation: use post-cutoff benchmarks (LiveBench, LiveCodeBench); track contamination with n-gram overlap detection
