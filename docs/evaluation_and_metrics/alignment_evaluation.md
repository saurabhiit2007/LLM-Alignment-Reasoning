## Overview

Evaluating whether an LLM is truly *aligned* is fundamentally harder than measuring raw capability. A model that scores well on academic benchmarks may still refuse reasonable requests, hallucinate facts, or produce subtly harmful outputs. This page covers the frameworks and benchmarks used to measure alignment.

---

## 1. The HHH Framework

Anthropic's Helpful–Harmless–Honest (HHH) triplet is the most widely adopted conceptual framework for alignment evaluation. Each axis targets a distinct failure mode:

| Axis | Failure it targets | Example signals |
|------|-------------------|-----------------|
| **Helpful** | Refusals on benign requests, low utility | Task success rate, user preference |
| **Harmless** | Harmful content, bias, toxicity | ToxiGen score, BBQ accuracy |
| **Honest** | Hallucination, overconfidence | TruthfulQA accuracy, calibration ECE |

Real tensions exist between axes: a model becomes more harmless by refusing more, but less helpful. Good alignment evaluation must measure all three simultaneously.

---

## 2. Human Preference Evaluation

### 2.1 Pairwise Comparisons

The most direct signal: show two model outputs to a human, ask which is better. This maps cleanly to the Bradley-Terry reward model objective used in RLHF.

**Data collection:**

- Typically 3–5 independent annotators per pair
- Inter-annotator agreement tracked via Fleiss' κ or Krippendorff's α
- Crowdsourced (MTurk, Prolific) for scale; expert annotators for safety-critical domains

**Limitations:**

- Expensive and slow to scale
- Annotators show *verbosity bias* — preferring longer answers even when shorter ones are better
- Cultural and individual variation in what counts as "helpful"

### 2.2 LMSYS Chatbot Arena

A live crowdsourced platform where users converse with two anonymous models and vote for the better response. Votes are aggregated into Elo ratings.

**Why it matters:**

- Covers real user queries (not curated benchmarks) — high ecological validity
- Continuous: new models enter the leaderboard without re-running static tests
- Elo scores provide a total ordering across hundreds of models

**Limitations:**

- Self-selected user population (skews technical)
- Expensive interactions are underrepresented (e.g., long-context tasks)
- Vote quality varies; no annotation guidelines

---

## 3. LLM-as-Judge Evaluation

Rather than expensive human annotation, a strong model (typically GPT-4) acts as an automated judge. This scales evaluation dramatically while retaining reasonable correlation with human preferences.

### 3.1 MT-Bench

**What it is:** 80 multi-turn questions across 8 categories (coding, math, reasoning, roleplay, writing, extraction, STEM, humanities).

**How it works:** GPT-4 scores each response 1–10. A second "turn" tests multi-turn coherence.

**Findings from the original paper (Zheng et al., 2023):**

- GPT-4-as-judge achieves >80% agreement with human preferences — matching the inter-human agreement level
- Identified key failure modes: position bias, verbosity bias, self-enhancement bias

### 3.2 AlpacaEval 2.0

**What it is:** 805 open-ended instructions evaluated by GPT-4-turbo; metric is *length-controlled win rate* against a GPT-4-turbo reference.

**Length-controlled win rate** adjusts for the verbosity bias of LLM judges — preventing models from gaming the score by simply being more verbose. It is now the preferred metric over raw win rate.

### 3.3 Arena-Hard-Auto

**What it is:** 500 technically challenging prompts curated from Chatbot Arena conversations using the BenchBuilder pipeline.

**Key properties:**

- 98.6% correlation with Chatbot Arena human preference rankings
- 3× higher model separation than MT-Bench (harder to tie)
- Fully automated: no human annotation needed at inference time

---

## 4. Key Alignment Benchmarks

### 4.1 TruthfulQA

**Scope:** 817 questions across 38 categories (health, law, finance, politics) designed around common human misconceptions.

**Metric:** % of responses that are both *truthful* and *informative* (measured by human annotation or fine-tuned classifiers).

**Why it's hard:** A model that refuses to answer or says "I don't know" scores well on truthfulness but poorly on informativeness. Models must balance both.

**Status:** Used in 8/16 model release papers that report safety evaluations (2024 survey). Frontier models are approaching the human baseline (~94%), but gains are uneven across categories.

### 4.2 BBQ (Bias Benchmark for QA)

**Scope:** 58,492 questions testing social biases across 9 categories (age, disability, gender, nationality, race/ethnicity, religion, SES, sexual orientation, physical appearance).

**Metric:** Accuracy on bias-identifying questions; a model is penalized for answering based on stereotypes rather than context.

### 4.3 IFEval (Instruction Following Evaluation)

**Scope:** ~500 prompts each with 1–3 *verifiable* constraints (e.g., "respond in fewer than 200 words", "include the word 'sustainability'", "use bullet points").

**Metric:** Rule-based verification — no LLM judge needed. Two variants: prompt-level accuracy (all constraints satisfied) and instruction-level accuracy.

**Why it matters:** Directly measures whether models follow explicit user instructions, a key alignment property that subjective benchmarks miss.

---

## 5. Reward Model Evaluation

### 5.1 RewardBench

A benchmark for evaluating reward models across four categories:

| Category | Description |
|----------|-------------|
| Chat | Helpfulness for conversational tasks |
| Chat Hard | Subtle preference signals (sycophancy traps) |
| Safety | Refusal of harmful content |
| Reasoning | Preference for correct solutions |

**Key finding:** High RewardBench scores do not reliably predict downstream alignment quality. Models scoring 90+ on RewardBench can still produce poorly calibrated or reward-hacked outputs when used in RL training.

**Spurious correlations:** Reward models show *length bias* (preferring longer responses in Chat) and *style bias* (preferring the writing style of powerful models like GPT-4 regardless of content quality). Models with high miscalibration (~40%) relative to human preferences have been observed.

### 5.2 RM-Bench

Extends RewardBench by including *subtlety* (near-tie preference pairs) and *style variants* (same content, different formatting). Explicitly designed to surface reward model reliance on shallow signals.

---

## 6. Known Biases in LLM-as-Judge

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Judge prefers the response shown first (or last) | Swap positions and average; use single-response scoring |
| **Verbosity bias** | Judge prefers longer responses regardless of quality | Length-controlled win rate (AlpacaEval 2.0) |
| **Self-enhancement bias** | A model judging itself rates its own outputs higher | Never use same model as both policy and judge |
| **Leniency bias** | Judges tend toward high scores, compressing discrimination | Calibrate with human reference scores |
| **Sycophancy** | Judge agrees with opinions stated in the response | Use outputs without stated opinions in test prompts |

---

## 7. Newer Evaluation Directions (2024–2025)

### Contamination-Limited Benchmarks

Static benchmarks suffer from *data contamination* — test questions leak into training sets, inflating reported performance. Solutions:

- **LiveBench:** Updates questions monthly using recent news/events; automatically verifiable answers; top models score below 70%.
- **AntiLeakBench:** Constructs questions referencing knowledge that appeared *after* a model's training cutoff.
- **Dynamic rephrasing:** Paraphrase benchmark questions at inference time to detect memorization.

### Aggregated Evaluation

**BenchHub (2025):** Aggregates 303,000 questions across 38 benchmarks into a unified evaluation ecosystem, enabling multi-benchmark testing with a single run. Reduces cherry-picking of favorable benchmarks in model releases.

### Evaluation of Reasoning Quality (Not Just Answers)

**CoT-Pass@k:** Evaluates correctness of reasoning steps, not only final answers. A model that reaches the right answer via flawed reasoning is penalized. Particularly important for math and science domains.

---

## 8. Benchmark Saturation Problem

Several widely-cited benchmarks have been saturated by frontier models:

| Benchmark | Saturation indicator |
|-----------|---------------------|
| MMLU | GPT-4-class models exceed 90%; human expert ~89% |
| GSM8K | Multiple models score 99%+ |
| HumanEval | Leading models exceed 90% pass@1 |

**Consequence:** These benchmarks can no longer distinguish among frontier models. The community has shifted to harder benchmarks: GPQA Diamond, MATH-500, AIME 2025/2026, SWE-bench Verified, LiveCodeBench Pro.

See [Reasoning & Evaluation Benchmarks](../references.md) for full benchmark listings with current SOTA scores.
