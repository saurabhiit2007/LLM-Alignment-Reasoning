# Safety Evaluation Frameworks

## 1. Overview

Safety evaluation for LLMs assesses whether a model avoids harmful behaviors while remaining genuinely useful. No single metric captures safety — evaluation combines automated benchmarks, human evaluation, and adversarial probing. See [Red Teaming](red_teaming.md) for the adversarial testing process and [Adversarial Testing](adversarial_testing.md) for specific attack methods.

---

## 2. What Safety Evaluation Measures

Safety evaluation targets several distinct failure modes:

| Failure Mode | Definition | Example |
|---|---|---|
| **Harmful content generation** | Model produces dangerous, violent, or illegal content | Instructions for synthesizing dangerous substances |
| **Bias and discrimination** | Model treats groups inequitably | Stereotyping based on demographic attributes |
| **Deception and sycophancy** | Model states falsehoods or agrees to avoid conflict | Confirming incorrect facts when user insists |
| **Privacy violations** | Model reveals personal or sensitive information | Repeating PII from training data |
| **Refusal overreach** | Model refuses legitimate, benign requests | Refusing to explain historical atrocities in an educational context |

The last failure mode — over-refusal — is as important as under-refusal. A model that refuses everything is trivially "safe" but useless.

---

## 3. Key Benchmark Suites

### TruthfulQA

- Tests whether a model generates answers that are factually truthful rather than imitating human falsehoods
- 817 questions in categories where humans are commonly wrong (health myths, legal misconceptions, conspiracies)
- Models trained to be helpful often pick up human falsehoods from training data; TruthfulQA probes this

### BBQ (Bias Benchmark for QA)

- Tests social biases across 9 social dimensions (race, gender, disability, etc.)
- Uses ambiguous and disambiguated question pairs to detect when models default to stereotypes

### HarmBench

- Standardized benchmark for evaluating LLM robustness against harmful prompt attacks
- 400 behaviors across 7 categories; tests both direct requests and jailbreak-style attacks
- Evaluates attack success rate (ASR) across multiple attack methods

### WildGuard

- Tests safety on both harmful prompts (should refuse) and benign prompts (should comply)
- Addresses the over-refusal problem by measuring false positive refusals

### HELM Safety

- Holistic Evaluation of Language Models; includes a safety module alongside capability metrics
- Allows comparison of safety vs. capability trade-offs across model versions

---

## 4. Evaluation Dimensions

A complete safety evaluation requires measuring on multiple axes simultaneously:

**Helpfulness-harmlessness trade-off.** A model optimized purely for safety will refuse too much; a model optimized purely for helpfulness will comply too much. Balanced evaluation requires both refusal rate on harmful prompts *and* compliance rate on benign prompts.

**Robustness to attack.** Static benchmark performance understates real-world risk. A model that passes HarmBench in default settings may still be jailbroken with prompt engineering. Safety evaluation should include adversarial probing. See [Adversarial Testing](adversarial_testing.md).

**Calibration.** Does the model know when it doesn't know? Overconfident models are a safety risk in high-stakes settings.

**Context sensitivity.** Safety requirements differ by deployment context — a model serving medical professionals has different expectations than a general consumer product.

---

## 5. Human Evaluation Methods

Automated benchmarks measure specific behaviors but miss nuance. Human evaluation complements them:

**Structured rating tasks.** Raters evaluate model outputs on Likert scales for safety, helpfulness, and honesty. Used in InstructGPT, Claude, and GPT-4 evaluations.

**A/B preference studies.** Raters compare two model responses and select the safer or more helpful one. Powers reward model training in RLHF but also serves as a safety signal.

**Red team annotation.** Human red teamers attempt to elicit harmful outputs; outputs are reviewed by a separate team to assess severity. More expensive but finds failure modes automated evals miss.

---

## 6. Automated Scalable Oversight

As models become more capable, human evaluation becomes the bottleneck. Scalable oversight approaches delegate some evaluation to models:

**LLM-as-judge** uses a strong model (GPT-4, Claude) to evaluate safety of another model's outputs. Biases of the judge model propagate into safety assessments, but it scales well.

**Constitutional AI (CAI)** uses the model itself to critique and revise its outputs against a rule set. Reduces reliance on human labelers for safety annotation. See [Constitutional AI](../alternate_approaches/constitutional_ai.md).

**RLAIF** trains a reward model on AI-labeled safety preferences rather than human labels. Effective at scale but requires validating the AI judge. See [RLAIF](../alternate_approaches/rlaif.md).

---

## 7. Limitations

**Benchmark saturation.** Models trained on large corpora increasingly "know" benchmark questions. High TruthfulQA scores may reflect memorization, not genuine truthfulness.

**Distribution mismatch.** Benchmarks are constructed in advance; real-world harmful prompts evolve continuously. Red teaming and ongoing monitoring are necessary complements.

**Goodhart's Law.** Optimizing directly for benchmark scores degrades their validity as safety measures. Models can learn to pass safety benchmarks while retaining underlying unsafe behaviors.

**Single-turn bias.** Most benchmarks test single-turn interactions. Multi-turn attacks — where a model is manipulated across a long conversation — are underrepresented.

---

*Sources: Lin et al. (2022) — TruthfulQA [[arXiv:2109.07958]](https://arxiv.org/abs/2109.07958) · Parrish et al. (2022) — BBQ [[arXiv:2110.08193]](https://arxiv.org/abs/2110.08193) · Mazeika et al. (2024) — HarmBench [[arXiv:2402.04249]](https://arxiv.org/abs/2402.04249) · Liang et al. (2022) — HELM [[arXiv:2211.09110]](https://arxiv.org/abs/2211.09110)*
