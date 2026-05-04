## 1. Overview

Chain-of-Thought (CoT) prompting elicits step-by-step reasoning from LLMs by including explicit reasoning steps in the prompt — either as worked examples (few-shot) or via a simple instruction (zero-shot). Instead of predicting the answer directly, the model generates a reasoning chain that conditions each step on the previous one.

Two papers introduced CoT in 2022:

- **Few-Shot CoT** — Wei et al. (NeurIPS 2022): include example problems with reasoning chains in the prompt
- **Zero-Shot CoT** — Kojima et al. (NeurIPS 2022): append *"Let's think step by step"* to any prompt; no examples needed

---

## 2. Why CoT Works

Standard prompting asks the model to map input → answer in one forward pass. For multi-step problems, this collapses all intermediate reasoning into latent space, giving the model no "scratch space" to work through sub-problems.

CoT addresses this by making intermediate steps explicit tokens:

- Each reasoning step conditions the next, allowing the model to decompose the problem progressively
- Errors in earlier steps are visible in the chain and can be caught (by the model or by the reader)
- The reasoning trace shifts difficult computation from a single prediction to a sequence of simpler predictions

**Critical finding — emergence at scale:** CoT is an *emergent capability*. Wei et al. found essentially no benefit for models below ~100B parameters; smaller models may even perform worse with CoT because they generate plausible-looking but incorrect reasoning chains. The technique became reliable only with large models like PaLM 540B and GPT-4.

---

## 3. Few-Shot vs Zero-Shot CoT

### Few-Shot CoT

Include 4–8 worked examples with full reasoning chains before the target question:

```
Q: Roger has 5 tennis balls. He buys 2 cans, each with 3 balls.
   How many does he have now?
A: Roger started with 5. 2 cans × 3 = 6 more. 5 + 6 = 11. Answer: 11.

Q: If John has 15 apples and gives away 40%, how many remain?
A: 40% of 15 = 6. 15 - 6 = 9. Answer: 9.

Q: [new problem]
A:
```

**Strengths:** Higher accuracy; the examples anchor the format and reasoning style.

**Weaknesses:** Requires manually crafted demonstrations; more tokens per call.

### Zero-Shot CoT

Append a trigger phrase — no examples needed:

```
Q: If John has 15 apples and gives away 40%, how many remain?
   Let's think step by step.
```

Kojima et al. showed that "Let's think step by step" alone is surprisingly effective — the phrase activates latent reasoning capabilities without any worked examples. Other triggers like "Let's work through this carefully" also work but "step by step" was the most robust across tasks.

---

## 4. Key Results

| Benchmark | Task type | Standard prompting | CoT prompting | Model |
|-----------|-----------|-------------------|--------------|-------|
| **GSM8K** | Math word problems | 17.9% | 56.9% | PaLM 540B |
| **SVAMP** | Math (robustness) | ~69% | ~79% | PaLM 540B |
| **StrategyQA** | Commonsense | ~75% | ~80% | PaLM 540B |
| **GSM8K** | Math word problems | — | ~92% | GPT-4 |

The GSM8K result (17.9% → 56.9%) is the headline finding: a 3× improvement on grade-school math from a prompt change alone.

---

## 5. Extensions

### Least-to-Most Prompting (Zhou et al., 2022)

Decomposes a hard problem into a sequence of easier sub-problems, solves each in order, and uses the answers as context for the next. Particularly effective for compositional generalisation tasks where the difficulty scales with problem length.

### Auto-CoT (Zhang et al., 2022)

Automatically generates reasoning demonstrations by clustering the training questions and sampling a representative from each cluster, then generating a chain via zero-shot CoT. Eliminates manual demonstration writing while preserving diversity.

### Program-Aided Language Models / PAL (Gao et al., 2022)

The model generates Python code as its reasoning chain instead of natural language steps. An interpreter executes the code to produce the final answer — offloading precise arithmetic and symbolic manipulation to a reliable executor rather than the model's token prediction.

### Multimodal CoT (Zhang et al., 2023)

Extends CoT to vision-language models. The model generates a rationale that incorporates both image features and text before producing the answer, improving performance on science QA tasks with diagrams.

---

## 6. Limitations

- **Emergent — model size dependent:** unreliable below ~100B parameters
- **Not self-verifying:** a fluent reasoning chain can lead to a wrong answer; the chain looks correct but contains a subtle error
- **Faithfulness:** the generated chain may not reflect the model's actual computation — it is a post-hoc rationalisation, not a ground-truth trace of internal reasoning
- **Cost:** 2–10× more output tokens than direct prompting

---

## 7. Best Practices

1. Use diverse, representative examples for few-shot CoT — avoid examples that all follow the same template
2. Lower temperature (0.3–0.7) gives more consistent reasoning chains
3. Combine with [Self-Consistency](self_consistency.md) (majority vote over multiple chains) for +10–20% on math tasks
4. For precise arithmetic, consider PAL to offload calculation to a code interpreter

---

*Sources: Wei et al. (2022) [[arXiv:2201.11903]](https://arxiv.org/abs/2201.11903) · Kojima et al. (2022) [[arXiv:2205.11916]](https://arxiv.org/abs/2205.11916)*
