## 1. Overview

System 2 Attention (S2A) is a prompting technique introduced by Weston & Sukhbaatar (Meta AI, 2023) that improves LLM reasoning by running the model twice: once to regenerate a cleaner context, and once to answer the question from that cleaned context. The name is drawn from Kahneman's dual-process theory — System 1 (fast, automatic) vs. System 2 (slow, deliberate).

**No fine-tuning required.** S2A works at inference time on any instruction-following LLM.

---

## 2. Motivation: Sycophancy and Irrelevant Context

The paper was motivated by a specific failure mode: **sycophancy**. When the input context contains an opinion or irrelevant statement, LLMs tend to incorporate it into their answer even when the question does not require it.

**Example:**

```
Context: "John thinks the answer is 42. A store has 15 items.
          Each item costs $3. How much does everything cost?"
Standard LLM → may anchor on 42 despite the math being straightforward
S2A          → strips "John thinks the answer is 42" before reasoning
```

Standard attention has no mechanism to ignore irrelevant tokens — all context competes for attention weight. S2A removes the distractor *before* the reasoning step rather than relying on the model to suppress it internally.

---

## 3. How S2A Works

```
Standard:  Answer = LLM(C, Q)

S2A:       C' = LLM_regenerate(C, Q)   ← strip irrelevant content
           Answer = LLM_reason(C', Q)   ← reason on clean context
```

**Stage 1 — Context Regeneration:**

The model is prompted to rewrite the context, keeping only what is necessary to answer the question:

> *"Given the following text: {context}. And the question: {question}. Rewrite the text so that it only contains information relevant to answering the question, and remove any opinions, distractors, or irrelevant details."*

**Stage 2 — Reasoning:**

The cleaned context C' replaces the original C. The model answers the question using only the filtered information.

The same model can be used for both stages with different prompts, or a smaller model can handle Stage 1 and a larger one Stage 2.

---

## 4. Evaluation Results

The paper evaluated on two benchmarks designed to stress-test context robustness:

| Benchmark | What it tests | S2A result |
|-----------|--------------|------------|
| **OpinionsQA** | Factual QA where the context contains a stated opinion that conflicts with the correct answer | Large accuracy gain vs. baseline; model stops deferring to stated opinions |
| **GSM-IC** | GSM8K math problems with an irrelevant sentence injected into the context | Maintains near-original accuracy; baseline degrades significantly |

Key finding: LLaMA-2-70B-chat with S2A recovered most of the accuracy lost due to irrelevant context injection, bringing performance close to the clean-context baseline.

---

## 5. Limitations

| Limitation | Detail |
|-----------|--------|
| **Latency** | Two LLM forward passes instead of one |
| **Information loss** | The regeneration step may accidentally drop relevant nuance |
| **Regeneration quality cap** | If Stage 1 removes the wrong content, Stage 2 cannot recover |
| **Scope** | Most effective when the distractor is clearly separable (opinion, off-topic sentence); less effective when relevant and irrelevant content are interleaved |

---

## 6. Relation to Other Techniques

- **RAG**: S2A can act as a post-retrieval filter — regenerate the retrieved chunks to strip noise before the reader LLM sees them
- **Self-Refine / Self-Critique**: Both use multi-pass LLM calls, but S2A's two passes target context quality, not answer quality
- **Constitutional AI**: CAI critiques the *output*; S2A critiques the *input context*

---

*Source: Weston & Sukhbaatar (2023) — "System 2 Attention (is something you might need too)." Meta AI. [[arXiv:2311.11829]](https://arxiv.org/abs/2311.11829)*
