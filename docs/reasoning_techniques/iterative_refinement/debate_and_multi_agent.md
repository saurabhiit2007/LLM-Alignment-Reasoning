## Overview

Multi-agent debate is a reasoning technique in which two or more LLM instances independently generate answers and then iteratively challenge each other's reasoning. A judge model (or majority vote) selects the final answer. The technique improves factuality and reduces hallucinations by forcing each model to defend its position against adversarial critique.

---

## How It Works

```
Round 1:  Agent A answers → Agent B answers
Round N:  Agent A critiques B → Agent B critiques A
Final:    Judge (or majority) selects winner
```

**Key design choices:**

- **Number of agents**: Typically 2–4; more agents improve quality but increase cost
- **Number of rounds**: 2–3 rounds typically capture most quality gain
- **Judge**: A separate model, or majority vote over the agents

---

## Why Debate Works

- Models are stronger critics than generators — identifying flaws in others' reasoning is easier than avoiding all flaws in one's own
- Adversarial pressure forces models to surface hidden assumptions
- Consensus under critique is more reliable than single-model confidence

---

## Comparison with Self-Critique

| Aspect | Self-Critique (Self-Refine) | Multi-Agent Debate |
|--------|----------------------------|--------------------|
| **Agents** | One model critiques itself | Multiple independent models |
| **Bias** | May reinforce own errors | Adversarial pressure breaks echo chambers |
| **Cost** | Moderate (2–3× calls) | High (N agents × R rounds) |
| **Best for** | Single-model refinement | Factual QA, complex reasoning |

---

## Limitations

- **Cost**: Scales as O(agents × rounds)
- **Convergence risk**: Agents can reach false consensus if they share the same systematic bias
- **Diminishing returns**: Most gain from round 1; additional rounds add less

---

## Relation to Alignment

Debate was proposed as an alignment mechanism by Irving et al. (2018): if two AI agents debate with a human judge, a weaker human can identify correct answers even for tasks beyond their own expertise, because the losing agent is incentivized to expose flaws. This connects debate to scalable oversight — a core open problem in AI safety.

See: [Self-Critic Methods](self_critic_methods.md) · [Agent Evaluation](https://saurabhiit2007.github.io/RAG_And_Context_Engineering/agents/agent_evaluation/)
