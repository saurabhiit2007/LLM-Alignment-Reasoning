# ReAct

ReAct (Reasoning + Acting) is covered in the [RAG and Agents repo](https://github.com/saurabhiit2007/RAG-and-Agents) under Agent Fundamentals — it is the foundational agentic pattern and is documented there in full.

**Quick summary:**

ReAct interleaves reasoning traces (Thought) with actions (Action) and their results (Observation) in a loop:

```
Thought: I need to find the population of Tokyo.
Action: search("Tokyo population 2024")
Observation: Tokyo population is approximately 13.96 million (city) / 37.4 million (metro).
Thought: I have the answer.
Action: finish("13.96 million in the city proper")
```

**Key properties:**

- Thought steps are not executed — they are internal scratchpad reasoning
- Actions call external tools (search, calculator, code executor, database)
- The loop continues until the agent emits a finish action
- Introduced by Yao et al. (2023); shown to outperform CoT-only and Act-only baselines on HotpotQA, Fever, AlfWorld

**In the context of alignment:** ReAct reasoning traces serve as interpretable audit logs — each decision step is visible, which supports human oversight and reward model training on reasoning quality.

**Reference:** Yao et al. (2023) — "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
