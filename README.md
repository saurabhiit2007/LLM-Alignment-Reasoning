# LLM Alignment & Reasoning

Technical reference and interview preparation for LLM alignment, RLHF, reasoning techniques, and evaluation methods. Covers the training pipelines and evaluation frameworks behind models like GPT-4, Claude, Gemini, and DeepSeek-R1.

**Live site:** [saurabhiit2007.github.io/LLM-Alignment-Reasoning](https://saurabhiit2007.github.io/LLM-Alignment-Reasoning/)

---

## Topics Covered

### Alignment Methods
- **RLHF pipeline** — preference data collection, reward model training, RL fine-tuning
- **PPO, DPO, GRPO** — RL optimization methods with trade-offs
- **KL penalty & reward hacking** — why policies drift and how to constrain them
- **RLAIF** — scaling feedback with AI rather than humans
- **Constitutional AI** — Anthropic's principle-based self-critique approach
- **Red teaming & adversarial testing** — finding jailbreaks and safety failures

### Reasoning Techniques
- **Prompting-based** — Chain-of-Thought, Tree of Thoughts, Self-Consistency, ReAct
- **Iterative refinement** — self-critic methods, debate and multi-agent reasoning
- **Advanced methods** — STAR (Self-Taught Reasoner), System 2 Attention
- **Test-time compute scaling** — compute-optimal inference, Best-of-N, ORMs & PRMs

### Evaluation & Metrics
- **Alignment evaluation** — HHH framework, LLM-as-judge (MT-Bench, AlpacaEval, Arena-Hard), TruthfulQA, IFEval, RewardBench, judge biases
- **Verification metrics** — pass@k, majority voting, RLVR, functional correctness, benchmark saturation

### Case Studies
- **DeepSeek-R1** — how GRPO + verifiable rewards produced a frontier reasoning model

---

## Running Locally

```bash
# Install dependencies
uv sync

# Serve with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Requires [uv](https://github.com/astral-sh/uv). The built site outputs to `site/` (not committed).
