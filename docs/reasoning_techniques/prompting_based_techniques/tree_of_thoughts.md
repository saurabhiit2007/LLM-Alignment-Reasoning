## 1. Overview

Tree of Thoughts (ToT) is a prompting framework introduced by Yao et al. (NeurIPS 2023) that frames problem-solving as a **search over a tree of intermediate reasoning steps** ("thoughts"). Unlike Chain-of-Thought, which commits to a single linear path, ToT generates multiple candidate thoughts at each step, evaluates them, and uses a search algorithm (BFS or DFS) to explore the most promising branches — backtracking when a path looks unpromising.

---

## 2. Motivation: Where CoT Falls Short

CoT works well when each step has an obvious next move. It fails on problems that require:

- **Lookahead** — the correct step now depends on what it enables later
- **Exploration** — multiple plausible next moves with very different outcomes
- **Backtracking** — recognising a path is wrong and trying a different one

**Paper benchmark:** Game of 24 (use four numbers and arithmetic to reach 24). CoT with GPT-4 solved only **4%** of problems. ToT solved **74%**.

---

## 3. How ToT Works

```
Problem
  ├── Thought A  →  eval: promising  →  expand
  │     ├── Thought A1  →  eval: good  →  solution
  │     └── Thought A2  →  eval: dead end  →  backtrack
  ├── Thought B  →  eval: weak  →  prune
  └── Thought C  →  eval: promising  →  expand
        └── ...
```

### 3.1 Thought Generation

Two strategies from the paper:

| Strategy | How | When to use |
|----------|-----|-------------|
| **Sampling** | Generate k thoughts independently in parallel (temperature > 0) | When thoughts are short and diverse (e.g. one sentence) |
| **Proposing** | Generate thoughts sequentially in a single prompt, using a "propose prompt" | When thoughts are longer or need to be distinct from each other |

### 3.2 Thought Evaluation

The LLM evaluates each thought to guide the search — this replaces the hand-crafted heuristic in classical search:

- **Value**: prompt the LLM to score each thought independently (e.g. 1–10, or sure/likely/impossible)
- **Vote**: prompt the LLM to compare thoughts and pick the most promising one

### 3.3 Search Algorithm

- **BFS**: expand all thoughts at depth d before going to d+1; keeps the top-k states at each level; better for problems needing global comparison
- **DFS**: explore one branch fully before backtracking; more memory-efficient but risks getting stuck

**Key hyperparameters:** tree depth, branching factor (number of thoughts per step), and number of states kept at each BFS level.

---

## 4. Results from the Paper

| Task | CoT (best) | ToT | Notes |
|------|-----------|-----|-------|
| **Game of 24** | 4% | 74% | GPT-4; ToT used BFS with k=5 |
| **Mini Crosswords** | 16% | 60% | 5×5 crossword; DFS with backtracking |
| **Creative Writing** | ~6.9/10 | ~7.6/10 | Human coherence rating; smaller gap |

The crossword result illustrates backtracking: the model fills a word, discovers it conflicts with a crossing word, and backs up to try a different fill — something a linear CoT cannot do.

---

## 5. Cost vs. Benefit

ToT is significantly more expensive than CoT:

- Game of 24 required ~$1 of API calls per problem (GPT-4 pricing at time of paper)
- Branching factor k=5 and depth=3 means up to 5³ = 125 LLM calls per problem vs. 1 for CoT

**When ToT is worth it:** problems with discrete, verifiable intermediate states where backtracking has clear value (puzzles, planning, formal reasoning). For open-ended generation or tasks where CoT already works well, the cost is rarely justified.

---

## 6. Extensions

**Graph of Thoughts (Besta et al., 2024)** — generalises ToT from a tree to an arbitrary directed graph. Thoughts from different branches can be merged (aggregated), allowing the model to combine partial solutions. Improves performance on sorting and set operations where results from parallel branches need to be combined.

---

## 7. Comparison with Related Techniques

| Technique | Structure | Evaluation | Backtracking | Cost |
|-----------|-----------|-----------|--------------|------|
| CoT | Linear chain | None | No | 1× |
| Self-Consistency | Parallel chains | Majority vote | No | k× |
| ToT | Tree | LLM value/vote | Yes | k^d × |
| GoT | Graph | LLM value | Yes | higher |

---

*Source: Yao et al. (2023) — "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023. [[arXiv:2305.10601]](https://arxiv.org/abs/2305.10601)*
