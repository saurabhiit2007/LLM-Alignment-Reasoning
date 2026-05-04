## 1. Overview

Self-critique methods enable LLMs to iteratively improve their outputs by evaluating and refining their own responses. This paradigm shifts from single-shot generation to multi-step refinement processes, improving accuracy and quality.

---

## 2. Core Concepts

### Self-Refinement Loop

1. **Generate** initial response
2. **Critique** the output (identify errors, weaknesses)
3. **Refine** based on critique
4. **Repeat** until satisfactory or max iterations

### Key Components

| Component | Role |
|-----------|------|
| **Critic Model** | Evaluates outputs against criteria (accuracy, completeness, consistency) |
| **Refinement Strategy** | How to incorporate feedback (rewrite, patch, restructure) |
| **Stopping Criteria** | When to halt iteration (quality threshold, iteration limit, convergence) |

---

## 3. Technical Approaches

### Reflexion

Agent architecture with episodic memory. Stores (trajectory, reflection, outcome) tuples and uses past reflections to improve future attempts. Effective for multi-step reasoning tasks where intermediate feedback is sparse.

### Self-Refine

The simplest form — the model critiques its own output and generates a revised version in a loop:

```
Loop:
  output = generate(input)
  feedback = critique(output, input)
  if feedback.is_satisfied(): break
  input = input + output + feedback
```

### CRITIC (2023)

Grounds critique in external evidence rather than parametric knowledge:

- Extracts verifiable claims from the model's output
- Searches the web or executes code to verify each claim
- Refines the answer based on discovered contradictions

### V-STaR (2024)

- Trains a separate verification model on correct/incorrect reasoning samples
- Uses the verifier to filter training data for iterative self-improvement

---

## 4. Advantages and Limitations

**Advantages**

- Catches errors and inconsistencies not visible in single-pass generation
- Works across tasks without task-specific fine-tuning
- Critique provides interpretability into why the output changed

**Limitations**

- Multiple LLM calls per query increases cost and latency
- Echo chambers: the model may reinforce its own blind spots
- Diminishing returns after 2–3 iterations
- Overconfident models may approve incorrect outputs without revision

---

## 5. Best Practices

1. Provide explicit evaluation criteria — vague critique requests produce vague critique
2. Ground critique in external tools (code execution, web search) when possible
3. Limit to 2–3 iterations; gains plateau quickly
4. Use lower temperature for the critique step, higher for generation

---

See also: [Tree of Thoughts](../prompting_based_techniques/tree_of_thoughts.md) · [STaR](../advanced_reasoning_models/STAR_self_taught_reasoner.md) · [Constitutional AI](../../alignment_methods/alternate_approaches/constitutional_ai.md)
