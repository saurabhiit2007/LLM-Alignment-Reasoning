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

### 3.1 Self-Consistency with Self-Verification

Self-consistency (Wang et al., 2022) samples N independent reasoning paths for the same prompt and takes a majority vote over the final answers. The key insight is that correct reasoning chains are more likely to converge on the same answer, while errors tend to be diverse.

Self-*verification* extends this: a second model pass scores each candidate answer for internal consistency and factual plausibility, then selects the highest-scoring answer rather than relying on raw vote count. This helps when the majority answer is confidently wrong.

```
For prompt x, sample N reasoning paths {r_1, ..., r_N}
Each r_i produces a final answer a_i
Return argmax_a  Σ  [a_i == a]   (majority vote)
                i
```

**When to use:** Tasks with discrete, verifiable answers (math, QA, classification). Degrades on open-ended generation where answers are rarely identical.

---

### 3.2 Self-Refine (Madaan et al., 2023)

The model acts as both generator and critic in a single-model loop. No fine-tuning required — the critique and refinement steps are driven purely by prompting.

```
output₀ = generate(prompt)

for each iteration t:
    feedback_t  = critique(prompt, output_t)   # same model, different prompt
    if feedback_t signals "no issues": stop
    output_{t+1} = refine(prompt, output_t, feedback_t)
```

**Key property:** The prompt passed to the refinement step is the *concatenation* of the original prompt, the previous output, and the critique. The model never sees a clean slate — it must incorporate the feedback rather than regenerate from scratch.

**Typical gains:** 5–20% improvement on coding, math, and writing tasks; gains plateau after 2–3 iterations.

---

### 3.3 Reflexion (Shinn et al., 2023)

Reflexion targets agentic tasks where the model takes actions in an environment (web navigation, code execution, game playing) and receives sparse end-of-episode feedback.

**Architecture — three components:**

| Component | Role |
|-----------|------|
| **Actor** | Executes actions in the environment using CoT |
| **Evaluator** | Scores the completed trajectory (task success / partial credit) |
| **Self-Reflection** | Generates a verbal summary of *why* the trajectory succeeded or failed |

**Episodic memory:** Each (trajectory, reflection, outcome) tuple is stored in a memory buffer. On the next attempt, relevant past reflections are prepended to the actor's context — so the model reads its own post-mortems before trying again.

**Why it works differently from Self-Refine:** Reflexion operates across *attempts* of a task (trial 1, trial 2, ...) rather than refining a single response in one session. It accumulates multi-attempt learning without parameter updates.

**Example:**
```
Attempt 1: Agent navigates website → fails → Evaluator: 0/1
Reflection: "I searched for the item by category but missed the filter
             for in-stock items. Next time apply the stock filter first."

Attempt 2: Agent reads reflection → applies stock filter early → succeeds
```

---

### 3.4 CRITIC (2023)

CRITIC breaks the echo-chamber problem by grounding critique in external evidence rather than the model's own parametric knowledge.

**Process:**

1. Generate an initial answer
2. Extract verifiable claims from the answer
3. For each claim: query a search engine, run a calculator, or execute code
4. If external evidence contradicts a claim, revise the answer

**Why this matters:** Self-Refine and Reflexion are limited by what the model already knows — they cannot detect hallucinations the model is confident about. CRITIC can.

---

### 3.5 V-STaR (2024)

Adds a trained *verifier* model to the self-improvement loop:

1. Generate multiple candidate reasoning chains for each problem
2. Label chains as correct/incorrect against ground truth
3. Train a verifier on these labeled chains
4. Use the verifier to filter high-quality chains → fine-tune the generator on filtered data
5. Repeat: the improved generator produces better chains for the next verifier training round

**Relationship to test-time compute:** At inference, V-STaR generates K candidates and uses the verifier to select the best one — equivalent to Best-of-N sampling with a learned verifier. See [ORMs & PRMs](../test_time_compute/orm_prm.md).

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
