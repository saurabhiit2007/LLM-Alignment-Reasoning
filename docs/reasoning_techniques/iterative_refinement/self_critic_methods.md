## 1. Overview

Self-critique methods enable LLMs to iteratively improve their outputs by evaluating and refining their own responses. This paradigm shifts from single-shot generation to multi-step refinement processes, improving accuracy and quality.

---

---

## 2. Core Concepts

### Self-Refinement Loop

1. **Generate** initial response
2. **Critique** the output (identify errors, weaknesses)
3. **Refine** based on critique
4. **Repeat** until satisfactory or max iterations

### Key Components

1. **Critic Model**: Evaluates outputs against criteria (accuracy, completeness, consistency)

2. **Refinement Strategy**: How to incorporate feedback (rewrite, patch, restructure)

3. **Stopping Criteria**: When to halt iteration (quality threshold, iteration limit, convergence)

---

---

## 3. Technical Approaches

### 1. Self-Consistency with Self-Verification
```python
# Pseudo-code pattern
responses = [model.generate(prompt) for _ in range(n)]
verified = [model.verify(r, criteria) for r in responses]
best = select_highest_confidence(responses, verified)
```
---

### 2. Reflexion

- Agent architecture with episodic memory
- Stores (trajectory, reflection, outcome) tuples
- Uses past reflections to improve future attempts
- Effective for multi-step reasoning tasks

---

### 3. Self-Refine
```
Loop:
  output = generate(input)
  feedback = critique(output, input)
  if feedback.is_satisfied(): break
  input = input + output + feedback
```

---

---

## 4. Recent Innovations (2023-2025)

### CRITIC (2023)

- Uses external tools for validation
- Searches web, executes code to verify claims
- Grounds critique in external evidence

### V-STaR (2024)

- Verifier-guided self-training
- Trains verification model on correct/incorrect samples
- Uses verifier to filter training data for refinement

See also: [Tree of Thoughts](../prompting_based_techniques/tree_of_thoughts.md) · [STaR](../advanced_reasoning_models/STAR_self_taught_reasoner.md) · [Constitutional AI](../../alignment_methods/alternate_approaches/constitutional_ai.md)

---

---

## 5. Implementation Patterns

### 5.1 Basic Self-Critique Template

```python
def self_refine(prompt, max_iterations=3):
    response = llm.generate(prompt)
    
    for i in range(max_iterations):
        critique = llm.generate(
            f"Critique this response:\n{response}\n"
            f"Identify errors and improvements."
        )
        
        if "no issues" in critique.lower():
            break
            
        response = llm.generate(
            f"Original: {prompt}\n"
            f"Previous: {response}\n"
            f"Critique: {critique}\n"
            f"Provide improved response."
        )
    
    return response
```

---

### 5.2 Rubric-Based Evaluation
```python
rubric = {
    "accuracy": "Are facts correct?",
    "completeness": "Are all aspects addressed?",
    "clarity": "Is it easy to understand?"
}

for criterion, question in rubric.items():
    score = llm.evaluate(response, question)
    if score < threshold:
        feedback = llm.critique(response, criterion)
        response = llm.refine(response, feedback)
```

---

---

## 6. Advantages

- **Improved Quality**: Catches errors and inconsistencies
- **Self-Correction**: Reduces hallucinations
- **Adaptability**: Works across tasks without task-specific training
- **Transparency**: Critique provides interpretability

---

---

## Challenges

- **Computational Cost**: Multiple LLM calls per query
- **Diminishing Returns**: May plateau after few iterations
- **Echo Chambers**: Model may reinforce own biases
- **Calibration**: Models may be overconfident in incorrect outputs

---

---

## Best Practices

1. **Specify Clear Criteria**: Provide explicit evaluation dimensions
2. **Use External Verification**: Ground critique in facts when possible
3. **Limit Iterations**: 2-3 iterations often optimal
4. **Temperature Tuning**: Lower for critique, higher for generation
5. **Prompt Engineering**: Frame critique as helpful, not adversarial

---

---

## Evaluation Metrics

- **Convergence Rate**: Iterations until stable output
- **Quality Improvement**: Δ score from initial to final
- **Critique Accuracy**: Does identified issue exist?
- **Cost Efficiency**: Quality gain vs. computational cost

---

---

## Applications

- **Code Generation**: Debug and optimize generated code
- **Math Problem Solving**: Verify steps and check answers
- **Creative Writing**: Improve style, coherence, grammar
- **Factual QA**: Verify claims against knowledge
- **Task Planning**: Validate and refine action sequences

---

---
