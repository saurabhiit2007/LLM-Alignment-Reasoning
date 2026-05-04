## 1. Overview

Chain of Thought is a prompting technique that encourages large language models (LLMs) to break down complex reasoning into intermediate steps, making their problem-solving process explicit and transparent. Instead of jumping directly to an answer, the model "thinks aloud" through the problem.

---

---

## 2. Core Concepts

### Basic CoT

- **Sequential Reasoning**: Breaking problems into logical steps
- **Explicit Thinking**: Making intermediate reasoning visible
- **Improved Accuracy**: Particularly effective for arithmetic, commonsense, and symbolic reasoning tasks

---

### Types of CoT

1. **Few-Shot CoT**: Providing examples with reasoning steps in the prompt
2. **Zero-Shot CoT**: Simply adding "Let's think step by step" to the prompt

See [Self-Consistency](self_consistency.md) and [Tree of Thoughts](tree_of_thoughts.md) for extensions that build on CoT.

---

---

## 3. Technical Implementation

### Zero-Shot Example
```
Prompt: "Solve: If John has 15 apples and gives away 40% to his friends, 
how many does he have left? Let's think step by step."

Response:
Step 1: Calculate 40% of 15 apples
Step 2: 15 × 0.40 = 6 apples given away
Step 3: 15 - 6 = 9 apples remaining
Answer: 9 apples
```

---

### Few-Shot Example
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans × 3 balls = 6 balls. 
5 + 6 = 11 balls. Answer: 11

Q: [New problem]
```

---

---

## 4. Recent Developments (2023-2025)

### 1. **Multimodal CoT**

- Extending CoT to vision-language models
- Incorporating visual reasoning steps
- Used in models like GPT-4V, Gemini, Claude 3+

### 2. **Automatic CoT (Auto-CoT)**

- Automatically generating diverse reasoning demonstrations
- Reduces manual prompt engineering
- Clustering questions for better coverage

### 3. **Program-Aided Language Models (PAL)**

- Combining CoT with code execution
- LLM generates reasoning + executable code
- Interpreter runs code for final answer

### 4. **Least-to-Most Prompting**

- Breaking problems into subproblems
- Solving simple cases first, building to complex
- Particularly effective for compositional generalization

---

---

## 5. Key Metrics & Performance

- **Accuracy Improvement**: 20-50% on complex reasoning tasks
- **Model Requirements**: Works better with larger models (>100B parameters)
- **Computational Cost**: 2-10x more tokens than direct answering
- **Error Analysis**: Makes reasoning failures more interpretable

---

---

## 6. Best Practices

1. **Prompt Design**: Use clear instructions like "explain your reasoning" or "work through this step by step"
2. **Example Selection**: Choose diverse, representative examples for few-shot CoT
3. **Temperature Settings**: Lower temperature (0.3-0.7) for more consistent reasoning
4. **Validation**: Verify critical answers through additional checks
5. **Error Handling**: Parse and validate reasoning steps when possible

---