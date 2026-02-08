## 1. Overview

Self-consistency is a prompting technique that improves the reliability and accuracy of large language models (LLMs) by generating multiple reasoning paths and selecting the most consistent answer through majority voting.

**Key Idea**: Instead of relying on a single greedy decode path, generate diverse reasoning paths and marginalize out the reasoning process to arrive at the most consistent final answer.

---

---

## 2. How It Works

### Basic Algorithm

1. **Generate Multiple Samples**: Use temperature-based sampling (T > 0) to generate N diverse reasoning paths for the same prompt
2. **Extract Answers**: Parse the final answer from each reasoning path
3. **Majority Vote**: Select the answer that appears most frequently across all samples

### Example

**Question**: "If there are 3 cars in the parking lot and 2 more arrive, how many cars are there?"

**Sample 1**: "Initially 3 cars. 2 more arrive. 3 + 2 = 5 cars" → **Answer: 5**

**Sample 2**: "We start with 3. Adding 2 gives us 3 + 2 = 5" → **Answer: 5**

**Sample 3**: "3 cars plus 2 cars equals 5 cars total" → **Answer: 5**

**Final Answer**: 5 (unanimous)

---

---

## 3. Technical Details

### Mathematical Formulation

Given a prompt `x`, instead of greedy decoding:
```
argmax p(r, a | x)
```

Self-consistency marginalizes over reasoning paths:
```
argmax Σ p(r, a | x)
  a    r∈R(a)
```

Where:
- `r` = reasoning path
- `a` = answer
- `R(a)` = set of reasoning paths leading to answer `a`

---

### Implementation Parameters

- **Temperature**: 0.5 - 1.0 (enables diverse sampling)
- **Number of Samples (N)**: 5-40 (typically 10-20 for good balance)
- **Sampling Method**: Temperature sampling or nucleus sampling (top-p)
- **Answer Extraction**: Regex patterns or structured output parsing

---

---

## 4. Advantages

- **Improved Accuracy**: 10-20% boost on arithmetic and commonsense reasoning tasks
- **Uncertainty Estimation**: Vote distribution indicates model confidence
- **No Fine-tuning Required**: Works with pre-trained models out-of-the-box
- **Robust to Prompt Variations**: Averages out biases from specific phrasings

---

---

## 5. Limitations

- **Computational Cost**: N times more expensive than single inference
- **Latency**: Parallel processing helps but still slower than greedy decode
- **Answer Extraction**: Requires reliable parsing of diverse outputs
- **Not Universal**: Most effective for tasks with discrete, verifiable answers

---

---

## 6. Recent Developments (2023-2025)

### Universal Self-Consistency (USC)
- Extends to open-ended generation tasks
- Uses semantic similarity instead of exact match
- Clusters similar responses and selects representative answer

### Self-Consistency with Chain-of-Thought (CoT)
- Combined with CoT prompting for complex reasoning
- Standard practice in modern LLM applications
- Implemented in frameworks like LangChain, DSPy

### Weighted Voting Schemes
- Confidence-weighted voting using log probabilities
- Quality-based weighting using separate verifier models
- Adaptive sample sizes based on initial agreement

### Integration with Tool Use
- Self-consistency over tool-augmented reasoning paths
- Multiple execution paths with external APIs/calculators
- Verification through diverse computational approaches

---

---

## 7. Common Interview Questions

### Q1: When should you use self-consistency over greedy decoding?

**Answer**: Use self-consistency when:
- Task has verifiable, discrete answers (math, QA, classification)
- Accuracy is more important than latency
- You need uncertainty estimates
- Single reasoning paths are unreliable

Avoid when:
- Open-ended creative generation is needed
- Real-time response is critical
- Budget constraints limit API calls

---

### Q2: How does self-consistency differ from ensemble methods?

**Answer**: 
- **Self-consistency**: Multiple samples from same model with different reasoning paths
- **Ensembles**: Multiple different models or model checkpoints

Self-consistency is cheaper (one model) but less diverse than true ensembles.

---

### Q3: What's the optimal number of samples?

**Answer**: Depends on task complexity and budget:
- **5-10 samples**: Simple tasks, diminishing returns after
- **10-20 samples**: Sweet spot for most applications
- **20-40 samples**: Complex reasoning, when accuracy is critical

Monitor accuracy vs. sample size curve to find the knee point.

---

### Q4: How do you handle cases where there's no clear majority?

**Answer**: Strategies include:
- **Threshold-based**: Require minimum vote percentage (e.g., 40%)
- **Tie-breaking**: Use confidence scores or select random winner
- **Abstention**: Return "uncertain" if no clear winner
- **Increase samples**: Generate more paths until convergence

---

### Q5: Can self-consistency be combined with other techniques?

**Answer**: Yes, commonly combined with:
- **Chain-of-Thought (CoT)**: Generate diverse reasoning chains
- **Few-shot prompting**: Provide examples in prompt
- **Prompt ensembling**: Different prompt formulations
- **Verification**: Use verifier model to score candidates

---

### Q6: How does temperature affect self-consistency performance?

**Answer**: 
- **Too low (< 0.3)**: Paths too similar, no diversity benefit
- **Optimal (0.5-0.8)**: Good balance of quality and diversity
- **Too high (> 1.0)**: Noisy outputs, parsing failures

Empirically, T=0.7 works well for most reasoning tasks.

---

### Q7: What are the failure modes of self-consistency?

**Answer**:
- **Systematic errors**: All paths make same mistake
- **Answer extraction errors**: Parsing fails on diverse formats
- **Adversarial inputs**: Designed to produce consistent wrong answers
- **Insufficient diversity**: Similar paths don't improve accuracy

---

---

## 8. Implementation Example

```python
def self_consistency(prompt, model, n_samples=10, temperature=0.7):
    """
    Implement self-consistency for LLM reasoning
    """
    # Generate diverse reasoning paths
    responses = []
    for _ in range(n_samples):
        response = model.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256
        )
        responses.append(response)
    
    # Extract answers
    answers = [extract_answer(r) for r in responses]
    
    # Majority vote
    from collections import Counter
    vote_counts = Counter(answers)
    most_common_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[most_common_answer] / n_samples
    
    return most_common_answer, confidence

def extract_answer(response):
    """Extract final answer from reasoning path"""
    # Pattern matching for common formats
    import re
    patterns = [
        r"(?:answer is|answer:|final answer:)\s*([^\n]+)",
        r"#### ([^\n]+)",  # Common in math problems
        r"\n\n([^\n]+)$"  # Last line
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()
```

---