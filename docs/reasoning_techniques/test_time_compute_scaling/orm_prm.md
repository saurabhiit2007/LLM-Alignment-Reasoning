## 1. Overview

**ORM (Outcome Reward Model)** and **PRM (Process Reward Model)** are two paradigms for training reward models in reinforcement learning from human feedback (RLHF), particularly for large language models.

---

---

## 2. Core Concepts

### Outcome Reward Models (ORMs)
- **Definition**: Provides a single reward signal based on the final output/answer
- **Evaluation**: Judges only whether the final solution is correct
- **Training**: Learns to predict if the end result satisfies the objective
- **Analogy**: Like grading only the final answer on a math test

---

### Process Reward Models (PRMs)
- **Definition**: Provides reward signals for each intermediate step in the reasoning process
- **Evaluation**: Judges the correctness of each reasoning step
- **Training**: Learns to identify which steps are valid/invalid
- **Analogy**: Like grading the work shown for each step in solving a problem

---

---

## 3. Key Differences

| Aspect | ORM | PRM |
|--------|-----|-----|
| **Granularity** | Coarse (end result only) | Fine (step-by-step) |
| **Feedback** | Single reward | Multiple rewards per solution |
| **Supervision** | Easier to collect | More labor-intensive |
| **Error Detection** | Poor at locating errors | Identifies where reasoning fails |
| **Search Efficiency** | Less effective for tree search | Excellent for beam/tree search |
| **Interpretability** | Black box | More transparent |

---

---

## 4. Technical Details

### ORM Implementation
```python
# Simplified ORM scoring
def orm_score(problem, solution):
    # Single scalar reward for complete solution
    is_correct = verify_final_answer(solution, ground_truth)
    return 1.0 if is_correct else 0.0
```

---

### PRM Implementation
```python
# Simplified PRM scoring
def prm_score(problem, reasoning_steps):
    step_rewards = []
    for step in reasoning_steps:
        # Score each intermediate step
        step_reward = verify_step_correctness(step, context)
        step_rewards.append(step_reward)
    return step_rewards  # Vector of rewards
```

---

### Training Data Requirements

**ORM Training**:
- Input: (problem, complete_solution)
- Label: binary (correct/incorrect) or scalar score
- Collection: Relatively straightforward

**PRM Training**:
- Input: (problem, step_1, step_2, ..., step_n)
- Label: correctness for each step
- Collection: Requires step-by-step human annotation

---

---

## 5. Recent Developments (2023-2025)

### Key Research Findings

1. **Let's Verify Step by Step (OpenAI, 2023)**
   - PRMs significantly outperform ORMs on math reasoning tasks
   - PRMs improve reliability in complex multi-step problems
   - Best-of-N sampling with PRMs shows major gains

2. **Math-Shepherd (2024)**
   - Automated PRM data collection using step-level annotations
   - Reduces human labeling costs while maintaining quality

3. **Hybrid Approaches**
   - Combining ORMs and PRMs for different task types
   - Outcome supervision for simple tasks, process supervision for complex reasoning

4. **Weak-to-Strong Generalization**
   - Using weaker models to provide process supervision for stronger models
   - Self-taught reasoner approaches

5. **Critique and Refinement**
   - PRMs used in iterative refinement loops
   - Integration with constitutional AI approaches

---

---

## 6. Use Cases

### When to Use ORMs
- Simple tasks with clear right/wrong answers
- Limited annotation budget
- Fast inference required
- End-to-end evaluation sufficient

### When to Use PRMs
- Complex multi-step reasoning (math, code, planning)
- Need for interpretability and debugging
- Tree search or beam search decoding
- Critical applications requiring verification

---

---

## 7. Common Interview Questions

### 1. **What are the main advantages of PRMs over ORMs?**
**Answer**: PRMs provide finer-grained feedback, making them better at: (a) identifying exactly where reasoning goes wrong, (b) guiding search algorithms through solution spaces, (c) improving sample efficiency during RL training, and (d) providing interpretability. However, they require more expensive step-level annotations.

---
### 2. **How would you implement best-of-N sampling with a PRM?**
**Answer**: Generate N complete solutions, score each step using the PRM, aggregate step scores (e.g., minimum, average, or product), and select the solution with the highest aggregate score. The key advantage is that PRM can detect subtle reasoning errors that might lead to accidentally correct answers.

---

### 3. **What are the data collection challenges for PRMs?**
**Answer**: PRMs require step-by-step human annotation, which is expensive and time-consuming. Challenges include: defining what constitutes a "step," ensuring annotator consistency, handling ambiguous intermediate states, and scaling to diverse domains. Recent work addresses this through automated labeling and weak supervision.

---

### 4. **Can you combine ORMs and PRMs? How?**
**Answer**: Yes, through ensemble methods (weighted combination of scores), hierarchical approaches (PRM for reasoning, ORM for final verification), or task-routing (PRM for complex tasks, ORM for simple ones). You can also use ORMs to bootstrap PRM training data.

---

### 5. **How do PRMs enable better search algorithms?**
**Answer**: PRMs provide value estimates at intermediate states, enabling tree search algorithms (like MCTS) to prune bad reasoning paths early. This makes exploration more efficient than ORMs, which only provide terminal rewards and can't distinguish good from bad partial solutions.

---

### 6. **What metrics would you use to evaluate ORM vs PRM performance?**
**Answer**: 
- **Accuracy**: Final answer correctness
- **Sample efficiency**: Performance vs. number of training samples
- **Search efficiency**: Solution quality vs. search budget
- **Calibration**: How well scores correlate with actual correctness
- **Interpretability**: Ability to identify error locations
- **Computational cost**: Inference time and memory

---

### 7. **Explain the training objective for a PRM.**
**Answer**: A PRM is typically trained with supervised learning to predict step-level correctness labels. The loss function is usually binary cross-entropy at each step: `L = -Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]` where y_i is the binary label (correct/incorrect) for step i and p_i is the model's predicted probability. Some approaches use Monte Carlo estimates of step value instead of binary labels.

---

---

## 8. Implementation Considerations

### Computational Trade-offs
- **ORMs**: Single forward pass per solution - O(1) calls
- **PRMs**: One forward pass per step - O(n) calls where n = number of steps
- **Mitigation**: Batch processing, caching, or step-level parallelization

### Integration with RLHF
- Use PRM scores as intermediate rewards in PPO/DPO training
- Allows for credit assignment to specific reasoning steps
- Enables more stable training compared to sparse ORM rewards

---