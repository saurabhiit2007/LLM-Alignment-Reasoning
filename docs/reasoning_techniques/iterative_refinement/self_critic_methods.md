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

### 2. Constitutional AI (CAI)
- Model critiques own outputs against principles
- Revises responses to align with constitutional rules
- Reduces harmfulness without human feedback per iteration

---

### 3. Reflexion
- Agent architecture with episodic memory
- Stores (trajectory, reflection, outcome) tuples
- Uses past reflections to improve future attempts
- Effective for multi-step reasoning tasks

---

### 4. Self-Refine
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

### Multi-Agent Debate
- Multiple model instances debate answers
- Consensus or judge model selects final output
- Improves factuality and reasoning

### Tree of Thoughts (ToT)
- Explores multiple reasoning paths simultaneously
- Self-evaluates intermediate steps
- Backtracks and explores alternatives
- Better for complex problems than chain-of-thought

### CRITIC (2023)
- Uses external tools for validation
- Searches web, executes code to verify claims
- Grounds critique in external evidence

### V-STaR (2024)
- Verifier-guided self-training
- Trains verification model on correct/incorrect samples
- Uses verifier to filter training data for refinement

### Self-Taught Reasoner (STaR)
- Generates reasoning chains
- Filters by correctness
- Retrains on successful chains
- Bootstraps reasoning ability

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

## Common Interview Questions & Answers

### Q1: Explain the difference between self-critique and RLHF (Reinforcement Learning from Human Feedback).

**Answer**: 
- **Timing**: Self-critique happens at inference time; RLHF occurs during training
- **Feedback Source**: Self-critique uses the model's own evaluation; RLHF uses human preferences
- **Flexibility**: Self-critique can adapt to new tasks without retraining; RLHF requires retraining for new objectives
- **Cost**: Self-critique has per-query computational cost; RLHF has upfront training cost
- **Use Case**: Self-critique for dynamic improvement; RLHF for aligning model behavior with human values

---

### Q2: What are the main failure modes of self-critique methods?

**Answer**:
1. **Blind Spots**: Model can't identify errors it doesn't understand
2. **Overconfidence**: May approve incorrect outputs it believes are correct
3. **Hallucination Reinforcement**: Critique may introduce new errors while fixing others
4. **Lack of Ground Truth**: Without external verification, stays within model's knowledge limitations
5. **Iteration Loops**: May oscillate between different wrong answers
6. **Computational Cost**: Multiple LLM calls can be expensive and slow

---
### Q3: How would you implement self-critique for a code generation task?

**Answer**:
```python
def self_critique_code(problem_description):
    # Step 1: Generate initial code
    code = llm.generate(f"Write code for: {problem_description}")
    
    # Step 2: Critique with specific checks
    critique_prompt = f"""
    Review this code for:
    1. Correctness (logic errors)
    2. Edge cases (empty inputs, null values)
    3. Efficiency (time/space complexity)
    4. Best practices (naming, structure)
    
    Code: {code}
    """
    critique = llm.generate(critique_prompt)
    
    # Step 3: Execute tests if possible
    test_results = run_unit_tests(code)
    
    # Step 4: Refine based on critique + test results
    if not test_results.all_passed or "issue" in critique.lower():
        refined_code = llm.generate(f"""
        Original problem: {problem_description}
        Previous code: {code}
        Issues found: {critique}
        Test failures: {test_results.failures}
        
        Provide corrected code.
        """)
        return refined_code
    
    return code
```

**Key points to mention**:
- Use external verification (test execution) when possible
- Specific evaluation criteria improve critique quality
- Limit iterations to avoid excessive cost

---

### Q4: Compare Tree of Thoughts (ToT) with standard Chain-of-Thought (CoT). When would you use each?

**Answer**:

| Aspect | Chain-of-Thought | Tree of Thoughts |
|--------|------------------|------------------|
| **Structure** | Linear reasoning path | Branching exploration |
| **Evaluation** | No intermediate evaluation | Self-evaluates each step |
| **Backtracking** | No backtracking | Can abandon poor paths |
| **Cost** | Single path, cheaper | Multiple paths, expensive |
| **Best For** | Straightforward problems | Complex multi-step reasoning |

**Use CoT when**: Problem has clear sequential steps, computational budget is limited
**Use ToT when**: Multiple approaches possible, need to explore alternatives, problem requires strategic planning (e.g., Game of 24, creative writing)

===

### Q5: How do you prevent a model from getting stuck in a self-critique loop?

**Answer**:
1. **Max Iterations**: Hard limit (typically 2-3)
2. **Convergence Detection**: Stop when output changes minimally between iterations
3. **Confidence Threshold**: Stop when critique indicates sufficient quality
4. **Diversity Penalty**: Penalize repetitive critiques
5. **External Validation**: Use external tools/models as circuit breakers

```python
def refined_generation(prompt, max_iter=3, similarity_threshold=0.95):
    responses = [llm.generate(prompt)]
    
    for i in range(max_iter):
        critique = llm.critique(responses[-1])
        if critique.quality_score > 0.9:  # Good enough
            break
            
        new_response = llm.refine(responses[-1], critique)
        
        # Check convergence
        if similarity(new_response, responses[-1]) > similarity_threshold:
            break  # Not changing meaningfully
            
        responses.append(new_response)
    
    return responses[-1]
```

---

### Q6: What is Constitutional AI and how does it use self-critique?

**Answer**:
Constitutional AI uses self-critique to align model behavior without human feedback per iteration.

**Process**:
1. **Principles**: Define "constitution" (rules/principles)
2. **Self-Critique**: Model evaluates own output against principles
3. **Self-Revision**: Model rewrites response to better align
4. **Training**: Fine-tune on revised outputs (RLAIF - RL from AI Feedback)

**Example**:
```
Principle: "Responses should be helpful, harmless, and honest"

Initial: [potentially problematic response]
Critique: "This could be harmful because..."
Revision: [improved response aligned with principles]
```

**Advantages**: Scales better than human feedback, more consistent, can encode complex values

---

### Q7: How would you evaluate the quality of a self-critique system?

**Answer**:

**Metrics**:
1. **Accuracy Improvement**: Compare initial vs. final output quality
2. **Critique Precision**: Do identified issues actually exist?
3. **Critique Recall**: Are all issues identified?
4. **Refinement Effectiveness**: Do revisions fix identified issues?
5. **Convergence Efficiency**: Iterations needed to reach quality threshold
6. **Cost-Benefit Ratio**: Quality gain per additional LLM call

**Evaluation Framework**:
```python
def evaluate_self_critique(test_cases):
    metrics = {
        'initial_quality': [],
        'final_quality': [],
        'iterations': [],
        'critique_accuracy': []
    }
    
    for case in test_cases:
        initial = model.generate(case.prompt)
        final, history = self_refine_with_history(case.prompt)
        
        metrics['initial_quality'].append(score(initial, case.gold))
        metrics['final_quality'].append(score(final, case.gold))
        metrics['iterations'].append(len(history))
        
        # Check if critique identified real issues
        real_issues = find_issues(initial, case.gold)
        identified = parse_critique(history[0].critique)
        metrics['critique_accuracy'].append(
            precision_recall(identified, real_issues)
        )
    
    return aggregate(metrics)
```

---

### Q8: Explain the Reflexion framework and its key innovation.

**Answer**:
Reflexion is an agent architecture that learns from failures through verbal self-reflection.

**Key Components**:
1. **Actor**: Takes actions in environment
2. **Evaluator**: Scores trajectory success
3. **Self-Reflection**: Generates verbal reflection on failures
4. **Memory**: Stores (trajectory, reflection, reward) tuples

**Innovation**: Uses **episodic memory** of past failures + reflections to improve future attempts without parameter updates.

**Process**:
```
Trial 1: Attempt task → Fail → Reflect on why
Trial 2: Retrieve relevant past reflections → Attempt with insights → Succeed
```

**Example (Code Debugging)**:
- Attempt 1: Code fails tests
- Reflection: "I didn't handle empty list case"
- Attempt 2: Uses reflection to add edge case handling
- Success!

**Advantage**: Works in long-horizon tasks where intermediate feedback is sparse

---

### Q9: What are the tradeoffs between inference-time compute (self-critique) vs. training-time compute (larger models)?

**Answer**:

| Approach | Inference-Time (Self-Critique) | Training-Time (Bigger Model) |
|----------|--------------------------------|------------------------------|
| **Cost Model** | Per-query cost | One-time training cost |
| **Latency** | Higher (multiple calls) | Lower (single call) |
| **Adaptability** | Task-agnostic | Task-specific knowledge |
| **Transparency** | Interpretable reasoning | Black-box improvement |
| **Quality Ceiling** | Limited by base model | Higher raw capability |

**When to Use Self-Critique**:
- Need interpretability
- Budget constraints on training
- Task-specific optimization
- Can tolerate latency

**When to Use Larger Model**:
- Latency-critical applications
- High query volume
- Need consistent quality without variability

---

### Q10: How would you combine self-critique with external tool use?

**Answer**: This is the **CRITIC approach** - using external tools to ground critique in facts.

**Implementation**:
```python
def critic_method(question):
    # Generate initial answer
    answer = llm.generate(question)
    
    # Extract verifiable claims
    claims = llm.extract_claims(answer)
    
    # Verify each claim with tools
    for claim in claims:
        if is_factual(claim):
            # Search web for verification
            evidence = web_search(claim)
            verification = llm.verify(claim, evidence)
            
            if not verification.is_correct:
                # Refine answer based on evidence
                answer = llm.refine(
                    answer, 
                    f"Claim '{claim}' is incorrect. Evidence: {evidence}"
                )
    
    # Check code/math with execution
    if has_code(answer):
        result = execute_code(extract_code(answer))
        if result.has_error:
            answer = llm.debug(answer, result.error)
    
    return answer
```

**Tools Used**:
- Web search: Verify facts
- Code execution: Test correctness
- Calculators: Check math
- Databases: Lookup data

**Advantage**: Breaks out of model's knowledge limitations and hallucination tendencies

---