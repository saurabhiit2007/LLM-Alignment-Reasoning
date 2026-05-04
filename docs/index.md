# LLM Alignment & Reasoning

Interview preparation and technical reference for LLM alignment, RLHF, reasoning techniques, and evaluation. Covers the methods used to train models like GPT-4, Claude, Gemini, and DeepSeek-R1.

---

## What's Inside

### Alignment Methods

How to get LLMs to behave the way we want — the training pipelines, optimization objectives, and alternative approaches.

| Topic | What it covers |
|-------|---------------|
| [RLHF Pipeline](alignment_methods/rlhf/rlhf_pipeline.md) | Preference data collection, reward model training, RL fine-tuning |
| [PPO](alignment_methods/rlhf/rl_optimization_methods/ppo.md) | Proximal Policy Optimization — the original RLHF optimizer |
| [DPO](alignment_methods/rlhf/rl_optimization_methods/dpo.md) | Direct Preference Optimization — bypasses the reward model |
| [GRPO](alignment_methods/rlhf/rl_optimization_methods/grpo.md) | Group Relative Policy Optimization — used in DeepSeek-R1 |
| [REINFORCE](alignment_methods/rlhf/rl_optimization_methods/reinforce.md) | Foundational policy gradient — simpler than PPO, surprisingly effective |
| [RLOO](alignment_methods/rlhf/rl_optimization_methods/rloo.md) | Leave-One-Out baseline — outperforms PPO at 2–3× the speed |
| [DAPO](alignment_methods/rlhf/rl_optimization_methods/dapo.md) | Asymmetric clipping + dynamic sampling — GRPO for long-CoT at scale |
| [KL Penalty & Reward Hacking](alignment_methods/rlhf/kl_penalty_reward_hacking.md) | Why the policy drifts and how to constrain it |
| [RLAIF](alignment_methods/alternate_approaches/rlaif.md) | Replacing human feedback with AI feedback |
| [Constitutional AI](alignment_methods/alternate_approaches/constitutional_ai.md) | Anthropic's principle-based self-critique method |
| [Red Teaming](alignment_methods/safety_and_evaluation/red_teaming.md) | Adversarial probing for safety failures and jailbreaks |

### Reasoning Techniques

How to elicit better reasoning from LLMs — at prompting time and at inference time.

| Topic | What it covers |
|-------|---------------|
| [Chain-of-Thought](reasoning_techniques/prompting_based_techniques/cot.md) | Step-by-step reasoning via prompting |
| [Tree of Thoughts](reasoning_techniques/prompting_based_techniques/tree_of_thoughts.md) | Search over reasoning paths |
| [Self-Consistency](reasoning_techniques/prompting_based_techniques/self_consistency.md) | Majority voting over multiple CoT samples |
| [ReAct](reasoning_techniques/prompting_based_techniques/react.md) | Interleaving reasoning and tool use |
| [Self-Critic Methods](reasoning_techniques/iterative_refinement/self_critic_methods.md) | Models revising their own outputs |
| [STAR](reasoning_techniques/advanced_reasoning_models/STAR_self_taught_reasoner.md) | Bootstrapping reasoning from rationales |
| [Compute-Optimal Inference](reasoning_techniques/test_time_compute/compute_optimal_inference.md) | Scaling test-time compute for better answers |
| [ORMs & PRMs](reasoning_techniques/test_time_compute/orm_prm.md) | Outcome vs. process reward models as verifiers |

### Evaluation & Metrics

How to measure whether alignment and reasoning actually work.

| Topic | What it covers |
|-------|---------------|
| [Alignment Evaluation](evaluation_and_metrics/alignment_evaluation.md) | HHH framework, LLM-as-judge, TruthfulQA, IFEval, RewardBench |
| [Verification Metrics](evaluation_and_metrics/verification_metrics.md) | pass@k, maj@k, RLVR, functional correctness, benchmark saturation |

### Case Studies

| Topic | What it covers |
|-------|---------------|
| [DeepSeek RL Fine-tuning](case_studies/deepseek_rl_finetuning.md) | How DeepSeek-R1 used GRPO + RLVR to develop reasoning |

---

## Key Concepts at a Glance

**RLHF pipeline:** SFT → reward model training (on human preferences) → RL optimization (PPO/DPO/GRPO) with KL penalty to prevent reward hacking.

**DPO vs PPO:** DPO eliminates the explicit reward model by reparameterizing the RL objective directly onto the policy. Simpler and more stable, but less flexible.

**GRPO vs PPO:** GRPO removes the value network by using group-relative baselines — key to DeepSeek-R1's scalable training without a critic model.

**RLVR:** Reinforcement Learning with Verifiable Rewards replaces the learned reward model with a deterministic verifier (math checker, test suite). No reward hacking possible.

**LLM-as-judge:** Automated evaluation using a strong model (GPT-4) as the judge. Scales human preference evaluation but introduces position, verbosity, and self-enhancement biases.

**pass@k:** Probability that at least one of k sampled completions passes all test cases — the standard metric for code and math evaluation.
