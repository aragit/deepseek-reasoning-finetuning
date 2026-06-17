# DeepSeek-R1 Medical Chain-of-Thought (CoT) Fine-Tuning Engine

[![Framework](https://img.shields.io/badge/Framework-Unsloth-orange.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-DeepSeek--R1--Distill--Qwen--7B-blue.svg)](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B)
[![Dataset](https://img.shields.io/badge/Dataset-medical--o1--reasoning--sft-green.svg)](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
[![Read Article](https://img.shields.io/badge/Medium-Deep%20Dive-black.svg)](https://medium.com/@anicomanesh/fine-tuning-deepseek-r1-reasoning-on-the-medical-chain-of-thought-dataset-922407121cc2)

An industrial, production-grade alignment pipeline engineered to optimize **DeepSeek-R1-Distill-Qwen-7B** for advanced clinical diagnostics and reasoning. By steering away from monolithic notebook structures, this repository isolates dataset mapping, hyperparameter configurations, and tracking engines into a clean, modular layout built for enterprise scaling.

For a comprehensive exploration of the theoretical foundations behind this system—including the shift from statistical text-generation to multi-modal deductive, inductive, and abductive clinical reasoning—read the full deep dive on Medium: 
👉 **[Fine-Tuning DeepSeek R1 Reasoning on Medical Chain of Thought Dataset](https://medium.com/@anicomanesh/fine-tuning-deepseek-r1-reasoning-on-the-medical-chain-of-thought-dataset-922407121cc2)**

---

## 🏗️ Architecture & Deep Technical Decisions

### 1. Raw Token Binding vs. Standard Chat Dictionaries
Standard fine-tuning pipelines often map datasets directly into abstract multi-turn chat templates (e.g., standard `Llama-3.1` or `Qwen` chat structural dictionaries). For distilled reasoning models like DeepSeek-R1, this approach breaks down. 

DeepSeek-R1 models rely heavily on recognizing exact structural prompt boundaries to correctly execute internal reasoning traces. This pipeline implements a precise raw token mapping structure via a deterministic string binding layout:

```python
TRAIN_PROMPT_STYLE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request. Before answering, think carefully about the question "
    "and create a step-by-step chain of thoughts to ensure a logical and accurate response."
    "### Instruction:\n"
    "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
    "Please answer the following medical question. \n\n"
    "### Question:\n{} \n\n"
    "### Response:\n"
    "<think>\n{}\n</think>\n{}"
)

- Eliminating Paradigm Drift: By embedding the <think> and </think> boundaries explicitly during data processing, we force the weight updates to respect the absolute segregation between the internal clinical monologue (Complex_CoT) and the final actionable diagnostic answer (Response).

- Enforcing Strict Termination: Hardcoding the tokenizer.eos_token (<｜end▁of▁sentence｜>) right after the output text prevents the model from generating trailing hallucinations or infinitely looping across long contexts during inference.
