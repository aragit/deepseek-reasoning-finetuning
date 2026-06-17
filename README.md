I'll organize this into a clean, professional README.md with proper Markdown formatting, corrected structure, and clear sections. Here's your final version:

```markdown
# DeepSeek-R1 Medical Reasoning Fine-Tuning

Fine-tuning pipeline for DeepSeek-R1-Distill-Qwen-7B on medical chain-of-thought datasets, optimized for constrained GPU environments.

---

## 🏗️ Architecture & Deep Technical Decisions

### 1. Raw Token Binding vs. Standard Chat Dictionaries

Standard fine-tuning pipelines map datasets into abstract multi-turn chat templates (e.g., Llama-3.1 or Qwen chat structures). For distilled reasoning models like DeepSeek-R1, this approach breaks down.

DeepSeek-R1 models rely on recognizing exact structural prompt boundaries to execute internal reasoning traces correctly. This pipeline implements precise raw token mapping via a deterministic string binding layout:

```python
TRAIN_PROMPT_STYLE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request. Before answering, think carefully about the question "
    "and create a step-by-step chain of thoughts to ensure a logical and accurate response."
    "\n\n### Instruction:\n"
    "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
    "Please answer the following medical question. \n\n"
    "### Question:\n{} \n\n"
    "### Response:\n"
    "\n{}\n\n{}"
)
```

**Eliminating Paradigm Drift:** By embedding explicit boundaries during data processing, weight updates respect the absolute segregation between the internal clinical monologue (`Complex_CoT`) and the final actionable diagnostic answer (`Response`).

**Enforcing Strict Termination:** Hardcoding the `tokenizer.eos_token` (`<｜end▁of▁sentence｜>`) immediately after the output text prevents trailing hallucinations or infinite loops during inference.

### 2. VRAM Optimization & Dual-T4 Memory Scaling

Fine-tuning a 7B parameter model with 4096-token sequences typically demands substantial GPU resources. This pipeline fits within constrained compute budgets (e.g., dual Tesla T4 instances with ~15GB VRAM per card) without sacrificing convergence:

| Technique | Benefit |
|-----------|---------|
| **Paged AdamW 8-bit** | Quantized optimizer states dramatically reduce memory footprint. Paged memory enables dynamic VRAM-to-CPU page-locking, mitigating OOM failures during backward passes. |
| **Full-Matrix Attention Targeting** | LoRA adapters inject across all attention and MLP blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), maximizing clinical feature internalization while keeping trainable parameters at ~0.58% (~40.3M). |
| **Smart Gradient Offloading** | Unsloth kernel optimizations dynamically manage gradients, enabling effective batch size of 16 (Batch: 2, Grad Accum: 4) for extended clinical contexts. |

### 3. Clinical Objectives: Judgment over Memorization

Conventional LLMs often recite textbook facts but fail on novel, edge-case presentations. Using the `FreedomIntelligence/medical-o1-reasoning-SFT` schema, this pipeline shifts the objective from data retention to cognitive structuring.

The model is penalized for jumping straight to diagnostic labels. Instead, it learns systematic multi-step internal monologue:

- **Abductive Inference** — Synthesizing disparate symptoms (e.g., sudden left-limb weakness + tender lower leg post-travel) to formulate differential diagnoses (e.g., paradoxical embolism vs. ischemic stroke).
- **Deductive Filtering** — Applying physiological laws to isolate mechanisms (e.g., how a DVT bypasses the pulmonary filter via PFO to access systemic arterial circulation).
- **Self-Verification** — Continuously evaluating intermediate hypotheses before committing to high-stakes diagnostic decisions.

---

## 📁 Repository Structure

```
deepseek-reasoning-finetuning/
├── configs/
│   └── lora_config.yaml          # Hyperparameters, training budgets, tracking
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Token injection & data-schema mapping
│   └── train.py                # Unsloth optimization engine & PEFT configs
├── main.py                       # CLI entry point
├── requirements.txt              # Pinned upstream dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create isolated environment
conda create -n r1-medical-finetuning python=3.10 -y
conda activate r1-medical-finetuning

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
export HF_TOKEN="your_huggingface_write_token"
export WANDB_API_KEY="your_wandb_telemetry_key"
```

### 3. Run Pipeline

```bash
python main.py --config configs/lora_config.yaml
```

---

## 📊 Performance & Telemetry

Training progress is reported to **Weights & Biases (wandb)**. Configuration:

- **Learning Rate:** 2e-4 with cosine scheduler
- **Loss Convergence:** Stabilizes gracefully across dataset subsets

**Output:** Finalized adapters and consolidated 16-bit merged weights are automatically pushed to Hugging Face Hub at:  
`Arnic/DeepSeek-R1-Distill-Qwen-7B_MedicalChain-Reasoning`

---

## 📜 References

- **Technical Deep Dive:** Nicoomanesh, A. (2025). *Fine-Tuning DeepSeek R1 Reasoning on Medical Chain of Thought Dataset*. Medium Article.
- **Unsloth Engine:** [Unsloth AI Framework](https://github.com/unslothai/unsloth)
- **Dataset:** [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
```

**Key improvements made:**
- Fixed broken code blocks and formatting
- Added proper Markdown hierarchy with headers
- Created a clean repository tree diagram
- Organized the architecture section with clear subsections
- Added a comparison table for VRAM techniques
- Standardized code block languages (bash, python)
- Removed redundant/fragmented text
- Fixed the prompt template formatting (it was garbled in the original)
- Added proper inline code styling for variables and tokens
- Cleaned up the Quick Start into numbered steps
- Added a References section with proper attribution formatting
