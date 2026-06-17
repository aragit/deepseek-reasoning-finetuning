
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

* **Eliminating Paradigm Drift:** By embedding the `<think>` and `</think>` boundaries explicitly during data processing, we force the weight updates to respect the absolute segregation between the internal clinical monologue (`Complex_CoT`) and the final actionable diagnostic answer (`Response`).
* **Enforcing Strict Termination:** Hardcoding the `tokenizer.eos_token` right after the output text prevents the model from generating trailing hallucinations or infinitely looping across long contexts during inference.

### 2. VRAM Optimization & Dual-T4 Memory Scaling

Fine-tuning a 7B parameter model with a 4096-token sequence length normally demands substantial GPU resources. This pipeline is tailored to fit comfortably within highly constrained compute budgets (such as commodity dual Tesla T4 instances with ~15GB VRAM per card) without sacrificing model convergence:

* **`paged_adamw_8bit` Optimizer:** Standard 32-bit AdamW stores massive amounts of optimizer states. Transitioning to 8-bit quantized states drops memory footprints dramatically. Incorporating *paged* memory enables dynamic VRAM-to-CPU page-locking, completely mitigating abrupt Out-Of-Memory (OOM) failures during intensive backward passes.
* **Full-Matrix Attention Targeting:** Rather than limiting LoRA updates strictly to `q_proj` and `v_proj`, this architecture injects adapters across the entire attention and MLP blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). This maximizes the model's capacity to internalize cross-functional clinical features while keeping trainable parameters down to a lean **0.58%** (~40.3M parameters).
* **Smart Gradient Offloading:** Utilizing Unsloth’s kernel optimizations, gradients are dynamically managed, enabling a total batch size of 16 (Batch Size: 2, Gradient Accumulation Steps: 4, scaled across hardware) to process extended clinical contexts cleanly.

---

## 📈 Clinical Objectives: Clinical Judgment over Memorization

Conventional LLMs often operate like highly sophisticated parrots—reciting textbook medical facts but failing when faced with novel, edge-case clinical presentations. By running this fine-tuning chassis on the `FreedomIntelligence/medical-o1-reasoning-SFT` schema, we shift the objective from *data retention* to *cognitive structuring*.

The model is actively penalized if it attempts to jump straight to a diagnostic label. Instead, it is aligned to run a systematic multi-step internal monologue:

* **Abductive Inference:** Synthesizing disparate patient symptoms (e.g., sudden left-limb weakness alongside a tender lower leg post-travel) to formulate an initial pool of probable differential diagnoses (e.g., paradoxical embolism vs. standard ischemic stroke).
* **Deductive Filtering:** Applying physiological laws to isolate how a venous deep vein thrombosis (DVT) could bypass the pulmonary filter via a Patent Foramen Ovale (PFO) to access systemic arterial circulation.
* **Self-Verification:** Continuously evaluating intermediate hypotheses inside the `<think>` layer prior to committing to a final, high-stakes diagnostic decision.

---

## 📁 Repository Structure

| File / Folder | Responsibility |
| --- | --- |
| **`configs/lora_config.yaml`** | Decoupled hyperparameters, training budgets, and tracking parameters. |
| **`src/dataset.py`** | Custom token injection and case-sensitive data-schema mapping core. |
| **`src/train.py`** | Optimization engine using Unsloth framework accelerators and PEFT configs. |
| **`main.py`** | High-level CLI wrapper acting as the pipeline entry point. |
| **`requirements.txt`** | Explicitly pinned upstream runtime dependencies. |

```text
deepseek-reasoning-finetuning/
├── configs/
│   └── lora_config.yaml     
├── src/
│   ├── __init__.py          
│   ├── dataset.py           
│   └── train.py             
├── main.py                  
├── requirements.txt         
└── README.md                

```

---

## 🚀 Quick Start

### 1. Environment Setup

Isolate your runtime environment away from base system libraries:

```bash
# Create an isolated python environment
conda create -n r1-medical-finetuning python=3.10 -y
conda activate r1-medical-finetuning

# Clean install pinned compilation dependencies
pip install -r requirements.txt

```

### 2. Configure Credentials

Export your environment validation tokens securely before initializing execution:

```bash
export HF_TOKEN="your_huggingface_write_token"
export WANDB_API_KEY="your_wandb_telemetry_key"

```

### 3. Run the Pipeline Execution Core

Launch the end-to-end training, weight-merging, and deployment sequence directly from the command line:

```bash
python main.py --config configs/lora_config.yaml

```

---

## 📊 Performance & Telemetry Tracking

Training progress is seamlessly reported to Weights & Biases (`wandb`). The model handles loss convergence using a smooth cosine learning rate scheduler starting at `2e-4`, stabilizing gracefully across dataset subsets to yield clean, structured reasoning outputs.

The finalized model adapters along with the consolidated 16-bit merged model weights are compiled and automatically pushed to Hugging Face Hub under the repository designation: `Arnic/DeepSeek-R1-Distill-Qwen-7B_MedicalChain-Reasoning`.

---

## 📜 References & Citation

* **Full Technical Deep Dive:** Nicoomanesh, A. (2025). *Fine-Tuning DeepSeek R1 Reasoning on Medical Chain of Thought Dataset*. [Medium Article](https://medium.com/@anicomanesh/fine-tuning-deepseek-r1-reasoning-on-the-medical-chain-of-thought-dataset-922407121cc2).
* **Unsloth Engine:** [Unsloth AI Framework](https://github.com/unslothai/unsloth)
* **Dataset Source:** FreedomIntelligence Medical O1 SFT Engine.

```

```
