from datasets import load_dataset

# Accurate string template reconstruction from Kaggle runtime core
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

def prepare_medical_cot_dataset(tokenizer, dataset_path: str, dataset_name: str, split_spec: str):
    """Loads and matches the raw Medical CoT parameters with exact case-sensitive keys."""
    dataset = load_dataset(dataset_path, dataset_name, split=split_spec, trust_remote_code=True)
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        
        for input_text, cot_text, output_text in zip(inputs, cots, outputs):
            # Strict formatting binding corresponding to target evaluation tokens
            text = TRAIN_PROMPT_STYLE.format(input_text, cot_text, output_text) + EOS_TOKEN
            texts.append(text)
            
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True)