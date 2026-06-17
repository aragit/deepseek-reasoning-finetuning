import os
import yaml
import torch
import wandb
from huggingface_hub import login
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from src.dataset import prepare_medical_cot_dataset

def run_finetuning(config_path: str = "configs/lora_config.yaml", hf_token: str = None, wandb_token: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Authentication Handlers
    if hf_token:
        login(token=hf_token)
    if wandb_token:
        wandb.login(key=wandb_token)
        wandb.init(project=cfg["training"]["wandb_project"], job_type="training", anonymous="allow")

    # FastLanguageModel Architecture Allocator
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["model"]["load_in_4bit"],
        token=hf_token
    )

    # Extracting optimization target paths 
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
        use_rslora=False,
        loftq_config=None
    )

    # Process localized datasets
    train_dataset = prepare_medical_cot_dataset(
        tokenizer=tokenizer,
        dataset_path=cfg["dataset"]["path"],
        dataset_name=cfg["dataset"]["name"],
        split_spec=cfg["dataset"]["split"]
    )

    # Framework Execution Core
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=cfg["training"]["batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            warmup_steps=cfg["training"]["warmup_steps"],
            max_steps=cfg["training"]["max_steps"],
            learning_rate=float(cfg["training"]["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=cfg["training"]["logging_steps"],
            optim=cfg["training"]["optim"],
            weight_decay=cfg["training"]["weight_decay"],
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            seed=cfg["training"]["seed"],
            output_dir=cfg["training"]["output_dir"],
            report_to="wandb" if wandb_token else "none"
        ),
    )

    trainer.train()

    # Remote target pushing deployment configurations
    hub_model_id = cfg["training"]["hub_model_id"]
    print(f"[INFO] Shipping structural LoRA adapters and consolidated 16bit weights to: {hub_model_id}")
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    model.push_to_hub_merged(hub_model_id, tokenizer, save_method="merged_16bit")