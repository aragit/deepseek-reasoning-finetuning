import os
import argparse
from src.train import run_finetuning

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1 Industrial Fine-Tuning Chassis")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml", help="YAML path specification")
    args = parser.parse_args()

    # Dynamic lookup resolution for credentials 
    hf_token = os.getenv("HF_TOKEN")
    wandb_token = os.getenv("WANDB_API_KEY")

    print(f"[START] Processing training workspace using layout parameters: {args.config}")
    run_finetuning(config_path=args.config, hf_token=hf_token, wandb_token=wandb_token)
    print("[SUCCESS] Consolidated weights exported cleanly.")

if __name__ == "__main__":
    main()