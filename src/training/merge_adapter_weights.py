"""
Merge LoRA adapter weights with the base model into a standalone model.

After QLoRA training, the adapter weights are saved separately. This script
merges them back into the base model for standalone inference/deployment.

Usage:
    python src/training/merge_adapter_weights.py \
        --peft_model_id runs/football-llm-qlora \
        --output_dir runs/football-llm-merged

    # Or push directly to HuggingFace Hub:
    python src/training/merge_adapter_weights.py \
        --peft_model_id runs/football-llm-qlora \
        --push_to_hub True \
        --repository_id <your-hf-username>/football-llm-merged
"""

import tempfile
from dataclasses import dataclass, field
from typing import Optional

import torch
from huggingface_hub import HfApi
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser


def save_model(model_path_or_id: str, save_dir: str, save_tokenizer: bool = True):
    """Load a PEFT model, merge adapter weights with base, and save."""
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # Merge LoRA adapters into the base model
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="3GB")

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        tokenizer.save_pretrained(save_dir)


@dataclass
class ScriptArguments:
    peft_model_id: str = field(
        metadata={"help": "Path to the PEFT/LoRA adapter directory or HuggingFace model ID."}
    )
    output_dir: Optional[str] = field(
        default="runs/football-llm-merged",
        metadata={"help": "Local directory to save the merged model."},
    )
    save_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to also save the tokenizer."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to push the merged model to HuggingFace Hub."},
    )
    repository_id: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace Hub repository ID (e.g., 'username/model-name')."},
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    api = HfApi()

    if args.push_to_hub:
        repo_id = args.repository_id if args.repository_id else args.peft_model_id.split("/")[-1]
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(args.peft_model_id, temp_dir, args.save_tokenizer)
            api.upload_large_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="model",
            )
        print(f"Model pushed to hub: {repo_id}")
    else:
        save_model(args.peft_model_id, args.output_dir, args.save_tokenizer)
        print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
