"""
Supervised fine-tuning script for football match prediction.

Follows the Phil Schmid blog pattern (https://www.philschmid.de/fine-tune-llms-in-2025)
using TRL's SFTTrainer with QLoRA support. Accepts a YAML config for all arguments.

Usage:
    python src/training/run_sft.py --config src/training/recipes/llama-3-1-8b-instruct-qlora.yaml

For distributed training with DeepSpeed:
    accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes 4 src/training/run_sft.py --config src/training/recipes/llama-3-1-8b-instruct-qlora.yaml
"""

from dataclasses import dataclass, field
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    """Custom arguments for the training script."""
    dataset_id_or_path: str = field(
        metadata={"help": "Path to local JSONL/JSON file or HuggingFace dataset ID."}
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation JSONL/JSON file. If not provided, no eval is run during training."}
    )
    dataset_splits: str = field(
        default="train",
        metadata={"help": "Dataset split to use when loading from HuggingFace Hub."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name or path. Defaults to model_name_or_path if not set."}
    )
    spectrum_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Spectrum SNR config YAML for selective layer unfreezing."}
    )


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


########################
# Helper functions
########################

def get_checkpoint(training_args: SFTConfig):
    """Check for existing checkpoints to resume training."""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def setup_model_for_spectrum(model, spectrum_config_path):
    """
    Apply Spectrum selective layer unfreezing.
    Freezes all parameters, then unfreezes only the top-SNR layers
    specified in the Spectrum config.
    """
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze Spectrum-selected parameters
    for name, param in model.named_parameters():
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True

    return model


def load_data(path: str, split: str = "train"):
    """Load dataset from local file or HuggingFace Hub."""
    if path.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=path, split="train")
    else:
        dataset = load_dataset(path, split=split)
    return dataset


###########################################################################
# Main training function
###########################################################################

def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""

    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    train_dataset = load_data(script_args.dataset_id_or_path, script_args.dataset_splits)
    logger.info(
        f"Loaded training dataset with {len(train_dataset)} samples "
        f"and features: {list(train_dataset.features.keys())}"
    )

    eval_dataset = None
    if script_args.eval_dataset_path:
        eval_dataset = load_data(script_args.eval_dataset_path)
        logger.info(f"Loaded eval dataset with {len(eval_dataset)} samples")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #######################
    # Load pretrained model
    #######################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),
        use_cache=False if training_args.gradient_checkpointing else True,
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,
    )

    # Set up 4-bit quantization if requested (QLoRA)
    if model_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )

    # Set up PEFT config (LoRA)
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    # Load model — use Liger kernels for efficiency if available and configured
    if training_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    training_args.distributed_state.wait_for_everyone()

    # Apply Spectrum selective unfreezing if configured
    if script_args.spectrum_config_path:
        model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    logger.info(
        f"*** Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"for {training_args.num_train_epochs} epochs ***"
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Saving model ***")
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save model card and push to hub on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["sft", "football-llm", "world-cup-prediction", "qlora"]})
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
