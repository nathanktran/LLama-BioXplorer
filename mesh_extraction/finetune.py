#!/usr/bin/env python3
"""
Fine-tune LLaMA for MeSH entity extraction using QLoRA
Based on research showing 20%+ improvement with task-specific fine-tuning
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import argparse
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from parser import MeSHParser


def prepare_training_data(mesh_file: str, max_samples: int = 500, skip: int = 0):
    """
    Prepare training data from MeSH dataset
    
    Args:
        mesh_file: Path to MeSH JSON file
        max_samples: Maximum training samples
        skip: Articles to skip (for train/test split)
    
    Returns:
        List of formatted training examples
    """
    print(f"Loading MeSH training data from {mesh_file}...")
    parser = MeSHParser(mesh_file)
    articles = parser.load_data(max_articles=max_samples, skip=skip)
    
    print(f"Preparing {len(articles)} training examples...")
    training_data = []
    
    for article in articles:
        # Skip articles without terms
        if not article.mesh_terms:
            continue
        
        # Create instruction-following format
        instruction = """Extract ALL MeSH (Medical Subject Headings) terms from this biomedical abstract.

IMPORTANT:
- Include ALL relevant entities: diseases, chemicals, organisms, procedures, demographics
- ALWAYS include: Humans, Animals, specific species when mentioned
- Include age groups (Adult, Child, Aged, Middle Aged) and gender when relevant
- Return as JSON array only

"""
        
        # Input text
        if article.title:
            input_text = f"Title: {article.title}\n\nAbstract: {article.abstract}"
        else:
            input_text = f"Abstract: {article.abstract}"
        
        # Output (ground truth MeSH terms)
        output_text = json.dumps(article.mesh_terms, ensure_ascii=False)
        
        training_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    print(f"Created {len(training_data)} training examples")
    return training_data


def format_instruction(sample, tokenizer):
    """Format training sample as chat template"""
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def create_datasets(training_data, tokenizer, val_split: float = 0.1):
    """Create train and validation datasets"""
    # Shuffle and split
    import random
    random.shuffle(training_data)
    
    split_idx = int(len(training_data) * (1 - val_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Format with chat template
    train_dataset = train_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer(model_path: str, use_4bit: bool = True):
    """
    Setup model with quantization and LoRA
    
    Args:
        model_path: Path to base model
        use_4bit: Whether to use 4-bit quantization (QLoRA)
    
    Returns:
        model, tokenizer
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Set reasonable max length (reduced for memory efficiency)
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 1536  # Reduced from 2048 to fit on 12GB GPUs
    
    print(f"Loading model with {'4-bit' if use_4bit else 'float16'} precision...")
    
    if use_4bit:
        # QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Better device mapping for multi-GPU (4x TITAN V)
        device_map = "auto"  # Let accelerate handle distribution
        max_memory = {i: "10GiB" for i in range(torch.cuda.device_count())}  # Reserve memory per GPU
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer


def setup_lora_config():
    """Configure LoRA parameters"""
    peft_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha (scaling factor)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return peft_config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLaMA for MeSH extraction')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to MeSH JSON file')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Base model path')
    parser.add_argument('--output_dir', type=str, default='./mesh_finetuned', help='Output directory for fine-tuned model')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum training samples')
    parser.add_argument('--skip', type=int, default=0, help='Articles to skip (for train/test split)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--use_4bit', action='store_true', default=True, help='Use 4-bit quantization (QLoRA)')
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    
    print("="*80)
    print("MeSH ENTITY EXTRACTION - FINE-TUNING WITH QLORA")
    print("="*80)
    print(f"Base model: {args.model_path}")
    print(f"Training samples: {args.max_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Prepare training data
    training_data = prepare_training_data(
        args.mesh_file,
        max_samples=args.max_samples,
        skip=args.skip
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_path, args.use_4bit)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(training_data, tokenizer, args.val_split)
    
    # Setup LoRA
    peft_config = setup_lora_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Initialize trainer (TRL 0.26+ API)
    from trl import SFTConfig
    
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,  # Reduced from args.batch_size (4) to 2 for memory
        per_device_eval_batch_size=2,   # Match train batch size
        gradient_accumulation_steps=8,  # Increased from 4 to maintain effective batch size
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=5,  # More frequent logging
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=3,
        max_length=1536,  # Reduced from 2048 to save memory
        packing=False,
        dataset_text_field="text",
        dataloader_num_workers=0,  # Avoid memory overhead from workers
        dataloader_pin_memory=True,  # Speed up data transfer
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,  # Changed from tokenizer
        args=sft_config,
    )
    
    # Train
    print("\nStarting training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\nSaving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("="*80)
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print("\nTo use the fine-tuned model:")
    print(f"  python3 extract.py --model_path {output_dir} --mode few-shot ...")
    print("="*80)


if __name__ == '__main__':
    main()
