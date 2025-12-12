#!/usr/bin/env python3
"""
Extract MeSH terms using fine-tuned model
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from parser import MeSHParser
from extract import parse_response


def load_finetuned_model(base_model_path: str, adapter_path: str):
    """Load base model with fine-tuned LoRA adapter"""
    print(f"Loading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"Loading fine-tuned adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    
    return model, tokenizer


def extract_with_finetuned(model, tokenizer, abstract: str, title: str = "", device: str = "cuda") -> str:
    """Extract entities using fine-tuned model"""
    
    instruction = """Extract ALL MeSH (Medical Subject Headings) terms from this biomedical abstract.

IMPORTANT:
- Include ALL relevant entities: diseases, chemicals, organisms, procedures, demographics
- ALWAYS include: Humans, Animals, specific species when mentioned
- Include age groups (Adult, Child, Aged, Middle Aged) and gender when relevant
- Return as JSON array only

"""
    
    if title:
        input_text = f"Title: {title}\n\nAbstract: {abstract}"
    else:
        input_text = f"Abstract: {abstract}"
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text}
    ]
    
    # Tokenize
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description='Extract MeSH terms using fine-tuned model')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to MeSH JSON file')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Base model path')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to fine-tuned adapter')
    parser.add_argument('--max_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--skip', type=int, default=0, help='Articles to skip')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()
    
    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"mesh_extraction_finetuned_{timestamp}.json"
    
    print("="*80)
    print("MeSH ENTITY EXTRACTION - FINE-TUNED MODEL")
    print("="*80)
    
    # Load model
    model, tokenizer = load_finetuned_model(args.base_model, args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    print(f"\nLoading MeSH data from {args.mesh_file}...")
    mesh_parser = MeSHParser(args.mesh_file)
    articles = mesh_parser.load_data(max_articles=args.max_samples, skip=args.skip)
    
    # Extract entities
    results = []
    print(f"\nExtracting entities from {len(articles)} articles...")
    
    for i, article in enumerate(articles):
        print(f"Processing {i+1}/{len(articles)}: {article.pmid}")
        
        response = extract_with_finetuned(
            model, tokenizer,
            article.abstract,
            article.title,
            device=device
        )
        
        predicted = parse_response(response)
        
        result = {
            'pmid': article.pmid,
            'title': article.title,
            'abstract': article.abstract,
            'ground_truth': article.mesh_terms,
            'predicted': predicted,
            'raw_response': response
        }
        results.append(result)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Extraction complete!")
    print(f"Processed {len(results)} articles")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
