#!/usr/bin/env python3
"""
Evaluation script for MeSH entity extraction
Calculates precision, recall, F1 scores for extracted MeSH terms
Implements fuzzy matching for semantic similarity
"""

import json
import argparse
from typing import List, Dict, Tuple, Set
from pathlib import Path
import sys
import re
from difflib import SequenceMatcher

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))
from parser import MeSHParser


def normalize_term(term: str) -> str:
    """Normalize MeSH term for better matching"""
    # Lowercase
    term = term.lower().strip()
    # Remove extra whitespace
    term = re.sub(r'\s+', ' ', term)
    # Normalize punctuation (commas, hyphens)
    term = term.replace(',', '').replace('-', ' ')
    # Remove common variations
    term = term.replace('type i', 'type 1').replace('type ii', 'type 2')
    return term


def fuzzy_match_score(term1: str, term2: str) -> float:
    """Calculate fuzzy matching score between two terms"""
    norm1 = normalize_term(term1)
    norm2 = normalize_term(term2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Check if one is contained in the other (partial match)
    if norm1 in norm2 or norm2 in norm1:
        return 0.9
    
    # Calculate sequence similarity
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Boost score if key words overlap
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if words1 and words2:
        word_overlap = len(words1 & words2) / max(len(words1), len(words2))
        similarity = max(similarity, word_overlap)
    
    return similarity


def find_best_matches(predicted: List[str], ground_truth: List[str], threshold: float = 0.85) -> Tuple[Set, Set, Set]:
    """Find best fuzzy matches between predicted and ground truth
    
    Args:
        predicted: List of predicted terms
        ground_truth: List of ground truth terms
        threshold: Minimum similarity score for a match
    
    Returns:
        (matched_pred_indices, matched_gt_indices, match_scores)
    """
    matched_pred = set()
    matched_gt = set()
    
    # First pass: exact matches after normalization
    pred_normalized = [(i, normalize_term(p)) for i, p in enumerate(predicted)]
    gt_normalized = [(i, normalize_term(g)) for i, g in enumerate(ground_truth)]
    
    for pred_idx, pred_norm in pred_normalized:
        for gt_idx, gt_norm in gt_normalized:
            if gt_idx not in matched_gt and pred_norm == gt_norm:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                break
    
    # Second pass: fuzzy matches for remaining terms
    for pred_idx, pred_term in enumerate(predicted):
        if pred_idx in matched_pred:
            continue
        
        best_score = 0
        best_gt_idx = None
        
        for gt_idx, gt_term in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            score = fuzzy_match_score(pred_term, gt_term)
            if score >= threshold and score > best_score:
                best_score = score
                best_gt_idx = gt_idx
        
        if best_gt_idx is not None:
            matched_pred.add(pred_idx)
            matched_gt.add(best_gt_idx)
    
    return matched_pred, matched_gt


def calculate_metrics(predicted: List[str], ground_truth: List[str], use_fuzzy: bool = True, threshold: float = 0.85) -> Tuple[float, float, float, int, int, int]:
    """
    Calculate precision, recall, F1 for entity extraction with fuzzy matching
    
    Args:
        predicted: List of predicted entities
        ground_truth: List of ground truth entities
        use_fuzzy: Whether to use fuzzy matching (default: True)
        threshold: Minimum similarity score for fuzzy match (default: 0.85)
    
    Returns:
        (precision, recall, f1, true_positives, false_positives, false_negatives)
    """
    # Clean inputs
    predicted = [e.strip() for e in predicted if e and e.strip()]
    ground_truth = [e.strip() for e in ground_truth if e and e.strip()]
    
    # Handle edge cases
    if len(predicted) == 0 and len(ground_truth) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    
    if len(predicted) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(ground_truth)
    
    if len(ground_truth) == 0:
        return 0.0, 0.0, 0.0, len(predicted), 0, 0
    
    if use_fuzzy:
        # Use fuzzy matching
        matched_pred, matched_gt = find_best_matches(predicted, ground_truth, threshold)
        true_positives = len(matched_pred)
        false_positives = len(predicted) - true_positives
        false_negatives = len(ground_truth) - len(matched_gt)
    else:
        # Use exact matching (normalized)
        pred_set = set([normalize_term(e) for e in predicted])
        gt_set = set([normalize_term(e) for e in ground_truth])
        
        true_positives = len(pred_set & gt_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
    
    # Calculate metrics
    precision = true_positives / len(predicted) if len(predicted) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, true_positives, false_positives, false_negatives


def evaluate_extraction_file(results_file: str) -> Dict:
    """
    Evaluate results from extraction file
    
    Args:
        results_file: Path to JSON file with extraction results
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading results from {results_file}...")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Evaluating {len(results)} samples...")
    
    # Calculate per-sample metrics
    detailed_results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, result in enumerate(results):
        predicted = result.get('predicted', [])
        ground_truth = result.get('ground_truth', [])
        
        precision, recall, f1, tp, fp, fn = calculate_metrics(predicted, ground_truth)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        detailed_result = {
            'index': i,
            'pmid': result.get('pmid', ''),
            'title': result.get('title', '')[:100] + '...',
            'n_predicted': len(predicted),
            'n_ground_truth': len(ground_truth),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'predicted': predicted,
            'ground_truth': ground_truth
        }
        detailed_results.append(detailed_result)
    
    # Calculate averages
    n = len(results)
    avg_precision = total_precision / n if n > 0 else 0
    avg_recall = total_recall / n if n > 0 else 0
    avg_f1 = total_f1 / n if n > 0 else 0
    
    # Calculate micro-averaged F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    evaluation = {
        'n_samples': n,
        'macro_avg': {
            'precision': round(avg_precision, 3),
            'recall': round(avg_recall, 3),
            'f1': round(avg_f1, 3)
        },
        'micro_avg': {
            'precision': round(micro_precision, 3),
            'recall': round(micro_recall, 3),
            'f1': round(micro_f1, 3)
        },
        'totals': {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'detailed_results': detailed_results
    }
    
    return evaluation


def evaluate_direct(mesh_file: str, predictions_file: str, max_samples: int = None, skip: int = 0) -> Dict:
    """
    Evaluate predictions directly against MeSH dataset
    
    Args:
        mesh_file: Path to MeSH JSON dataset
        predictions_file: Path to predictions JSON (list of {"pmid": "...", "predicted": [...]})
        max_samples: Maximum samples to evaluate
        skip: Articles to skip
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading MeSH data from {mesh_file}...")
    mesh_parser = MeSHParser(mesh_file)
    articles = mesh_parser.load_data(max_articles=max_samples, skip=skip)
    
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # Create PMID to predictions mapping
    predictions_map = {}
    for pred in predictions_data:
        pmid = pred.get('pmid', '')
        predictions_map[pmid] = pred.get('predicted', [])
    
    print(f"Evaluating {len(articles)} samples...")
    
    # Calculate metrics
    detailed_results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, article in enumerate(articles):
        predicted = predictions_map.get(article.pmid, [])
        ground_truth = article.mesh_terms
        
        precision, recall, f1, tp, fp, fn = calculate_metrics(predicted, ground_truth)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        detailed_result = {
            'index': i,
            'pmid': article.pmid,
            'title': article.title[:100] + '...',
            'n_predicted': len(predicted),
            'n_ground_truth': len(ground_truth),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        detailed_results.append(detailed_result)
    
    # Calculate averages
    n = len(articles)
    avg_precision = total_precision / n if n > 0 else 0
    avg_recall = total_recall / n if n > 0 else 0
    avg_f1 = total_f1 / n if n > 0 else 0
    
    # Calculate micro-averaged F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    evaluation = {
        'n_samples': n,
        'macro_avg': {
            'precision': round(avg_precision, 3),
            'recall': round(avg_recall, 3),
            'f1': round(avg_f1, 3)
        },
        'micro_avg': {
            'precision': round(micro_precision, 3),
            'recall': round(micro_recall, 3),
            'f1': round(micro_f1, 3)
        },
        'totals': {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'detailed_results': detailed_results
    }
    
    return evaluation


def print_evaluation_summary(evaluation: Dict, mode: str = ""):
    """Print evaluation summary to console"""
    print("\n" + "="*70)
    if mode:
        print(f"MESH EVALUATION RESULTS - {mode.upper()}")
    else:
        print("MESH EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nSamples Evaluated: {evaluation['n_samples']}")
    
    print("\nMACRO-AVERAGED METRICS (average per sample):")
    print(f"  Precision: {evaluation['macro_avg']['precision']:.3f}")
    print(f"  Recall:    {evaluation['macro_avg']['recall']:.3f}")
    print(f"  F1 Score:  {evaluation['macro_avg']['f1']:.3f}")
    
    print("\nMICRO-AVERAGED METRICS (overall):")
    print(f"  Precision: {evaluation['micro_avg']['precision']:.3f}")
    print(f"  Recall:    {evaluation['micro_avg']['recall']:.3f}")
    print(f"  F1 Score:  {evaluation['micro_avg']['f1']:.3f}")
    
    print("\nTOTALS:")
    print(f"  True Positives:  {evaluation['totals']['true_positives']}")
    print(f"  False Positives: {evaluation['totals']['false_positives']}")
    print(f"  False Negatives: {evaluation['totals']['false_negatives']}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MeSH entity extraction results')
    parser.add_argument('--results_file', type=str, required=True, help='Path to extraction results JSON')
    parser.add_argument('--output_file', type=str, default=None, help='Output file for evaluation results')
    parser.add_argument('--mode', type=str, default='', help='Mode label for output (e.g., zero-shot, few-shot)')
    args = parser.parse_args()
    
    # Default output file
    if args.output_file is None:
        base_name = Path(args.results_file).stem
        args.output_file = f"evaluation_{base_name}.json"
    
    # Run evaluation
    evaluation = evaluate_extraction_file(args.results_file)
    
    # Print summary
    print_evaluation_summary(evaluation, args.mode)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
