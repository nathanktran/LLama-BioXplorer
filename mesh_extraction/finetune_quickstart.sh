#!/bin/bash
# Quick start script for fine-tuning MeSH extraction model

set -e

echo "============================================================"
echo "MeSH Entity Extraction - Fine-Tuning Quick Start"
echo "============================================================"
echo ""

# Default parameters
MESH_FILE="../data/allMeSH_2022/allMeSH_2022_100mb.json"
MAX_SAMPLES=500
EPOCHS=3
BATCH_SIZE=4
SKIP=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip)
            SKIP="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --samples N      Number of training samples (default: 500)"
            echo "  --epochs N       Number of training epochs (default: 3)"
            echo "  --batch-size N   Batch size (default: 4)"
            echo "  --skip N         Articles to skip (default: 0)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Quick experiment (500 samples, 3 epochs)"
            echo "  $0 --samples 1000 --epochs 5 # Full training"
            echo "  $0 --batch-size 2            # Reduce for low VRAM"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Training samples: $MAX_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Skip: $SKIP articles"
echo ""

# Check if data file exists
if [ ! -f "$MESH_FILE" ]; then
    echo "❌ Data file not found: $MESH_FILE"
    exit 1
fi
echo "✓ Data file found"
echo ""

# Start fine-tuning
echo "Starting fine-tuning..."
echo "This will take approximately 2-4 hours for 500 samples"
echo "Press Ctrl+C to cancel"
echo ""
sleep 2

python3 finetune.py \
    --mesh_file "$MESH_FILE" \
    --max_samples "$MAX_SAMPLES" \
    --skip "$SKIP" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir ./mesh_finetuned

echo ""
echo "============================================================"
echo "Fine-tuning complete!"
echo ""
echo "Next steps:"
echo "1. Find your model in: ./mesh_finetuned_*/"
echo "2. Test it with:"
echo "   python3 extract_finetuned.py --adapter_path ./mesh_finetuned_* --max_samples 10"
echo "3. Evaluate on test set:"
echo "   python3 extract_finetuned.py --adapter_path ./mesh_finetuned_* --max_samples 100 --skip 1000"
echo "   python3 evaluate.py --results_file mesh_extraction_finetuned_*.json"
echo "============================================================"
