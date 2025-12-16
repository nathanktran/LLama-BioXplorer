# BioXplorer Application Implementation Guide

## How the Application Works

### Architecture Overview
```
Frontend (React) → API Server (FastAPI) → ML Models (BioGPT + NER)
```

### Component Breakdown

#### 1. **Named Entity Recognition (NER)**
- **Model**: `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Purpose**: Extract entities from text
- **Current Labels**: PER (Person), ORG (Organization), LOC (Location), MISC (Miscellaneous)
- **Limitation**: This is a general NER model, NOT biomedical-specific

#### 2. **BioGPT**
- **Model**: `microsoft/biogpt`
- **Purpose**: Generate biomedical text completions and summaries
- **Max Input**: 1024 tokens (truncated to 800 to leave room for generation)

### Current Workflow

1. **User uploads PDF or enters text**
2. **Extract Keywords**: Sends text to `/ner` endpoint
   - NER model identifies entities
   - Returns deduplicated list of entity texts
3. **Analyze with BioGPT**: Sends text to `/generate` endpoint
   - BioGPT generates continuation/summary
4. **Click on Entity**: Sends prompt to `/generate` asking for description

## Issues and Solutions

### Issue 1: Processing Entire Document Instead of Abstract
**Problem**: Currently processes all text
**Solution**: Extract and process only the abstract section

### Issue 2: Generic NER Model
**Problem**: Using CoNLL03 model (news domain), not biomedical
**Solution**: Switch to biomedical NER model like:
- `dmis-lab/biobert-base-cased-v1.2`
- `allenai/scibert_scivocab_uncased`
- BC5CDR-disease or BC5CDR-chem models

### Issue 3: No Entity Categorization
**Problem**: Entities not grouped by type (drug, disease, gene, etc.)
**Solution**: Use biomedical NER that provides entity types

### Issue 4: UI Too "GPT-like"
**Problem**: Emoji-heavy, rounded, gradient design
**Solution**: Professional medical/scientific interface

## Recommended Changes

### For Better Biomedical Entity Extraction:
Use a model like `d4data/biomedical-ner-all` which provides:
- Disease entities
- Chemical/Drug entities  
- Gene/Protein entities
- And more biomedical categories

This will automatically solve Issue #3 (categorization).
