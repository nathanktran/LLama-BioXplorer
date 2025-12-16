# BioXplorer: Detailed Technical Explanation

## Complete Workflow Overview

```
User Input (Text/PDF) 
    ↓
[Abstract Extraction] (optional, default ON)
    ↓
Two Paths:
    1. "Analyze Text" → BioGPT → Text Generation/Summary
    2. "Extract Entities" → NER Model → Entity Recognition → Categorization
```

---

## 1. ANALYZE TEXT BUTTON - How BioGPT Works

### What Happens When You Click "Analyze Text"

**Step 1: Input Processing**
```javascript
// Frontend (BioGPTComponent.js)
const handleProcess = async () => {
    // Sends full input text to backend
    fetch('http://localhost:5001/generate', {
        method: 'POST',
        body: JSON.stringify({ 
            text: inputText,           // Your full text
            max_new_tokens: 200        // Generate up to 200 new tokens
        })
    });
}
```

**Step 2: Backend Processing (API Server)**
```python
# Backend (app.py)
@app.post("/generate")
def generate(req: GenerateRequest):
    # 1. Tokenize the input text
    inputs = tokenizer(req.text, truncation=True, max_length=800, return_tensors="pt")
    
    # 2. Decode back to text (ensures it's within token limit)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    # 3. Generate text using BioGPT
    out = generator(input_text, max_new_tokens=200, do_sample=False)
    
    # 4. Return generated text
    return {"result": out[0]["generated_text"]}
```

### What is BioGPT Actually Doing?

**BioGPT Model**: `microsoft/biogpt`
- **Architecture**: GPT-2 style transformer (15 billion parameters)
- **Training**: Pre-trained on 15 million PubMed abstracts (biomedical literature)
- **Purpose**: Generate or complete biomedical text

**How Text Generation Works**:

1. **Tokenization**: 
   - Converts your text into numerical tokens (word pieces)
   - Example: "COVID-19 patients" → [12345, 67890, 11223]

2. **Encoding**:
   - BioGPT processes tokens through 12 transformer layers
   - Each layer applies:
     - Self-attention (understands context between words)
     - Feed-forward neural networks
   - Creates rich representations of biomedical concepts

3. **Generation**:
   - Predicts next token based on context
   - Uses greedy decoding (`do_sample=False`) - picks most likely token
   - Repeats for up to 200 new tokens
   - Stops at sentence boundaries or max length

4. **Why It Works for Biology**:
   - Trained on PubMed = knows biomedical terminology
   - Understands relationships (drug-disease, gene-protein)
   - Can complete sentences about medical concepts accurately

**Example**:
```
Input: "Patients with diabetes often develop"
BioGPT: "Patients with diabetes often develop cardiovascular complications, 
         including coronary artery disease and peripheral neuropathy, which 
         require regular monitoring and management."
```

---

## 2. EXTRACT ENTITIES BUTTON - How NER Works

### What Happens When You Click "Extract Entities"

**Step 1: Abstract Extraction (if enabled)**
```python
def extract_abstract(text: str) -> str:
    # Uses regex to find "Abstract:" section
    pattern = r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|keywords|...))'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()  # Returns just the abstract
    else:
        return text[:2000]  # Fallback: first 2000 chars
```

**Step 2: NER Processing**
```javascript
// Frontend sends request
fetch('http://localhost:5001/ner', {
    method: 'POST',
    body: JSON.stringify({ 
        text: inputText,
        extract_abstract_only: true  // Your checkbox setting
    })
});
```

**Step 3: Backend NER Pipeline**
```python
@app.post("/ner")
def ner(req: NERRequest):
    # 1. Extract abstract if requested
    text_to_process = extract_abstract(req.text) if req.extract_abstract_only else req.text
    
    # 2. Tokenize and truncate to 512 tokens (BERT limit)
    inputs = tokenizer(text_to_process, truncation=True, max_length=512, return_tensors="pt")
    truncated_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    # 3. Run NER model
    ents = ner_pipeline(truncated_text)
    
    # 4. Process and categorize entities
    for e in ents:
        clean_text = e.get('word').replace('##', '').strip()
        category = e.get('entity_group') or e.get('entity')  # Drug, Disease, Gene, etc.
        score = e.get('score')  # Confidence score (0-1)
        
        out.append({
            'text': clean_text,
            'label': category,
            'category': category,
            'score': score
        })
    
    return {'entities': out}
```

### What is NER (Named Entity Recognition)?

**NER Model**: `d4data/biomedical-ner-all`
- **Architecture**: BERT-based token classifier
- **Training**: Fine-tuned on biomedical literature for entity recognition
- **Output**: Identifies and categorizes biomedical terms

**How NER Works Step-by-Step**:

1. **Tokenization**:
   ```
   Input: "Penicillin treats bacterial infections"
   Tokens: ["Penicillin", "treats", "bacterial", "infections"]
   ```

2. **BERT Encoding**:
   - Each token passes through 12 transformer layers
   - Creates contextual embeddings (768-dimensional vectors)
   - Token "Penicillin" gets different embedding based on context

3. **Classification Head**:
   - Final layer is a classifier for each token
   - Predicts entity type: `[B-Chemical, O, O, B-Disease]`
   - B- = Beginning of entity, I- = Inside entity, O = Outside (not an entity)

4. **Entity Labels (Categories)**:
   The biomedical NER model recognizes:
   - **Chemical/Drug**: Medications, compounds (e.g., "aspirin", "dopamine")
   - **Disease**: Medical conditions (e.g., "diabetes", "cancer")
   - **Gene**: Gene names (e.g., "BRCA1", "TP53")
   - **Protein**: Protein names (e.g., "insulin", "hemoglobin")
   - **Cell Type**: Cell types (e.g., "T-cell", "neuron")
   - **Cell Line**: Laboratory cell lines
   - **DNA**: DNA sequences or references
   - **RNA**: RNA molecules
   - **Species**: Organism names (e.g., "E. coli", "human")

5. **Confidence Scores**:
   - Each entity has a score (0-1) indicating model confidence
   - Example: "aspirin" = 0.98 (very confident it's a chemical)
   - Low scores filtered out to reduce false positives

**Example**:
```
Input: "COVID-19 is caused by SARS-CoV-2 virus and treated with remdesivir."

NER Output:
- "COVID-19" → Disease (score: 0.96)
- "SARS-CoV-2" → Species/Virus (score: 0.94)
- "remdesivir" → Chemical/Drug (score: 0.98)
```

---

## 3. ENTITY CATEGORIZATION - How Grouping Works

### Backend Categorization
The NER model automatically assigns categories:
```python
entity = {
    'text': 'penicillin',
    'category': 'Chemical',  # Assigned by NER model
    'score': 0.95
}
```

### Frontend Grouping
```javascript
// Group entities by category
const grouped = {};
entityList.forEach(entity => {
    const category = entity.category || entity.label || 'Other';
    
    if (!grouped[category]) {
        grouped[category] = [];
    }
    
    grouped[category].push(entity);
});

setEntitiesByCategory(grouped);

// Result:
// {
//   "Chemical": [{text: "penicillin", score: 0.95}, ...],
//   "Disease": [{text: "infection", score: 0.89}, ...],
//   "Gene": [{text: "BRCA1", score: 0.97}, ...]
// }
```

### Display
```javascript
{Object.entries(entitiesByCategory).map(([category, entityList]) => (
    <div>
        <div>{category}</div>  // "Chemical", "Disease", etc.
        {entityList.map(entity => (
            <div onClick={() => handleEntityClick(entity.text)}>
                {entity.text} ({entity.score * 100}%)
            </div>
        ))}
    </div>
))}
```

---

## 4. ENTITY CLICK - BioGPT Description

When you click an entity:
```javascript
const handleEntityClick = async (entityText) => {
    // Creates a prompt for BioGPT
    const prompt = `Provide a concise biomedical description of "${entityText}" 
                    (1-3 sentences) and mention common contexts where it appears.`;
    
    // Sends to BioGPT
    fetch('http://localhost:5001/generate', {
        body: JSON.stringify({ text: prompt, max_new_tokens: 120 })
    });
    
    // BioGPT generates description based on its PubMed training
}
```

---

## 5. WHY ABSTRACT-ONLY EXTRACTION?

**Problem**: Full papers can be 10,000+ words
- Exceeds model token limits (512 for NER, 1024 for BioGPT)
- Slows processing
- Introduces noise (methods, references aren't relevant entities)

**Solution**: Extract abstract
- Abstracts are 200-300 words
- Contain key findings and entities
- Within token limits
- Faster and more accurate

**Regex Logic**:
```python
# Looks for patterns like:
"Abstract: ..."
"Summary: ..."
"ABSTRACT\n..."

# Stops at:
"Introduction", "Methods", "Keywords", numbered sections
```

---

## 6. CONFIDENCE SCORES

NER model outputs probability for each prediction:
```
"aspirin" → Chemical (confidence: 0.98) ✓ High confidence, show it
"the" → Entity (confidence: 0.23) ✗ Low confidence, likely false positive
```

Displayed as percentage: `aspirin (98%)`

---

## Technical Summary

| Component | Model | Purpose | Input Limit |
|-----------|-------|---------|-------------|
| **BioGPT** | microsoft/biogpt | Text generation | 1024 tokens (~800 words) |
| **NER** | d4data/biomedical-ner-all | Entity extraction | 512 tokens (~400 words) |
| **Tokenizer** | BioGPT tokenizer | Convert text↔numbers | - |
| **Abstract Extractor** | Regex | Find abstract section | - |

### Processing Flow
1. User inputs text → Extract abstract (optional)
2. **Analyze**: Text → BioGPT → Generate summary/completion
3. **Extract**: Text → NER → Find entities → Group by category → Display
4. **Click Entity**: Entity name → BioGPT (with prompt) → Generate description

### Why It Works
- **Domain-specific models**: Trained on biomedical literature
- **Transformer architecture**: Understands context and relationships
- **Token limits**: Keep processing fast and within GPU memory
- **Abstract extraction**: Focus on relevant content
- **Categorization**: Biomedical NER trained to recognize specific entity types
