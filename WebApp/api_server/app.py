"""
FastAPI server to run microsoft/biogpt locally on the server GPU.

Endpoints:
 - POST /generate { "text": "...", "max_new_tokens": 200 }
 - POST /ner { "text": "..." }
 - POST /extract_abstract { "text": "..." }

Requirements: see requirements.txt. You must install a torch build compatible with your CUDA.
This server loads the model onto GPU (device_map='auto' or .to('cuda')) when available.
"""
import os
import re
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()
logger = logging.getLogger("uvicorn")

# Enable CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_abstract(text: str) -> str:
    """
    Extract the abstract section from a scientific paper.
    Looks for common abstract markers and extracts the relevant portion.
    """
    text_lower = text.lower()
    
    # Common abstract section markers
    abstract_patterns = [
        r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|keywords|background|methods|results|\d+\.|1\s+introduction))',
        r'summary\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|keywords|background|methods|results|\d+\.|1\s+introduction))',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:  # Ensure it's not just a header
                return abstract
    
    # If no explicit abstract found, take first 1500 characters (likely contains abstract)
    lines = text.split('\n')
    # Skip title/header lines (usually short), start from meatier content
    content_start = 0
    for i, line in enumerate(lines):
        if len(line.strip()) > 100:  # Found a substantial paragraph
            content_start = i
            break
    
    # Return first substantial paragraphs (likely the abstract)
    remaining_text = '\n'.join(lines[content_start:])
    return remaining_text[:2000] if len(remaining_text) > 2000 else remaining_text


class GenerateRequest(BaseModel):
    text: str
    max_new_tokens: Optional[int] = 200


class NERRequest(BaseModel):
    text: str
    extract_abstract_only: Optional[bool] = True


class AbstractRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_models():
    global tokenizer, model, generator, ner_pipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    model_id = os.environ.get("HF_BIOGPT_MODEL", "microsoft/biogpt")
    # Use biomedical NER model for better entity categorization (drug, disease, gene, etc.)
    ner_model = os.environ.get("HF_NER_MODEL", "d4data/biomedical-ner-all")

    use_cuda = torch.cuda.is_available()
    logger.info(f"CUDA available: {use_cuda}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model on single GPU to avoid multi-device tensor errors
    if use_cuda:
        # Load model to single GPU (cuda:0) with float16 for efficiency
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        model.to("cuda:0")
        logger.info("Loaded model on cuda:0 with float16")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        logger.info("Loaded model on CPU")

    # NER pipeline (can be swapped to a biomedical NER model via HF_NER_MODEL env var)
    try:
        ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_model, aggregation_strategy="simple", device=0 if use_cuda else -1)
        logger.info(f"Loaded NER model {ner_model}")
    except Exception:
        # Fallback to default transformers behavior
        ner_pipeline = pipeline("ner", model=ner_model, aggregation_strategy="simple", device=0 if use_cuda else -1)

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post('/process_file')
async def process_file(file: UploadFile = File(...)):
    """Accepts file upload (multipart/form-data) and returns extracted text, entities and a short summary from BioGPT."""
    try:
        # Read file content
        content = await file.read()
        filename = file.filename
        
        # Lazy import for PDF processing
        import io
        text = ''
        if filename and filename.lower().endswith('.pdf'):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=io.BytesIO(content), filetype='pdf')
                pages = []
                for p in doc:
                    pages.append(p.get_text())
                text = '\n'.join(pages)
            except Exception as e:
                logger.exception('PDF parsing failed')
                raise HTTPException(status_code=500, detail=f'PDF parsing failed: {e}')
        else:
            try:
                text = content.decode('utf-8')
            except Exception:
                # fallback: try latin-1
                text = content.decode('latin-1')

        # Truncate very long texts for NER/generation to avoid timeouts - process first 20000 chars
        proc_text = text if len(text) <= 20000 else text[:20000]

        # Run NER
        ents_raw = ner_pipeline(proc_text)
        seen = set()
        entities = []
        for e in ents_raw:
            word = e.get('word') or e.get('entity') or str(e)
            clean = str(word).replace('##', '').strip()
            key = (clean, e.get('entity_group') or e.get('entity'))
            if key not in seen and clean:
                seen.add(key)
                # ensure score is a native float
                score = e.get('score')
                try:
                    score = float(score) if score is not None else None
                except Exception:
                    score = None
                entities.append({'text': clean, 'label': e.get('entity_group') or e.get('entity'), 'score': score})

        # Generate a short summary / features using BioGPT generator
        prompt = f"Extract 6 concise keywords/phrases and a 2-sentence summary from the following article:\n\n{proc_text[:4000]}"
        gen = generator(prompt, max_new_tokens=150, do_sample=False)
        summary = ''
        if isinstance(gen, list) and len(gen) > 0:
            summary = gen[0].get('generated_text', '')
        else:
            summary = str(gen)

        result = {'text': proc_text, 'entities': entities, 'summary': summary}
        encoded = jsonable_encoder(
            result,
            custom_encoder={
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.ndarray: lambda x: x.tolist(),
                torch.Tensor: lambda t: t.detach().cpu().tolist(),
            },
        )
        return JSONResponse(content=encoded)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('process_file error')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Missing text")
    try:
        # Truncate input to prevent exceeding model max length (1024 for BioGPT)
        # Reserve space for generation by limiting input to 800 tokens
        inputs = tokenizer(req.text, truncation=True, max_length=800, return_tensors="pt")
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        # Use the generation pipeline (input is already truncated)
        out = generator(input_text, max_new_tokens=min(req.max_new_tokens, 200), do_sample=False)
        # pipeline returns list of dicts with 'generated_text'
        if isinstance(out, list) and len(out) > 0:
            res = {"result": out[0].get("generated_text", "")}
        else:
            res = {"result": str(out)}
        encoded = jsonable_encoder(
            res,
            custom_encoder={
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.ndarray: lambda x: x.tolist(),
                torch.Tensor: lambda t: t.detach().cpu().tolist(),
            },
        )
        return JSONResponse(content=encoded)
    except Exception as e:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner")
def ner(req: NERRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Missing text")
    try:
        # Extract abstract if requested (default: True)
        text_to_process = extract_abstract(req.text) if req.extract_abstract_only else req.text
        
        # Tokenize and truncate to model's max length to avoid CUDA errors
        # Most BERT-based NER models have 512 token limit
        inputs = tokenizer(text_to_process, truncation=True, max_length=512, return_tensors="pt")
        truncated_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        ents = ner_pipeline(truncated_text)
        # Normalize and deduplicate
        seen = set()
        out = []
        for e in ents:
            text = e.get('word') or e.get('entity') or e.get('entity_group') or str(e)
            clean = text.replace('##', '').strip()
            label = e.get('entity_group') or e.get('entity')
            key = (clean, label)
            if key not in seen and clean and len(clean) > 1:  # Filter single characters
                seen.add(key)
                score = e.get('score')
                try:
                    score = float(score) if score is not None else None
                except Exception:
                    score = None
                out.append({
                    'text': clean, 
                    'label': label,
                    'category': label,  # Will be drug, disease, gene, etc. with biomedical model
                    'score': score
                })
        encoded = jsonable_encoder(
            {'entities': out},
            custom_encoder={
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.ndarray: lambda x: x.tolist(),
                torch.Tensor: lambda t: t.detach().cpu().tolist(),
            },
        )
        return JSONResponse(content=encoded)
    except Exception as e:
        logger.exception("NER error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_abstract")
def extract_abstract_endpoint(req: AbstractRequest):
    """Extract just the abstract section from a full paper text."""
    if not req.text:
        raise HTTPException(status_code=400, detail="Missing text")
    try:
        abstract = extract_abstract(req.text)
        return {"abstract": abstract, "length": len(abstract)}
    except Exception as e:
        logger.exception("Abstract extraction error")
        raise HTTPException(status_code=500, detail=str(e))
