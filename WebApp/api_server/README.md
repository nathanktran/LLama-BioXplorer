Local BioGPT API server
======================

This folder contains a FastAPI server that loads `microsoft/biogpt` on the server GPU and exposes two endpoints:

- POST /generate { "text": "...", "max_new_tokens": 200 }
- POST /ner { "text": "..." }

Setup
-----

1. Create and activate a Python virtualenv.

2. Install a PyTorch build compatible with your server CUDA. Example for CUDA 11.8 (adjust for your machine):

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install the rest of requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Set environment variables to select models:

   ```bash
   export HF_BIOGPT_MODEL="microsoft/biogpt"
   export HF_NER_MODEL="dbmdz/bert-large-cased-finetuned-conll03-english"
   ```

Run
---

Start the server (bind to 0.0.0.0 to make accessible externally):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Notes
-----
- The first model load will download model weights to the cache (~ several GB). Make sure you have enough disk space.
- Installing `torch` with the matching CUDA build is required to use the GPU. `accelerate` is used implicitly when device_map='auto' is selected.
- The `ner` endpoint uses a token-classification HF model; swap to a biomedical NER model if you need domain-specific entities.
