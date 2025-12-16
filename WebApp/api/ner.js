// Serverless endpoint to run token-classification (NER) via Hugging Face Inference API
// Expects POST { text: string }
// Requires HF_API_KEY and optionally HF_NER_MODEL in env (default: dbmdz/bert-large-cased-finetuned-conll03-english)

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  const { text } = req.body || {};
  if (!text) {
    res.status(400).json({ error: 'Missing text in request body' });
    return;
  }

  try {
    // If USE_LOCAL_BIOGPT=1 use the local Python server ner endpoint
    if (process.env.USE_LOCAL_BIOGPT === '1') {
      const localUrl = process.env.LOCAL_BIOGPT_URL_NER || 'http://127.0.0.1:8000/ner';
      const localRes = await fetch(localUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const j = await localRes.json();
      if (!localRes.ok) {
        res.status(localRes.status).json(j);
        return;
      }
      res.status(200).json(j);
      return;
    }

    const HF_API_KEY = process.env.HF_API_KEY;
    const model = process.env.HF_NER_MODEL || 'dbmdz/bert-large-cased-finetuned-conll03-english';

    if (!HF_API_KEY) {
      res.status(500).json({ error: 'HF_API_KEY not configured on server' });
      return;
    }

    const hfRes = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs: text }),
    });

    if (!hfRes.ok) {
      const txt = await hfRes.text();
      res.status(502).json({ error: 'Hugging Face API error', details: txt });
      return;
    }

    const json = await hfRes.json();
    const entities = [];
    if (Array.isArray(json)) {
      const seen = new Set();
      for (const e of json) {
        const textEnt = e.word || e.entity || e.entity_group || JSON.stringify(e);
        const clean = String(textEnt).replace(/##/g, '').trim();
        const key = `${clean}::${e.entity_group || e.entity || 'UNK'}`;
        if (!seen.has(key) && clean) {
          seen.add(key);
          entities.push({ text: clean, label: e.entity_group || e.entity || 'UNK', score: e.score || 0, start: e.start, end: e.end });
        }
      }
    }

    res.status(200).json({ entities });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error', details: String(err) });
  }
}
