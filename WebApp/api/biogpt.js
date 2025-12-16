// Serverless endpoint to proxy text generation to Hugging Face Inference API
// Expects POST { text: string, max_new_tokens?: number }
// Requires HF_API_KEY and optionally HF_BIOGPT_MODEL in env (default: microsoft/BioGPT-Large)

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  const { text, max_new_tokens = 200 } = req.body || {};
  if (!text) {
    res.status(400).json({ error: 'Missing text in request body' });
    return;
  }
  try {
    // If USE_LOCAL_BIOGPT=1 is set in the environment, proxy to the local Python server
    if (process.env.USE_LOCAL_BIOGPT === '1') {
      const localUrl = process.env.LOCAL_BIOGPT_URL || 'http://127.0.0.1:8000/generate';
      const localRes = await fetch(localUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, max_new_tokens }),
      });
      const j = await localRes.json();
      if (!localRes.ok) {
        res.status(localRes.status).json(j);
        return;
      }
      res.status(200).json({ result: j.result });
      return;
    }

    // Default behaviour: proxy to Hugging Face Inference API (cloud)
    const HF_API_KEY = process.env.HF_API_KEY;
    const model = process.env.HF_BIOGPT_MODEL || 'microsoft/BioGPT-Large';

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
      body: JSON.stringify({ inputs: text, parameters: { max_new_tokens } }),
    });

    if (!hfRes.ok) {
      const txt = await hfRes.text();
      res.status(502).json({ error: 'Hugging Face API error', details: txt });
      return;
    }

    const json = await hfRes.json();
    let output = '';
    if (Array.isArray(json) && json.length > 0) {
      if (typeof json[0] === 'string') output = json[0];
      else if (json[0].generated_text) output = json[0].generated_text;
      else output = JSON.stringify(json);
    } else if (typeof json === 'object' && json.generated_text) {
      output = json.generated_text;
    } else if (typeof json === 'string') {
      output = json;
    } else {
      output = JSON.stringify(json);
    }

    res.status(200).json({ result: output });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error', details: String(err) });
  }
}
