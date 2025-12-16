// Proxy endpoint to forward multipart file uploads to local BioGPT API server when enabled.
export const config = { api: { bodyParser: false } };

import formidable from 'formidable';
import fs from 'fs';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (process.env.USE_LOCAL_BIOGPT !== '1') {
    res.status(400).json({ error: 'Local BioGPT server not enabled (set USE_LOCAL_BIOGPT=1)' });
    return;
  }

  const form = new formidable.IncomingForm();
  form.parse(req, async (err, fields, files) => {
    if (err) {
      res.status(500).json({ error: 'Error parsing form' });
      return;
    }

    const fileKey = Object.keys(files || {})[0];
    if (!fileKey) {
      res.status(400).json({ error: 'No file uploaded' });
      return;
    }

    const uploaded = files[fileKey];
    try {
      const buffer = fs.readFileSync(uploaded.filepath);
      const localUrl = process.env.LOCAL_BIOGPT_URL_PROCESS || 'http://127.0.0.1:8000/process_file';
      const fetchRes = await fetch(localUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': uploaded.originalFilename || uploaded.newFilename },
        body: buffer,
      });

      const j = await fetchRes.json();
      if (!fetchRes.ok) {
        res.status(fetchRes.status).json(j);
        return;
      }
      res.status(200).json(j);
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });
}
