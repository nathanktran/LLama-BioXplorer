const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf'); // Correct import path

const app = express();
const upload = multer({ dest: path.join(__dirname, '..', 'uploads/') });

app.use(cors({
  origin: '*', // Allow all origins
}));

app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    const filePath = path.join(__dirname, '..', 'uploads', req.file.filename);
    const dataBuffer = fs.readFileSync(filePath);

    const loadingTask = pdfjsLib.getDocument({ data: dataBuffer });
    const pdfDoc = await loadingTask.promise;
    const firstPage = await pdfDoc.getPage(1);
    const textContent = await firstPage.getTextContent();

    let title = '';
    let maxFontSize = 0;    

    textContent.items.forEach((item) => {
      const fontSize = item.transform[0];
    
      if (fontSize > maxFontSize) {
        maxFontSize = fontSize;
        title = item.str;  // Start a new title
      } else if (fontSize === maxFontSize) {
        title += ` ${item.str}`;
      }
    });
    title = title.trim();

    const text = textContent.items.map(item => item.str).join('\n');
    const abstractMatch = text.match(/ABSTRACT\s*\n(.*(?:\n(?!KEYWORDS|INTRODUCTION).*)*)/i);
    const abstract = abstractMatch ? abstractMatch[1].trim() : 'Abstract not found';

    fs.unlinkSync(filePath);

    res.json({ title, abstract });

  } catch (error) {
    console.error('Error processing PDF:', error);
    res.status(500).send('Error processing PDF');
  }
});

module.exports = app;
