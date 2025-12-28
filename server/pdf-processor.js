import { createRequire } from 'module';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const pdf = require('pdf-parse');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * PDF 파일을 텍스트로 변환하고 청킹
 */
async function processPDF(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const pdfData = await pdf(dataBuffer);
    
    // 텍스트 추출
    const text = pdfData.text;
    
    // 청킹 (500자 단위로 분할, 겹치는 부분 포함)
    const chunks = chunkText(text, 500, 100);
    
    return {
      text,
      chunks,
      pageCount: pdfData.numpages,
      metadata: {
        title: pdfData.info?.Title || path.basename(filePath),
        author: pdfData.info?.Author || 'Unknown',
      }
    };
  } catch (error) {
    throw new Error(`PDF 처리 실패: ${error.message}`);
  }
}

/**
 * 텍스트를 청크로 분할
 */
function chunkText(text, chunkSize = 500, overlap = 100) {
  const chunks = [];
  let start = 0;
  
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end).trim();
    
    if (chunk.length > 0) {
      chunks.push({
        text: chunk,
        start,
        end,
      });
    }
    
    start = end - overlap; // 겹치는 부분
  }
  
  return chunks;
}

export { processPDF, chunkText };

