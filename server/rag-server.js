import express from 'express';
import multer from 'multer';
import path from 'path';
import { promises as fs } from 'fs';
import { GoogleGenAI } from '@google/genai';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { processPDF } from './pdf-processor.js';
import SimpleVectorStore from './vector-store.js';

// .env 파일 로드
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// CORS 설정
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Multer 설정 (PDF 업로드) - 개발자용, 일반 사용자는 ML pdf 폴더에 직접 넣기
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '..', 'ML pdf');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('PDF 파일만 업로드 가능합니다.'));
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

// Gemini AI 초기화
const geminiApiKey = process.env.GEMINI_API_KEY;
if (!geminiApiKey || geminiApiKey === 'your_gemini_api_key_here') {
  console.warn('⚠️  GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 설정해주세요.');
}
const genAI = new GoogleGenAI({ apiKey: geminiApiKey });

// Storage 폴더 생성
const storageDir = path.join(__dirname, '..', 'storage');
const pdfsDir = path.join(__dirname, '..', 'ML pdf'); // ML pdf 폴더 사용
const vectorsDir = path.join(storageDir, 'vectors');

fs.mkdir(pdfsDir, { recursive: true }).catch(console.error);
fs.mkdir(vectorsDir, { recursive: true }).catch(console.error);

// 벡터 스토어 초기화
const vectorStore = new SimpleVectorStore(vectorsDir);

// 서버 시작 시 PDF 폴더의 모든 PDF를 자동으로 스캔하고 벡터화
async function initializePDFs() {
  try {
    await vectorStore.load();
    
    // PDF 폴더의 모든 파일 스캔
    const files = await fs.readdir(pdfsDir);
    const pdfFiles = files.filter(file => file.endsWith('.pdf'));
    
    if (pdfFiles.length === 0) {
      console.log('PDF 폴더에 파일이 없습니다. PDF를 "ML pdf" 폴더에 넣어주세요.');
      return;
    }
    
    console.log(`${pdfFiles.length}개의 PDF 파일을 발견했습니다. 벡터화를 시작합니다...`);
    
    // 이미 벡터화된 파일 목록 확인
    const processedFiles = new Set(
      vectorStore.embeddings
        .map(doc => doc.metadata.fileName)
        .filter(Boolean)
    );
    
    let processedCount = 0;
    let skippedCount = 0;
    
    for (const fileName of pdfFiles) {
      // 이미 처리된 파일은 스킵
      if (processedFiles.has(fileName)) {
        skippedCount++;
        continue;
      }
      
      try {
        const filePath = path.join(pdfsDir, fileName);
        console.log(`처리 중: ${fileName}`);
        
        // PDF 처리 및 청킹
        const { chunks, metadata } = await processPDF(filePath);
        
        // 각 청크를 벡터 스토어에 추가
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          await vectorStore.addDocument(
            `${fileName}-chunk-${i}`,
            chunk.text,
            {
              fileName,
              chunkIndex: i,
              ...metadata,
            }
          );
        }
        
        await vectorStore.save();
        processedCount++;
        console.log(`✓ ${fileName} 처리 완료 (${chunks.length}개 청크)`);
      } catch (error) {
        console.error(`✗ ${fileName} 처리 실패:`, error.message);
      }
    }
    
    console.log(`\nPDF 초기화 완료:`);
    console.log(`  - 새로 처리된 파일: ${processedCount}개`);
    console.log(`  - 이미 처리된 파일: ${skippedCount}개`);
    console.log(`  - 총 벡터 수: ${vectorStore.embeddings.length}개\n`);
  } catch (error) {
    console.error('PDF 초기화 실패:', error);
  }
}

// 서버 시작 시 PDF 초기화
initializePDFs();

// ============================================================================
// PDF 업로드 및 벡터화
// ============================================================================
app.post('/api/rag/upload-pdf', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'PDF 파일이 필요합니다.' });
    }

    const filePath = req.file.path;
    const fileName = req.file.originalname;

    // PDF 처리 및 청킹
    const { chunks, metadata } = await processPDF(filePath);

    // 각 청크를 벡터 스토어에 추가
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      await vectorStore.addDocument(
        `${fileName}-chunk-${i}`,
        chunk.text,
        {
          fileName,
          chunkIndex: i,
          ...metadata,
        }
      );
    }

    await vectorStore.save();

    res.json({
      success: true,
      message: 'PDF가 성공적으로 업로드되고 벡터화되었습니다.',
      chunks: chunks.length,
      metadata,
    });
  } catch (error) {
    console.error('PDF 업로드 실패:', error);
    res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// RAG 질문 및 답변
// ============================================================================
app.post('/api/rag/ask', async (req, res) => {
  try {
    const { question, contextData, dataInfo } = req.body;

    if (!question) {
      return res.status(400).json({ error: '질문이 필요합니다.' });
    }

    // 1. 벡터 스토어에서 관련 문서 검색
    const relevantDocs = await vectorStore.search(question, 5);

    // 2. 컨텍스트 구성
    const context = relevantDocs
      .map(doc => `[출처: ${doc.metadata.fileName || '문서'}]\n${doc.text}`)
      .join('\n\n---\n\n');

    // 3. 사용자 데이터 컨텍스트 추가
    let dataContext = '';
    if (dataInfo) {
      dataContext = `\n\n[사용자 데이터 정보]\n- 컬럼 수: ${dataInfo.columnCount || 'N/A'}\n- 행 수: ${dataInfo.rowCount || 'N/A'}\n- 컬럼 목록: ${dataInfo.columns ? dataInfo.columns.join(', ') : 'N/A'}\n- 데이터 타입: ${dataInfo.dataTypes ? JSON.stringify(dataInfo.dataTypes, null, 2) : 'N/A'}`;
    }

    // 4. 프롬프트 구성
    const prompt = `당신은 데이터 분석 전문가입니다. 다음 문서들을 참고하여 질문에 답변해주세요.

[참고 문서]
${context}
${dataContext}

[질문]
${question}

[답변 지침]
1. 참고 문서의 내용을 기반으로 정확하고 상세하게 답변하세요.
2. 참고 문서에 없는 내용은 추측하지 마세요.
3. 가능하면 구체적인 예시나 단계를 포함하세요.
4. 한국어로 답변하세요.
5. 이 앱의 모듈을 사용해서 분석 예시를 보여주세요. 예를 들어 "Load Data → Select Data → Statistics → ..." 같은 순서도를 제시하세요.
6. 전처리(공백 처리, 정규화, 결측치 처리 등)에 대한 구체적인 제안도 포함해주세요.`;

    // 5. Gemini API 호출
    console.log('Gemini API 호출 시작...');
    console.log('프롬프트 길이:', prompt.length);
    console.log('참고 문서 수:', relevantDocs.length);
    
    let answer;
    try {
      const result = await genAI.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
      });
      answer = result.text;
      console.log('Gemini API 호출 성공');
    } catch (apiError) {
      console.error('Gemini API 호출 오류:', apiError);
      throw new Error(`Gemini API 호출 실패: ${apiError.message}`);
    }

    // 6. 응답 반환
    res.json({
      answer,
      sources: relevantDocs.map(doc => ({
        fileName: doc.metadata.fileName,
        chunkIndex: doc.metadata.chunkIndex,
        similarity: doc.similarity,
        preview: doc.text.substring(0, 200) + '...',
      })),
    });
  } catch (error) {
    console.error('RAG 질문 처리 실패:', error);
    console.error('오류 스택:', error.stack);
    res.status(500).json({ 
      error: error.message || '알 수 없는 오류가 발생했습니다.',
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// ============================================================================
// 업로드된 PDF 목록 조회
// ============================================================================
app.get('/api/rag/pdfs', async (req, res) => {
  try {
    // 폴더가 없으면 생성
    await fs.mkdir(pdfsDir, { recursive: true });
    const files = await fs.readdir(pdfsDir);
    const pdfs = files
      .filter(file => file.endsWith('.pdf'))
      .map(file => ({
        fileName: file,
        uploadDate: new Date().toISOString(), // 실제로는 파일 메타데이터에서 가져와야 함
      }));

    res.json({ pdfs });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// PDF 삭제
// ============================================================================
app.delete('/api/rag/pdfs/:fileName', async (req, res) => {
  try {
    const fileName = req.params.fileName;
    const filePath = path.join(__dirname, '..', 'ML pdf', fileName);
    
    await fs.unlink(filePath);
    
    // 벡터 스토어에서도 제거
    await vectorStore.removeDocumentsByFileName(fileName);
    await vectorStore.save();

    res.json({ success: true, message: 'PDF가 삭제되었습니다.' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.RAG_SERVER_PORT || 3002;

// 요청 로깅 미들웨어
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

app.listen(PORT, () => {
  console.log(`RAG 서버가 포트 ${PORT}에서 실행 중입니다.`);
  console.log(`API 엔드포인트:`);
  console.log(`  GET  /api/rag/pdfs`);
  console.log(`  POST /api/rag/upload-pdf`);
  console.log(`  POST /api/rag/ask`);
  console.log(`  DELETE /api/rag/pdfs/:fileName`);
});

export default app;

