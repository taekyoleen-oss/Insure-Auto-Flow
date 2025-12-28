import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class SimpleVectorStore {
  constructor(storagePath = './storage/vectors') {
    this.storagePath = storagePath;
    this.embeddings = []; // {id, text, vector, metadata}
  }

  /**
   * 간단한 텍스트 임베딩 (해시 기반)
   * 실제 프로덕션에서는 OpenAI Embedding API나 다른 embedding 서비스를 사용하는 것이 좋습니다.
   */
  simpleEmbedding(text) {
    const words = text.toLowerCase().split(/\s+/);
    const vector = new Array(384).fill(0);
    words.forEach((word) => {
      const hash = this.hashCode(word);
      vector[hash % 384] += 1;
    });
    // 정규화
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      return vector.map(val => val / magnitude);
    }
    return vector;
  }

  hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  /**
   * 코사인 유사도 계산
   */
  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;
    return dotProduct / denominator;
  }

  /**
   * 문서 추가
   */
  async addDocument(id, text, metadata = {}) {
    const vector = this.simpleEmbedding(text);
    this.embeddings.push({
      id,
      text,
      vector,
      metadata,
    });
    await this.save();
  }

  /**
   * 유사한 문서 검색
   */
  async search(query, topK = 5) {
    const queryVector = this.simpleEmbedding(query);
    
    const results = this.embeddings.map(doc => ({
      ...doc,
      similarity: this.cosineSimilarity(queryVector, doc.vector),
    }));
    
    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK)
      .filter(doc => doc.similarity > 0.1); // 최소 유사도 필터
  }

  /**
   * 벡터 스토어 저장
   */
  async save() {
    await fs.mkdir(this.storagePath, { recursive: true });
    await fs.writeFile(
      path.join(this.storagePath, 'vectors.json'),
      JSON.stringify(this.embeddings, null, 2)
    );
  }

  /**
   * 벡터 스토어 로드
   */
  async load() {
    try {
      const data = await fs.readFile(
        path.join(this.storagePath, 'vectors.json'),
        'utf-8'
      );
      this.embeddings = JSON.parse(data);
    } catch (error) {
      console.log('벡터 스토어가 없습니다. 새로 생성합니다.');
      this.embeddings = [];
    }
  }

  /**
   * 특정 파일명으로 문서 제거
   */
  async removeDocumentsByFileName(fileName) {
    this.embeddings = this.embeddings.filter(
      doc => doc.metadata.fileName !== fileName
    );
    await this.save();
  }
}

export default SimpleVectorStore;

