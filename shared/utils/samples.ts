/**
 * Samples 폴더에서 샘플 파일을 로드하는 유틸리티 함수
 */

/**
 * Samples 폴더의 모든 샘플 파일 목록을 가져옵니다.
 * @param apiUrl API 엔드포인트 URL (예: "http://localhost:3002/api/samples/list")
 * @returns 샘플 파일 목록 배열
 */
export async function loadFolderSamples(apiUrl: string): Promise<Array<{ filename: string; name: string; data: any }>> {
  try {
    console.log(`Fetching samples from: ${apiUrl}`);
    const response = await fetch(apiUrl);
    console.log(`Response status: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      console.error(`Failed to fetch samples list: ${response.status} ${response.statusText}`, errorText);
      if (response.status === 0 || errorText.includes('Failed to fetch')) {
        console.error('서버가 실행 중이지 않거나 연결할 수 없습니다. 서버를 시작하세요: pnpm run server');
      }
      return [];
    }
    const samples = await response.json();
    console.log(`Received ${Array.isArray(samples) ? samples.length : 0} samples from server`);
    
    if (!Array.isArray(samples)) {
      console.error("Samples API returned non-array response:", samples);
      return [];
    }
    
    if (samples.length > 0) {
      console.log('Sample names:', samples.map((s: any) => s.name || s.filename));
    }
    
    return samples;
  } catch (error: any) {
    console.error("Error loading folder samples:", error);
    console.error("Error details:", error.message, error.stack);
    if (error.message && (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {
      console.error('서버 연결 실패. 서버를 시작하세요: pnpm run server 또는 pnpm run dev:full');
    }
    return [];
  }
}

/**
 * Samples 폴더에서 특정 샘플 파일을 로드합니다.
 * @param filename 샘플 파일 이름 (예: "StatModel.mla")
 * @param apiBaseUrl API 기본 URL (예: "http://localhost:3002/api/samples")
 * @returns 샘플 모델 데이터
 */
export async function loadSampleFromFolder(
  filename: string,
  apiBaseUrl: string
): Promise<any | null> {
  try {
    // URL 인코딩 처리 (공백, 특수문자 등)
    const encodedFilename = encodeURIComponent(filename);
    const url = `${apiBaseUrl}/${encodedFilename}`;
    
    console.log(`Loading sample from: ${url}`);
    const response = await fetch(url);
    
    if (!response.ok) {
      console.error(`Failed to load sample file: ${response.status} ${response.statusText}`);
      return null;
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Error loading sample file ${filename}:`, error);
    return null;
  }
}




