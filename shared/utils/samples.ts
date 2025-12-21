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
    const response = await fetch(apiUrl);
    if (!response.ok) {
      console.error(`Failed to fetch samples list: ${response.status} ${response.statusText}`);
      return [];
    }
    const samples = await response.json();
    return Array.isArray(samples) ? samples : [];
  } catch (error) {
    console.error("Error loading folder samples:", error);
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

