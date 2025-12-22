/**
 * Node.js를 사용하여 Python 스크립트를 실행하는 유틸리티
 * Pyodide가 실패하거나 타임아웃될 때 사용하는 폴백 방법
 */

/**
 * Node.js를 통해 Python 스크립트를 실행하여 데이터를 분할합니다
 */
export async function splitDataWithNode(
    data: any[],
    trainSize: number,
    randomState: number,
    shuffle: boolean
): Promise<{ trainIndices: number[], testIndices: number[] }> {
    // Node.js 환경에서만 작동 (서버 사이드)
    if (typeof window !== 'undefined') {
        throw new Error('Node.js split_data는 서버 사이드에서만 작동합니다');
    }

    // Python 스크립트 실행은 서버 사이드에서 처리되어야 함
    // 현재는 브라우저 환경이므로 에러 발생
    throw new Error('Node.js 방법은 서버 사이드에서만 사용 가능합니다');
}




















































