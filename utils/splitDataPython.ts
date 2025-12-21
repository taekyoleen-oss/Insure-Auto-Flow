/**
 * Python의 sklearn.train_test_split과 동일한 결과를 내기 위한 유틸리티
 * 
 * 참고: Python의 sklearn.train_test_split은 numpy의 Mersenne Twister를 사용합니다.
 * JavaScript에서 완전히 재현하기는 어렵지만, 최대한 유사하게 구현합니다.
 * 
 * 정확한 결과를 위해서는 Python 백엔드 API를 호출하는 것을 권장합니다.
 */

/**
 * Python의 numpy.random과 유사한 결과를 내기 위한 시드 기반 랜덤 생성기
 * Mersenne Twister를 완전히 재현하기는 어렵지만, 최대한 유사하게 구현
 */
class PythonLikeRandom {
    private seed: number;
    private mt: number[] = [];
    private index: number = 0;
    private readonly N = 624;
    private readonly M = 397;
    private readonly MATRIX_A = 0x9908b0df;
    private readonly UPPER_MASK = 0x80000000;
    private readonly LOWER_MASK = 0x7fffffff;

    constructor(seed: number) {
        this.seed = seed;
        this.mt[0] = seed >>> 0;
        for (let i = 1; i < this.N; i++) {
            const s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
            this.mt[i] = (((((s & 0xffff0000) >>> 16) * 1812433253) << 16) + (s & 0x0000ffff) * 1812433253) + i;
            this.mt[i] = this.mt[i] >>> 0;
        }
        this.index = 0;
    }

    private generateNumbers() {
        for (let i = 0; i < this.N; i++) {
            const y = (this.mt[i] & this.UPPER_MASK) + (this.mt[(i + 1) % this.N] & this.LOWER_MASK);
            this.mt[i] = this.mt[(i + this.M) % this.N] ^ (y >>> 1);
            if (y % 2 !== 0) {
                this.mt[i] = this.mt[i] ^ this.MATRIX_A;
            }
        }
    }

    random(): number {
        if (this.index === 0) {
            this.generateNumbers();
        }

        let y = this.mt[this.index];
        y = y ^ (y >>> 11);
        y = y ^ ((y << 7) & 0x9d2c5680);
        y = y ^ ((y << 15) & 0xefc60000);
        y = y ^ (y >>> 18);

        this.index = (this.index + 1) % this.N;
        return (y >>> 0) / 4294967296.0;
    }

    randint(min: number, max: number): number {
        return Math.floor(this.random() * (max - min)) + min;
    }
}

/**
 * Python의 sklearn.train_test_split과 동일한 방식으로 데이터를 분할합니다.
 * 
 * @param data 데이터 배열
 * @param trainSize 훈련 세트 비율 (0.0 ~ 1.0)
 * @param randomState 랜덤 시드
 * @param shuffle 셔플 여부
 * @returns [trainIndices, testIndices]
 */
export function splitDataLikePython(
    dataLength: number,
    trainSize: number,
    randomState: number,
    shuffle: boolean
): [number[], number[]] {
    const indices = Array.from({ length: dataLength }, (_, i) => i);

    if (shuffle) {
        // Python의 numpy.random과 유사한 랜덤 생성기 사용
        const rng = new PythonLikeRandom(randomState);
        
        // Fisher-Yates shuffle with Python-like random
        for (let i = indices.length - 1; i > 0; i--) {
            const j = rng.randint(0, i + 1);
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
    }

    const trainCount = Math.floor(dataLength * trainSize);
    const trainIndices = indices.slice(0, trainCount);
    const testIndices = indices.slice(trainCount);

    return [trainIndices, testIndices];
}



















































