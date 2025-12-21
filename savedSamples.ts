// 공유 가능한 Samples 목록
// 이 파일은 커밋/푸시에 포함되어 모든 사용자가 공유할 수 있습니다.

import { ModuleType } from "./types";

export interface SavedSample {
  name: string;
  modules: Array<{
    type: ModuleType;
    position: { x: number; y: number };
    name: string;
    parameters?: Record<string, any>;
  }>;
  connections: Array<{
    fromModuleIndex: number;
    fromPort: string;
    toModuleIndex: number;
    toPort: string;
  }>;
}

export const SAVED_SAMPLES: SavedSample[] = [];
