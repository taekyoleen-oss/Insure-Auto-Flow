import { CanvasModule, Connection } from '../types';
import { getModuleCode } from '../codeSnippets';

/**
 * 모듈 간 의존성을 기반으로 실행 순서를 결정합니다 (위상 정렬)
 */
function getExecutionOrder(
  modules: CanvasModule[],
  connections: Connection[]
): CanvasModule[] {
  const moduleMap = new Map<string, CanvasModule>();
  modules.forEach((m) => moduleMap.set(m.id, m));

  // 각 모듈의 의존성 계산
  const dependencies = new Map<string, Set<string>>();
  modules.forEach((m) => dependencies.set(m.id, new Set()));

  connections.forEach((conn) => {
    const fromModule = moduleMap.get(conn.from.moduleId);
    const toModule = moduleMap.get(conn.to.moduleId);
    if (fromModule && toModule) {
      const deps = dependencies.get(toModule.id);
      if (deps) {
        deps.add(fromModule.id);
      }
    }
  });

  // 위상 정렬
  const ordered: CanvasModule[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  function visit(moduleId: string) {
    if (visiting.has(moduleId)) {
      // 순환 참조 감지 (무시하고 계속 진행)
      return;
    }
    if (visited.has(moduleId)) {
      return;
    }

    visiting.add(moduleId);
    const deps = dependencies.get(moduleId);
    if (deps) {
      deps.forEach((depId) => visit(depId));
    }
    visiting.delete(moduleId);
    visited.add(moduleId);

    const module = moduleMap.get(moduleId);
    if (module) {
      ordered.push(module);
    }
  }

  modules.forEach((m) => visit(m.id));

  return ordered;
}

/**
 * 전체 파이프라인의 Python 코드를 생성합니다
 */
export function generateFullPipelineCode(
  modules: CanvasModule[],
  connections: Connection[]
): string {
  if (modules.length === 0) {
    return '# 파이프라인이 비어있습니다.';
  }

  // 실행 순서 결정
  const executionOrder = getExecutionOrder(modules, connections);

  // 코드 생성
  const codeLines: string[] = [];
  codeLines.push('# ============================================================================');
  codeLines.push('# 전체 파이프라인 코드');
  codeLines.push('# ============================================================================');
  codeLines.push('');
  codeLines.push('import pandas as pd');
  codeLines.push('import numpy as np');
  codeLines.push('');

  // 각 모듈의 코드 생성
  const variableMap = new Map<string, string>(); // moduleId -> 변수명

  executionOrder.forEach((module, index) => {
    codeLines.push(`# ============================================================================`);
    codeLines.push(`# ${index + 1}. ${module.name} (${module.type})`);
    codeLines.push(`# ============================================================================`);
    codeLines.push('');

    // 모듈 코드 가져오기
    let moduleCode = getModuleCode(module, modules, connections);

    // 변수명 매핑
    // 입력 변수명 찾기
    const inputConnections = connections.filter(
      (c) => c.to.moduleId === module.id
    );
    const inputVars: Record<string, string> = {};

    inputConnections.forEach((conn) => {
      const fromModuleId = conn.from.moduleId;
      const fromPort = conn.from.portName;
      const toPort = conn.to.portName;

      // 이전 모듈의 출력 변수명 찾기
      const prevVarName = variableMap.get(fromModuleId);
      if (prevVarName) {
        // 포트 타입에 따라 변수명 결정
        if (toPort === 'data_in') {
          inputVars['dataframe'] = prevVarName;
        } else if (toPort === 'model_in') {
          inputVars['model'] = prevVarName;
        } else if (toPort === 'handler_in') {
          inputVars['handler'] = prevVarName;
        } else if (toPort === 'dist_in') {
          inputVars['dist'] = prevVarName;
        } else if (toPort === 'curve_in') {
          inputVars['curve'] = prevVarName;
        } else if (toPort === 'contract_in') {
          inputVars['contract'] = prevVarName;
        }
      }
    });

    // 모듈 코드의 주석 처리된 부분을 실제 변수로 교체
    if (inputVars['dataframe']) {
      moduleCode = moduleCode.replace(
        /# Assuming 'dataframe' is passed from the previous step/g,
        `# Using data from previous step`
      );
      moduleCode = moduleCode.replace(
        /dataframe = /g,
        `${inputVars['dataframe']} = `
      );
    }

    // 출력 변수명 결정
    let outputVarName = '';
    if (module.outputs.length > 0) {
      const outputType = module.outputs[0].type;
      if (outputType === 'data') {
        outputVarName = `data_${module.id.slice(0, 8)}`;
      } else if (outputType === 'model') {
        outputVarName = `model_${module.id.slice(0, 8)}`;
      } else if (outputType === 'handler') {
        outputVarName = `handler_${module.id.slice(0, 8)}`;
      } else if (outputType === 'evaluation') {
        outputVarName = `eval_${module.id.slice(0, 8)}`;
      } else {
        outputVarName = `result_${module.id.slice(0, 8)}`;
      }
    }

    // 모듈 코드에 출력 변수명 추가
    if (outputVarName) {
      // 주석 처리된 실행 부분을 실제 실행 코드로 변환
      moduleCode = moduleCode.replace(
        /# Execution\n# (.+)/g,
        `# Execution\n${outputVarName} = $1`
      );

      // 결과 할당이 없는 경우 추가
      if (!moduleCode.includes('=') || moduleCode.match(/^[^=]*$/)) {
        // 모듈 타입에 따라 적절한 출력 생성
        if (module.type === 'LoadData') {
          moduleCode += `\n${outputVarName} = dataframe`;
        } else if (module.type === 'SelectData') {
          moduleCode += `\n${outputVarName} = selected_data`;
        } else if (module.type === 'HandleMissingValues') {
          moduleCode += `\n${outputVarName} = cleaned_data`;
        } else if (module.type === 'EncodeCategorical') {
          moduleCode += `\n${outputVarName} = encoded_data`;
        } else if (module.type === 'NormalizeData') {
          moduleCode += `\n${outputVarName} = normalized_data`;
        } else if (module.type === 'TransformData') {
          moduleCode += `\n${outputVarName} = transformed_data`;
        } else if (module.type === 'TransitionData') {
          moduleCode += `\n${outputVarName} = transformed_data`;
        } else if (module.type === 'ResampleData') {
          moduleCode += `\n${outputVarName} = resampled_data`;
        } else if (module.type === 'SplitData') {
          moduleCode += `\n${outputVarName}_train = train_data\n${outputVarName}_test = test_data`;
        } else if (module.type === 'TrainModel') {
          moduleCode += `\n${outputVarName} = trained_model`;
        } else if (module.type === 'ScoreModel') {
          moduleCode += `\n${outputVarName} = scored_data`;
        } else if (module.type === 'EvaluateModel') {
          moduleCode += `\n${outputVarName} = evaluation_metrics`;
        } else if (module.type === 'ResultModel') {
          moduleCode += `\n${outputVarName} = model_results`;
        } else if (module.type === 'PredictModel') {
          moduleCode += `\n${outputVarName} = predicted_data`;
        }
      }
    }

    codeLines.push(moduleCode);
    codeLines.push('');

    // 변수명 저장
    if (outputVarName) {
      variableMap.set(module.id, outputVarName);
    }
  });

  codeLines.push('# ============================================================================');
  codeLines.push('# 파이프라인 실행 완료');
  codeLines.push('# ============================================================================');

  return codeLines.join('\n');
}
