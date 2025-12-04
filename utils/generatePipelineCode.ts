import { CanvasModule, Connection, ModuleType, ModuleStatus } from '../types';
import { getModuleCode } from '../codeSnippets';

/**
 * 위상 정렬을 사용하여 모듈의 실행 순서를 결정합니다.
 */
function getExecutionOrder(modules: CanvasModule[], connections: Connection[]): CanvasModule[] {
  // 인접 리스트와 진입 차수 계산
  const adj: Record<string, string[]> = {};
  const inDegree: Record<string, number> = {};
  
  modules.forEach(m => {
    adj[m.id] = [];
    inDegree[m.id] = 0;
  });

  connections.forEach(conn => {
    if (adj[conn.from.moduleId] && inDegree[conn.to.moduleId] !== undefined) {
      adj[conn.from.moduleId].push(conn.to.moduleId);
      inDegree[conn.to.moduleId]++;
    }
  });

  // 위상 정렬
  const queue = modules.filter(m => inDegree[m.id] === 0).map(m => m.id);
  const sortedModuleIds: string[] = [];
  const tempInDegree = { ...inDegree };
  
  while (queue.length > 0) {
    const u = queue.shift()!;
    sortedModuleIds.push(u);
    (adj[u] || []).forEach(v => {
      tempInDegree[v]--;
      if (tempInDegree[v] === 0) {
        queue.push(v);
      }
    });
  }

  // 순서대로 모듈 반환
  const moduleMap = new Map(modules.map(m => [m.id, m]));
  const sortedModules = sortedModuleIds.map(id => moduleMap.get(id)!);
  
  // 연결되지 않은 모듈 추가
  const connectedIds = new Set(sortedModuleIds);
  modules.forEach(m => {
    if (!connectedIds.has(m.id)) {
      sortedModules.push(m);
    }
  });

  return sortedModules;
}

/**
 * 모듈 타입에 따라 변수명을 생성합니다.
 */
function generateVariableName(module: CanvasModule, index: number): string {
  const typeMap: Record<string, string> = {
    [ModuleType.LoadData]: 'data',
    [ModuleType.SplitData]: 'split',
    [ModuleType.TrainModel]: 'model',
    [ModuleType.ScoreModel]: 'scored',
    [ModuleType.EvaluateModel]: 'evaluation',
  };
  
  const baseName = typeMap[module.type] || `module${index}`;
  return baseName;
}

/**
 * 코드에서 print 문, 함수 정의, 불필요한 주석을 제거합니다.
 */
function cleanCode(code: string): string {
  const lines = code.split('\n');
  const cleaned: string[] = [];
  let inFunction = false;
  
  for (const line of lines) {
    const trimmed = line.trim();
    
    // 함수 정의 시작/끝 추적
    if (trimmed.startsWith('def ') || trimmed.startsWith('async def ')) {
      inFunction = true;
      continue;
    }
    
    if (inFunction && (trimmed === '' || !trimmed.startsWith('#'))) {
      if (trimmed && !trimmed.startsWith('def ') && !trimmed.startsWith('async def ')) {
        inFunction = false;
      } else {
        continue;
      }
    }
    
    // print 문 제거
    if (trimmed.startsWith('print(') || trimmed.startsWith('print ')) {
      continue;
    }
    
    // matplotlib, seaborn 시각화 코드 제거
    if (trimmed.includes('plt.') || trimmed.includes('sns.') || trimmed.includes('matplotlib')) {
      continue;
    }
    
    // 불필요한 주석 제거 (단, 중요한 주석은 유지)
    if (trimmed.startsWith('#') && (
      trimmed.includes('Parameters') || 
      trimmed.includes('Execution') ||
      trimmed.includes('Note:') ||
      trimmed.length < 10
    )) {
      continue;
    }
    
    cleaned.push(line);
  }
  
  return cleaned.join('\n');
}

/**
 * 전체 파이프라인을 하나의 파이썬 코드로 생성합니다.
 * 실행된 모듈(status === Success)만 포함합니다.
 */
export function generateFullPipelineCode(
  modules: CanvasModule[],
  connections: Connection[]
): string {
  if (modules.length === 0) {
    return '# No modules in the pipeline.';
  }

  // 실행된 모듈만 필터링
  const executedModules = modules.filter(m => m.status === ModuleStatus.Success);
  
  if (executedModules.length === 0) {
    return '# No executed modules found. Please run modules first.';
  }

  // 실행 순서 결정 (실행된 모듈만 사용)
  const executionOrder = getExecutionOrder(executedModules, connections);
  
  // 실행된 모듈만 변수명 매핑에 포함
  const moduleVariableMap = new Map<string, string>();
  executionOrder.forEach((module, index) => {
    if (module.status === ModuleStatus.Success) {
      moduleVariableMap.set(module.id, generateVariableName(module, index));
    }
  });

  // 전체 코드 생성 (간소화된 헤더)
  let fullCode = `# Pipeline Code - Data Input/Output Only\n`;
  fullCode += `# Generated from executed modules only\n\n`;

  // 필요한 import 문 수집
  const imports = new Set<string>();
  imports.add('import pandas as pd');
  imports.add('import numpy as np');
  
  executionOrder.forEach(module => {
    if (module.status !== ModuleStatus.Success) return;
    
    const code = getModuleCode(module);
    // 간단한 import 추출 (실제로는 더 정교한 파싱이 필요할 수 있음)
    if (code.includes('import pandas')) imports.add('import pandas as pd');
    if (code.includes('import numpy')) imports.add('import numpy as np');
    if (code.includes('from sklearn')) imports.add('from sklearn import *');
    if (code.includes('from statsmodels')) imports.add('from statsmodels import *');
    if (code.includes('from scipy')) imports.add('from scipy import *');
    if (code.includes('from imblearn')) imports.add('from imblearn import *');
    if (code.includes('from io import StringIO')) imports.add('from io import StringIO');
  });

  // import 문 추가
  Array.from(imports).sort().forEach(imp => {
    fullCode += `${imp}\n`;
  });
  fullCode += '\n';

  // 모델 정의 모듈 타입 정의
  const MODEL_DEFINITION_TYPES = [
    ModuleType.LinearRegression,
    ModuleType.LogisticRegression,
    ModuleType.PoissonRegression,
    ModuleType.NegativeBinomialRegression,
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.SVM,
    ModuleType.LinearDiscriminantAnalysis,
    ModuleType.NaiveBayes,
    ModuleType.KNN,
  ];

  // 각 모듈의 코드를 순서대로 추가 (실행된 모듈만)
  let stepIndex = 0;
  const processedModuleIds = new Set<string>(); // 이미 처리된 모듈 ID 추적
  
  executionOrder.forEach((module) => {
    // 실행되지 않은 모듈은 건너뛰기
    if (module.status !== ModuleStatus.Success) {
      return;
    }
    
    // 모델 정의 모듈은 TrainModel에서 처리하므로 여기서는 건너뛰기
    if (MODEL_DEFINITION_TYPES.includes(module.type)) {
      return;
    }
    
    stepIndex++;
    const varName = moduleVariableMap.get(module.id)!;
    const moduleCode = getModuleCode(module);
    
    // TrainModel 모듈인 경우, 연결된 모델 정의 모듈의 코드를 먼저 포함
    if (module.type === ModuleType.TrainModel) {
      // model_in 포트로 연결된 모델 정의 모듈 찾기
      const modelInputConn = connections.find(c => 
        c.to.moduleId === module.id && 
        c.to.portName === 'model_in' &&
        executedModules.some(m => m.id === c.from.moduleId && m.status === ModuleStatus.Success)
      );
      
      if (modelInputConn) {
        const modelDefinitionModule = executedModules.find(m => m.id === modelInputConn.from.moduleId);
        if (modelDefinitionModule && modelDefinitionModule.status === ModuleStatus.Success) {
          if (MODEL_DEFINITION_TYPES.includes(modelDefinitionModule.type)) {
            // 모델 정의 모듈의 코드를 TrainModel 바로 앞에 추가
            const modelDefVarName = moduleVariableMap.get(modelDefinitionModule.id)!;
            const modelDefCode = getModuleCode(modelDefinitionModule);
            let processedModelDefCode = cleanCode(modelDefCode);
            
            // 모델 정의 모듈의 출력 변수명 치환
            processedModelDefCode = processedModelDefCode.replace(/\bmodel\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\bdecision_tree_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\brandom_forest_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\bsvm_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\bknn_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\bnaive_bayes_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\blda_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\bpoisson_model\b/g, modelDefVarName);
            processedModelDefCode = processedModelDefCode.replace(/\bnegative_binomial_model\b/g, modelDefVarName);
            
            // 변수 할당이 없는 경우 추가
            if (!processedModelDefCode.match(new RegExp(`${modelDefVarName}\\s*=`))) {
              // 함수 호출이 있는 경우 찾아서 변수에 할당
              const functionCallMatch = processedModelDefCode.match(/(create_\w+\([^)]*\))/);
              if (functionCallMatch) {
                processedModelDefCode += `\n${modelDefVarName} = ${functionCallMatch[1]}`;
              } else {
                processedModelDefCode += `\n${modelDefVarName} = model`;
              }
            }
            
            // 모델 정의 모듈 구분선 및 코드 추가 (TrainModel 바로 앞)
            fullCode += `\n${'='.repeat(60)}\n`;
            fullCode += `# Model Definition: ${modelDefinitionModule.name} (${modelDefinitionModule.type})\n`;
            fullCode += `${'='.repeat(60)}\n\n`;
            fullCode += processedModelDefCode;
            fullCode += '\n\n';
            
            // 처리된 모듈로 표시
            processedModuleIds.add(modelDefinitionModule.id);
          }
        }
      }
    }

    // 모듈 구분선 및 이름 추가
    fullCode += `\n${'='.repeat(60)}\n`;
    fullCode += `# Module: ${module.name} (${module.type})\n`;
    fullCode += `${'='.repeat(60)}\n\n`;

    // 이전 모듈의 출력을 현재 모듈의 입력으로 연결
    const inputConnections = connections.filter(c => 
      c.to.moduleId === module.id && 
      executedModules.some(m => m.id === c.from.moduleId && m.status === ModuleStatus.Success)
    );
    
    // 입력 연결 처리 - 이전 모듈의 출력 변수를 현재 모듈의 입력 변수로 할당
    if (inputConnections.length > 0 && module.type !== ModuleType.LoadData && module.type !== ModuleType.XolLoading) {
      inputConnections.forEach(conn => {
        const fromModule = executedModules.find(m => m.id === conn.from.moduleId);
        if (fromModule && fromModule.status === ModuleStatus.Success) {
          const fromModuleVar = moduleVariableMap.get(conn.from.moduleId);
          if (fromModuleVar) {
            // 포트 타입에 따라 변수명 결정
            const outputPort = fromModule.outputs.find(p => p.name === conn.from.portName);
            const inputPort = module.inputs.find(p => p.name === conn.to.portName);
            
            if (outputPort?.type === 'data' && inputPort?.type === 'data') {
              // 데이터 포트 연결 - 이전 모듈의 출력을 현재 모듈의 입력으로
              if (module.type === ModuleType.SplitData) {
                // SplitData는 특별 처리 불필요 (모듈 코드에서 처리)
              } else {
                fullCode += `# Input from ${fromModule.name}\n`;
                fullCode += `dataframe = ${fromModuleVar}\n\n`;
              }
            } else if (outputPort?.type === 'handler' && inputPort?.type === 'handler') {
              fullCode += `# Handler from ${fromModule.name}\n`;
              fullCode += `handler = ${fromModuleVar}_handler\n\n`;
            } else if (outputPort?.type === 'model' && inputPort?.type === 'model') {
              fullCode += `# Model from ${fromModule.name}\n`;
              fullCode += `model = ${fromModuleVar}\n\n`;
            }
          }
        }
      });
    }

    // 각 모듈의 속성에 있는 코드를 순서대로 나열 (getModuleCode로 가져온 코드)
    // outputData를 사용하지 않고 모듈 코드를 그대로 사용
    let processedCode = moduleCode;
    
    // 코드 정리 (print 문, 불필요한 주석 제거)
    processedCode = cleanCode(processedCode);
    
    // LoadData/XolLoading의 경우 fileContent 처리
    if (module.type === ModuleType.LoadData || module.type === ModuleType.XolLoading) {
      const fileContent = module.parameters.fileContent as string | undefined;
      const source = module.parameters.source as string | undefined;
      
      if (fileContent) {
        // fileContent가 있으면 파일 이름으로 대체
        // csv_data 부분을 제거하고 file_path를 사용하도록 수정
        const fileName = source || 'data.csv';
        
        // csv_data = """...""" 부분을 찾아서 제거
        processedCode = processedCode.replace(
          /csv_data\s*=\s*"""[^"]*"""/gs,
          ''
        );
        processedCode = processedCode.replace(
          /csv_data\s*=\s*'''[^']*'''/gs,
          ''
        );
        
        // file_path를 파일 이름으로 설정
        processedCode = processedCode.replace(
          /file_path\s*=\s*[^\n]+/,
          `file_path = "${fileName}"`
        );
        
        // StringIO 사용 부분을 일반 file_path로 변경
        processedCode = processedCode.replace(
          /pd\.read_csv\(StringIO\(csv_data\)\)/g,
          `pd.read_csv(file_path)`
        );
        processedCode = processedCode.replace(
          /pd\.read_csv\(StringIO\([^)]+\)\)/g,
          `pd.read_csv(file_path)`
        );
        
        // StringIO import 제거 (더 이상 필요 없음)
        processedCode = processedCode.replace(
          /from io import StringIO\n?/g,
          ''
        );
        processedCode = processedCode.replace(
          /import StringIO\n?/g,
          ''
        );
      }
    }
    
    // 출력 변수명 치환 - 각 모듈의 출력을 varName으로 설정
    if (module.outputs.length > 0) {
      const outputPort = module.outputs[0];
      if (outputPort.type === 'data') {
        // dataframe 변수를 varName으로 치환
        processedCode = processedCode.replace(/\bdataframe\b/g, varName);
        processedCode = processedCode.replace(/\bdf\b/g, varName);
        processedCode = processedCode.replace(/\bselected_df\b/g, varName);
        processedCode = processedCode.replace(/\bdf_processed\b/g, varName);
        processedCode = processedCode.replace(/\bdf_normalized\b/g, varName);
        processedCode = processedCode.replace(/\bdf_transformed\b/g, varName);
        processedCode = processedCode.replace(/\bdf_result\b/g, varName);
        processedCode = processedCode.replace(/\bdf_encoded\b/g, varName);
        processedCode = processedCode.replace(/\bresampled_df\b/g, varName);
        processedCode = processedCode.replace(/\bfiltered_df\b/g, varName);
        processedCode = processedCode.replace(/\bceded_df\b/g, varName);
        processedCode = processedCode.replace(/\blarge_claims_df\b/g, varName);
        processedCode = processedCode.replace(/\bxol_dataframe\b/g, varName);
        
        // 변수 할당이 없는 경우 추가 (모듈 코드에 이미 있는 경우는 그대로 사용)
        if (!processedCode.match(new RegExp(`${varName}\\s*=`))) {
          // 모듈 코드의 마지막 실행 결과를 varName에 할당
          const lines = processedCode.split('\n').reverse();
          const lastExecutableLine = lines.find(line => {
            const trimmed = line.trim();
            return trimmed && 
                   !trimmed.startsWith('#') && 
                   !trimmed.startsWith('import') && 
                   !trimmed.startsWith('from') &&
                   !trimmed.match(/^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*$/); // 빈 할당 제외
          });
          if (lastExecutableLine && !lastExecutableLine.includes('=')) {
            processedCode += `\n${varName} = ${lastExecutableLine.trim()}`;
          } else if (!processedCode.match(new RegExp(`${varName}\\s*=`))) {
            // 기본값으로 dataframe 할당
            processedCode += `\n${varName} = dataframe`;
          }
        }
      } else if (outputPort.type === 'handler') {
        processedCode = processedCode.replace(/\bhandler\b/g, `${varName}_handler`);
        if (!processedCode.match(new RegExp(`${varName}_handler\\s*=`))) {
          processedCode += `\n${varName}_handler = handler`;
        }
      } else if (outputPort.type === 'model') {
        processedCode = processedCode.replace(/\bmodel\b/g, varName);
        processedCode = processedCode.replace(/\btrained_model\b/g, varName);
        if (!processedCode.match(new RegExp(`${varName}\\s*=`))) {
          processedCode += `\n${varName} = model`;
        }
      }
    }

    // 특수 케이스 처리
    if (module.type === ModuleType.SplitData) {
      processedCode = processedCode.replace(/train_data, test_data = train_test_split/m, `${varName}_train, ${varName}_test = train_test_split`);
    } else if (module.type === ModuleType.TrainModel) {
      processedCode = processedCode.replace(/trained_model = model.fit/m, `${varName} = model.fit`);
    } else if (module.type === ModuleType.ScoreModel) {
      processedCode = processedCode.replace(/scored_data = second_data.copy/m, `${varName} = second_data.copy`);
      // test_data를 찾아서 연결된 변수로 치환
      const testDataConn = connections.find(c => 
        c.to.moduleId === module.id && c.to.portName === 'data_in'
      );
      if (testDataConn) {
        const testDataModule = executedModules.find(m => m.id === testDataConn.from.moduleId);
        if (testDataModule && testDataModule.status === ModuleStatus.Success && testDataModule.type === ModuleType.SplitData) {
          const testDataVar = moduleVariableMap.get(testDataConn.from.moduleId);
          if (testDataVar) {
            processedCode = processedCode.replace(/second_data/g, `${testDataVar}_test`);
          }
        }
      }
    } else if (module.type === ModuleType.EvaluateModel) {
      processedCode = processedCode.replace(/scored_data/g, varName);
      // scored_data를 찾아서 연결된 변수로 치환
      const scoredDataConn = connections.find(c => 
        c.to.moduleId === module.id && c.to.portName === 'data_in'
      );
      if (scoredDataConn) {
        const scoredDataVar = moduleVariableMap.get(scoredDataConn.from.moduleId);
        if (scoredDataVar) {
          processedCode = processedCode.replace(/scored_data/g, scoredDataVar);
        }
      }
    }

    // 빈 줄 정리
    processedCode = processedCode.replace(/\n{3,}/g, '\n\n').trim();
    
    if (processedCode) {
      fullCode += processedCode;
      fullCode += '\n';
    }
    
    // 처리된 모듈로 표시
    processedModuleIds.add(module.id);
  });

  // 최종 결과 출력
  fullCode += `\n${'='.repeat(60)}\n`;
  fullCode += `# Pipeline execution complete\n`;
  fullCode += `${'='.repeat(60)}\n`;

  return fullCode;
}
