/**
 * Node.js Express 서버 - SplitData API & Samples API
 * Pyodide가 타임아웃되거나 실패할 때 사용하는 백엔드 서버
 * Samples API도 함께 제공
 */

import express from 'express';
import { spawn } from 'child_process';
import { exec } from 'child_process';
import { promisify } from 'util';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// samples 폴더 경로
const SAMPLES_DIR = path.join(__dirname, '..', 'samples');

// samples 폴더가 없으면 생성
if (!fs.existsSync(SAMPLES_DIR)) {
  fs.mkdirSync(SAMPLES_DIR, { recursive: true });
}

app.post('/api/split-data', async (req, res) => {
    try {
        const { data, train_size, random_state, shuffle, stratify, stratify_column } = req.body;

        // Python 스크립트 실행
        const pythonScript = `
import sys
import json
import traceback
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    # sklearn의 train_test_split을 사용하여 데이터를 분할합니다.
    input_data = json.loads(sys.stdin.read())
    dataframe = pd.DataFrame(input_data['data'])
    
    # DataFrame 인덱스를 명시적으로 0부터 시작하도록 리셋
    dataframe.index = range(len(dataframe))
    
    # Parameters from UI
    p_train_size = float(input_data['train_size'])
    p_random_state = int(input_data['random_state'])
    p_shuffle = bool(input_data['shuffle'])
    p_stratify = bool(input_data.get('stratify', False))
    p_stratify_column = input_data.get('stratify_column', None)
    
    # Stratify 배열 준비
    stratify_array = None
    if p_stratify and p_stratify_column and p_stratify_column != 'None' and p_stratify_column is not None:
        if p_stratify_column not in dataframe.columns:
            raise ValueError(f"Stratify column '{p_stratify_column}' not found in DataFrame")
        stratify_array = dataframe[p_stratify_column]
    
    # 데이터 분할
    train_data, test_data = train_test_split(
        dataframe,
        train_size=p_train_size,
        random_state=p_random_state,
        shuffle=p_shuffle,
        stratify=stratify_array
    )
    
    result = {
        'train_indices': train_data.index.tolist(),
        'test_indices': test_data.index.tolist(),
        'train_count': len(train_data),
        'test_count': len(test_data)
    }
    
    print(json.dumps(result))
except Exception as e:
    error_info = {
        'error': True,
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_traceback': ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    }
    print(json.dumps(error_info), file=sys.stderr)
    sys.exit(1)
`;

        const pythonProcess = spawn('python', ['-c', pythonScript], {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        const inputData = JSON.stringify({
            data,
            train_size: parseFloat(train_size),
            random_state: parseInt(random_state),
            shuffle: shuffle === true || shuffle === 'True',
            stratify: stratify === true || stratify === 'True',
            stratify_column: stratify_column || null
        });

        let output = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        // Promise로 래핑하여 비동기 처리
        await new Promise<void>((resolve, reject) => {
            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    console.error('Python error:', error);
                    try {
                        // stderr에서 에러 정보 파싱 시도
                        const errorInfo = JSON.parse(error);
                        if (errorInfo.error) {
                            return reject(new Error(`Python execution failed: ${errorInfo.error_message || error}`));
                        }
                    } catch {
                        // JSON 파싱 실패 시 원본 에러 사용
                    }
                    return reject(new Error(`Python execution failed with code ${code}: ${error || output}`));
                }

                try {
                    const result = JSON.parse(output);
                    res.status(200).json(result);
                    resolve();
                } catch (e) {
                    console.error('JSON parse error:', e);
                    reject(new Error(`Failed to parse Python output: ${output}`));
                }
            });

            pythonProcess.on('error', (err) => {
                console.error('Python process error:', err);
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });

            pythonProcess.stdin.write(inputData);
            pythonProcess.stdin.end();
        });

    } catch (error) {
        console.error('API error:', error);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
});

// ============================================================================
// Samples API
// ============================================================================

// 샘플 파일 목록 가져오기
app.get('/api/samples/list', (req, res) => {
  try {
    if (!fs.existsSync(SAMPLES_DIR)) {
      console.log(`Samples directory does not exist: ${SAMPLES_DIR}`);
      return res.json([]);
    }

    const files = fs.readdirSync(SAMPLES_DIR);
    console.log(`Found ${files.length} files in samples directory:`, files);
    
    const sampleFiles = files
      .filter(file => {
        const isJson = file.endsWith('.json');
        const isIns = file.endsWith('.ins');
        const isMla = file.endsWith('.mla');
        if (isJson || isIns || isMla) {
          const ext = isIns ? '.ins' : (isMla ? '.mla' : '.json');
          console.log(`Processing file: ${file} (${ext})`);
          return true;
        }
        return false;
      })
      .map(file => {
        const filePath = path.join(SAMPLES_DIR, file);
        try {
          console.log(`Reading file: ${filePath}`);
          const content = fs.readFileSync(filePath, 'utf-8');
          
          if (!content || content.trim().length === 0) {
            console.warn(`File ${file} is empty`);
            return null;
          }
          
          const data = JSON.parse(content);
          console.log(`Parsed file ${file}, has modules: ${!!data.modules}, has connections: ${!!data.connections}, has projectName: ${!!data.projectName}`);
          
          // .ins 또는 .mla 파일 형식 (Save 버튼으로 저장한 파일) 변환
          if ((file.endsWith('.ins') || file.endsWith('.mla')) && data.modules && data.connections) {
            const ext = file.endsWith('.ins') ? '.ins' : '.mla';
            const projectName = data.projectName || file.replace(ext, '');
            console.log(`Converting ${ext} file: ${file} -> ${projectName}`);
            console.log(`  Modules count: ${data.modules.length}, Connections count: ${data.connections.length}`);
            
            const convertedData = {
              filename: file,
              name: projectName,
              data: {
                name: projectName,
                modules: data.modules.map((m) => ({
                  type: m.type,
                  position: m.position || { x: 0, y: 0 },
                  name: m.name || m.type,
                  parameters: m.parameters || {},
                })),
                connections: data.connections.map((c) => {
                  const fromIndex = data.modules.findIndex((m) => m.id === c.from.moduleId);
                  const toIndex = data.modules.findIndex((m) => m.id === c.to.moduleId);
                  if (fromIndex < 0 || toIndex < 0) {
                    console.warn(`Invalid connection in ${file}: fromIndex=${fromIndex}, toIndex=${toIndex}`);
                  }
                  return {
                    fromModuleIndex: fromIndex >= 0 ? fromIndex : -1,
                    fromPort: c.from.portName,
                    toModuleIndex: toIndex >= 0 ? toIndex : -1,
                    toPort: c.to.portName,
                  };
                }).filter((c) => c.fromModuleIndex >= 0 && c.toModuleIndex >= 0),
              }
            };
            console.log(`Successfully converted ${ext} file: ${file} -> ${convertedData.name} (${convertedData.data.modules.length} modules, ${convertedData.data.connections.length} connections)`);
            return convertedData;
          }
          
          // .json 파일 형식 (기존 samples 형식)
          const jsonData = {
            filename: file,
            name: data.name || file.replace('.json', '').replace('.ins', '').replace('.mla', ''),
            data: data
          };
          console.log(`Loaded .json file: ${file} -> ${jsonData.name}`);
          return jsonData;
        } catch (error) {
          console.error(`Error reading file ${file}:`, error.message);
          console.error(`Stack:`, error.stack);
          return null;
        }
      })
      .filter(file => file !== null);

    console.log(`Returning ${sampleFiles.length} sample files`);
    res.json(sampleFiles);
  } catch (error) {
    console.error('Error listing samples:', error);
    res.status(500).json({ error: 'Failed to list samples' });
  }
});

// 특정 샘플 파일 읽기
app.get('/api/samples/:filename', (req, res) => {
  try {
    // URL 디코딩 처리 (공백, 특수문자 등)
    const filename = decodeURIComponent(req.params.filename);
    const filePath = path.join(SAMPLES_DIR, filename);
    
    console.log(`Reading sample file: ${filename}`);
    console.log(`Full path: ${filePath}`);

    if (!fs.existsSync(filePath)) {
      console.error(`File not found: ${filePath}`);
      return res.status(404).json({ error: 'File not found' });
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(content);
    
    // .ins 또는 .mla 파일 형식 (Save 버튼으로 저장한 파일) 변환
    // projectName이 없어도 파일명에서 추출 가능하도록 수정
    if ((filename.endsWith('.ins') || filename.endsWith('.mla')) && data.modules && data.connections) {
      const ext = filename.endsWith('.ins') ? '.ins' : '.mla';
      console.log(`Converting ${ext} file: ${filename}`);
      const projectName = data.projectName || filename.replace(ext, '').trim();
      const convertedData = {
        name: projectName,
        modules: data.modules.map((m) => ({
          type: m.type,
          position: m.position,
          name: m.name,
          parameters: m.parameters || {},
        })),
        connections: data.connections.map((c) => {
          const fromIndex = data.modules.findIndex((m) => m.id === c.from.moduleId);
          const toIndex = data.modules.findIndex((m) => m.id === c.to.moduleId);
          return {
            fromModuleIndex: fromIndex >= 0 ? fromIndex : -1,
            fromPort: c.from.portName,
            toModuleIndex: toIndex >= 0 ? toIndex : -1,
            toPort: c.to.portName,
          };
        }).filter((c) => c.fromModuleIndex >= 0 && c.toModuleIndex >= 0),
      };
      console.log(`Converted ${convertedData.modules.length} modules and ${convertedData.connections.length} connections`);
      return res.json(convertedData);
    }
    
    // .json 파일 형식 (기존 samples 형식)
    console.log(`Returning .json file: ${filename}`);
    res.json(data);
  } catch (error) {
    console.error('Error reading sample:', error);
    res.status(500).json({ error: 'Failed to read sample' });
  }
});

// 샘플 파일 저장
app.post('/api/samples/save', (req, res) => {
  try {
    const { name, modules, connections } = req.body;

    if (!name || !modules || !connections) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // 파일명 생성 (특수문자 제거)
    const safeFilename = name.replace(/[^a-zA-Z0-9가-힣\s]/g, '_').replace(/\s+/g, '_') + '.json';
    const filePath = path.join(SAMPLES_DIR, safeFilename);

    const sampleData = {
      name,
      modules,
      connections
    };

    fs.writeFileSync(filePath, JSON.stringify(sampleData, null, 2), 'utf-8');
    res.json({ success: true, filename: safeFilename });
  } catch (error) {
    console.error('Error saving sample:', error);
    res.status(500).json({ error: 'Failed to save sample' });
  }
});

// 샘플 파일 삭제
app.delete('/api/samples/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    const filePath = path.join(SAMPLES_DIR, filename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    fs.unlinkSync(filePath);
    res.json({ success: true });
  } catch (error) {
    console.error('Error deleting sample:', error);
    res.status(500).json({ error: 'Failed to delete sample' });
  }
});

// PPT 생성 API
const execAsync = promisify(exec);
app.post('/api/generate-ppts', async (req, res) => {
  try {
    const { projectData } = req.body;

    if (!projectData || !projectData.modules) {
      return res.status(400).json({ error: 'Missing projectData or modules' });
    }

    // 임시 파일로 저장
    const tempDir = path.join(__dirname, '..', 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    const tempFile = path.join(tempDir, `project_${Date.now()}.json`);
    fs.writeFileSync(tempFile, JSON.stringify(projectData, null, 2), 'utf-8');

    // Python 스크립트 경로
    const scriptPath = path.join(__dirname, '..', 'scripts', 'generate_module_ppts.py');
    
    // Python 실행 명령어 (Windows와 Unix 모두 지원)
    let pythonCmd = 'python3';
    if (process.platform === 'win32') {
      pythonCmd = 'python';
    }
    
    // output_dir을 지정하지 않으면 Python 스크립트가 다운로드 폴더를 사용
    // 명시적으로 전달하지 않아도 Python 스크립트가 자동으로 다운로드 폴더를 찾음
    const command = `${pythonCmd} "${scriptPath}" "${tempFile}"`;

    console.log(`Executing: ${command}`);
    console.log(`Working directory: ${path.join(__dirname, '..')}`);
    
    let stdout = '';
    try {
      const result = await execAsync(command, {
        cwd: path.join(__dirname, '..'),
        maxBuffer: 10 * 1024 * 1024, // 10MB
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      });
      stdout = result.stdout;
      if (result.stderr && !result.stderr.includes('경고') && !result.stderr.includes('Warning')) {
        console.error('Python script stderr:', result.stderr);
      }
      
      console.log('Python script stdout:', stdout);
    } catch (execError) {
      // Python 명령어가 실패하면 py 시도 (Windows)
      if (process.platform === 'win32' && pythonCmd === 'python') {
        console.log('Python command failed, trying py...');
        try {
          const result = await execAsync(`py "${scriptPath}" "${tempFile}"`, {
            cwd: path.join(__dirname, '..'),
            maxBuffer: 10 * 1024 * 1024,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
          });
          stdout = result.stdout;
          
          if (result.stderr && !result.stderr.includes('경고') && !result.stderr.includes('Warning')) {
            console.error('Python script stderr:', result.stderr);
          }
          
          console.log('Python script stdout:', stdout);
        } catch (pyError) {
          throw new Error(`Python execution failed: ${execError.message}. Py launcher also failed: ${pyError.message}`);
        }
      } else {
        throw execError;
      }
    }

    // Python 스크립트 출력에서 파일 경로 추출
    let generatedFiles = [];
    let outputPath = null;
    
    if (stdout) {
      // "생성 완료:" 또는 "저장 위치:" 패턴으로 경로 추출
      const pathMatch = stdout.match(/(?:생성 완료|저장 위치):\s*(.+\.pptx|.+)/);
      if (pathMatch) {
        outputPath = pathMatch[1].trim();
        const filename = path.basename(outputPath);
        generatedFiles = [{
          filename: filename,
          filepath: outputPath,
          downloadPath: outputPath
        }];
        console.log(`PPT 파일이 저장되었습니다: ${outputPath}`);
      }
    }

    // 임시 파일 정리
    try {
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
    } catch (cleanupError) {
      console.warn('Failed to cleanup temp file:', cleanupError);
    }

    // stdout에서 저장 위치 정보도 추출
    let storageLocation = null;
    if (stdout) {
      const locationMatch = stdout.match(/저장 위치:\s*(.+)/);
      if (locationMatch) {
        storageLocation = locationMatch[1].trim();
      }
    }

    res.json({
      success: true,
      files: generatedFiles,
      message: stdout || 'PPT 생성 완료',
      downloadPath: outputPath,
      storageLocation: storageLocation || outputPath ? path.dirname(outputPath) : null
    });

  } catch (error) {
    console.error('PPT 생성 오류:', error);
    console.error('Error stack:', error.stack);
    res.status(500).json({
      success: false,
      error: error.message,
      details: error.stack,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// ============================================================================
// 서버 시작
// ============================================================================

app.listen(PORT, () => {
    console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
    console.log(`- SplitData API: http://localhost:${PORT}/api/split-data`);
    console.log(`- Samples API: http://localhost:${PORT}/api/samples/list`);
    console.log(`- PPT 생성 API: http://localhost:${PORT}/api/generate-ppts`);
});
