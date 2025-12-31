import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.SAMPLES_SERVER_PORT || 3003;

app.use(cors());
app.use(express.json());

// samples 폴더 경로
const SAMPLES_DIR = path.join(__dirname, '..', 'samples');

// samples 폴더가 없으면 생성
if (!fs.existsSync(SAMPLES_DIR)) {
  fs.mkdirSync(SAMPLES_DIR, { recursive: true });
}

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

    // 샘플 목록 정렬: 특정 순서를 지정하고 나머지는 알파벳 순으로 정렬
    const sortOrder = [
      'Linear Regression',
      'Logistic Regression',
      'Decision Tree',
      'Random Forest',
      'Neural Network',
      'SVM',
      'KNN',
      'Naive Bayes',
      'LDA',
      'GLM Model',
      'Stat Model'
    ];

    sampleFiles.sort((a, b) => {
      const nameA = a.name || '';
      const nameB = b.name || '';
      
      const indexA = sortOrder.findIndex(order => nameA.includes(order));
      const indexB = sortOrder.findIndex(order => nameB.includes(order));
      
      // 둘 다 순서에 있으면 순서대로 정렬
      if (indexA !== -1 && indexB !== -1) {
        return indexA - indexB;
      }
      // A만 순서에 있으면 A를 앞으로
      if (indexA !== -1) {
        return -1;
      }
      // B만 순서에 있으면 B를 앞으로
      if (indexB !== -1) {
        return 1;
      }
      // 둘 다 순서에 없으면 알파벳 순으로 정렬
      return nameA.localeCompare(nameB);
    });

    console.log(`Returning ${sampleFiles.length} sample files (sorted)`);
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
    // output_dir을 지정하지 않으면 다운로드 폴더에 저장됨
    let pythonCmd = 'python3';
    if (process.platform === 'win32') {
      pythonCmd = 'python';
    }
    
    // output_dir을 지정하지 않으면 다운로드 폴더에 저장
    const command = `${pythonCmd} "${scriptPath}" "${tempFile}"`;

    console.log(`Executing: ${command}`);
    console.log(`Working directory: ${path.join(__dirname, '..')}`);
    
    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: path.join(__dirname, '..'),
        maxBuffer: 10 * 1024 * 1024, // 10MB
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      });

      if (stderr && !stderr.includes('경고') && !stderr.includes('Warning')) {
        console.error('Python script stderr:', stderr);
      }
      
      console.log('Python script stdout:', stdout);
    } catch (execError) {
      // Python 명령어가 실패하면 py 시도 (Windows)
      if (process.platform === 'win32' && pythonCmd === 'python') {
        console.log('Python command failed, trying py...');
        try {
          const { stdout, stderr } = await execAsync(`py "${scriptPath}" "${tempFile}"`, {
            cwd: path.join(__dirname, '..'),
            maxBuffer: 10 * 1024 * 1024,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
          });
          
          if (stderr && !stderr.includes('경고') && !stderr.includes('Warning')) {
            console.error('Python script stderr:', stderr);
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
    // stdout에서 "생성 완료: [경로]" 형식으로 출력됨
    let generatedFiles = [];
    let outputPath = null;
    
    if (stdout) {
      // stdout에서 파일 경로 추출
      const pathMatch = stdout.match(/생성 완료:\s*(.+\.pptx)/);
      if (pathMatch) {
        outputPath = pathMatch[1].trim();
        const filename = path.basename(outputPath);
        generatedFiles = [{
          filename: filename,
          filepath: outputPath,
          downloadPath: outputPath
        }];
        console.log(`PPT 파일이 다운로드 폴더에 저장되었습니다: ${outputPath}`);
      }
      
      // 다운로드 폴더 경로도 추출
      const downloadFolderMatch = stdout.match(/다운로드 폴더에 저장되었습니다:\s*(.+)/);
      if (downloadFolderMatch) {
        console.log(`다운로드 폴더: ${downloadFolderMatch[1].trim()}`);
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

    res.json({
      success: true,
      files: generatedFiles,
      message: stdout || 'PPT 생성 완료',
      downloadPath: outputPath
    });

  } catch (error) {
    console.error('PPT 생성 오류:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// 생성된 PPT 파일 다운로드
app.get('/api/ppts/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    const filePath = path.join(__dirname, '..', 'output', 'ppts', filename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'PPT file not found' });
    }

    res.download(filePath, filename, (err) => {
      if (err) {
        console.error('Error downloading PPT:', err);
        res.status(500).json({ error: 'Failed to download PPT' });
      }
    });
  } catch (error) {
    console.error('Error serving PPT:', error);
    res.status(500).json({ error: 'Failed to serve PPT' });
  }
});

app.listen(PORT, () => {
  console.log(`Samples server running on http://localhost:${PORT}`);
});

