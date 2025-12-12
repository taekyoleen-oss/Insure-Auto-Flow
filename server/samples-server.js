import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3002;

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
        const isMla = file.endsWith('.mla');
        if (isJson || isMla) {
          console.log(`Processing file: ${file} (${isMla ? '.mla' : '.json'})`);
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
          
          // .mla 파일 형식 (Save 버튼으로 저장한 파일) 변환
          if (file.endsWith('.mla') && data.modules && data.connections) {
            const projectName = data.projectName || file.replace('.mla', '');
            console.log(`Converting .mla file: ${file} -> ${projectName}`);
            console.log(`  Modules count: ${data.modules.length}, Connections count: ${data.connections.length}`);
            
            const convertedData = {
              filename: file,
              name: projectName,
              data: {
                name: projectName,
                modules: data.modules.map((m: any) => ({
                  type: m.type,
                  position: m.position || { x: 0, y: 0 },
                  name: m.name || m.type,
                  parameters: m.parameters || {},
                })),
                connections: data.connections.map((c: any) => {
                  const fromIndex = data.modules.findIndex((m: any) => m.id === c.from.moduleId);
                  const toIndex = data.modules.findIndex((m: any) => m.id === c.to.moduleId);
                  if (fromIndex < 0 || toIndex < 0) {
                    console.warn(`Invalid connection in ${file}: fromIndex=${fromIndex}, toIndex=${toIndex}`);
                  }
                  return {
                    fromModuleIndex: fromIndex >= 0 ? fromIndex : -1,
                    fromPort: c.from.portName,
                    toModuleIndex: toIndex >= 0 ? toIndex : -1,
                    toPort: c.to.portName,
                  };
                }).filter((c: any) => c.fromModuleIndex >= 0 && c.toModuleIndex >= 0),
              }
            };
            console.log(`Successfully converted .mla file: ${file} -> ${convertedData.name} (${convertedData.data.modules.length} modules, ${convertedData.data.connections.length} connections)`);
            return convertedData;
          }
          
          // .json 파일 형식 (기존 samples 형식)
          const jsonData = {
            filename: file,
            name: data.name || file.replace('.json', '').replace('.mla', ''),
            data: data
          };
          console.log(`Loaded .json file: ${file} -> ${jsonData.name}`);
          return jsonData;
        } catch (error: any) {
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
    
    // .mla 파일 형식 (Save 버튼으로 저장한 파일) 변환
    // projectName이 없어도 파일명에서 추출 가능하도록 수정
    if (filename.endsWith('.mla') && data.modules && data.connections) {
      console.log(`Converting .mla file: ${filename}`);
      const projectName = data.projectName || filename.replace('.mla', '').trim();
      const convertedData = {
        name: projectName,
        modules: data.modules.map((m: any) => ({
          type: m.type,
          position: m.position,
          name: m.name,
          parameters: m.parameters || {},
        })),
        connections: data.connections.map((c: any) => {
          const fromIndex = data.modules.findIndex((m: any) => m.id === c.from.moduleId);
          const toIndex = data.modules.findIndex((m: any) => m.id === c.to.moduleId);
          return {
            fromModuleIndex: fromIndex >= 0 ? fromIndex : -1,
            fromPort: c.from.portName,
            toModuleIndex: toIndex >= 0 ? toIndex : -1,
            toPort: c.to.portName,
          };
        }).filter((c: any) => c.fromModuleIndex >= 0 && c.toModuleIndex >= 0),
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

app.listen(PORT, () => {
  console.log(`Samples server running on http://localhost:${PORT}`);
});

