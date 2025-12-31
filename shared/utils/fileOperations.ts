/**
 * File operations using File System Access API
 */

export interface SaveOptions {
  extension: string;
  description: string;
  onSuccess?: (fileName: string) => void;
  onError?: (error: Error) => void;
}

export async function savePipeline(
  data: any,
  options: SaveOptions
): Promise<void> {
  try {
    if (!('showSaveFilePicker' in window)) {
      throw new Error(
        'File System Access API is not supported in this browser.'
      );
    }

    // 사용자 제스처 컨텍스트에서 직접 호출되도록 보장
    const fileHandle = await (window as any).showSaveFilePicker({
      suggestedName: `${data.projectName || 'pipeline'}${options.extension}`,
      types: [
        {
          description: options.description,
          accept: {
            'application/json': [options.extension],
          },
        },
      ],
      excludeAcceptAllOption: false,
    });

    const writable = await fileHandle.createWritable();
    await writable.write(JSON.stringify(data, null, 2));
    await writable.close();

    const fileName = fileHandle.name;
    if (options.onSuccess) {
      options.onSuccess(fileName);
    }
  } catch (error: any) {
    if (error.name === 'AbortError') {
      // User cancelled the save dialog
      return;
    }
    if (options.onError) {
      options.onError(error);
    } else {
      throw error;
    }
  }
}

export async function loadPipeline(): Promise<any> {
  try {
    if (!('showOpenFilePicker' in window)) {
      throw new Error(
        'File System Access API is not supported in this browser.'
      );
    }

    const [fileHandle] = await (window as any).showOpenFilePicker({
      types: [
        {
          description: 'ML Pipeline Files',
          accept: {
            'application/json': ['.ins', '.mla', '.json'],
          },
        },
      ],
    });

    const file = await fileHandle.getFile();
    const content = await file.text();
    return JSON.parse(content);
  } catch (error: any) {
    if (error.name === 'AbortError') {
      // User cancelled the open dialog
      return null;
    }
    throw error;
  }
}

