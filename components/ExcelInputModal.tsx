import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { XCircleIcon } from './icons';

interface ExcelInputModalProps {
  onClose: () => void;
  onApply: (csvContent: string) => void;
  initialData?: string[][];
}

export const ExcelInputModal: React.FC<ExcelInputModalProps> = ({
  onClose,
  onApply,
  initialData,
}) => {
  const [data, setData] = useState<string[][]>(() => {
    if (initialData && initialData.length > 0) {
      return initialData;
    }
    // 기본 20행 10열 그리드
    return Array(20)
      .fill(null)
      .map(() => Array(10).fill(''));
  });

  const [selectedCells, setSelectedCells] = useState<Set<string>>(new Set()); // "row,col" 형식으로 저장
  const [anchorCell, setAnchorCell] = useState<{ row: number; col: number } | null>(null); // Shift 선택의 기준점
  const [editingCell, setEditingCell] = useState<{ row: number; col: number } | null>(null);
  const [editValue, setEditValue] = useState('');
  const gridRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [modalPosition, setModalPosition] = useState<{ x: number | string; y: number | string } | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; row: number; col: number } | null>(null);

  // 그리드 크기 조정
  const [rowCount, setRowCount] = useState(20);
  const [colCount, setColCount] = useState(10);

  // 열 헤더 생성 (A, B, C, ...)
  const getColumnHeader = (index: number): string => {
    let result = '';
    let num = index;
    while (num >= 0) {
      result = String.fromCharCode(65 + (num % 26)) + result;
      num = Math.floor(num / 26) - 1;
    }
    return result;
  };

  // 데이터 확장
  const ensureGridSize = useCallback((rows: number, cols: number) => {
    setData((prev) => {
      const newData = prev.map((row) => [...row]);
      
      // 행 확장
      while (newData.length < rows) {
        newData.push(Array(Math.max(cols, newData[0]?.length || 10)).fill(''));
      }
      
      // 열 확장
      const maxCols = Math.max(...newData.map((row) => row.length), cols);
      return newData.map((row) => {
        const newRow = [...row];
        while (newRow.length < maxCols) {
          newRow.push('');
        }
        return newRow;
      });
    });
  }, []);

  // 셀 키 문자열 생성
  const getCellKey = (row: number, col: number): string => {
    return `${row},${col}`;
  };

  // 셀이 선택되어 있는지 확인
  const isCellSelected = (row: number, col: number): boolean => {
    return selectedCells.has(getCellKey(row, col));
  };

  // 직사각형 영역의 모든 셀 선택
  const selectRange = (startRow: number, startCol: number, endRow: number, endCol: number) => {
    const newSelection = new Set<string>();
    const minRow = Math.min(startRow, endRow);
    const maxRow = Math.max(startRow, endRow);
    const minCol = Math.min(startCol, endCol);
    const maxCol = Math.max(startCol, endCol);

    for (let r = minRow; r <= maxRow; r++) {
      for (let c = minCol; c <= maxCol; c++) {
        newSelection.add(getCellKey(r, c));
      }
    }
    setSelectedCells(newSelection);
  };

  // 첫 번째 선택된 셀 가져오기 (키보드 네비게이션용)
  const getFirstSelectedCell = (): { row: number; col: number } | null => {
    if (selectedCells.size === 0) return null;
    const firstKey = Array.from(selectedCells)[0];
    const [row, col] = firstKey.split(',').map(Number);
    return { row, col };
  };

  // 셀 클릭 (선택만)
  const handleCellClick = (e: React.MouseEvent, row: number, col: number) => {
    // 더블클릭 방지
    if (e.detail === 2) return;

    const cellKey = getCellKey(row, col);

    if (e.shiftKey && anchorCell) {
      // Shift 키: 범위 선택
      selectRange(anchorCell.row, anchorCell.col, row, col);
    } else if (e.ctrlKey || e.metaKey) {
      // Ctrl/Cmd 키: 다중 선택 (토글)
      setSelectedCells((prev) => {
        const newSelection = new Set(prev);
        if (newSelection.has(cellKey)) {
          newSelection.delete(cellKey);
          // 마지막 셀을 제거하면 anchor도 제거
          if (newSelection.size === 0) {
            setAnchorCell(null);
          }
        } else {
          newSelection.add(cellKey);
          // 첫 번째 선택이면 anchor 설정
          if (prev.size === 0) {
            setAnchorCell({ row, col });
          }
        }
        return newSelection;
      });
    } else {
      // 일반 클릭: 단일 선택
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row, col });
    }
    // 편집 모드는 더블클릭으로만 진입
  };

  // 셀 더블클릭 (편집 모드 진입)
  const handleCellDoubleClick = (row: number, col: number) => {
    const cellKey = getCellKey(row, col);
    setSelectedCells(new Set([cellKey]));
    setAnchorCell({ row, col });
    setEditingCell({ row, col });
    setEditValue(data[row]?.[col] || '');
  };

  // 셀 편집 완료
  const handleCellBlur = () => {
    if (editingCell) {
      const newData = data.map((r) => [...r]);
      if (!newData[editingCell.row]) {
        newData[editingCell.row] = [];
      }
      newData[editingCell.row][editingCell.col] = editValue;
      setData(newData);
      setEditingCell(null);
    }
  };

  // 키보드 이벤트 처리
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // 편집 중이면 입력 필드에서 처리
    if (editingCell) {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleCellBlur();
        // 다음 행으로 이동
        const firstCell = getFirstSelectedCell();
        if (firstCell && firstCell.row < rowCount - 1) {
          const nextRow = firstCell.row + 1;
          const cellKey = getCellKey(nextRow, firstCell.col);
          setSelectedCells(new Set([cellKey]));
          setAnchorCell({ row: nextRow, col: firstCell.col });
          setEditingCell({ row: nextRow, col: firstCell.col });
          setEditValue(data[nextRow]?.[firstCell.col] || '');
        }
      } else if (e.key === 'Tab') {
        e.preventDefault();
        handleCellBlur();
        // 다음 열로 이동
        const firstCell = getFirstSelectedCell();
        if (firstCell && firstCell.col < colCount - 1) {
          const nextCol = firstCell.col + 1;
          const cellKey = getCellKey(firstCell.row, nextCol);
          setSelectedCells(new Set([cellKey]));
          setAnchorCell({ row: firstCell.row, col: nextCol });
          setEditingCell({ row: firstCell.row, col: nextCol });
          setEditValue(data[firstCell.row]?.[nextCol] || '');
        }
      } else if (e.key === 'Escape') {
        setEditingCell(null);
        setEditValue('');
      }
      return;
    }

    // 편집 모드가 아닐 때
    const firstCell = getFirstSelectedCell();
    if (!firstCell) {
      // 선택된 셀이 없으면 첫 번째 셀 선택
      const cellKey = getCellKey(0, 0);
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row: 0, col: 0 });
      return;
    }

    // Delete 또는 Del 키로 선택된 모든 셀 내용 삭제
    if (e.key === 'Delete' || e.key === 'Del' || e.key === 'Backspace') {
      e.preventDefault();
      e.stopPropagation();
      const newData = data.map((r) => [...r]);
      selectedCells.forEach((cellKey) => {
        const [row, col] = cellKey.split(',').map(Number);
        if (!newData[row]) {
          newData[row] = [];
        }
        newData[row][col] = '';
      });
      setData(newData);
      return;
    }

    // 문자나 숫자 입력 시 자동으로 편집 모드 진입 (첫 번째 선택된 셀만)
    if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      // 일반 문자나 숫자
      e.preventDefault();
      setEditingCell({ row: firstCell.row, col: firstCell.col });
      setEditValue(e.key);
      return;
    }

    if (e.key === 'Enter') {
      e.preventDefault();
      // Enter 키로 편집 모드 진입 (첫 번째 선택된 셀만)
      setEditingCell({ row: firstCell.row, col: firstCell.col });
      setEditValue(data[firstCell.row]?.[firstCell.col] || '');
    } else if (e.key === 'Tab') {
      e.preventDefault();
      // 다음 열로 이동
      if (firstCell.col < colCount - 1) {
        const cellKey = getCellKey(firstCell.row, firstCell.col + 1);
        setSelectedCells(new Set([cellKey]));
        setAnchorCell({ row: firstCell.row, col: firstCell.col + 1 });
      }
    } else if (e.key === 'ArrowUp' && firstCell.row > 0) {
      e.preventDefault();
      const cellKey = getCellKey(firstCell.row - 1, firstCell.col);
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row: firstCell.row - 1, col: firstCell.col });
    } else if (e.key === 'ArrowDown' && firstCell.row < rowCount - 1) {
      e.preventDefault();
      const cellKey = getCellKey(firstCell.row + 1, firstCell.col);
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row: firstCell.row + 1, col: firstCell.col });
    } else if (e.key === 'ArrowLeft' && firstCell.col > 0) {
      e.preventDefault();
      const cellKey = getCellKey(firstCell.row, firstCell.col - 1);
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row: firstCell.row, col: firstCell.col - 1 });
    } else if (e.key === 'ArrowRight' && firstCell.col < colCount - 1) {
      e.preventDefault();
      const cellKey = getCellKey(firstCell.row, firstCell.col + 1);
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row: firstCell.row, col: firstCell.col + 1 });
    } else if (e.key === 'Escape') {
      setEditingCell(null);
      setEditValue('');
      setSelectedCells(new Set());
      setAnchorCell(null);
    }
  };

  // 붙여넣기 처리 함수 (텍스트 직접 전달)
  const pasteData = useCallback(
    async (text: string, targetRow?: number, targetCol?: number) => {
      if (!text || text.trim() === '') return;

      // 편집 모드면 취소
      if (editingCell) {
        setEditingCell(null);
      }

      // 선택된 셀이 없으면 첫 번째 셀에 붙여넣기
      const firstCell = getFirstSelectedCell();
      const targetCell = targetRow !== undefined && targetCol !== undefined
        ? { row: targetRow, col: targetCol }
        : firstCell || { row: 0, col: 0 };
      
      if (!firstCell && targetRow === undefined) {
        const cellKey = getCellKey(targetCell.row, targetCell.col);
        setSelectedCells(new Set([cellKey]));
        setAnchorCell(targetCell);
      }

      const startRow = targetCell.row;
      const startCol = targetCell.col;

      // 엑셀 데이터 파싱 (탭으로 구분된 셀들)
      const lines = text.split(/\r?\n/);
      const pastedRows: string[][] = [];

      lines.forEach((line) => {
        // 빈 줄도 행으로 추가 (엑셀에서 빈 행을 복사한 경우)
        if (line.trim() === '') {
          pastedRows.push(['']);
          return;
        }
        
        // 탭으로 구분 (엑셀에서 복사한 데이터는 탭으로 구분됨)
        // 탭이 없으면 쉼표로 구분
        const delimiter = line.includes('\t') ? '\t' : ',';
        const cells = line.split(delimiter);
        
        // 각 셀의 따옴표 제거 및 공백 정리
        const processedCells = cells.map((cell) => {
          let processed = cell.trim();
          // 따옴표로 감싸진 경우 제거 (CSV 형식)
          if ((processed.startsWith('"') && processed.endsWith('"')) ||
              (processed.startsWith("'") && processed.endsWith("'"))) {
            processed = processed.slice(1, -1);
          }
          // 이스케이프된 따옴표 복원
          processed = processed.replace(/""/g, '"');
          return processed;
        });
        
        pastedRows.push(processedCells);
      });

      if (pastedRows.length === 0) return;

      // 필요한 행/열 확장
      const maxColsInPaste = Math.max(...pastedRows.map((row) => row.length), 1);
      const maxRow = Math.max(startRow + pastedRows.length - 1, rowCount - 1);
      const maxCol = Math.max(startCol + maxColsInPaste - 1, colCount - 1);

      ensureGridSize(maxRow + 1, maxCol + 1);

      // 데이터 붙여넣기 (선택된 셀을 기준으로)
      setData((prevData) => {
        const newData = prevData.map((r) => [...r]);
        
        pastedRows.forEach((row, rowIndex) => {
          row.forEach((cell, colIndex) => {
            const targetRow = startRow + rowIndex;
            const targetCol = startCol + colIndex;
            
            // 그리드 크기 확장
            if (!newData[targetRow]) {
              newData[targetRow] = [];
            }
            
            // 셀 데이터 설정
            newData[targetRow][targetCol] = cell;
          });
        });

        setRowCount((prev) => Math.max(prev, maxRow + 1));
        setColCount((prev) => Math.max(prev, maxCol + 1));
        
        // 붙여넣기 후 마지막 셀로 선택 이동
        const lastRow = startRow + pastedRows.length - 1;
        const lastCol = startCol + Math.max(...pastedRows.map(r => r.length)) - 1;
        const lastCellKey = getCellKey(lastRow, lastCol);
        setSelectedCells(new Set([lastCellKey]));
        setAnchorCell({ row: lastRow, col: lastCol });
        
        return newData;
      });
    },
    [editingCell, rowCount, colCount, ensureGridSize]
  );

  // 붙여넣기 처리 (전역 이벤트)
  const handlePaste = useCallback(
    async (e: React.ClipboardEvent | ClipboardEvent) => {
      // 편집 모드가 아닐 때만 붙여넣기 처리
      if (editingCell) return;

      e.preventDefault();
      e.stopPropagation();
      const text = e.clipboardData.getData('text');
      await pasteData(text);
    },
    [editingCell, pasteData]
  );

  // 오른쪽 클릭 처리
  const handleContextMenu = useCallback((e: React.MouseEvent, row: number, col: number) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({ x: e.clientX, y: e.clientY, row, col });
    const cellKey = getCellKey(row, col);
    setSelectedCells(new Set([cellKey]));
    setAnchorCell({ row, col });
  }, []);

  // 컨텍스트 메뉴 닫기
  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // 컨텍스트 메뉴에서 붙여넣기
  const handleContextMenuPaste = useCallback(async () => {
    if (!contextMenu) return;
    
    try {
      const text = await navigator.clipboard.readText();
      await pasteData(text, contextMenu.row, contextMenu.col);
    } catch (err) {
      console.error('Failed to read clipboard:', err);
      // 폴백: 사용자에게 입력 요청
      const text = prompt('붙여넣을 텍스트를 입력하세요:');
      if (text) {
        await pasteData(text, contextMenu.row, contextMenu.col);
      }
    }
    setContextMenu(null);
  }, [contextMenu, pasteData]);

  // 모달이 열릴 때 포커스 설정
  useEffect(() => {
    if (modalRef.current) {
      modalRef.current.focus();
    }
  }, []);

  // 전역 붙여넣기 이벤트 리스너
  useEffect(() => {
    const handleGlobalPaste = async (e: ClipboardEvent) => {
      // 모달이 열려있고 편집 모드가 아닐 때만 처리
      if (modalRef.current && !editingCell) {
        e.preventDefault();
        e.stopPropagation();
        // 선택된 셀이 없으면 첫 번째 셀 선택
        if (selectedCells.size === 0) {
          const cellKey = getCellKey(0, 0);
          setSelectedCells(new Set([cellKey]));
          setAnchorCell({ row: 0, col: 0 });
        }
        const text = e.clipboardData.getData('text');
        await pasteData(text);
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenu && !(e.target as Element).closest('.context-menu')) {
        setContextMenu(null);
      }
    };

    document.addEventListener('paste', handleGlobalPaste, true);
    document.addEventListener('click', handleClickOutside, true);
    return () => {
      document.removeEventListener('paste', handleGlobalPaste, true);
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [editingCell, selectedCells, contextMenu, pasteData]);

  // CSV로 변환
  const convertToCSV = (): string => {
    // 빈 행 제거
    const nonEmptyRows = data.filter((row) =>
      row.some((cell) => cell.trim() !== '')
    );

    if (nonEmptyRows.length === 0) return '';

    // 각 행의 마지막 빈 셀 제거
    const trimmedRows = nonEmptyRows.map((row) => {
      const trimmed = [...row];
      while (trimmed.length > 0 && trimmed[trimmed.length - 1].trim() === '') {
        trimmed.pop();
      }
      return trimmed;
    });

    return trimmedRows
      .map((row) => {
        return row
          .map((cell) => {
            const str = String(cell).trim();
            if (str.includes(',') || str.includes('"') || str.includes('\n')) {
              return `"${str.replace(/"/g, '""')}"`;
            }
            return str;
          })
          .join(',');
      })
      .join('\n');
  };

  // 적용
  const handleApply = () => {
    const csvContent = convertToCSV();
    if (csvContent.trim() === '') {
      alert('입력된 데이터가 없습니다.');
      return;
    }
    onApply(csvContent);
    onClose();
  };

  // 초기화
  const handleClear = () => {
    if (confirm('모든 데이터를 삭제하시겠습니까?')) {
      setData(Array(20).fill(null).map(() => Array(10).fill('')));
      setRowCount(20);
      setColCount(10);
      setSelectedCells(new Set());
      setAnchorCell(null);
      setEditingCell(null);
    }
  };

  // 입력 필드 포커스
  useEffect(() => {
    if (editingCell && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingCell]);

  // 드래그 핸들러 (헤더와 툴바 영역)
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    // 버튼이나 입력 필드가 아닐 때만 드래그 가능
    const isButton = target.closest('button') || target.tagName === 'BUTTON';
    const isInput = target.closest('input') || target.tagName === 'INPUT';
    const isSelect = target.closest('select') || target.tagName === 'SELECT';
    
    if (isButton || isInput || isSelect) {
      return;
    }
    
    if (modalRef.current) {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
      const rect = modalRef.current.getBoundingClientRect();
      // 드래그 시작 시점의 마우스 위치와 모달 위치의 차이를 계산
      const currentX = typeof modalPosition?.x === 'number' ? modalPosition.x : rect.left;
      const currentY = typeof modalPosition?.y === 'number' ? modalPosition.y : rect.top;
      setDragOffset({
        x: e.clientX - currentX,
        y: e.clientY - currentY,
      });
      // 초기 위치가 설정되지 않았다면 현재 위치를 설정
      if (!modalPosition) {
        setModalPosition({
          x: rect.left,
          y: rect.top,
        });
        setDragOffset({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }
    }
  }, [modalPosition]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging && modalRef.current) {
        e.preventDefault();
        e.stopPropagation();
        // 마우스 위치에서 드래그 오프셋을 빼서 모달 위치 계산
        const newX = e.clientX - dragOffset.x;
        const newY = e.clientY - dragOffset.y;
        setModalPosition({ x: newX, y: newY });
      }
    };

    const handleMouseUp = (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove, true);
    document.addEventListener('mouseup', handleMouseUp, true);
    document.body.addEventListener('mouseleave', handleMouseUp, true);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove, true);
      document.removeEventListener('mouseup', handleMouseUp, true);
      document.body.removeEventListener('mouseleave', handleMouseUp, true);
    };
  }, [isDragging, dragOffset]);

  // 실제 데이터가 있는 행/열 수 계산
  const actualRowCount = Math.max(
    rowCount,
    data.findIndex((row, idx) => {
      const reversedIdx = data.length - 1 - idx;
      return data[reversedIdx]?.some((cell) => cell.trim() !== '');
    }) + 1 || 1
  );

  const actualColCount = Math.max(
    colCount,
    ...data.map((row) => {
      const lastNonEmpty = row
        .map((cell, idx) => (cell.trim() !== '' ? idx : -1))
        .filter((idx) => idx >= 0);
      return lastNonEmpty.length > 0 ? Math.max(...lastNonEmpty) + 1 : 0;
    })
  );

  const displayRowCount = Math.max(actualRowCount, 20);
  const displayColCount = Math.max(actualColCount, 10);

  const modalContent = (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 z-50"
      onMouseDown={(e) => {
        // 배경 클릭을 무시 (모달은 닫기 버튼으로만 닫힘)
        e.preventDefault();
        e.stopPropagation();
      }}
      onClick={(e) => {
        // 배경 클릭을 무시 (모달은 닫기 버튼으로만 닫힘)
        e.preventDefault();
        e.stopPropagation();
      }}
    >
      <div
        ref={modalRef}
        className="bg-white rounded-lg shadow-xl w-[90vw] max-w-6xl max-h-[90vh] flex flex-col"
        onClick={(e) => {
          e.stopPropagation();
          // 모달 클릭 시 포커스 설정
          if (modalRef.current) {
            modalRef.current.focus();
          }
        }}
        onPaste={(e) => {
          if (!editingCell) {
            handlePaste(e);
          }
        }}
        onKeyDown={handleKeyDown}
        tabIndex={0}
        style={{
          position: 'fixed',
          left: modalPosition ? (typeof modalPosition.x === 'number' ? `${modalPosition.x}px` : modalPosition.x) : '50%',
          top: modalPosition ? (typeof modalPosition.y === 'number' ? `${modalPosition.y}px` : modalPosition.y) : '50%',
          transform: modalPosition ? 'none' : 'translate(-50%, -50%)',
          cursor: isDragging ? 'grabbing' : 'default',
        }}
      >
        {/* 헤더 */}
        <div 
          className="modal-header flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0 cursor-move"
          onMouseDown={handleMouseDown}
          onClick={(e) => e.stopPropagation()}
        >
          <div>
            <h2 className="text-xl font-bold text-gray-800">엑셀 데이터 입력</h2>
            <p className="text-sm text-gray-500 mt-1">
              엑셀에서 복사한 데이터를 붙여넣거나 직접 입력하세요
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-800 transition-colors"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <XCircleIcon className="w-6 h-6" />
          </button>
        </div>

        {/* 툴바 */}
        <div 
          className="modal-toolbar flex items-center justify-between p-3 border-b border-gray-200 bg-gray-50 flex-shrink-0 cursor-move"
          onMouseDown={handleMouseDown}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex gap-2">
            <button
              onClick={handleClear}
              className="px-3 py-1.5 text-sm bg-gray-200 hover:bg-gray-300 rounded-md font-semibold text-gray-700 transition-colors"
              onMouseDown={(e) => e.stopPropagation()}
            >
              초기화
            </button>
          </div>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-1.5 text-sm bg-gray-200 hover:bg-gray-300 rounded-md font-semibold text-gray-700 transition-colors"
              onMouseDown={(e) => e.stopPropagation()}
            >
              닫기
            </button>
            <button
              onClick={handleApply}
              className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 rounded-md font-semibold text-white transition-colors"
              onMouseDown={(e) => e.stopPropagation()}
            >
              저장하기
            </button>
          </div>
        </div>

        {/* 스프레드시트 그리드 */}
        <div
          ref={gridRef}
          className="flex-grow overflow-auto bg-white"
          style={{ maxHeight: 'calc(90vh - 140px)' }}
        >
          <div className="inline-block min-w-full">
            <table className="border-collapse">
              <thead className="sticky top-0 bg-gray-100 z-10">
                <tr>
                  <th className="w-12 h-8 border border-gray-300 bg-gray-200 text-center text-xs font-semibold text-gray-600">
                    {/* 빈 헤더 셀 */}
                  </th>
                  {Array(displayColCount)
                    .fill(null)
                    .map((_, colIndex) => (
                      <th
                        key={colIndex}
                        className="w-24 h-8 border border-gray-300 bg-gray-200 text-center text-xs font-semibold text-gray-600"
                      >
                        {getColumnHeader(colIndex)}
                      </th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {Array(displayRowCount)
                  .fill(null)
                  .map((_, rowIndex) => (
                    <tr key={rowIndex}>
                      <td className="w-12 h-8 border border-gray-300 bg-gray-100 text-center text-xs font-semibold text-gray-600">
                        {rowIndex + 1}
                      </td>
                      {Array(displayColCount)
                        .fill(null)
                        .map((_, colIndex) => {
                          const isSelected = isCellSelected(rowIndex, colIndex);
                          const isEditing =
                            editingCell?.row === rowIndex &&
                            editingCell?.col === colIndex;
                          const cellValue = data[rowIndex]?.[colIndex] || '';

                          return (
                            <td
                              key={colIndex}
                              className={`min-w-24 h-8 border border-gray-300 p-0 ${
                                isSelected
                                  ? 'bg-blue-100 ring-2 ring-blue-500'
                                  : 'bg-white hover:bg-gray-50'
                              }`}
                              onClick={(e) => handleCellClick(e, rowIndex, colIndex)}
                              onDoubleClick={() => handleCellDoubleClick(rowIndex, colIndex)}
                              onContextMenu={(e) => handleContextMenu(e, rowIndex, colIndex)}
                            >
                              {isEditing ? (
                                <input
                                  ref={inputRef}
                                  type="text"
                                  value={editValue}
                                  onChange={(e) => setEditValue(e.target.value)}
                                  onBlur={handleCellBlur}
                                  onKeyDown={handleKeyDown}
                                  className="w-full h-full px-1 text-sm border-0 outline-none bg-transparent text-gray-700"
                                  autoFocus
                                />
                              ) : (
                                <div className="w-full h-full px-1 text-sm flex items-center overflow-hidden text-ellipsis whitespace-nowrap text-black">
                                  {cellValue}
                                </div>
                              )}
                            </td>
                          );
                        })}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 푸터 */}
        <div className="p-3 border-t border-gray-200 bg-gray-50 flex-shrink-0">
          <p className="text-xs text-gray-500">
            팁: 엑셀에서 데이터를 복사한 후 이 창에 붙여넣기(Ctrl+V)하거나, 셀을 클릭하여 직접 입력할 수 있습니다.
          </p>
        </div>
      </div>

      {/* 컨텍스트 메뉴 */}
      {contextMenu && (
        <div
          className="context-menu fixed bg-white border border-gray-300 rounded-md shadow-lg z-50 py-1 min-w-[120px]"
          style={{
            left: `${contextMenu.x}px`,
            top: `${contextMenu.y}px`,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={handleContextMenuPaste}
            className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
          >
            붙여넣기
          </button>
        </div>
      )}
    </div>
  );

  // React Portal을 사용하여 body에 직접 렌더링하여 다른 요소의 클릭 이벤트와 격리
  return typeof document !== 'undefined' 
    ? createPortal(modalContent, document.body)
    : null;
};
















