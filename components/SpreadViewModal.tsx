import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { XCircleIcon } from './icons';

interface SpreadViewModalProps {
  onClose: () => void;
  data: Array<Record<string, any>>;
  columns: Array<{ name: string; type?: string }>;
  title?: string;
}

export const SpreadViewModal: React.FC<SpreadViewModalProps> = ({
  onClose,
  data,
  columns,
  title = "Spread View",
}) => {
  const [selectedCells, setSelectedCells] = useState<Set<string>>(new Set());
  const [anchorCell, setAnchorCell] = useState<{ row: number; col: number } | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; row: number; col: number } | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [modalPosition, setModalPosition] = useState<{ x: number | string; y: number | string } | null>(null);

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

  // 첫 번째 선택된 셀 가져오기
  const getFirstSelectedCell = (): { row: number; col: number } | null => {
    if (selectedCells.size === 0) return null;
    const firstKey = Array.from(selectedCells)[0];
    const [row, col] = firstKey.split(',').map(Number);
    return { row, col };
  };

  // 셀 클릭 (선택만)
  const handleCellClick = (e: React.MouseEvent, row: number, col: number) => {
    if (e.detail === 2) return;

    const cellKey = getCellKey(row, col);

    if (e.shiftKey && anchorCell) {
      selectRange(anchorCell.row, anchorCell.col, row, col);
    } else if (e.ctrlKey || e.metaKey) {
      setSelectedCells((prev) => {
        const newSelection = new Set(prev);
        if (newSelection.has(cellKey)) {
          newSelection.delete(cellKey);
          if (newSelection.size === 0) {
            setAnchorCell(null);
          }
        } else {
          newSelection.add(cellKey);
          if (prev.size === 0) {
            setAnchorCell({ row, col });
          }
        }
        return newSelection;
      });
    } else {
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row, col });
    }
  };

  // 오른쪽 클릭 처리
  const handleContextMenu = useCallback((e: React.MouseEvent, row: number, col: number) => {
    e.preventDefault();
    e.stopPropagation();
    
    // 이미 선택된 셀들이 있으면 선택을 유지 (변경하지 않음)
    // 선택된 셀이 없을 때만 오른쪽 클릭한 셀을 선택
    if (selectedCells.size === 0) {
      const cellKey = getCellKey(row, col);
      setSelectedCells(new Set([cellKey]));
      setAnchorCell({ row, col });
    }
    // 이미 선택된 셀들이 있으면 선택을 변경하지 않음
    
    setContextMenu({ x: e.clientX, y: e.clientY, row, col });
  }, [selectedCells]);

  // 컨텍스트 메뉴 닫기
  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // 선택된 셀들을 CSV 형식으로 복사
  const copySelectedCells = useCallback(async () => {
    if (selectedCells.size === 0) {
      const firstCell = getFirstSelectedCell();
      if (!firstCell) return;
      const cellKey = getCellKey(firstCell.row, firstCell.col);
      setSelectedCells(new Set([cellKey]));
    }

    const selected = Array.from(selectedCells);
    const cellPositions = selected.map(key => {
      const [row, col] = key.split(',').map(Number);
      return { row, col };
    });

    if (cellPositions.length === 0) return;

    // 행과 열의 범위 계산
    const rows = cellPositions.map(p => p.row);
    const cols = cellPositions.map(p => p.col);
    const minRow = Math.min(...rows);
    const maxRow = Math.max(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);

    // 선택된 영역의 데이터 추출
    const copiedData: string[][] = [];
    for (let r = minRow; r <= maxRow; r++) {
      const row: string[] = [];
      for (let c = minCol; c <= maxCol; c++) {
        const cellKey = getCellKey(r, c);
        if (selectedCells.has(cellKey)) {
          if (r === 0) {
            // 헤더 행
            row.push(columns[c]?.name || '');
          } else {
            // 데이터 행
            const dataRow = data[r - 1];
            const colName = columns[c]?.name;
            const value = dataRow?.[colName] ?? '';
            row.push(String(value));
          }
        } else {
          row.push('');
        }
      }
      copiedData.push(row);
    }

    // CSV 형식으로 변환
    const csvContent = copiedData
      .map((row) => {
        return row
          .map((cell) => {
            const str = String(cell).trim();
            if (str.includes(',') || str.includes('"') || str.includes('\n')) {
              return `"${str.replace(/"/g, '""')}"`;
            }
            return str;
          })
          .join('\t'); // 탭으로 구분 (엑셀 붙여넣기용)
      })
      .join('\n');

    try {
      await navigator.clipboard.writeText(csvContent);
      setContextMenu(null);
    } catch (err) {
      console.error('Failed to copy:', err);
      // 폴백: 텍스트 영역 사용
      const textArea = document.createElement('textarea');
      textArea.value = csvContent;
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setContextMenu(null);
    }
  }, [selectedCells, data, columns]);

  // 컨텍스트 메뉴에서 복사하기
  const handleContextMenuCopy = useCallback(async () => {
    await copySelectedCells();
  }, [copySelectedCells]);

  // 모달이 열릴 때 포커스 설정
  useEffect(() => {
    if (modalRef.current) {
      modalRef.current.focus();
    }
  }, []);

  // 컨텍스트 메뉴 외부 클릭 시 닫기
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenu && !(e.target as Element).closest('.context-menu')) {
        setContextMenu(null);
      }
    };

    document.addEventListener('click', handleClickOutside, true);
    return () => {
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [contextMenu]);

  // 드래그 핸들러
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
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
      const currentX = typeof modalPosition?.x === 'number' ? modalPosition.x : rect.left;
      const currentY = typeof modalPosition?.y === 'number' ? modalPosition.y : rect.top;
      setDragOffset({
        x: e.clientX - currentX,
        y: e.clientY - currentY,
      });
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

  // 셀 값 가져오기
  const getCellValue = (row: number, col: number): string => {
    if (row === 0) {
      // 헤더 행
      return columns[col]?.name || '';
    }
    // 데이터 행
    const dataRow = data[row - 1];
    const colName = columns[col]?.name;
    return dataRow?.[colName] != null ? String(dataRow[colName]) : '';
  };

  const displayRowCount = data.length + 1; // 헤더 포함
  const displayColCount = columns.length;

  const modalContent = (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 z-50"
      onMouseDown={(e) => {
        e.preventDefault();
        e.stopPropagation();
      }}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
      }}
    >
      <div
        ref={modalRef}
        className="bg-white rounded-lg shadow-xl w-[90vw] max-w-6xl max-h-[90vh] flex flex-col"
        onClick={(e) => {
          e.stopPropagation();
          if (modalRef.current) {
            modalRef.current.focus();
          }
        }}
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
            <h2 className="text-xl font-bold text-gray-800">{title}</h2>
            <p className="text-sm text-gray-500 mt-1">
              셀을 선택하고 오른쪽 클릭하여 복사할 수 있습니다
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

        {/* 스프레드시트 그리드 */}
        <div
          className="flex-grow overflow-auto bg-white"
          style={{ maxHeight: 'calc(90vh - 100px)' }}
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
                        {rowIndex === 0 ? '' : rowIndex}
                      </td>
                      {Array(displayColCount)
                        .fill(null)
                        .map((_, colIndex) => {
                          const isSelected = isCellSelected(rowIndex, colIndex);
                          const cellValue = getCellValue(rowIndex, colIndex);

                          return (
                            <td
                              key={colIndex}
                              className={`min-w-24 h-8 border border-gray-300 p-0 ${
                                isSelected
                                  ? 'bg-blue-100 ring-2 ring-blue-500'
                                  : rowIndex === 0
                                  ? 'bg-gray-50 hover:bg-gray-100'
                                  : 'bg-white hover:bg-gray-50'
                              }`}
                              onClick={(e) => handleCellClick(e, rowIndex, colIndex)}
                              onContextMenu={(e) => handleContextMenu(e, rowIndex, colIndex)}
                            >
                              <div className="w-full h-full px-1 text-sm flex items-center overflow-hidden text-ellipsis whitespace-nowrap text-black">
                                {cellValue}
                              </div>
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
            팁: 셀을 선택하고 오른쪽 클릭하여 복사할 수 있습니다. Shift+Click으로 범위 선택, Ctrl+Click으로 다중 선택이 가능합니다.
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
            onClick={handleContextMenuCopy}
            className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
          >
            복사하기
          </button>
        </div>
      )}
    </div>
  );

  return typeof document !== 'undefined' 
    ? createPortal(modalContent, document.body)
    : null;
};

