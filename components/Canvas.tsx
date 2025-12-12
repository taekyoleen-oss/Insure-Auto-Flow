import React, { useState, useCallback, useRef, MouseEvent, TouchEvent, useEffect } from 'react';
import { CanvasModule, Connection, ModuleType } from '../types';
import { ComponentRenderer as ModuleNode } from './ComponentRenderer';
import { ShapeRenderer } from './ShapeRenderer';

interface CanvasProps {
  modules: CanvasModule[];
  connections: Connection[];
  setConnections: React.Dispatch<React.SetStateAction<Connection[]>>;
  selectedModuleIds: string[];
  setSelectedModuleIds: React.Dispatch<React.SetStateAction<string[]>>;
  updateModulePositions: (updates: { id: string, position: { x: number, y: number } }[]) => void;
  onModuleDrop: (type: ModuleType, position: { x: number; y: number }) => void;
  scale: number;
  setScale: React.Dispatch<React.SetStateAction<number>>;
  pan: { x: number, y: number };
  setPan: React.Dispatch<React.SetStateAction<{ x: number, y: number }>>;
  canvasContainerRef: React.RefObject<HTMLDivElement>;
  onViewDetails: (moduleId: string) => void;
  onModuleDoubleClick: (moduleId: string) => void;
  onRunModule: (moduleId: string) => void;
  onDeleteModule: (moduleId: string) => void;
  onUpdateModuleName: (id: string, newName: string) => void;
  onUpdateModule: (id: string, updates: Partial<CanvasModule>) => void;
  suggestion: { module: CanvasModule, connection: Connection } | null;
  onAcceptSuggestion: () => void;
  onClearSuggestion: () => void;
  onStartSuggestion: (moduleId: string, portName: string) => void;
  areUpstreamModulesReady: (moduleId: string, allModules: CanvasModule[], allConnections: Connection[]) => boolean;
}

export const Canvas: React.FC<CanvasProps> = ({ 
    modules, connections, setConnections, selectedModuleIds, setSelectedModuleIds, 
    updateModulePositions, onModuleDrop, scale, setScale, pan, setPan, 
    canvasContainerRef, onViewDetails, onModuleDoubleClick, onRunModule, 
    onDeleteModule, onUpdateModuleName, onUpdateModule, suggestion, onAcceptSuggestion, 
    onClearSuggestion, onStartSuggestion, areUpstreamModulesReady
}) => {
  const [dragConnection, setDragConnection] = useState<{ from: { moduleId: string, portName: string, isInput: boolean }, to: { x: number, y: number } } | null>(null);
  const [isSuggestionDrag, setIsSuggestionDrag] = useState(false);
  const [tappedSourcePort, setTappedSourcePort] = useState<{ moduleId: string; portName: string; } | null>(null);
  const portRefs = useRef(new Map<string, HTMLDivElement>());
  const modulesLayerRef = useRef<HTMLDivElement>(null);
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0 });
  const [selectionBox, setSelectionBox] = useState<{ x1: number, y1: number, x2: number, y2: number } | null>(null);
  const isSelecting = useRef(false);
  const isSpacePressed = useRef(false);
  
  // Refs for optimized dragging
  const dragInfoRef = useRef<{
    draggedModuleIds: string[];
    startPositions: Map<string, { x: number, y: number }>;
    dragStartPoint: { x: number, y: number };
  } | null>(null);
  
  const touchDragInfoRef = useRef<{
    draggedModuleIds: string[];
    startPositions: Map<string, { x: number, y: number }>;
    dragStartPoint: { x: number, y: number };
    touchIdentifier: number;
  } | null>(null);

  const latestMousePosRef = useRef<{ x: number, y: number } | null>(null);
  const requestRef = useRef<number | null>(null);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === ' ' || e.key === 'Space') {
                e.preventDefault();
                isSpacePressed.current = true;
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            if ((e.key === 'Control' || e.key === 'Meta') && isSuggestionDrag) {
                onClearSuggestion();
                setDragConnection(null);
                setIsSuggestionDrag(false);
            }
            if (e.key === ' ' || e.key === 'Space') {
                e.preventDefault();
                isSpacePressed.current = false;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [isSuggestionDrag, onClearSuggestion]);
    
    const cancelDragConnection = useCallback(() => {
        setDragConnection(null);
    }, []);
    
    useEffect(() => {
        // When an AI suggestion is created, clear any existing drag connection
        // to avoid visual glitches or conflicting states.
        if (suggestion) {
            cancelDragConnection();
        }
    }, [suggestion, cancelDragConnection]);

  const getPortPosition = useCallback((
    module: CanvasModule,
    portName: string,
    isInput: boolean,
  ) => {
    const portEl = portRefs.current.get(`${module.id}-${portName}-${isInput ? 'in' : 'out'}`);
    if (!portEl || !canvasContainerRef.current) {
        const portIndex = isInput ? module.inputs.findIndex(p => p.name === portName) : module.outputs.findIndex(p => p.name === portName);
        const portCount = isInput ? module.inputs.length : module.outputs.length;
        const moduleWidth = 256; // Updated to match component width
        return { 
            x: module.position.x + (moduleWidth / (portCount + 1)) * (portIndex + 1), 
            y: module.position.y + (isInput ? -10 : 110)
        };
    }

    const portRect = portEl.getBoundingClientRect();
    const canvasRect = canvasContainerRef.current.getBoundingClientRect();
    
    // Reverse the transformation to get canvas-space ("world") coordinates
    return {
        x: (portRect.left + portRect.width / 2 - canvasRect.left - pan.x) / scale,
        y: (portRect.top + portRect.height / 2 - canvasRect.top - pan.y) / scale
    };
  }, [scale, pan, canvasContainerRef]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onClearSuggestion();
    
    if (!canvasContainerRef.current) {
      console.error('Canvas container ref is not available');
      return;
    }
    
    const typeString = e.dataTransfer.getData('application/reactflow');
    // Check if type is valid ModuleType (non-empty string)
    if (!typeString || typeString.trim() === '') {
      console.error('Invalid module type received from drag:', typeString);
      return;
    }
    
    // Validate that the type is a valid ModuleType enum value
    const type = typeString as ModuleType;
    const validTypes = Object.values(ModuleType);
    if (!validTypes.includes(type)) {
      console.error('Module type is not a valid ModuleType enum value:', type);
      return;
    }
    
    const canvasBounds = canvasContainerRef.current.getBoundingClientRect();
    const moduleWidth = 256; // Updated to match component width
    
    const position = {
      x: (e.clientX - canvasBounds.left - pan.x) / scale,
      y: (e.clientY - canvasBounds.top - pan.y) / scale,
    };
    onModuleDrop(type, {x: position.x - moduleWidth / 2, y: position.y});
  };

  // The animate function for rAF
  const animateDrag = useCallback(() => {
      if (!dragInfoRef.current || !latestMousePosRef.current || !canvasContainerRef.current) {
          return;
      }

      const { dragStartPoint, startPositions } = dragInfoRef.current;
      const currentMousePos = latestMousePosRef.current;
      const canvasBounds = canvasContainerRef.current.getBoundingClientRect();
      
      // Convert screen coordinates to canvas coordinates
      const startCanvasX = (dragStartPoint.x - canvasBounds.left - pan.x) / scale;
      const startCanvasY = (dragStartPoint.y - canvasBounds.top - pan.y) / scale;
      const currentCanvasX = (currentMousePos.x - canvasBounds.left - pan.x) / scale;
      const currentCanvasY = (currentMousePos.y - canvasBounds.top - pan.y) / scale;
      
      const dx = currentCanvasX - startCanvasX;
      const dy = currentCanvasY - startCanvasY;

      const updates: { id: string, position: { x: number, y: number } }[] = [];
      startPositions.forEach((startPos, id) => {
          updates.push({
              id,
              position: {
                  x: startPos.x + dx,
                  y: startPos.y + dy,
              }
          });
      });

      if (updates.length > 0) {
          updateModulePositions(updates);
      }
      
      requestRef.current = null;
  }, [scale, pan, canvasContainerRef, updateModulePositions]);

  const handleDragMove = useCallback((e: globalThis.MouseEvent) => {
      if (!dragInfoRef.current) return;
      
      // Just update the ref, don't trigger state update directly
      latestMousePosRef.current = { x: e.clientX, y: e.clientY };
      
      // Schedule animation frame if not already scheduled
      if (!requestRef.current) {
          requestRef.current = requestAnimationFrame(animateDrag);
      }

  }, [animateDrag]);

  const handleDragEnd = useCallback(() => {
      dragInfoRef.current = null;
      latestMousePosRef.current = null;
      if (requestRef.current) {
          cancelAnimationFrame(requestRef.current);
          requestRef.current = null;
      }
      window.removeEventListener('mousemove', handleDragMove);
      window.removeEventListener('mouseup', handleDragEnd);
  }, [handleDragMove]);

  const handleModuleDragStart = useCallback((draggedModuleId: string, e: MouseEvent) => {
    if (e.button !== 0) return;

    const isShift = e.shiftKey;
    const alreadySelected = selectedModuleIds.includes(draggedModuleId);
    let idsToDrag = selectedModuleIds;

    if (isShift) {
        const newSelection = alreadySelected
            ? selectedModuleIds.filter(id => id !== draggedModuleId)
            : [...selectedModuleIds, draggedModuleId];
        setSelectedModuleIds(newSelection);
        idsToDrag = newSelection;
    } else if (!alreadySelected) {
        setSelectedModuleIds([draggedModuleId]);
        idsToDrag = [draggedModuleId];
    }
    
    const startPositions = new Map<string, { x: number, y: number }>();
    modules.forEach(m => {
        if (idsToDrag.includes(m.id)) {
            startPositions.set(m.id, m.position);
        }
    });

    dragInfoRef.current = {
        draggedModuleIds: idsToDrag,
        startPositions,
        dragStartPoint: { x: e.clientX, y: e.clientY },
    };
    latestMousePosRef.current = { x: e.clientX, y: e.clientY };

    window.addEventListener('mousemove', handleDragMove);
    window.addEventListener('mouseup', handleDragEnd);
  }, [modules, selectedModuleIds, setSelectedModuleIds, handleDragMove, handleDragEnd]);
  
  // Similar optimization for Touch events could be added, 
  // but sticking to mouse for now as per request priority.
  const handleTouchMove = useCallback((e: globalThis.TouchEvent) => {
    if (!touchDragInfoRef.current || !canvasContainerRef.current) return;
    
    let currentTouch: Touch | null = null;
    for (let i = 0; i < e.touches.length; i++) {
        if (e.touches[i].identifier === touchDragInfoRef.current.touchIdentifier) {
            currentTouch = e.touches[i];
            break;
        }
    }
    if (!currentTouch) return;

    e.preventDefault();

    const { dragStartPoint, startPositions } = touchDragInfoRef.current;
    const canvasBounds = canvasContainerRef.current.getBoundingClientRect();
    
    // Convert screen coordinates to canvas coordinates
    const startCanvasX = (dragStartPoint.x - canvasBounds.left - pan.x) / scale;
    const startCanvasY = (dragStartPoint.y - canvasBounds.top - pan.y) / scale;
    const currentCanvasX = (currentTouch.clientX - canvasBounds.left - pan.x) / scale;
    const currentCanvasY = (currentTouch.clientY - canvasBounds.top - pan.y) / scale;
    
    const dx = currentCanvasX - startCanvasX;
    const dy = currentCanvasY - startCanvasY;

    const updates: { id: string, position: { x: number, y: number } }[] = [];
    startPositions.forEach((startPos, id) => {
        updates.push({
            id,
            position: { x: startPos.x + dx, y: startPos.y + dy }
        });
    });

    if (updates.length > 0) {
        updateModulePositions(updates);
    }
  }, [scale, pan, canvasContainerRef, updateModulePositions]);

  const handleTouchEnd = useCallback(() => {
    if (touchDragInfoRef.current) {
        touchDragInfoRef.current = null;
        window.removeEventListener('touchmove', handleTouchMove);
        window.removeEventListener('touchend', handleTouchEnd);
    }
  }, [handleTouchMove]);
  
  const handleModuleTouchDragStart = useCallback((draggedModuleId: string, e: TouchEvent) => {
    if (e.touches.length !== 1) return;
    const touch = e.touches[0];

    const isShift = e.shiftKey;
    const alreadySelected = selectedModuleIds.includes(draggedModuleId);
    let idsToDrag = selectedModuleIds;

    if (isShift) {
        const newSelection = alreadySelected
            ? selectedModuleIds.filter(id => id !== draggedModuleId)
            : [...selectedModuleIds, draggedModuleId];
        setSelectedModuleIds(newSelection);
        idsToDrag = newSelection;
    } else if (!alreadySelected) {
        setSelectedModuleIds([draggedModuleId]);
        idsToDrag = [draggedModuleId];
    }
    
    const startPositions = new Map<string, { x: number, y: number }>();
    modules.forEach(m => {
        if (idsToDrag.includes(m.id)) {
            startPositions.set(m.id, m.position);
        }
    });

    touchDragInfoRef.current = {
        draggedModuleIds: idsToDrag,
        startPositions,
        dragStartPoint: { x: touch.clientX, y: touch.clientY },
        touchIdentifier: touch.identifier,
    };

    window.addEventListener('touchmove', handleTouchMove, { passive: false });
    window.addEventListener('touchend', handleTouchEnd);
  }, [modules, selectedModuleIds, setSelectedModuleIds, handleTouchMove, handleTouchEnd]);
  
  const handleCanvasMouseDown = (e: MouseEvent) => {
    // Panning with middle mouse button
    if (e.button === 1) {
        e.preventDefault();
        isPanning.current = true;
        panStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
        (e.currentTarget as HTMLElement).style.cursor = 'grabbing';
        return;
    }

    if (e.target === e.currentTarget && e.button === 0) {
        onClearSuggestion();
        setTappedSourcePort(null);
        
        if (e.shiftKey) {
            // Shift 키를 누른 상태에서만 패닝
            isPanning.current = true;
            panStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
            (e.currentTarget as HTMLElement).style.cursor = 'grabbing';
        } else {
            // 일반 드래그 시 선택 박스 모드 (Space 키 없이도 작동)
            if (!e.shiftKey) {
                setSelectedModuleIds([]);
            }
            isSelecting.current = true;
            const canvasRect = canvasContainerRef.current!.getBoundingClientRect();
            const startX = e.clientX - canvasRect.left;
            const startY = e.clientY - canvasRect.top;
            setSelectionBox({ x1: startX, y1: startY, x2: startX, y2: startY });
        }
    }
  };

  const handleCanvasMouseMove = (e: MouseEvent) => {
      // 모듈 드래그 중이면 패닝하지 않음
      if (dragInfoRef.current) {
          return;
      }
      
      if (dragConnection && canvasContainerRef.current) {
        const canvasRect = canvasContainerRef.current.getBoundingClientRect();
        setDragConnection(prev => prev ? ({
            ...prev,
            to: { 
                x: (e.clientX - canvasRect.left - pan.x) / scale, 
                y: (e.clientY - canvasRect.top - pan.y) / scale
            },
        }) : null);
      } else if (isPanning.current) {
          e.preventDefault();
          setPan({
              x: e.clientX - panStart.current.x,
              y: e.clientY - panStart.current.y
          });
      } else if (isSelecting.current && selectionBox && canvasContainerRef.current) {
          const canvasRect = canvasContainerRef.current.getBoundingClientRect();
          const currentX = e.clientX - canvasRect.left;
          const currentY = e.clientY - canvasRect.top;
          setSelectionBox(prev => prev ? { ...prev, x2: currentX, y2: currentY } : null);
      }
  };

  const handleCanvasMouseUp = (e: MouseEvent) => {
      if (isSuggestionDrag) {
        onAcceptSuggestion();
      }
      setIsSuggestionDrag(false);
      setDragConnection(null);

      if(isPanning.current) {
          isPanning.current = false;
          if (e.currentTarget) {
              (e.currentTarget as HTMLElement).style.cursor = 'grab';
          }
      }

      if (isSelecting.current) {
        isSelecting.current = false;
        if (selectionBox) {
            const { x1, y1, x2, y2 } = selectionBox;

            const selectionRect = {
                minX: (Math.min(x1, x2) - pan.x) / scale,
                minY: (Math.min(y1, y2) - pan.y) / scale,
                maxX: (Math.max(x1, x2) - pan.x) / scale,
                maxY: (Math.max(y1, y2) - pan.y) / scale,
            };

            const moduleWidth = 256; // Updated to match component width
            const moduleHeight = 120; 

            const newlySelectedIds = modules
                .filter(module => {
                    const moduleRect = {
                        x: module.position.x,
                        y: module.position.y,
                        width: moduleWidth,
                        height: moduleHeight,
                    };
                    return (
                        moduleRect.x < selectionRect.maxX &&
                        moduleRect.x + moduleRect.width > selectionRect.minX &&
                        moduleRect.y < selectionRect.maxY &&
                        moduleRect.y + moduleRect.height > selectionRect.minY
                    );
                })
                .map(m => m.id);

            if (newlySelectedIds.length > 0) {
                 if (e.shiftKey) {
                    setSelectedModuleIds(prev => {
                        const newSet = new Set(prev);
                        newlySelectedIds.forEach(id => newSet.add(id));
                        return Array.from(newSet);
                    });
                } else {
                    setSelectedModuleIds(newlySelectedIds);
                }
            }
        }
        setSelectionBox(null);
    }
  };
  
  const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        const delta = e.deltaY * -0.001;
        const newScale = Math.max(0.2, Math.min(2, scale + delta));
        
        if (!canvasContainerRef.current) return;
        const canvasRect = canvasContainerRef.current.getBoundingClientRect();
        const mousePoint = { x: e.clientX - canvasRect.left, y: e.clientY - canvasRect.top };
        
        const canvasPoint = {
            x: (mousePoint.x - pan.x) / scale,
            y: (mousePoint.y - pan.y) / scale,
        };

        const newPan = {
            x: mousePoint.x - canvasPoint.x * newScale,
            y: mousePoint.y - canvasPoint.y * newScale,
        };

        setScale(newScale);
        setPan(newPan);
  };

  const handleStartConnection = useCallback((moduleId: string, portName: string, clientX: number, clientY: number, isInput: boolean) => {
    if (!canvasContainerRef.current) return;
    const canvasRect = canvasContainerRef.current.getBoundingClientRect();
    const to = { 
        x: (clientX - canvasRect.left - pan.x) / scale, 
        y: (clientY - canvasRect.top - pan.y) / scale
    };
    
    setDragConnection({ from: { moduleId, portName, isInput }, to });
  }, [scale, pan, canvasContainerRef]);

    const handleStartSuggestionDrag = useCallback((moduleId: string, portName: string, clientX: number, clientY: number, isInput: boolean) => {
        setIsSuggestionDrag(true);
        onStartSuggestion(moduleId, portName);
    }, [onStartSuggestion]);

  const handleEndConnection = useCallback((moduleId: string, portName: string, dropOnIsInput: boolean) => {
    if (isSuggestionDrag) {
        onAcceptSuggestion();
    } else if (dragConnection) {
        const fromModule = modules.find(m => m.id === dragConnection.from.moduleId);
        const toModule = modules.find(m => m.id === moduleId);

        if (!fromModule || !toModule || fromModule.id === toModule.id) {
            setDragConnection(null);
            return;
        }

        const dragFromIsInput = dragConnection.from.isInput;

        if (dragFromIsInput && !dropOnIsInput) { // Drag from INPUT to OUTPUT
            const fromPort = toModule.outputs.find(p => p.name === portName);
            const toPort = fromModule.inputs.find(p => p.name === dragConnection.from.portName);
            if (fromPort && toPort && fromPort.type === toPort.type) {
                const newConnection: Connection = {
                    id: `conn-${Date.now()}`,
                    from: { moduleId: toModule.id, portName: fromPort.name },
                    to: { moduleId: fromModule.id, portName: toPort.name },
                };
                setConnections(prev => [
                    ...prev.filter(c => !(c.to.moduleId === fromModule.id && c.to.portName === toPort.name)),
                    newConnection,
                ]);
            }
        } else if (!dragFromIsInput && dropOnIsInput) { // Drag from OUTPUT to INPUT
            const fromPort = fromModule.outputs.find(p => p.name === dragConnection.from.portName);
            const toPort = toModule.inputs.find(p => p.name === portName);
            if (fromPort && toPort && fromPort.type === toPort.type) {
                const newConnection: Connection = {
                    id: `conn-${Date.now()}`,
                    from: { moduleId: fromModule.id, portName: fromPort.name },
                    to: { moduleId: toModule.id, portName: toPort.name },
                };
                setConnections(prev => [
                    ...prev.filter(c => !(c.to.moduleId === toModule.id && c.to.portName === toPort.name)),
                    newConnection,
                ]);
            }
        }
    }
    setDragConnection(null);
    setIsSuggestionDrag(false);
  }, [dragConnection, isSuggestionDrag, modules, setConnections, onAcceptSuggestion]);

  const handleTapPort = useCallback((moduleId: string, portName: string, isInput: boolean) => {
    cancelDragConnection(); 

    if (isInput) {
        if (tappedSourcePort) {
            const sourceModule = modules.find(m => m.id === tappedSourcePort.moduleId);
            const targetModule = modules.find(m => m.id === moduleId);
            if (!sourceModule || !targetModule || sourceModule.id === targetModule.id) {
                setTappedSourcePort(null);
                return;
            }

            const sourcePort = sourceModule.outputs.find(p => p.name === tappedSourcePort.portName);
            const targetPort = targetModule.inputs.find(p => p.name === portName);

            if (sourcePort && targetPort && sourcePort.type === targetPort.type) {
                const newConnection: Connection = {
                    id: `conn-${Date.now()}`,
                    from: tappedSourcePort,
                    to: { moduleId, portName },
                };
                setConnections(prev => [
                    ...prev.filter(c => !(c.to.moduleId === moduleId && c.to.portName === portName)),
                    newConnection,
                ]);
            }
            setTappedSourcePort(null);
        }
    } else {
        if (tappedSourcePort && tappedSourcePort.moduleId === moduleId && tappedSourcePort.portName === portName) {
            setTappedSourcePort(null);
        } else {
            setTappedSourcePort({ moduleId, portName });
        }
    }
  }, [tappedSourcePort, modules, setConnections, cancelDragConnection]);
  
    const handleCanvasTouchEnd = (e: React.TouchEvent) => {
        if (dragConnection) {
            cancelDragConnection();
        }
    }

    const handleConnectionDoubleClick = useCallback((connectionId: string) => {
        if (suggestion && suggestion.connection.id === connectionId) {
            return;
        }
        setConnections(prev => prev.filter(c => c.id !== connectionId));
    }, [setConnections, suggestion]);

    const allModules = suggestion ? [...modules, suggestion.module] : modules;
    const allConnections = suggestion ? [...connections, suggestion.connection] : connections;

  return (
    <div
      className="w-full h-full relative"
      style={{ cursor: isPanning.current ? 'grabbing' : 'grab' }}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onMouseDown={handleCanvasMouseDown}
      onMouseMove={handleCanvasMouseMove}
      onMouseUp={handleCanvasMouseUp}
      onMouseLeave={handleCanvasMouseUp}
      onTouchEnd={handleCanvasTouchEnd}
      onWheel={handleWheel}
    >
      {/* Layer 1: Modules (HTML elements) */}
      <div
        ref={modulesLayerRef}
        className="absolute top-0 left-0"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
          transformOrigin: 'top left',
        }}
      >
        {allModules.map(module => {
          // Render shapes separately
          if (module.type === ModuleType.TextBox || module.type === ModuleType.GroupBox) {
            return (
              <ShapeRenderer
                key={module.id}
                module={module}
                isSelected={selectedModuleIds.includes(module.id)}
                onDragStart={handleModuleDragStart}
                onDelete={onDeleteModule}
                onUpdateModule={onUpdateModule}
                scale={scale}
              />
            );
          }
          // Render regular modules
          return (
            <ModuleNode 
              key={module.id} 
              module={module} 
              isSelected={selectedModuleIds.includes(module.id)} 
              onDragStart={handleModuleDragStart}
              onTouchDragStart={handleModuleTouchDragStart}
              onDoubleClick={onModuleDoubleClick}
              portRefs={portRefs}
              onStartConnection={handleStartConnection}
              onEndConnection={handleEndConnection}
              onViewDetails={onViewDetails}
              scale={scale}
              onRunModule={onRunModule}
              tappedSourcePort={tappedSourcePort}
              onTapPort={handleTapPort}
              cancelDragConnection={cancelDragConnection}
              onDelete={onDeleteModule}
              onModuleNameChange={onUpdateModuleName}
              isSuggestion={!!suggestion && suggestion.module.id === module.id}
              onAcceptSuggestion={onAcceptSuggestion}
              onStartSuggestion={handleStartSuggestionDrag}
              dragConnection={dragConnection}
              areUpstreamModulesReady={areUpstreamModulesReady}
              allModules={allModules}
              allConnections={allConnections}
            />
          );
        })}
      </div>

       {selectionBox && (
            <div
                className="absolute border-2 border-dashed border-blue-500 bg-blue-500 bg-opacity-20 pointer-events-none z-30"
                style={{
                    left: Math.min(selectionBox.x1, selectionBox.x2),
                    top: Math.min(selectionBox.y1, selectionBox.y2),
                    width: Math.abs(selectionBox.x1 - selectionBox.x2),
                    height: Math.abs(selectionBox.y1 - selectionBox.y2),
                }}
            />
        )}

      {/* Layer 2: Connections (SVG overlay) */}
      <svg
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      >
        <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#FFFFFF" />
            </marker>
             <marker id="arrow-drag" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#a78bfa" />
            </marker>
        </defs>
        {/* This group element applies the same pan and zoom to the connections as the modules */}
        <g style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`}}>
            {allConnections.map(conn => {
                const fromModule = allModules.find(m => m.id === conn.from.moduleId);
                const toModule = allModules.find(m => m.id === conn.to.moduleId);
                if (!fromModule || !toModule) return null;

                const start = getPortPosition(fromModule, conn.from.portName, false);
                const end = getPortPosition(toModule, conn.to.portName, true);
                const isSuggestionConn = !!suggestion && suggestion.connection.id === conn.id;
                const pathD = `M${start.x},${start.y} C${start.x},${start.y + 75} ${end.x},${end.y - 75} ${end.x},${end.y}`;

                return (
                    <g key={conn.id} onDoubleClick={() => handleConnectionDoubleClick(conn.id)}>
                        <path
                            d={pathD}
                            stroke={isSuggestionConn ? "#a78bfa" : "#FFFFFF"}
                            strokeWidth="3"
                            fill="none"
                            strokeDasharray={isSuggestionConn ? "6,6" : undefined}
                            markerEnd="url(#arrow)"
                            style={{ pointerEvents: 'none' }}
                        />
                        <path
                            d={pathD}
                            stroke="transparent"
                            strokeWidth="20"
                            fill="none"
                            style={{ cursor: 'pointer', pointerEvents: 'stroke' }}
                            title="Double-click to delete connection"
                        />
                    </g>
                )
            })}
            {dragConnection && (
                () => {
                    const fromModule = modules.find(m => m.id === dragConnection.from.moduleId);
                    if (!fromModule) return null;
                    
                    const isInput = dragConnection.from.isInput;
                    const start = getPortPosition(fromModule, dragConnection.from.portName, isInput);
                    const end = dragConnection.to;
                    
                    const path = isInput
                        ? `M${end.x},${end.y} C${end.x},${end.y + 75} ${start.x},${start.y - 75} ${start.x},${start.y}`
                        : `M${start.x},${start.y} C${start.x},${start.y + 75} ${end.x},${end.y - 75} ${end.x},${end.y}`;

                    return (
                         <path
                            d={path}
                            stroke="#a78bfa"
                            strokeWidth="3"
                            fill="none"
                            strokeDasharray="6,6"
                            markerEnd={!isInput ? "url(#arrow-drag)" : undefined}
                            markerStart={isInput ? "url(#arrow-drag)" : undefined}
                        />
                    )
                }
            )()}
        </g>
      </svg>
    </div>
  );
};