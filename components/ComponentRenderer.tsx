
import React, { MouseEvent, TouchEvent, useRef, useState, useEffect, useCallback } from 'react';
import { CanvasModule, ModuleStatus, Port, ModuleType, Connection } from '../types';
import { CheckCircleIcon, CogIcon, XCircleIcon, PlayIcon, XMarkIcon } from './icons';

interface PortComponentProps {
  port: Port;
  isInput: boolean;
  moduleId: string;
  portRefs: React.MutableRefObject<Map<string, HTMLDivElement>>;
  onStartConnection: (moduleId: string, portName: string, clientX: number, clientY: number, isInput: boolean) => void;
  onStartSuggestion: (moduleId: string, portName: string, clientX: number, clientY: number, isInput: boolean) => void;
  onEndConnection: (moduleId: string, portName: string, isInput: boolean) => void;
  isTappedSource: boolean;
  onTapPort: (moduleId: string, portName: string, isInput: boolean) => void;
  cancelDragConnection: () => void;
  style: React.CSSProperties;
  dragConnection: { from: { moduleId: string; portName: string; isInput: boolean; }; to: { x: number; y: number; }; } | null;
}

interface ModuleNodeProps {
  module: CanvasModule;
  isSelected: boolean;
  onDoubleClick: (id: string) => void;
  onDragStart: (moduleId: string, e: MouseEvent) => void;
  onTouchDragStart: (moduleId: string, e: TouchEvent) => void;
  portRefs: React.MutableRefObject<Map<string, HTMLDivElement>>;
  onStartConnection: (moduleId: string, portName: string, clientX: number, clientY: number, isInput: boolean) => void;
  onStartSuggestion: (moduleId: string, portName: string, clientX: number, clientY: number, isInput: boolean) => void;
  onEndConnection: (moduleId: string, portName: string, isInput: boolean) => void;
  onViewDetails: (moduleId: string) => void;
  scale: number;
  onRunModule: (moduleId: string) => void;
  tappedSourcePort: { moduleId: string; portName: string; } | null;
  onTapPort: (moduleId: string, portName: string, isInput: boolean) => void;
  cancelDragConnection: () => void;
  onDelete: (id: string) => void;
  onModuleNameChange: (id: string, newName: string) => void;
  isSuggestion: boolean;
  onAcceptSuggestion: () => void;
  dragConnection: { from: { moduleId: string; portName: string; isInput: boolean; }; to: { x: number; y: number; }; } | null;
  areUpstreamModulesReady: (moduleId: string, allModules: CanvasModule[], allConnections: Connection[]) => boolean;
  allModules: CanvasModule[];
  allConnections: Connection[];
}

const statusStyles = {
    [ModuleStatus.Pending]: 'border-gray-500',
    [ModuleStatus.Running]: 'border-blue-500 animate-pulse',
    [ModuleStatus.Success]: 'border-blue-500',
    [ModuleStatus.Error]: 'border-red-500',
};

const statusDotColors = {
    [ModuleStatus.Pending]: 'bg-gray-400',
    [ModuleStatus.Running]: 'bg-blue-400 animate-pulse',
    [ModuleStatus.Success]: 'bg-green-400',
    [ModuleStatus.Error]: 'bg-red-400',
};

const statusIcons = {
    [ModuleStatus.Pending]: <CogIcon className="w-4 h-4 text-gray-400" />,
    [ModuleStatus.Running]: <CogIcon className="w-4 h-4 text-blue-400 animate-spin" />,
    [ModuleStatus.Success]: <CheckCircleIcon className="w-4 h-4 text-green-400" />,
    [ModuleStatus.Error]: <XCircleIcon className="w-4 h-4 text-red-400" />,
};

const PortComponent: React.FC<PortComponentProps> = ({ port, isInput, moduleId, portRefs, onStartConnection, onStartSuggestion, onEndConnection, isTappedSource, onTapPort, cancelDragConnection, style, dragConnection }) => {
    const touchStartRef = useRef<{x: number, y: number, time: number, isDragging: boolean} | null>(null);
    const mouseDragStartRef = useRef<{x: number, y: number, isDragging: boolean } | null>(null);
    const DRAG_THRESHOLD = 8; // pixels for drag detection

    const handleWindowMouseMove = useCallback((e: globalThis.MouseEvent) => {
        if (!mouseDragStartRef.current) return;
        const dx = Math.abs(e.clientX - mouseDragStartRef.current.x);
        const dy = Math.abs(e.clientY - mouseDragStartRef.current.y);
        if (!mouseDragStartRef.current.isDragging && (dx > DRAG_THRESHOLD || dy > DRAG_THRESHOLD)) {
            mouseDragStartRef.current.isDragging = true;
            // Start the connection visually only when dragging starts
            onStartConnection(moduleId, port.name, e.clientX, e.clientY, isInput);
        }
        if(mouseDragStartRef.current.isDragging){
            // Update connection line while dragging
            onStartConnection(moduleId, port.name, e.clientX, e.clientY, isInput);
        }
    }, [moduleId, port.name, isInput, onStartConnection]);

    const handleWindowMouseUp = useCallback((e: globalThis.MouseEvent) => {
        window.removeEventListener('mousemove', handleWindowMouseMove);
        window.removeEventListener('mouseup', handleWindowMouseUp);
        if (!mouseDragStartRef.current?.isDragging) {
            cancelDragConnection();
        }
        mouseDragStartRef.current = null;
    }, [handleWindowMouseMove, cancelDragConnection]);

    useEffect(() => {
        // This effect cleans up listeners if the drag is cancelled from outside (e.g., dropping on canvas)
        if (mouseDragStartRef.current && dragConnection === null) {
            window.removeEventListener('mousemove', handleWindowMouseMove);
            window.removeEventListener('mouseup', handleWindowMouseUp);
            mouseDragStartRef.current = null;
        }
    }, [dragConnection, handleWindowMouseMove, handleWindowMouseUp]);

    const handleMouseDown = (e: MouseEvent) => {
        e.stopPropagation();
        if (e.button !== 0) return;

        mouseDragStartRef.current = { x: e.clientX, y: e.clientY, isDragging: false };
        window.addEventListener('mousemove', handleWindowMouseMove);
        window.addEventListener('mouseup', handleWindowMouseUp);
        
        if (e.ctrlKey || e.metaKey) {
            onStartSuggestion(moduleId, port.name, e.clientX, e.clientY, isInput);
        }
    };
    
    const handlePortMouseUp = (e: MouseEvent) => {
        e.stopPropagation();
        const wasTap = mouseDragStartRef.current && !mouseDragStartRef.current.isDragging;
        
        if (wasTap) {
            onTapPort(moduleId, port.name, isInput);
        } else {
            onEndConnection(moduleId, port.name, isInput);
        }
        handleWindowMouseUp(e.nativeEvent);
    };
    
    const handleTouchStart = (e: React.TouchEvent) => {
        e.stopPropagation();
        const touch = e.touches[0];
        touchStartRef.current = {
            x: touch.clientX,
            y: touch.clientY,
            time: Date.now(),
            isDragging: false, 
        };

        const handleWindowTouchMove = (moveEvent: globalThis.TouchEvent) => {
            if (!touchStartRef.current) return;
            const currentTouch = moveEvent.touches[0];
            const dx = Math.abs(currentTouch.clientX - touchStartRef.current.x);
            const dy = Math.abs(currentTouch.clientY - touchStartRef.current.y);
            
            if (!touchStartRef.current.isDragging && (dx > DRAG_THRESHOLD || dy > DRAG_THRESHOLD)) {
                touchStartRef.current.isDragging = true;
            }

            if (touchStartRef.current.isDragging) {
                moveEvent.preventDefault();
                onStartConnection(moduleId, port.name, currentTouch.clientX, currentTouch.clientY, isInput);
            }
        };

        const handleWindowTouchEnd = (endEvent: globalThis.TouchEvent) => {
            window.removeEventListener('touchmove', handleWindowTouchMove);
            window.removeEventListener('touchend', handleWindowTouchEnd);

            if (touchStartRef.current && !touchStartRef.current.isDragging) {
                onTapPort(moduleId, port.name, isInput);
            }
            // The drop target's touchend or the canvas's touchend will cancel the connection line
            touchStartRef.current = null;
        };

        window.addEventListener('touchmove', handleWindowTouchMove, { passive: false });
        window.addEventListener('touchend', handleWindowTouchEnd, { once: true });
    };

    const handleTouchEnd = (e: React.TouchEvent) => {
        e.stopPropagation();
        onEndConnection(moduleId, port.name, isInput);
    };

    const portDotClasses = `w-4 h-4 rounded-full border-2 cursor-crosshair z-10 
                           ${isTappedSource ? 'bg-red-500 border-red-400 ring-2 ring-red-300' : 'bg-gray-600 border-gray-400 hover:bg-blue-500'}`;
    
    const portRefCallback = (el: HTMLDivElement | null) => {
      const key = `${moduleId}-${port.name}-${isInput ? 'in' : 'out'}`;
      if (el) {
          portRefs.current.set(key, el);
      } else {
          portRefs.current.delete(key);
      }
    };
    
    return (
        <div 
             ref={portRefCallback}
             style={style}
             className={portDotClasses}
             onMouseDown={handleMouseDown}
             onMouseUp={handlePortMouseUp}
             onTouchStart={handleTouchStart}
             onTouchEnd={handleTouchEnd}
        />
    );
};


const noRunButtonTypes = [
    ModuleType.LinearRegression,
    ModuleType.LogisticRegression,
    ModuleType.PoissonRegression,
    ModuleType.NegativeBinomialRegression,
    ModuleType.OLSModel,
    ModuleType.LogisticModel,
    ModuleType.PoissonModel,
    ModuleType.QuasiPoissonModel,
    ModuleType.NegativeBinomialModel,
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.SVM,
    ModuleType.LinearDiscriminantAnalysis,
    ModuleType.NaiveBayes,
    ModuleType.KNN,
    ModuleType.StatModels,
    // Deprecated
    ModuleType.LogisticTradition,
];

export const ComponentRenderer: React.FC<ModuleNodeProps> = ({ module, isSelected, onDoubleClick, onDragStart, onTouchDragStart, portRefs, onStartConnection, onStartSuggestion, onEndConnection, onViewDetails, scale, onRunModule, tappedSourcePort, onTapPort, cancelDragConnection, onDelete, onModuleNameChange, isSuggestion, onAcceptSuggestion, dragConnection, areUpstreamModulesReady, allModules, allConnections }) => {
  const isDraggingRef = useRef(false);
  const lastTapRef = useRef(0);

  const handleDelete = (e: MouseEvent | TouchEvent) => {
    e.stopPropagation();
    onDelete(module.id);
  };

  const handleMouseDown = (e: MouseEvent) => {
    if (isSuggestion) {
        e.stopPropagation();
        onAcceptSuggestion();
        return;
    }
    e.stopPropagation();
    onDragStart(module.id, e);
  };
  
  const handleDoubleClick = (e: MouseEvent) => {
    if (isSuggestion) return;
    e.stopPropagation();
    onDoubleClick(module.id);
  };

  const handleTouchStart = (e: TouchEvent) => {
    if (isSuggestion) {
        e.stopPropagation();
        e.preventDefault();
        onAcceptSuggestion();
        return;
    }
    e.stopPropagation();
    const now = Date.now();
    const DOUBLE_TAP_DELAY = 300;

    if (now - lastTapRef.current < DOUBLE_TAP_DELAY) {
      // Double tap detected
      lastTapRef.current = 0; // Reset tap timer
      e.preventDefault(); // Prevent zoom
      onDoubleClick(module.id);
      return; // Prevent starting a drag
    }

    lastTapRef.current = now;
    
    onTouchDragStart(module.id, e);
  };

  const { position, status } = module;
  const isRunnable = !noRunButtonTypes.includes(module.type) && areUpstreamModulesReady(module.id, allModules, allConnections);
  const getBackgroundColor = () => {
    if (status === ModuleStatus.Success) {
      return 'bg-blue-900/50';
    } else if (status === ModuleStatus.Pending && isRunnable) {
      return 'bg-green-900/30';
    }
    return 'bg-gray-800';
  };
  const getBorderColor = () => {
    if (status === ModuleStatus.Success) {
      return 'border-blue-500';
    } else if (status === ModuleStatus.Pending && isRunnable) {
      return 'border-green-600';
    }
    return statusStyles[status];
  };
  const suggestionClasses = isSuggestion ? 'opacity-70 border-dashed border-purple-500 animate-pulse' : '';
  const wrapperClasses = `absolute w-48 ${getBackgroundColor()} ${getBorderColor()} border-2 rounded-lg shadow-lg flex flex-col ${isSuggestion ? 'cursor-pointer' : 'cursor-move'} ${isSelected ? 'ring-2 ring-offset-2 ring-offset-gray-900 ring-blue-500' : ''} ${suggestionClasses}`;
  
  const componentStyle: React.CSSProperties = {
    transform: `translate(${position.x}px, ${position.y}px)`,
    userSelect: 'none',
  };

  return (
    <div
      style={componentStyle}
      className={wrapperClasses}
      onMouseDown={handleMouseDown}
      onDoubleClick={handleDoubleClick}
      onTouchStart={handleTouchStart}
    >
      {/* Ports are now absolutely positioned relative to the main wrapper */}
      {module.inputs.map((port, index) => {
          const total = module.inputs.length;
          const style: React.CSSProperties = { 
              position: 'absolute',
              top: '-9px', // w-4 is 16px, border is 2px. Half of 16px is 8px.
              left: `${((index + 1) * 100) / (total + 1)}%`,
              transform: 'translateX(-50%)',
          };
          return <PortComponent 
                    key={port.name} 
                    port={port} 
                    isInput={true} 
                    moduleId={module.id} 
                    portRefs={portRefs} 
                    onStartConnection={onStartConnection} 
                    onEndConnection={onEndConnection} 
                    isTappedSource={tappedSourcePort?.moduleId === module.id && tappedSourcePort.portName === port.name}
                    onTapPort={onTapPort}
                    cancelDragConnection={cancelDragConnection}
                    style={style}
                    onStartSuggestion={onStartSuggestion} 
                    dragConnection={dragConnection}
                  />;
      })}

      {isSelected && !isSuggestion && (
          <button
              onClick={handleDelete}
              onTouchEnd={handleDelete}
              className="absolute -top-2 -right-2 z-20 p-0.5 bg-gray-700 rounded-full text-gray-300 hover:bg-red-600 hover:text-white transition-colors"
              title="Delete Module"
              aria-label="Delete Module"
          >
              <XMarkIcon className="w-3.5 h-3.5" />
          </button>
      )}
      
      <div className="module-header bg-gray-700 px-3 py-1">
          <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 truncate">
                  <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${statusDotColors[status]}`} title={`Status: ${status}`}></span>
                  <span 
                      className="font-bold text-lg truncate" 
                      title={module.name}
                  >
                      {module.name}
                  </span>
              </div>
              {statusIcons[status]}
          </div>
      </div>
      
      <div className="flex-grow flex items-center justify-center px-2 py-6 relative">
          {isSuggestion ? (
              <div className="text-center">
                  <p className="text-xs text-purple-300 font-semibold">AI Suggestion</p>
                  <button
                      onClick={(e) => { e.stopPropagation(); onAcceptSuggestion(); }}
                      className="mt-2 px-2 py-1 text-xs bg-purple-600 hover:bg-purple-500 rounded text-white"
                  >
                      Accept
                  </button>
              </div>
          ) : (
              <>
                  {!noRunButtonTypes.includes(module.type) && (() => {
                      const canRun = areUpstreamModulesReady(module.id, allModules, allConnections);
                      const getRunButtonColor = () => {
                          if (status === ModuleStatus.Success) {
                              return 'text-green-500';
                          } else if (status === ModuleStatus.Pending && canRun) {
                              return 'text-blue-500';
                          } else {
                              return 'text-gray-600';
                          }
                      };
                      const buttonColor = getRunButtonColor();
                      const isDisabled = !canRun && status !== ModuleStatus.Success;
                      return (
                          <button
                              onClick={(e) => { e.stopPropagation(); onRunModule(module.id); }}
                              disabled={isDisabled}
                              className={`absolute left-2 top-1/2 -translate-y-1/2 p-2 rounded-full transition-colors z-10 ${
                                  isDisabled 
                                      ? `bg-gray-800 ${buttonColor} cursor-not-allowed opacity-50` 
                                      : `bg-gray-700 hover:bg-green-600 ${buttonColor} hover:text-white`
                              }`}
                              title={canRun ? "Run this module" : "Upstream modules must be executed first"}
                          >
                              <PlayIcon className="w-6 h-6" />
                          </button>
                      );
                  })()}
                  <button
                      onClick={(e) => { e.stopPropagation(); onViewDetails(module.id); }}
                      className="text-xs text-gray-400 hover:text-white disabled:text-gray-600 disabled:cursor-not-allowed"
                      disabled={module.status !== ModuleStatus.Success}
                  >
                      {module.status === ModuleStatus.Success ? 'View Details' : module.status}
                  </button>
              </>
          )}
      </div>

      {module.outputs.map((port, index) => {
          const total = module.outputs.length;
          const style: React.CSSProperties = { 
              position: 'absolute',
              bottom: '-9px',
              left: `${((index + 1) * 100) / (total + 1)}%`,
              transform: 'translateX(-50%)',
          };
          return <PortComponent 
                    key={port.name} 
                    port={port} 
                    isInput={false} 
                    moduleId={module.id} 
                    portRefs={portRefs} 
                    onStartConnection={onStartConnection} 
                    onEndConnection={onEndConnection} 
                    isTappedSource={tappedSourcePort?.moduleId === module.id && tappedSourcePort.portName === port.name}
                    onTapPort={onTapPort}
                    cancelDragConnection={cancelDragConnection}
                    style={style}
                    onStartSuggestion={onStartSuggestion} 
                    dragConnection={dragConnection}
                    />;
      })}
    </div>
  );
};
