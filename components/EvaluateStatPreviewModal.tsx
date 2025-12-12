import React from 'react';
import { CanvasModule, EvaluateStatOutput } from '../types';
import { XCircleIcon } from './icons';

interface EvaluateStatPreviewModalProps {
  module: CanvasModule;
  projectName: string;
  onClose: () => void;
}

export const EvaluateStatPreviewModal: React.FC<EvaluateStatPreviewModalProps> = ({
  module,
  projectName,
  onClose,
}) => {
  const output = module.outputData as EvaluateStatOutput;
  if (!output || output.type !== 'EvaluateStatOutput') return null;

  const { modelType, metrics, residuals, deviance, pearsonChi2, dispersion, aic, bic, logLikelihood } = output;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">
            Evaluate Stat - {module.name}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <XCircleIcon className="w-6 h-6" />
          </button>
        </div>

        <div className="overflow-y-auto p-6 flex-1">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-4">Model Type: {modelType}</h3>
          </div>

          <div className="mb-6">
            <h3 className="text-md font-semibold text-white mb-3">Metrics</h3>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-400">{key}:</span>
                    <span className="text-white font-mono">
                      {typeof value === 'number' ? value.toFixed(6) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {(deviance !== undefined || pearsonChi2 !== undefined || dispersion !== undefined || 
            aic !== undefined || bic !== undefined || logLikelihood !== undefined) && (
            <div className="mb-6">
              <h3 className="text-md font-semibold text-white mb-3">Additional Statistics</h3>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4">
                  {deviance !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Deviance:</span>
                      <span className="text-white font-mono">{deviance.toFixed(6)}</span>
                    </div>
                  )}
                  {pearsonChi2 !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pearson Chi²:</span>
                      <span className="text-white font-mono">{pearsonChi2.toFixed(6)}</span>
                    </div>
                  )}
                  {dispersion !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Dispersion (φ):</span>
                      <span className="text-white font-mono">{dispersion.toFixed(6)}</span>
                    </div>
                  )}
                  {aic !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">AIC:</span>
                      <span className="text-white font-mono">{aic.toFixed(6)}</span>
                    </div>
                  )}
                  {bic !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">BIC:</span>
                      <span className="text-white font-mono">{bic.toFixed(6)}</span>
                    </div>
                  )}
                  {logLikelihood !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Log-Likelihood:</span>
                      <span className="text-white font-mono">{logLikelihood.toFixed(6)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {residuals && residuals.length > 0 && (
            <div className="mb-6">
              <h3 className="text-md font-semibold text-white mb-3">
                Residuals (First 100)
              </h3>
              <div className="bg-gray-900 rounded-lg p-4 max-h-64 overflow-y-auto">
                <div className="text-sm text-gray-300 font-mono">
                  {residuals.slice(0, 100).map((res, idx) => (
                    <div key={idx} className="py-1">
                      {res.toFixed(6)}
                    </div>
                  ))}
                  {residuals.length > 100 && (
                    <div className="text-gray-500 mt-2">
                      ... and {residuals.length - 100} more
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-4 border-t border-gray-700 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

