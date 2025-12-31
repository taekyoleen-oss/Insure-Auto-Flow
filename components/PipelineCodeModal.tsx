import React, { useState, useEffect } from 'react';
import { XCircleIcon, CodeBracketIcon, ClipboardIcon, CheckIcon } from './icons';

interface PipelineCodeModalProps {
    isOpen: boolean;
    onClose: () => void;
    code: string;
}

export const PipelineCodeModal: React.FC<PipelineCodeModalProps> = ({ isOpen, onClose, code }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy code:', err);
        }
    };

    useEffect(() => {
        if (!isOpen) {
            setCopied(false);
        }
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-gray-800 text-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <CodeBracketIcon className="w-6 h-6 text-blue-400" />
                        전체 파이프라인 코드
                    </h2>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleCopy}
                            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded-md transition-colors"
                            title="코드 복사"
                        >
                            {copied ? (
                                <>
                                    <CheckIcon className="w-4 h-4 text-green-400" />
                                    <span className="text-green-400">복사됨</span>
                                </>
                            ) : (
                                <>
                                    <ClipboardIcon className="w-4 h-4" />
                                    <span>복사</span>
                                </>
                            )}
                        </button>
                        <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    </div>
                </header>
                <main className="flex-1 overflow-auto p-4">
                    <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm font-mono text-gray-200 whitespace-pre-wrap">
                        <code>{code}</code>
                    </pre>
                </main>
                <footer className="flex justify-end p-4 bg-gray-900 rounded-b-lg border-t border-gray-700 flex-shrink-0">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-semibold text-white bg-gray-700 hover:bg-gray-600 rounded-md"
                    >
                        닫기
                    </button>
                </footer>
            </div>
        </div>
    );
};









































