import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface ModelConfig {
  provider: 'gemini' | 'openai' | 'anthropic' | 'ollama';
  model: string;
  apiKey: string;
  endpoint?: string; // For Ollama custom endpoints
}

interface ModelContextType {
  currentModel: ModelConfig | null;
  availableModels: Record<string, string[]>;
  setModel: (config: ModelConfig) => void;
  removeModel: () => void;
}

const ModelContext = createContext<ModelContextType | undefined>(undefined);

const defaultModels = {
  gemini: [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-2.0-flash-exp',
    'gemini-1.0-pro'
  ],
  openai: [
    'gpt-4',
    'gpt-4-turbo',
    'gpt-3.5-turbo',
    'gpt-4o',
    'gpt-4o-mini'
  ],
  anthropic: [
    'claude-3-5-sonnet-20241022',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229'
  ],
  ollama: [
    'llama3:8b',
    'llama3:70b',
    'llama3.2:3b',
    'llama2:7b',
    'mistral:7b',
    'codellama:7b',
    'phi3:3.8b',
    'custom'
  ]
};

export const ModelProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentModel, setCurrentModel] = useState<ModelConfig | null>(null);
  const [availableModels] = useState(defaultModels);

  useEffect(() => {
    // Load saved model from localStorage
    const saved = localStorage.getItem('datagenesis-model-config');
    if (saved) {
      try {
        const config = JSON.parse(saved);
        setCurrentModel(config);
      } catch (error) {
        console.error('Failed to load saved model config:', error);
      }
    }
  }, []);

  const setModel = (config: ModelConfig) => {
    setCurrentModel(config);
    localStorage.setItem('datagenesis-model-config', JSON.stringify(config));
  };

  const removeModel = () => {
    setCurrentModel(null);
    localStorage.removeItem('datagenesis-model-config');
  };

  return (
    <ModelContext.Provider value={{
      currentModel,
      availableModels,
      setModel,
      removeModel
    }}>
      {children}
    </ModelContext.Provider>
  );
};

export const useModel = () => {
  const context = useContext(ModelContext);
  if (context === undefined) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return context;
};