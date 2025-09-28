import { useState } from 'react';
import type {FormEvent, ChangeEvent} from 'react';
import './App.css';

type ApiResponse = {
  label: string;
  confidence: number;
  suggested_reply: string;
};

function App() {
  // Estados para gerenciar a aplicação
  const [emailText, setEmailText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState('');
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setEmailText(e.target.value);
    if (selectedFile) {
      setSelectedFile(null);
      setFileName('');
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files ? e.target.files[0] : null;
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
      setEmailText('');
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    let endpoint = '';
    let body: BodyInit;
    let headers: HeadersInit = {};

    if (emailText.trim()) {
      endpoint = 'http://127.0.0.1:8000/api/process-email';
      headers = { 'Content-Type': 'application/json' };
      body = JSON.stringify({ text: emailText });
    } else if (selectedFile) {
      endpoint = 'http://127.0.0.1:8000/api/process-file';
      const formData = new FormData();
      formData.append('file', selectedFile);
      body = formData;
    } else {
      setError('Por favor, insira um texto ou selecione um arquivo.');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ocorreu um erro na API.');
      }

      const data: ApiResponse = await response.json();
      setResult(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Ocorreu um erro desconhecido.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Classificador de E-mails</h1>
      <p className="description">
        Cole o conteúdo de um e-mail abaixo ou envie um arquivo (.txt ou .pdf) para que nossa IA o classifique e sugira uma resposta.
        <br />
        Nossa IA espera um texto no formato Assunto: [Assunto do E-mail] Corpo: [Corpo do E-mail]. Seguindo esse formato o resultado será mais assertivo.
      </p>
      
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Cole o texto do e-mail aqui..."
          value={emailText}
          onChange={handleTextChange}
          disabled={isLoading}
        />
        <div className="separator"><span>OU</span></div>
        <label 
          htmlFor="file-upload" 
          className={`file-upload-label ${selectedFile ? 'file-selected' : ''}`}
        >
          {fileName || 'Escolher Arquivo'}
        </label>
        <input
          id="file-upload"
          type="file"
          accept=".pdf,.txt"
          onChange={handleFileChange}
          disabled={isLoading}
        />
        
        <button type="submit" disabled={isLoading || (!emailText.trim() && !selectedFile)}>
          {isLoading ? 'Analisando...' : 'Classificar'}
        </button>
      </form>
      
      {/* --- Área de Resultados --- */}
      {isLoading && <div className="spinner"></div>}
      {error && <div className="error-alert">{error}</div>}
      {result && (
        <div className="result-card">
          <div className="result-section">
            <h3>Classificação</h3>
            <p>
              <span className={`badge ${result.label.toLowerCase()}`}>
                {result.label}
              </span>
              (Confiança: {(result.confidence * 100).toFixed(2)}%)
            </p>
          </div>
          <div className="result-section">
            <h3>Sugestão de Resposta</h3>
            <p className="suggestion">{result.suggested_reply}</p>
          </div>
        </div>
      )}
      <p className="description">
        Nossa IA pode cometer erros. Sempre revise as sugestões antes de enviar qualquer resposta.
      </p>
    </div>
  );
}

export default App;