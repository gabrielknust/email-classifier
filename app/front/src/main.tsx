// src/main.tsx

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css' // Importa nosso estilo global

// Este código encontra a 'div' com id="root" no seu index.html
// e renderiza nosso componente App dentro dela.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)