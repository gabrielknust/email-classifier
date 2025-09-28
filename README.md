# Classificador de e-mails

Esse projeto consiste em um classificador de e-mails que tem como objetivo usar Inteligência Artificial para classificar emails como produtivos ou improdutivos e sugerir uma resposta. O projeto está organizado em três partes principais:

### Fine-Tuning

Para classificar os e-mails foi necessário realizar um procedimento de Fine-Tuning em um modelo já pronto (https://huggingface.co/neuralmind/bert-base-portuguese-cased). Para realizar esse procedimento foi necessário criar uma base de dados fictícia, fazer o tratamento dos dados utilizando técnicas de Natural Language Processing (NLP) e treinar o modelo com essa base de dados. No final do treinamento temos um modelo treinado pronto para ser utilizado na API do projeto.

### API

A API do projeto consiste em duas rotas simples, uma (/api/process-email) processa o texto enviado, aplicando as mesmas técnicas de NLP usadas no treinamento, e usa o modelo criado para classificar e usa a API do Gemini para sugerir uma resposta e a outra /api/process-file processa um arquivo .txt ou .pdf para extrair os textos e utilizar o mesmo processo da rota anterior. Foi escolhido o modelo models/gemini-flash-latest para gerar as respostas sugeridas.

### Frontend

O Frontend do projeto é feito utilizando React e consiste em uma single page simples com um campo de texto e um campo para envio de arquivos.

## Organização de pastas

O projeto está dividido em três pastas principais:

### APP

Aqui é onde está o back(API) e o frontend(front) da aplicação.

### Examples

Arquivos de exemplo para a utilização em testes.

### Fine-Tuning

Aqui é onde está amazenado os arquivos resposáveis por realizar o Fine-Tuning no modelo e criar um novo modelo.

## Tecnologias utilizadas

Foi utilizado Python com FastAPI e Uvicorn para criar a API e React com Typescript e Vite para criar o Frontend.

## Instruções de uso

### Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/)

- [Node.js 18+ e npm](https://nodejs.org/en/download)

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Criação das variáveis de ambiente

Primeiro precisamos criar um arquivo de .env com as variáveis de ambiente necessárias:

``
    HUGGINGFACE_API_KEY
``
[Instruções para obter a chave.](https://huggingface.co/docs/hub/main/en/security-tokens)

``
    GEMINI_API_KEY
``
[Instruções para obter a chave.](https://ai.google.dev/gemini-api/docs/api-key)

### Criação do ambiente virtual

Navegue para a pasta app/api e crie um ambiente virtual:

``
    uv venv
``

Ative o ambiente virtual:

``
    source .venv/bin/activate
``

Instale os pacotes necessários:

``
    uv pip install -r requirements.txt
``

Volte para a pasta raiz do projeto:
``
    cd ../../
``

### Execução do Fine-Tuning

Ainda com o ambiente virtual ativo execute o comando:

``
    python -m Fine-Tuning.main
``

Aguarde o script ser finalizado (Pode demorar um pouco caso não seja possível utilizar uma GPU).

Isso irá criar a pasta do modelo necessária para a execução da API.

### Execução da API

Agora execute o comando:

``
    uvicorn app.api.src.main:app --reload
``

Isso irá executar a API na porta 8000. Aguarde até receber a mensagem "Application startup complete."

### Execução do front-end

Navegue para a pasta app/front

Execute o comando de instalação dos pacotes do React:

``
    npm install
``

Agora execute o comando de inicialização:

``
    npm run dev
``
___
Desenvolvido por [Gabriel Silva Knust](https://www.linkedin.com/in/gabrielknust/)
