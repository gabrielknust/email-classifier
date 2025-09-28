from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.src.controller import endpoints

app = FastAPI(
    title="API de Classificação de Texto",
    description="Uma API para classificar textos como produtivos ou improdutivos usando IA.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173", # Endereço padrão do Vite (React)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todos os cabeçalhos
)

@app.get("/")
def read_root():
    return {"status": "API está no ar!"}

app.include_router(endpoints.router, prefix="/api")