from fastapi import FastAPI
from app.api.src.controller import endpoints

app = FastAPI(
    title="API de Classificação de Texto",
    description="Uma API para classificar textos como produtivos ou improdutivos usando IA.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "API está no ar!"}

app.include_router(endpoints.router, prefix="/api")