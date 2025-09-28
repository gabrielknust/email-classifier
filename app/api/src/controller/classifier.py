from transformers import pipeline
from typing import Dict, Any
import os

caminho_script_atual = os.path.abspath(__file__)

caminho_pasta_app = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(caminho_script_atual))))

DIRETORIO_DO_MODELO = os.path.join(caminho_pasta_app, 'modelo_final_classificador')

classifier_pipeline = None

if not os.path.exists(DIRETORIO_DO_MODELO):
    print(f"ERRO CRÍTICO: O diretório do modelo treinado '{DIRETORIO_DO_MODELO}' não foi encontrado.")
else:
    try:
        classifier_pipeline = pipeline(
            "text-classification", 
            model=DIRETORIO_DO_MODELO,
            device=-1
        )
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível carregar o modelo de classificação.")
        print(f"Erro: {e}")

def predict(texto_email: str) -> Dict[str, Any]:
    if not classifier_pipeline:
        raise RuntimeError("O modelo de classificação não está disponível ou falhou ao carregar.")

    resultado = classifier_pipeline(texto_email)
    
    previsao = resultado[0]

    return {
        "label": previsao['label'],
        "confidence": previsao['score']
    }