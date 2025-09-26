import httpx
import json
from config import settings

def test_huggingface_connection():
    """
    Função para testar a conexão com a API de inferência da Hugging Face.
    Usa as configurações do arquivo config.py.
    """
    print("--- Iniciando teste de conexão com a API da Hugging Face ---")
    
    # 1. Pega a URL da API e a Chave do nosso arquivo de configuração
    api_url = settings.HF_API_URL
    api_key = settings.HUGGINGFACE_API_KEY
    
    # 2. Monta o cabeçalho da requisição com a autorização
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # 3. Cria um payload de exemplo para o teste de classificação
    payload = {
        "inputs": "Hoje o dia está lindo e ensolarado, perfeito para um passeio no parque.",
        "parameters": {
            "candidate_labels": ["clima", "trabalho", "esportes", "notícia"]
        }
    }
    
    print(f"Enviando requisição para: {api_url}")
    print(f"Payload de teste: {payload}")
    
    try:
        # 4. Faz a requisição POST para a API
        response = httpx.post(api_url, headers=headers, json=payload)
        
        # 5. Verifica se a requisição teve algum erro HTTP (ex: 401, 404, 500)
        response.raise_for_status()
        
        # 6. Se tudo deu certo, imprime a resposta
        print("\n✅ Conexão bem-sucedida!")
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        
        print("\nResposta da API:")
        # O json.dumps formata o JSON para uma leitura mais fácil
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except httpx.HTTPStatusError as e:
        print(f"\n❌ Erro HTTP ao conectar com a API: {e.response.status_code}")
        print("A resposta do servidor foi:")
        print(e.response.text)
        print("\nPossíveis causas:")
        print("- Sua HUGGINGFACE_API_KEY no arquivo .env está incorreta ou inválida.")
        print("- O modelo pode estar carregando (tente novamente em um minuto).")
        
    except Exception as e:
        print(f"\n❌ Ocorreu um erro inesperado: {e}")
        print("Verifique sua conexão com a internet ou o nome do modelo no config.py.")

# Executa a função de teste
if __name__ == "__main__":
    test_huggingface_connection()