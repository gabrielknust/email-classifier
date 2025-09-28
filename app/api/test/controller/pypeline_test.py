import json
import httpx
from app.api.src.controller.pypeline import run_classification_pipeline

# --- DADOS PARA O TESTE ---
# Sinta-se à vontade para alterar o texto e os rótulos abaixo.

# Cenário 1: Texto Válido (deve funcionar)
text_to_test = "Assunto: Proposta de Orçamento. Olá, anexo a proposta de orçamento para o projeto de marketing digital. Por favor, revisar e aprovar."

# Cenário 2: Texto Inválido (deve dar erro de idioma)
# Para testar este cenário, comente a linha acima e descomente a linha abaixo.
# text_to_test = "This is an english text and it should be blocked by the language validation step."

candidate_labels = [
    "proposta comercial",
    "fatura ou cobrança",
    "conversa pessoal",
    "spam"
]

# --- EXECUÇÃO DO PIPELINE ---

def main():
    """
    Função principal que executa o pipeline completo e imprime o resultado ou o erro.
    """
    print("--- Iniciando teste de integração do Pipeline de Classificação ---")
    print(f"\nTexto a ser processado: '{text_to_test}'")
    
    try:
        # Chama a função principal do nosso pipeline
        result = run_classification_pipeline(text=text_to_test, labels=candidate_labels)
        
        print("\n[✔] Pipeline executado com sucesso!")
        
        # Imprime o resultado formatado
        print("Resultado da API:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        print(f"\n🏆 Principal classificação: '{top_label}' com pontuação de {top_score:.2%}")

    # Captura os erros específicos que o nosso pipeline pode levantar
    except ValueError as e:
        print(f"\n[❌] ERRO DE VALIDAÇÃO NO PIPELINE:")
        print(f"   -> Mensagem: {e}")
    except httpx.HTTPStatusError as e:
        print(f"\n[❌] ERRO NA API DE CLASSIFICAÇÃO:")
        print(f"   -> Status: {e.response.status_code}")
        print(f"   -> Resposta: {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n[❌] ERRO DE CONEXÃO:")
        print(f"   -> Mensagem: Não foi possível conectar à API. Verifique a internet.")
    except Exception as e:
        print(f"\n[❌] OCORREU UM ERRO INESPERADO:")
        print(f"   -> Tipo: {type(e).__name__}, Mensagem: {e}")

if __name__ == "__main__":
    main()