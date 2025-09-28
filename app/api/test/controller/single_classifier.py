import json
from app.api.src.controller.classifier import classify_text

sample_email_text = """
Assunto: Alinhamento Urgente: Entrega do Projeto Phoenix
Corpo: Olá equipe, precisamos fazer uma reunião rápida hoje às 15h para 
alinhar os últimos pontos da entrega. Por favor, confirmem a presença. Obrigado!
"""

candidate_labels = [
    "tarefa de trabalho", 
    "relatório de dados", 
    "agendamento de reunião",
    "discussão de projeto", 
    "comunicação com cliente",
    "spam ou promoção",
    "newsletter informativa",
    "conversa pessoal"
]

def main():
    print("--- Iniciando teste real da função classify_text ---")
    print("\nTexto a ser classificado:")
    print(f"'{sample_email_text.strip()}'")
    print("\nRótulos candidatos:")
    print(candidate_labels)
    
    try:
        print("\n[+] Enviando requisição para a API da Hugging Face...")

        result = classify_text(text=sample_email_text, labels=candidate_labels)
        
        print("\n[✔] Resposta recebida com sucesso!")
        
        print(json.dumps(result, indent=2, ensure_ascii=False))

        top_label = result['labels'][0]
        top_score = result['scores'][0]
        print(f"\n🏆 Principal classificação: '{top_label}' com pontuação de {top_score:.2%}")

    except Exception as e:
        print(f"\n[❌] Ocorreu um erro durante a execução.")
        print(f"Tipo do erro: {type(e).__name__}")
        print("Verifique a mensagem de erro que apareceu durante a execução da função classify_text.")
        print("Possíveis causas: Chave de API inválida no .env, falta de conexão com a internet ou a API da Hugging Face está fora do ar.")

if __name__ == "__main__":
    main()