import google.generativeai as genai
from app.api.config import settings

try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel('models/gemini-flash-latest')
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o modelo Gemini. Verifique a API Key. Erro: {e}")
    model = None

PROMPT_TEMPLATE = """
Você é um assistente de IA para uma empresa do setor financeiro. Sua tarefa é gerar uma sugestão de resposta curta e profissional para um e-mail.

O e-mail foi classificado como: "{label}"
O conteúdo do e-mail original é:
---
{original_text}
---

Baseado na categoria e no conteúdo, gere uma sugestão de resposta adequada.
- Se a categoria for 'Produtivo', a resposta deve ser útil, como confirmar o recebimento ou pedir mais informações.
- Se a categoria for 'Improdutivo', a resposta pode ser uma sugestão de ação interna (como 'Sugerir arquivar') ou uma resposta educada e curta para encerrar o assunto.
- A resposta deve ser apenas o texto da sugestão, sem introduções como "Aqui está a sugestão:".
"""

# --- FUNÇÃO DE GERAÇÃO DE RESPOSTA ---
async def suggest_reply(original_text: str, label: str) -> str:
    if not model:
        return "Erro: O modelo de geração de respostas não está disponível."
    
    try:
        # Preenche o prompt com os dados do e-mail específico
        prompt = PROMPT_TEMPLATE.format(label=label, original_text=original_text)
        
        # Chama a API do Gemini de forma assíncrona
        response = await model.generate_content_async(prompt)
        
        # Retorna o texto gerado pela IA
        return response.text.strip()
        
    except Exception as e:
        print(f"ERRO ao gerar resposta com Gemini: {e}")
        return "Não foi possível gerar uma sugestão de resposta."