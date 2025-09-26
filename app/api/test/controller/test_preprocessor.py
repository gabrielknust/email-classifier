import pytest
from src.controller.preprocessor import validate_language, process_text

def test_validate_language_com_texto_em_portugues():
    validate_language("Este é um texto de exemplo em português.")
    assert True

def test_validate_language_falha_com_texto_em_ingles():
    with pytest.raises(ValueError) as excinfo:
        validate_language("This is an English text.")
    assert "Idioma não suportado" in str(excinfo.value)

def test_validate_language_falha_com_texto_curto_nao_portugues():
    with pytest.raises(ValueError) as excinfo:
        validate_language("Ok")
    assert "Idioma não suportado" in str(excinfo.value)

def test_process_text_com_email_produtivo():
    email_produtivo = """
    Olá Maria, tudo bem? Anexei a versão final da proposta comercial para o cliente XYZ. 
    Você poderia, por favor, fazer a revisão final da seção de custos?
    """
    resultado_esperado = [
        'olá', 'Maria', 'anexei', 'versão', 'proposta', 'comercial', 
        'cliente', 'xyz', 'poder', 'revisão', 'seção', 'custo'
    ]
    tokens_processados = process_text(email_produtivo)
    assert tokens_processados == resultado_esperado

def test_process_text_com_email_improdutivo():
    email_improdutivo = "Clique aqui e aproveite nossa oferta de 70% OFF em notebooks!"
    resultado_esperado = [
        'clique', 'aproveitar', 'oferta', 'off', 'notebook'
    ]
    tokens_processados = process_text(email_improdutivo)
    assert tokens_processados == resultado_esperado

def test_process_text_com_email_ambiguo():
    email_ambiguo = "André, valeu pelos dados! Vamos sair no fim de semana?"
    resultado_esperado = ['André', 'valer', 'dado', 'ir', 'sair', 'semana']
    tokens_processados = process_text(email_ambiguo)
    assert tokens_processados == resultado_esperado