import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import os
from tqdm import tqdm

# Importamos nossa função de pré-processamento
from app.api.src.controller.preprocessor import process_text

# --- 0. CONFIGURAÇÕES ---
NOME_ARQUIVO_DADOS = 'fine-tunning/emails.csv'
MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
DIRETORIO_SAIDA_MODELO = './modelo_processado_classificador' # Novo nome para não confundir

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Usando dispositivo: {device.upper()} ---")

# --- 1. CARREGAR E PREPARAR O DATASET ---
def preparar_dataset():
    print(f"\n--- 1. Carregando e Preparando o Dataset de '{NOME_ARQUIVO_DADOS}' ---")
    df = pd.read_csv(NOME_ARQUIVO_DADOS)
    df.dropna(inplace=True)
    
    # Limpeza de rótulos (labels)
    df['label'] = df['label'].astype(str).str.strip()
    label_map = {'Improdutivo': 0, 'Produtivo': 1}
    df['label'] = df['label'].map(label_map)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # --- MUDANÇA PRINCIPAL AQUI ---
    # 1. Aplicamos o pré-processamento do spaCy primeiro
    print("\n[INFO] Aplicando pré-processamento de NLP (spaCy) em todos os textos...")
    tqdm.pandas(desc="Processando com spaCy")
    df['processed_tokens'] = df['text'].progress_apply(process_text)
    
    # 2. Juntamos os tokens processados em uma única string
    # É ESTE TEXTO LIMPO QUE USAREMOS PARA O TREINAMENTO
    df['text'] = df['processed_tokens'].apply(lambda tokens: ' '.join(tokens))
    print("[INFO] Textos pré-processados e prontos para o treinamento.")
    # -----------------------------

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    print("\nDataset preparado!")
    print(f"Tamanho do treino: {len(train_dataset)}, Tamanho da validação: {len(eval_dataset)}")
    return train_dataset, eval_dataset

# (O resto do código permanece o mesmo, pois ele já espera uma coluna 'text' e 'label')
# --- 2. CARREGAR MODELO BASE E TOKENIZER ---
def carregar_modelo_e_tokenizer():
    print(f"\n--- 2. Carregando Modelo Base '{MODELO_BASE}' ---")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    id2label = {0: "Improdutivo", 1: "Produtivo"}
    label2id = {"Improdutivo": 0, "Produtivo": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE, num_labels=2, id2label=id2label, label2id=label2id
    )
    return model, tokenizer

# --- 3. TOKENIZAR OS DADOS ---
def tokenizar_datasets(train_dataset, eval_dataset, tokenizer):
    print("\n--- 3. Tokenizando os Datasets ---")
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
    return train_tokenized, eval_tokenized

# --- 4. FUNÇÃO DE MÉTRICAS E TREINAMENTO ---
def treinar_modelo(model, train_dataset, eval_dataset, tokenizer):
    print("\n--- 4. Configurando e Iniciando o Treinamento ---")
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer

# --- PONTO DE ENTRADA DO SCRIPT ---
if __name__ == "__main__":
    train_ds, eval_ds = preparar_dataset()
    if train_ds:
        model, tokenizer = carregar_modelo_e_tokenizer()
        train_tokenized, eval_tokenized = tokenizar_datasets(train_ds, eval_ds, tokenizer)
        trainer = treinar_modelo(model, train_tokenized, eval_tokenized, tokenizer)
        
        print(f"\n--- 5. Salvando o Modelo Final em '{DIRETORIO_SAIDA_MODELO}' ---")
        trainer.save_model(DIRETORIO_SAIDA_MODELO)
        print("Modelo salvo com sucesso!")