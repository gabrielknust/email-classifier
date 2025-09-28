import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import os

NOME_ARQUIVO_DADOS = 'emails.csv'
MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
DIRETORIO_SAIDA_MODELO = './modelo_final_classificador'

device = "cuda" if torch.cuda.is_available() else "cpu"

def preparar_dataset():
    if not os.path.exists(NOME_ARQUIVO_DADOS):
        print(f"ERRO: Arquivo '{NOME_ARQUIVO_DADOS}' não encontrado.")
        return None, None

    df = pd.read_csv(NOME_ARQUIVO_DADOS)
    df.dropna(subset=['text'], inplace=True)
    df['label'] = df['label'].astype(str).str.strip()
    label_map = {'Improdutivo': 0, 'Produtivo': 1}
    df['label'] = df['label'].map(label_map)

    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    df = df.rename(columns={'text': 'text'})

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    return train_dataset, eval_dataset

def carregar_modelo_e_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    id2label = {0: "Improdutivo", 1: "Produtivo"}
    label2id = {"Improdutivo": 0, "Produtivo": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE, num_labels=2, id2label=id2label, label2id=label2id
    )
    return model, tokenizer


def tokenizar_datasets(train_dataset, eval_dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
    return train_tokenized, eval_tokenized

def treinar_modelo(model, train_dataset, eval_dataset, tokenizer):
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
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="epoch",
        save_strategy="epoch",
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

if __name__ == "__main__":
    train_ds, eval_ds = preparar_dataset()
    if train_ds:
        model, tokenizer = carregar_modelo_e_tokenizer()
        train_tokenized, eval_tokenized = tokenizar_datasets(train_ds, eval_ds, tokenizer)
        trainer = treinar_modelo(model, train_tokenized, eval_tokenized, tokenizer)
        
        trainer.save_model(DIRETORIO_SAIDA_MODELO)
