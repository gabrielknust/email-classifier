# backend/train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Imports do TensorFlow/Keras para a Rede Neural
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# Importa nossa função de pré-processamento
from app.api.src.controller.preprocessor import process_text

# --- CONFIGURAÇÕES ---
NOME_ARQUIVO_DADOS = 'emails.csv'
VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 10 
BATCH_SIZE = 32

# --- FUNÇÕES DE PREPARAÇÃO DE DADOS (Passos 1, 2, 3) ---

def carregar_dados():
    print(f"--- PASSO 1: Carregando o Dataset '{NOME_ARQUIVO_DADOS}' ---")
    df = pd.read_csv(NOME_ARQUIVO_DADOS)
    df.dropna(subset=['text'], inplace=True)
    print(f"Arquivo CSV carregado com sucesso com {len(df)} e-mails.")
    return df

def pre_processar_textos(df):
    from tqdm import tqdm
    print(f"\n--- PASSO 2: Pré-processando os Textos com spaCy ---")
    tqdm.pandas(desc="Processando E-mails")
    df['processed_tokens'] = df['text'].progress_apply(process_text)
    return df

def vetorizar_dados(df):
    print(f"\n--- PASSO 3: Vetorizando Dados para o Modelo ---")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_tokens'])
    X = tokenizer.texts_to_sequences(df['processed_tokens'])
    X_padded = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    print("\n[INFO] Salvando o Tokenizer e o LabelEncoder...")
    joblib.dump(tokenizer, 'tokenizer.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print(" -> Arquivos 'tokenizer.joblib' e 'label_encoder.joblib' salvos.")
    return X_padded, y, label_encoder.classes_

# --- FUNÇÃO DE CONSTRUÇÃO DO MODELO (Passo 4) ---

def construir_modelo():
    print("\n--- PASSO 4: Construindo a Arquitetura do Modelo LSTM ---")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM), # Removido 'input_length' obsoleto
        SpatialDropout1D(0.3),
        LSTM(units=LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# --- Ponto de Entrada do Script ---

if __name__ == "__main__":
    # Passos 1, 2 e 3: Preparar todos os dados
    dataframe = carregar_dados()
    dataframe_processado = pre_processar_textos(dataframe)
    X_dados, y_rotulos, classes = vetorizar_dados(dataframe_processado)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_dados, y_rotulos, test_size=0.2, random_state=42, stratify=y_rotulos
    )
    
    # Passo 4: Construir o modelo
    model = construir_modelo()
    
    # --- PASSO 5: TREINAR O MODELO ---
    print("\n--- PASSO 5: Iniciando o Treinamento do Modelo ---")
    callbacks = [
        # Para o treinamento se a perda na validação não melhorar por 2 épocas seguidas
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )
    
    # --- PASSO 6: AVALIAR O MODELO ---
    print("\n--- PASSO 6: Avaliando a Performance Final ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia no conjunto de teste: {accuracy:.2%}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # --- PASSO 7: SALVAR O MODELO TREINADO ---
    print("\n--- PASSO 7: Salvando o Modelo Treinado ---")
    model.save('classifier_lstm.h5')
    print(" -> Modelo salvo como 'classifier_lstm.h5'")
    
    print("\n\nPROCESSO DE TREINAMENTO CONCLUÍDO COM SUCESSO!")