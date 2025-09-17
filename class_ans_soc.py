import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# --- 1. BLOCO DE CONFIGURAÇÃO E HIPERPARÂMETROS ---
RANDOM_SEED = 42
N_SPLITS = 10  # R-Fold Cross Validation com r = 10
BATCH_SIZE = 32
NUM_EPOCHS = 70
LEARNING_RATE = 0.1
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- 2. DEFINIÇÃO DAS FEATURES E ALVO ---

# Variáveis Demográficas
demographic_features = ['Age', 'Gender', 'Occupation']

# Variáveis de Estilo de Vida
lifestyle_features = [
    'Sleep Hours', 'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)',
    'Alcohol Consumption (drinks/week)', 'Smoking'
]

# Indicadores Fisiológicos e Mentais
physiological_features = [
    'Heart Rate (bpm)', 'Breathing Rate (breaths/min)', 'Sweating Level (1-5)',
    'Dizziness', 'Stress Level (1-10)'
]

# Histórico Psicológico
psychological_features = [
    'Family History of Anxiety', 'Medication', 'Therapy Sessions (per month)'
]

# Eventos de Vida
life_events_features = ['Recent Major Life Event']

# Variáveis categóricas (conforme proposta)
categorical_features = [
    'Gender', 'Occupation', 'Smoking', 'Family History of Anxiety',
    'Dizziness', 'Medication', 'Recent Major Life Event'
]

# Variáveis numéricas (conforme proposta)
numerical_features = [
    'Age', 'Sleep Hours', 'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)',
    'Alcohol Consumption (drinks/week)', 'Stress Level (1-10)', 'Heart Rate (bpm)',
    'Breathing Rate (breaths/min)', 'Sweating Level (1-5)', 'Therapy Sessions (per month)'
]

# Adicionando Diet Quality se existir no dataset
extended_numerical_features = numerical_features + ['Diet Quality (1-10)']

# Variável alvo
target_variable = 'Anxiety Level (1-10)'

# --- 3. DEFINIÇÃO DO MODELO HÍBRIDO ---


class HybridMLPLogisticModel(nn.Module):
    """
    Modelo Híbrido que combina MLP e Regressão Logística

    Arquitetura:
    - Branch MLP: Input → 64 → 32 → output_features
    - Branch Logística: Input → output_features  
    - Combinação: Soma ponderada dos outputs + camada final
    """

    def __init__(self, input_dim, output_dim, combination_method='weighted_sum'):
        super(HybridMLPLogisticModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.combination_method = combination_method

        # --- BRANCH MLP ---
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, output_dim)
        )

        # --- BRANCH REGRESSÃO LOGÍSTICA ---
        self.logistic_branch = nn.Linear(input_dim, output_dim)

        # --- CAMADA DE COMBINAÇÃO ---
        if combination_method == 'weighted_sum':
            # Pesos aprendíveis para combinar os branches
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Peso para MLP
            self.beta = nn.Parameter(torch.tensor(0.5))   # Peso para Logística
        elif combination_method == 'concatenation':
            # Concatenar outputs e passar por uma camada final
            self.final_layer = nn.Linear(output_dim * 2, output_dim)
        elif combination_method == 'attention':
            # Mecanismo de atenção para combinar os branches
            self.attention = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Tanh(),
                nn.Linear(output_dim, 2),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        # Forward pass nos dois branches
        mlp_output = self.mlp_branch(x)
        logistic_output = self.logistic_branch(x)

        # Combinação dos outputs
        if self.combination_method == 'weighted_sum':
            # Soma ponderada com pesos aprendíveis
            combined_output = self.alpha * mlp_output + self.beta * logistic_output

        elif self.combination_method == 'concatenation':
            # Concatenação + camada final
            concatenated = torch.cat([mlp_output, logistic_output], dim=1)
            combined_output = self.final_layer(concatenated)

        elif self.combination_method == 'attention':
            # Mecanismo de atenção
            concatenated = torch.cat([mlp_output, logistic_output], dim=1)
            attention_weights = self.attention(
                concatenated)  # Shape: (batch_size, 2)

            # Aplicar pesos de atenção
            mlp_weighted = attention_weights[:, 0:1] * mlp_output
            logistic_weighted = attention_weights[:, 1:2] * logistic_output
            combined_output = mlp_weighted + logistic_weighted

        return combined_output

    def get_branch_outputs(self, x):
        """Retorna os outputs de cada branch separadamente para análise"""
        with torch.no_grad():
            mlp_output = self.mlp_branch(x)
            logistic_output = self.logistic_branch(x)
        return mlp_output, logistic_output

    def get_logistic_weights(self):
        """Retorna os pesos da regressão logística para análise de importância"""
        return self.logistic_branch.weight.data.numpy()

    def get_mlp_input_weights(self):
        """Retorna os pesos da primeira camada do MLP para análise de importância"""
        # A primeira camada é self.mlp_branch[0]
        return self.mlp_branch[0].weight.data.numpy()

    def get_combination_weights(self):
        """Retorna os pesos de combinação dos branches"""
        if self.combination_method == 'weighted_sum':
            return {
                'alpha (MLP)': float(self.alpha.data),
                'beta (Logistic)': float(self.beta.data)
            }
        return None


class AnxietyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- 4. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO ---


def train_model(model, train_loader, criterion, optimizer, model_name="Hybrid"):
    """Treina o modelo híbrido"""
    model.train()
    total_loss = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        total_loss += epoch_loss

        # Print de progresso a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(
                f"  {model_name} - Época {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

            # Mostrar pesos de combinação se disponível
            if hasattr(model, 'get_combination_weights'):
                weights = model.get_combination_weights()
                if weights:
                    print(
                        f"    Pesos de combinação: α={weights['alpha (MLP)']:.3f}, β={weights['beta (Logistic)']:.3f}")


def evaluate_model(model, test_loader):
    """Avalia o modelo com todas as métricas da proposta"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Métricas conforme proposta
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds,
                           average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, f1_macro, f1_weighted, conf_matrix


def analyze_hybrid_model(model, test_loader, feature_names, class_names):
    """Análise específica do modelo híbrido"""
    print("\n" + "="*60)
    print("ANÁLISE DO MODELO HÍBRIDO")
    print("="*60)

    # 1. Análise dos pesos da regressão logística
    print("\n--- Importância das Variáveis (Branch Regressão Logística) ---")
    # Shape: (num_classes, num_features)
    logistic_weights = model.get_logistic_weights()

    # Importância média absoluta por feature
    avg_importance_logistic = np.mean(np.abs(logistic_weights), axis=0)
    feature_importance_logistic = list(
        zip(feature_names, avg_importance_logistic))
    feature_importance_logistic.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 variáveis mais importantes (Branch Logística):")
    for i, (feature, importance) in enumerate(feature_importance_logistic[:10]):
        print(f"{i+1:2d}. {feature:<35} {importance:.4f}")

    # 2. Análise dos pesos da primeira camada do MLP
    print("\n--- Importância das Variáveis (Branch MLP - Pesos da 1ª Camada) ---")
    mlp_input_weights = model.get_mlp_input_weights()
    # Para o MLP, a importância pode ser inferida pela magnitude dos pesos da primeira camada
    # A média absoluta dos pesos para cada feature de entrada através dos neurônios da primeira camada
    avg_importance_mlp = np.mean(np.abs(mlp_input_weights), axis=0)
    feature_importance_mlp = list(zip(feature_names, avg_importance_mlp))
    feature_importance_mlp.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 variáveis mais importantes (Branch MLP):")
    for i, (feature, importance) in enumerate(feature_importance_mlp[:10]):
        print(f"{i+1:2d}. {feature:<35} {importance:.4f}")

    # 3. Análise dos pesos de combinação
    combination_weights = model.get_combination_weights()
    if combination_weights:
        print(f"\n--- Pesos de Combinação dos Branches ---")
        print(f"Peso do MLP (α): {combination_weights['alpha (MLP)']:.4f}")
        print(
            f"Peso da Regressão Logística (β): {combination_weights['beta (Logistic)']:.4f}")

        # Interpretação
        total_weight = combination_weights['alpha (MLP)'] + \
            combination_weights['beta (Logistic)']
        mlp_contribution = combination_weights['alpha (MLP)'] / \
            total_weight * 100
        logistic_contribution = combination_weights['beta (Logistic)'] / \
            total_weight * 100

        print(f"\nContribuição Relativa:")
        print(f"MLP: {mlp_contribution:.1f}%")
        print(f"Regressão Logística: {logistic_contribution:.1f}%")

    # 4. Análise de gradientes para o modelo híbrido
    print(f"\n--- Análise de Gradientes (Modelo Híbrido Completo) ---")
    model.eval()
    gradients = []

    for features, labels in test_loader:
        features.requires_grad_(True)
        outputs = model(features)

        # Calcular gradiente médio
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                if outputs[i, j].requires_grad:
                    # Garantir que o gradiente é calculado apenas uma vez por elemento de saída
                    grad = torch.autograd.grad(outputs[i, j], features,
                                               # Retain graph only if more computations are needed
                                               retain_graph=True if (
                                                   i < outputs.shape[0] - 1 or j < outputs.shape[1] - 1) else False,
                                               create_graph=False)[0]
                    if grad is not None:
                        gradients.append(torch.abs(grad[i]).detach().numpy())

        # Limitar para não sobrecarregar
        if len(gradients) > 1000:  # Limite para 1000 amostras para velocidade
            break

    if gradients:
        avg_gradients = np.mean(gradients, axis=0)
        gradient_importance = list(zip(feature_names, avg_gradients))

        # Ordenar por importância (decrescente) para as mais sensíveis
        gradient_importance_sorted_desc = sorted(
            gradient_importance, key=lambda x: x[1], reverse=True)
        print("Top 10 variáveis mais sensíveis (Gradientes):")
        for i, (feature, importance) in enumerate(gradient_importance_sorted_desc[:10]):
            print(f"{i+1:2d}. {feature:<35} {importance:.6f}")

        # Ordenar por importância (crescente) para as menos sensíveis
        gradient_importance_sorted_asc = sorted(
            gradient_importance, key=lambda x: x[1], reverse=False)
        print("\nTop 5 variáveis menos sensíveis (Gradientes):")
        for i, (feature, importance) in enumerate(gradient_importance_sorted_asc[:5]):
            print(f"{i+1:2d}. {feature:<35} {importance:.6f}")

    return feature_importance_logistic, feature_importance_mlp, combination_weights


# --- 5. FUNÇÃO PRINCIPAL ---


def main(csv_filepath, combination_method='weighted_sum'):
    try:
        df = pd.read_csv(csv_filepath)
        print(f"Arquivo '{csv_filepath}' carregado com sucesso.")
        print(f"Shape do dataset: {df.shape}")
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{csv_filepath}' não foi encontrado.")
        return

    # Verificar se Diet Quality existe e ajustar features numéricas
    if 'Diet Quality (1-10)' in df.columns:
        final_numerical_features = extended_numerical_features
    else:
        final_numerical_features = numerical_features
        print("Nota: Coluna 'Diet Quality (1-10)' não encontrada, continuando sem ela.")

    # Verificar colunas obrigatórias
    required_cols = final_numerical_features + \
        categorical_features + [target_variable]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERRO: Colunas obrigatórias não encontradas: {missing_cols}")
        return

    # Processamento da variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_variable])
    class_names = le.classes_
    num_classes = len(class_names)
    print(f"Variável alvo '{target_variable}' processada.")
    print(f"Classes encontradas: {class_names}")
    print(f"Número de classes: {num_classes}")

    # Preparação das features
    X = df[final_numerical_features + categorical_features]
    print(
        f"Features utilizadas: {len(final_numerical_features)} numéricas + {len(categorical_features)} categóricas")

    # Preprocessamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',
             sparse_output=False), categorical_features)
        ])

    # K-Fold Cross Validation
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Inicialização das listas de resultados
    accuracies, f1_macros, f1_weighteds = [], [], []
    best_accuracy = 0.0
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    best_model = None
    feature_names = None

    print(f"\n--- Iniciando {N_SPLITS}-Fold Cross Validation ---")
    print(f"Método de combinação: {combination_method}")

    for fold, (train_index, test_index) in enumerate(kf.split(X, y_encoded)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Preprocessamento
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        input_dim = X_train_processed.shape[1]

        # Obter nomes das features processadas (apenas no primeiro fold)
        if feature_names is None:
            num_feature_names = final_numerical_features
            cat_feature_names = preprocessor.named_transformers_[
                'cat'].get_feature_names_out(categorical_features)
            feature_names = num_feature_names + list(cat_feature_names)

        # Datasets e DataLoaders
        train_dataset = AnxietyDataset(X_train_processed, y_train)
        test_dataset = AnxietyDataset(X_test_processed, y_test)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Dimensão de entrada: {input_dim}")
        print(f"Tamanho do conjunto de treino: {len(train_dataset)}")
        print(f"Tamanho do conjunto de teste: {len(test_dataset)}")

        # --- MODELO HÍBRIDO ---
        print(f"\nTreinando Modelo Híbrido (MLP + Regressão Logística)...")
        hybrid_model = HybridMLPLogisticModel(input_dim, output_dim=num_classes,
                                              combination_method=combination_method)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(hybrid_model.parameters(), lr=LEARNING_RATE)

        train_model(hybrid_model, train_loader,
                    criterion, optimizer, "Híbrido")
        acc, f1m, f1w, conf_matrix = evaluate_model(hybrid_model, test_loader)

        accuracies.append(acc)
        f1_macros.append(f1m)
        f1_weighteds.append(f1w)
        total_confusion_matrix += conf_matrix

        print(
            f"Modelo Híbrido - Accuracy: {acc:.4f}, F1 (Macro): {f1m:.4f}, F1 (Weighted): {f1w:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = hybrid_model
            torch.save(hybrid_model.state_dict(),
                       f'best_hybrid_model_{combination_method}.pth')
            print(f"Novo melhor modelo salvo com accuracy de {acc:.4f}!")

    # --- RESULTADOS FINAIS ---
    print("\n" + "="*80)
    print("RESULTADOS FINAIS - MODELO HÍBRIDO (MLP + Regressão Logística)")
    print("="*80)

    print(f"\nMÉTRICAS DE PERFORMANCE (Média ± Desvio Padrão):")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1 (Macro): {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
    print(
        f"F1 (Weighted): {np.mean(f1_weighteds):.4f} ± {np.std(f1_weighteds):.4f}")

    # --- ANÁLISE DO MODELO HÍBRIDO ---
    if best_model and feature_names:
        # Recriar test_loader para análise
        X_processed = preprocessor.fit_transform(X)
        test_dataset_full = AnxietyDataset(X_processed, y_encoded)
        test_loader_full = DataLoader(
            test_dataset_full, batch_size=BATCH_SIZE, shuffle=False)

        analyze_hybrid_model(best_model, test_loader_full,
                             feature_names, class_names)

    # --- VISUALIZAÇÕES ---
    # Matriz de Confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.title(
        f'Matriz de Confusão - Modelo Híbrido ({combination_method})\n(Agregada - Todos os Folds)')
    plt.tight_layout()
    plt.show()

    # Gráfico de Performance por Fold
    plt.figure(figsize=(15, 5))

    folds = list(range(1, N_SPLITS + 1))

    plt.subplot(1, 3, 1)
    plt.plot(folds, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                alpha=0.7, label=f'Média: {np.mean(accuracies):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy por Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    plt.plot(folds, f1_macros, 'go-', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(f1_macros), color='r', linestyle='--',
                alpha=0.7, label=f'Média: {np.mean(f1_macros):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('F1 (Macro)')
    plt.title('F1 Macro por Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(1, 3, 3)
    plt.plot(folds, f1_weighteds, 'mo-', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(f1_weighteds), color='r', linestyle='--',
                alpha=0.7, label=f'Média: {np.mean(f1_weighteds):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('F1 (Weighted)')
    plt.title('F1 Weighted por Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print(
        f"\nModelo híbrido salvo: best_hybrid_model_{combination_method}.pth")
    print(f"Melhor accuracy: {best_accuracy:.4f}")


# --- 6. INICIALIZAÇÃO DO SCRIPT ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Treina modelo híbrido (MLP + Regressão Logística) para classificação de ansiedade social")
    parser.add_argument("csv_filepath", type=str,
                        help="Caminho para o arquivo CSV do Social Anxiety Dataset")
    parser.add_argument("--combination", type=str, default="weighted_sum",
                        choices=["weighted_sum", "concatenation", "attention"],
                        help="Método de combinação dos branches (default: weighted_sum)")
    args = parser.parse_args()

    print("="*80)
    print("CLASSIFICADOR HÍBRIDO DE ANSIEDADE SOCIAL")
    print("MLP + Regressão Logística")
    print("="*80)

    main(args.csv_filepath, args.combination)
