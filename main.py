#Importa bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt

#configurar visualização
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 12

#Fixar seed para reprodutibilidade
np.random.seed(42)

print("Bibliotecas carregadas com sucesso!")

class Perceptron:
    """
        Implementação do perceptron de Rosenblatt (1958)
        Um neuronio artificial binario, que aprende a classificar dados linearmente separáveis.
    """
    def __init__(self, learning_rate=0.1, n_interations=100):
        """
        Inicializa o Perceptron
        Parametros:
        ----------
        learning_rate: float
            Taxa de aprendizado (n) - controla o tamanho dos ajustes nos pesos
            Valores típicos: 0.01 a 1.0
        n_interations: int
            Número máximo de épocas (passagens completas pelos dados de treino)
        """
        self.learning_rate = learning_rate
        self.n_interations - n_interations
        self.weights = None     # Peso (w) - serão inicializado no treino
        self.bias = None        # Bias (b) - será inicializado no treino
        self.erros_history = [] # Histórico de erros por época (para análise)

    def step_function(self, z):
        """
            Função de ativação: Função Degrau (Step function)
            Converte a soma ponderada em um decisão binária.
            É a função de ativação mais simples possível.

            Parametros:
            ----------
            z: float ou array
                Soma ponderada (produto escalar entre entradas e pesos + bias)

            Retorna:
            -------
            int ou array: 1 se z >= 0, caso contrario 0

            Por que usar essa função?
             - O perceptron original de Rosemblant usava essa função
             - Produz saidas binarias (0 ou 1)
             - Simples de entender e implementar
        """
        return np.where(z>= 0, 1,0)
    def predict(self, X):
        """
        Faz previsões para novos dados.
        Processo:
        1. Calcula o produto escalar entre entradas (X) e pesos (w)
        2. Adiciona o bias (b)
        3. Aplica a função de ativação

        Parametros:
        -----------
        X: array de shape (n_amostras, n_features)
            Dados de entrada para classificar

        Retorna:
        --------
        array: Previsões (0 ou 1 ) para cada amostra

        Fórmula:
        --------
        Ŷ = f(x.w+b)
        onde f é a função degrau
        """
        # Produto escalar: soma ponderada das entradas.
        linear_output = np.dot(X, self.weights) + self.bias
        # Aplicar função de ativação
        predictions = self.step_function(linear_output)
        return predictions
    def fit(self, X, y ):
        """
         Treina o perceptron usando os dados fornecidos

         Algoritmo de Treinamento:
         1. Inicializar pesos e bias com zeros
         2. Para cada época:
         a. Para cada exemplo (x, y):
         - Calcular previsão ŷ
         - Calcular erro: e = y - ŷ
         - Atualizar pesos: w = w + η × e × x
         - Atualizar bias: b = b + η × e
         3. Repetir até convergir ou atingir n_iterations

         Parâmetros:
         -----------
         X : array de shape (n_amostras, n_features)
         Dados de treinamento (entradas)

         y : array de shape (n_amostras,)
         Rótulos verdadeiros (saídas esperadas: 0 ou 1)

         Retorna:
         --------
         self : objeto Perceptron treinado
         """
        n_samples =n_features = X.shape





