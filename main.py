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
        self.n_interations = n_interations
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
    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        print("Iniciando treinamento...")
        print(f"Pesos iniciais: {self.weights}")
        print(f"Bias inicial: {self.bias}")
        print("_" * 50)

        for epoch in range(self.n_interations):
            erros = 0

            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i.reshape(1, -1))[0]
                error = y[idx] - prediction

                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * x_i
                    self.bias += update
                    erros += 1

            self.erros_history.append(erros)

            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1} | Erros: {erros}")

            if erros == 0:
                print("\nConvergiu com sucesso!")
                break

        if erros > 0:
            print("\n⚠ Treinamento finalizado sem convergência completa")

        return self

# Teste rápido da classe
print("\n" + "="*50)
print("CLASSE PERCEPTRON IMPLEMENTADA COM SUCESSO!")
print("="*50)

#Implementação: Treinando para AND
print("\n" + "="*60)
print("EXPERIMENTO 1: PORTA LÓGICA AND")
print("="*60)
# Criar dataset para AND
X_and = np.array([
 [0, 0], # Entrada 1
 [0, 1], # Entrada 2
 [1, 0], # Entrada 3
 [1, 1] # Entrada 4
])
y_and = np.array([0, 0, 0, 1]) # Saídas esperadas
print("\n📊 Dataset AND:")
print("Entradas (X):")
print(X_and)
print("\nSaídas esperadas (y):")
print(y_and)
print("\n" + "-"*60)

# Criar e treinar o perceptron
perceptron_and = Perceptron(learning_rate=0.1, n_interations=100)
perceptron_and.fit(X_and, y_and)
print("\n" + "-"*60)
print("📈 RESULTADOS DO TREINAMENTO")
print("-"*60)

# Fazer previsões
predictions_and = perceptron_and.predict(X_and)
print("\n🎯 Comparação: Esperado vs Previsto")
print("-"*40)
for i in range(len(X_and)):
    print(f"Entrada: {X_and[i]} | Esperado: {y_and[i]} | "
    f"Previsto: {predictions_and[i]} | "
    f"{'✅ CORRETO' if y_and[i] == predictions_and[i] else '❌ERRADO'}")
# Calcular acurácia
accuracy_and = np.mean(predictions_and == y_and) * 100
print(f"\n🎯 Acurácia: {accuracy_and:.2f}%")

def plot_decision_boundary(X, y, perceptron, title):
    """
    Plota os pontos de dados e a fronteira de decisão do perceptron

    A fr
    onteira de decisão é a linha que separa as duas classes.
    Matematicamente: w₁x₁ + w₂x₂ + b = 0
    """
    # Configurar o gráfico
    plt.figure(figsize=(10, 7))

    # Plotar pontos de dados
    # Classe 0 (círculos vermelhos)
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
    color='red', marker='o', s=200,
    edgecolors='black', linewidths=2,
    label='Classe 0', alpha=0.7)

    # Classe 1 (estrelas verdes)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
    color='green', marker='*', s=400,
    edgecolors='black', linewidths=2,
    label='Classe 1', alpha=0.7)
    
    #Calcular e plotar fronteira de decisão
    # A fronteira é definida por: w₁x₁ + w₂x₂ + b = 0
    # Resolvendo para x₂: x₂ = -(w₁x₁ + b) / w₂

    # Criar pontos x₁ de -0.5 a 1.5 (margem ao redor dos dados)
    x1_boundary = np.linspace(-0.5, 1.5, 100)

    # Calcular x₂ correspondentes
    # Evitar divisão por zero
    if perceptron.weights[1] != 0:
        x2_boundary = -(perceptron.weights[0] * x1_boundary + perceptron.bias) / perceptron.weights[1]
        plt.plot(x1_boundary, x2_boundary, 'b-', linewidth=2,
        label='Fronteira de Decisão')

    # Configurações do gráfico
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x₁ (Entrada 1)', fontsize=14, fontweight='bold')
    plt.ylabel('x₂ (Entrada 2)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

    # Adicionar anotações com os pesos
    equation = f'Fronteira: {perceptron.weights[0]:.2f}x₁ +{perceptron.weights[1]:.2f}x₂ + {perceptron.bias:.2f} = 0'
    plt.text(0.02, 0.98, equation, transform=plt.gca().transAxes,
    fontsize=11, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()
    # Plotar resultado para AND
    plot_decision_boundary(X_and, y_and, perceptron_and,
    'Perceptron - Porta Lógica AND')
    print("\n📊 Interpretação do Gráfico:")
    print("-" * 60)
    print("• A linha azul é a FRONTEIRA DE DECISÃO")
    print("• Tudo acima/à direita da linha: perceptron prevê 1")
    print("• Tudo abaixo/à esquerda da linha: perceptron prevê 0")
    print("• Círculos vermelhos: classe 0 (AND retorna 0)")
    print("• Estrelas verdes: classe 1 (AND retorna 1)")
    print("\n✅ Note que a linha separa perfeitamente as duas classes!")
    # Plotar evolução dos erros
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perceptron_and.errors_history) + 1),
    perceptron_and.errors_history,
    marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Época', fontsize=14, fontweight='bold')
    plt.ylabel('Número de Erros', fontsize=14, fontweight='bold')
    plt.title('Curva de Aprendizado - AND', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n📊 Análise da Curva de Aprendizado:")
    print("-" * 60)
    print(f"• Número total de épocas: {len(perceptron_and.errors_history)}")
    print(f"• Erros na época 1: {perceptron_and.errors_history[0]}")
    print(f"• Erros na última época: {perceptron_and.errors_history[-1]}")
    print(f"• Convergência: {'SIM ✅' if perceptron_and.errors_history[-1] == 0
    else 'NÃO ❌'}")
    print("\n💡 O que isso significa:")
    print(" O perceptron começou errando algumas previsões,")
    print(" mas aprendeu rapidamente e chegou a zero erros.")
    print(" Isso acontece porque AND é linearmente separável!")
    
    print("\n" + "="*60)
    print("EXPERIMENTO 2: PORTA LÓGICA OR")
    print("="*60)
    # Criar dataset para OR
    X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ])
    y_or = np.array([0, 1, 1, 1]) # Saídas esperadas
    print("\n📊 Dataset OR:")
    print("Entradas (X):")
    print(X_or)
    print("\nSaídas esperadas (y):")
    print(y_or)
    print("\n" + "-"*60)
    # Criar e treinar o perceptron
    perceptron_or = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron_or.fit(X_or, y_or)
    print("\n" + "-"*60)
    print("📈 RESULTADOS DO TREINAMENTO")
    print("-"*60)
    # Fazer previsões
    predictions_or = perceptron_or.predict(X_or)
    print("\n🎯 Comparação: Esperado vs Previsto")
    print("-"*40)
    for i in range(len(X_or)):
        print(f"Entrada: {X_or[i]} | Esperado: {y_or[i]} | "
        f"Previsto: {predictions_or[i]} | "
        f"{'✅ CORRETO' if y_or[i] == predictions_or[i] else '❌ ERRADO'}")
    # Calcular acurácia
    accuracy_or = np.mean(predictions_or == y_or)
    
    # Visualizar
    y(X_or, y_or, perceptron_or,'Perceptron - Porta Lógica OR')
    print("\n" + "="*60)
    print("COMPARAÇÃO: AND vs OR")
    print("="*60)
    print("\n📊 Pesos Finais:")
    print("-" * 40)
    print(f"AND - Peso 1: {perceptron_and.weights[0]:.4f}")
    print(f"AND - Peso 2: {perceptron_and.weights[1]:.4f}")
    print(f"AND - Bias: {perceptron_and.bias:.4f}")
    print()
    print(f"OR - Peso 1: {perceptron_or.weights[0]:.4f}")
    print(f"OR - Peso 2: {perceptron_or.weights[1]:.4f}")
    print(f"OR - Bias: {perceptron_or.bias:.4f}")
    print("\n💡 Interpretação:")
    print("-" * 60)
    print("• Para AND: pesos maiores e bias mais negativo")
    print(" → Precisa de MAIS evidência para ativar (ambas entradas)")
    print()
    print("• Para OR: pesos menores e bias menos negativo")
    print(" → Precisa de MENOS evidência para ativar (qualquer entrada)")
    print()
    print("• Ambos convergem rapidamente porque são linearmente separáveis!")
    
    print("\n" + "="*60)
    print("EXPERIMENTO 3: PORTA LÓGICA XOR")
    print("="*60)
    # Criar dataset para XOR
    X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ])
    y_xor = np.array([0, 1, 1, 0]) # Saídas esperadas
    print("\n📊 Dataset XOR:")
    print("Entradas (X):")
    print(X_xor)
    print("\nSaídas esperadas (y):")
    print(y_xor)
    print("\n⚠ ATENÇÃO: XOR é NÃO linearmente separável!")
    print(" O perceptron NÃO deve conseguir aprender este padrão.")
    print("\n" + "-"*60)
    # Criar e treinar o perceptron
    perceptron_xor = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron_xor.fit(X_xor, y_xor)
    print("\n" + "-"*60)
    print("📈 RESULTADOS DO TREINAMENTO")
    print("-"*60)
    # Fazer previsões
    predictions_xor = perceptron_xor.predict(X_xor)
    print("\n🎯 Comparação: Esperado vs Previsto")
    print("-"*40)
    for i in range(len(X_xor)):
        print(f"Entrada: {X_xor[i]} | Esperado: {y_xor[i]} | "
        f"Previsto: {predictions_xor[i]} | "
        f"{'✅ CORRETO' if y_xor[i] == predictions_xor[i] else '❌ERRADO'}")
    # Calcular acurácia
    accuracy_xor = np.mean(predictions_xor == y_xor) * 100
    print(f"\n🎯 Acurácia: {accuracy_xor:.2f}%")
    if accuracy_xor < 100:
        print("\n⚠ FALHA ESPERADA!")
        print(" O perceptron não conseguiu aprender XOR perfeitamente.")
    # Visualizar
    plot_decision_boundary(X_xor, y_xor, perceptron_xor,'Perceptron - Porta Lógica XOR (FALHA ESPERADA)')
    print("\n" + "="*60)
    print("POR QUE O PERCEPTRON FALHA NO XOR?")
    print("="*60)
    print("\n🔍 Análise Geométrica:")
    print("-" * 60)
    print("1. O perceptron só consegue desenhar LINHAS RETAS")
    print("2. XOR precisa de uma separação em FORMATO DE CRUZ ou CURVA")
    print("3. Matematicamente: XOR é NÃO linearmente separável")
    print()
    print("Visualizando:")
    print()
    print(" x₂")
    print(" 1│ [0] [1]")
    print(" │")
    print(" 0│ [0] [1]")
    print(" └────────────── x₁")
    print(" 0 1")
    print()
    print("Impossível separar com uma linha reta!")
    print()
    print("💡 Solução Histórica:")
    print(" • 1969: Minsky & Papert publicam 'Perceptrons'")
    print(" • Provam matematicamente a limitação")
    print(" • Causa o primeiro 'inverno da IA'")
    print(" • Solução: Redes Neurais Multicamadas (1986)")
    
    # Plotar curva de aprendizado do XOR
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perceptron_xor.errors_history) + 1),
    perceptron_xor.errors_history,
    marker='o', linewidth=2, markersize=8, color='red')
    plt.xlabel('Época', fontsize=14, fontweight='bold')
    plt.ylabel('Número de Erros', fontsize=14, fontweight='bold')
    plt.title('Curva de Aprendizado - XOR (Não Converge)',
    fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n📊 Análise da Curva:")
    print("-" * 60)
    print("• Note que o número de erros NÃO chega a zero")
    print("• O perceptron fica 'preso' tentando aprender")
    print("• Não há configuração de pesos que resolva XOR com uma linha")
    print("• Este foi um resultado CRUCIAL na história da IA")
    
    print("\n" + "="*60)
    print("EXPERIMENTO 4: IMPACTO DA TAXA DE APRENDIZADO")
    print("="*60)
    # Testar diferentes taxas de aprendizado
    learning_rates = [0.01, 0.1, 1.0]
    colors = ['blue', 'green', 'red']
    plt.figure(figsize=(14, 5))
    for idx, lr in enumerate(learning_rates):
        print(f"\n{'='*60}")
        print(f"Taxa de Aprendizado: {lr}")
        print(f"{'='*60}")

        # Treinar perceptron com esta taxa
        perceptron_lr = Perceptron(learning_rate=lr, n_iterations=100)
        perceptron_lr.fit(X_and, y_and)

        # Plotar curva de aprendizado
        plt.subplot(1, 3, idx + 1)
        plt.plot(range(1, len(perceptron_lr.errors_history) + 1),
        perceptron_lr.errors_history,
        marker='o', linewidth=2, markersize=6, color=colors[idx])
        plt.xlabel('Época', fontsize=12, fontweight='bold')
        plt.ylabel('Erros', fontsize=12, fontweight='bold')
        plt.title(f'η = {lr}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.5, 5)
    plt.tight_layout()
    plt.show()
    print("\n" + "="*60)
    print("ANÁLISE DOS RESULTADOS")
    print("="*60)
    print("\n🐌 Taxa Baixa (η = 0.01):")
    print("-" * 60)
    print("• Convergência LENTA")
    print("• Passos muito pequenos")
    print("• Seguro, mas ineficiente")
    print("• Use quando: dados ruidosos ou instabilidade")
    print("\n⚖ Taxa Moderada (η = 0.1):")
    print("-" * 60)
    print("• EQUILÍBRIO ideal")
    print("• Convergência rápida e estável")
    print("• Valor PADRÃO recomendado")
    print("• Use quando: situação típica")
    print("\n🚀 Taxa Alta (η = 1.0):")
    print("-" * 60)
    print("• Convergência muito rápida OU instabilidade")
    print("• Passos grandes podem 'pular' a solução")
    print("• Pode oscilar sem convergir em problemas complexos")
    print("• Use quando: problema simples e quer velocidade")
    print("\n💡 Regra Prática:")
    print("-" * 60)
    print("• Comece com η = 0.1 (valor padrão)")
    print("• Se não convergir: reduza (ex: 0.01)")
    print("• Se convergir muito devagar: aumente (ex: 0.5)")
    print("• Problemas difíceis: taxas menores (0.001 - 0.1)")
    print("• Problemas simples: taxas maiores (0.1 - 1.0)")
    
    # Criar tabela comparativa
    print("\n" + "="*60)
    print("TABELA COMPARATIVA: IMPACTO DA TAXA DE APRENDIZADO")
    print("="*60)
    results = []
    for lr in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        perceptron_test = Perceptron(learning_rate=lr, n_iterations=100)
        perceptron_test.fit(X_and, y_and)

        results.append({
        'Taxa': lr,
        'Épocas': len(perceptron_test.errors_history),
        'Convergiu': perceptron_test.errors_history[-1] == 0,
        'Erros_Finais': perceptron_test.errors_history[-1]
        })
    print("\n{:<10} {:<10} {:<12} {:<15}".format(
    "Taxa (η)", "Épocas", "Convergiu?", "Erros Finais"
    ))
    
    print("-" * 60)
    for r in results:
        print("{:<10} {:<10} {:<12} {:<15}".format(
        r['Taxa'],
        r['Épocas'],
        "✅ Sim" if r['Convergiu'] else "❌ Não",
        r['Erros_Finais']
        ))
    print("\n" + "="*60)
    print("\n" + "="*60)
    print("DO PERCEPTRON AO DEEP LEARNING")
    print("="*60)
    print(""" 📜 LINHA DO TEMPO E EVOLUÇÃO:
    1943 - McCulloch & Pitts
    │ Primeiro modelo matemático de neurônio
    │
    1958 - Rosenblatt
    │ PERCEPTRON (o que implementamos hoje!)
    │ ✅ Aprende padrões lineares
    │ ❌ Não resolve XOR
    │
    1969 - Minsky & Papert
    │ Livro "Perceptrons" → Primeiro Inverno da IA
    │ Provam limitações matemáticas
        │
    1986 - Rumelhart, Hinton & Williams
    │ BACKPROPAGATION + Redes Multicamadas
    │ ✅ Resolve XOR com camada oculta
    │ ✅ Funções não-lineares
    │
    2012 - Krizhevsky (AlexNet)
    │ Deep Learning vence ImageNet
    │ Revolução da IA moderna
    │
    2017-2024 - Era Transformer
    │ GPT, BERT, ChatGPT, Gemini
    │ Bilhões de parâmetros
    │ Mas o PRINCÍPIO é o mesmo! ⚡
    """)
    print("🧠 Como o Perceptron se Conecta com Redes Modernas:")
    print("-" * 60)
    print()
    print("1. NEURÔNIO BÁSICO (Perceptron):")
    print(" ŷ = f(w·x + b)")
    print()
    print("2. REDE NEURAL MULTICAMADAS:")
    print(" • Empilhar múltiplos perceptrons")
    print(" • Adicionar camadas ocultas")
    print(" • Resolver problemas não-lineares (XOR!)")
    print()
    print("3. DEEP LEARNING:")
    print(" • Muitas camadas (10, 50, 100+)")
    print(" • Milhões/bilhões de pesos")
    print(" • Mesma regra de atualização (gradient descent)")
    print()
    print("💡 CONCLUSÃO:")
    print(" O perceptron é o 'átomo' das redes neurais modernas!")
    print(" Tudo que você aprendeu hoje escala para GPT, BERT, etc.")
    print("\n" + "="*60)
    print("COMO RESOLVER XOR: VISÃO GERAL")
    print("="*60)
    print(""" ❌ Perceptron Simples (1 camada):

    x₁ ─────┐
    ├──→ Σ ──→ f() ──→ ŷ
    Pode:
    • AND, OR, NOT
    • Qualquer função linearmente separável

    Não Pode:
    • XOR
    • Problemas complexos
    ✅ Rede Neural Multicamadas (2+ camadas):

    Camada Oculta Camada Saída
    x₁ ─────┬──→ h₁ ─────┐
    │ ├──→ ŷ
    x₂ ─────┼──→ h₂ ─────┘
    │
    └──→ h₃

    Pode:
    • XOR ✓
    • Funções não-lineares complexas ✓
    • Classificação de imagens ✓
    • Reconhecimento de fala ✓
    🔑 DIFERENÇA FUNDAMENTAL:

    • Camada oculta cria REPRESENTAÇÕES INTERMEDIÁRIAS
    • Neurônios ocultos aprendem FEATURES úteis
    • Combina múltiplas fronteiras lineares
    • Resultado: capacidade de modelar NÃO-LINEARIDADE
    📚 Para Aprender Mais:

    • Próximas aulas: implementaremos MLP (Multi-Layer Perceptron)
    • Veremos backpropagation (como treinar redes profundas)
    • Aplicações práticas: MNIST, classificação de imagens
    """)
    print("\n" + "="*60)
    print("QUANDO USAR (E NÃO USAR) PERCEPTRON")
    print("="*60)
    print("\n✅ USE PERCEPTRON QUANDO:")
    print("-" * 60)
    print("• Problema é linearmente separável")
    print("• Classificação binária simples")
    print("• Poucos features (< 100)")
    print("• Precisa de modelo interpretável")
    print("• Dados pequenos/limitados")
    print("• Quer entender fundamentos de ML")
    print()
    print("Exemplos:")
    print(" → Classificar emails (spam: sim/não) com palavras-chave")
    print(" → Aprovar crédito baseado em 3-4 variáveis simples")
    print(" → Detectar fraude com regras lineares")
    print("\n❌ NÃO USE PERCEPTRON QUANDO:")
    print("-" * 60)
    print("• Problema é não-linear (ex: XOR)")
    print("• Múltiplas classes (>2)")
    print("• Dados complexos (imagens, áudio, texto)")
    print("• Padrões sutis e hierárquicos")
    print("• Grande volume de dados")
    print("• Estado da arte é necessário")
    print()
    print("Exemplos:")
    print(" → Reconhecimento facial → Use CNN (Redes Convolucionais)")
    print(" → Análise de sentimento de texto → Use Transformers")
    print(" → Jogar xadrez → Use Redes Profundas + RL")
    print(" → Classificar 1000 objetos → Use ResNet, EfficientNet")
    print("\n🎯 REGRA DE OURO:")
    print("-" * 60)
    print(" Se você pode visualizar os dados em 2D/3D")
    print(" e separar com uma linha/plano,")
    print(" então o perceptron PODE funcionar.")
    print()
    print(" Se não consegue visualizar uma separação linear,")
    print(" você PRECISA de redes mais profundas.")
    print("\n" + "="*60)
    print("CHECKLIST: O QUE VOCÊ APRENDEU HOJE")
    print("="*60)
    checklist = [
        ("Conceito de neurônio artificial",
        "Unidade básica que processa informação"),

        ("Arquitetura do perceptron",
        "Entradas, pesos, bias, soma ponderada, ativação"),

        ("Função de ativação (step function)",
        "Converte números em decisões binárias"),
        ("Regra de aprendizado do perceptron",
    "w = w + η × erro × x"),

    ("Implementação do zero em Python",
    "Classe completa sem bibliotecas de ML"),

    ("Problemas linearmente separáveis",
    "AND, OR - perceptron RESOLVE"),

    ("Problemas não-linearmente separáveis",
    "XOR - perceptron FALHA"),

    ("Taxa de aprendizado (η)",
    "Controla velocidade e estabilidade"),

    ("Fronteira de decisão",
    "Linha que separa classes"),

    ("Convergência",
    "Quando o modelo para de errar"),

    ("Limitação histórica",
    "Inverno da IA (1969) por causa do XOR"),

    ("Conexão com Deep Learning",
    "Perceptron é o bloco básico de redes modernas")
    ]
    for i, (concept, description) in enumerate(checklist, 1):
        print(f"\n{i:2d}. ✅ {concept}")
        print(f" → {description}")
    print("\n" + "="*60)
    print("\n" + "="*60)
    print("EXERCÍCIOS PARA PRATICAR")
    print("="*60)
    print(""" 🎯 NÍVEL BÁSICO (Consolidar Conceitos):
    1. Modifique a taxa de aprendizado para 0.5 e treine AND
    • Compare com η=0.1
    • Qual converge mais rápido?
    2. Implemente a função lógica NAND (NOT AND)
    • Tabela verdade: (0,0)→1, (0,1)→1, (1,0)→1, (1,1)→0
    • O perceptron consegue aprender?
    3. Teste o perceptron com dados NOR (NOT OR)
    • Tabela verdade: (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→0
    • Plote a fronteira de decisão
    🎯 NÍVEL INTERMEDIÁRIO (Explorar Limites):
    4. Crie um dataset com 3 features (ao invés de 2)
    • Exemplo: [x1, x2, x3] → y
    • Treine o perceptron
    • Como visualizar em 3D?
    5. Adicione ruído aos dados
    • Use: X + np.random.normal(0, 0.1, X.shape)
    • O perceptron ainda converge?
    • Teste diferentes níveis de ruído
    6. Implemente uma função de ativação diferente
    • Sigmoid: σ(z) = 1 / (1 + e^(-z))
    • Compare com step function
    🎯 NÍVEL AVANÇADO (Aprofundar):
    7. Implemente validação cruzada
    • Divida dados em treino/teste
    • Avalie generalização
    8. Adicione regularização L2
    • Penalizar pesos grandes
    • w = w - η × λ × w (a cada época)
    9. Crie visualização interativa
    • Use ipywidgets no Colab
    • Sliders para η, número de épocas
    • Atualização em tempo real
    10. Pesquise o ADALINE (Adaptive Linear Neuron)
    • Diferença para o Perceptron
    • Implemente e compare
    """)
    print("\n" + "="*60)
    print("\n" + "="*60)
    print("RECURSOS PARA APROFUNDAMENTO")
    print("="*60)
    print(""" 📚 LEITURA RECOMENDADA:
    • Livros:
    1. "Neural Networks and Deep Learning" - Michael Nielsen
    → http://neuralnetworksanddeeplearning.com (GRATUITO!)

    2. "Deep Learning" - Goodfellow, Bengio, Courville
    → https://www.deeplearningbook.org (GRATUITO!)

    3. "Make Your Own Neural Network" - Tariq Rashid
    → Excelente para iniciantes
    • Papers Históricos:
    1. Rosenblatt (1958) - "The Perceptron"
    2. Minsky & Papert (1969) - "Perceptrons"
    3. Rumelhart et al. (1986) - "Learning representations by backpropagation"
    🎥 VÍDEOS:
    • 3Blue1Brown - Neural Networks Series
    → Visualizações INCRÍVEIS de como redes neurais funcionam

    • StatQuest - Neural Networks Clearly Explained
    → Didático e acessível
    • Andrew Ng - Machine Learning Course (Coursera)
    → Curso completo, gratuito para auditar
    💻 CÓDIGO E PRÁTICA:
    • Kaggle - Learn Machine Learning
    → Tutoriais interativos gratuitos

    • Google Colab - Notebooks prontos
    → Busque "perceptron implementation"

    • GitHub - Projetos de exemplo
    → Procure: "perceptron python implementation"
    🔧 FERRAMENTAS PROFISSIONAIS:
    • scikit-learn: biblioteca de ML do Python
    → from sklearn.linear_model import Perceptron

    • TensorFlow/PyTorch: frameworks de Deep Learning
    → Próximos passos após dominar o básico
    • Keras: API de alto nível para redes neurais
     → Facilita construir modelos complexos
    """)
    print("\n" + "="*60)