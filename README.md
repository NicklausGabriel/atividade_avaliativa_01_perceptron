# atividade_avaliativa_01_perceptron (Do Perceptron ao Aprendizado de Máquina)

1º atividade avaliativa inteligencia artificial proposta pelo professor: **Alexandre de Oliveira**

Implementando um Perceptron do Zero em Python

O **perceptron** é o modelo mais simples de **neurônio artificial**, criado por **Frank Rosenblatt** em 1958.
Imagine-o como um "tomador de decisões binário": ele recebe várias informações de entrada (inputs),
pondera a importância de cada uma, e decide entre duas opções **(sim/não, 0/1, verdadeiro/falso)**

**Analogia do Mundo Real:**

Pense em um porteiro de uma festa VIP. Ele observa várias características de cada pessoa:
 - Está na lista? (peso 0.8)
 - Está bem vestida? (peso 0.5)
 - Tem convite? (peso 0.9)

Se a soma ponderada dessas características ultrapassar um limite, a pessoa entra. Caso contrário, fica do
lado de fora. Esse é exatamente o princípio do perceptron!

**O que vamos implementar?**

Neste tutorial, você irá:
  1. ✅ Construir uma classe Perceptron do zero (sem bibliotecas de ML)
  2. ✅ Treinar o perceptron para aprender funções lógicas (AND, OR)
  3. ✅ Testar a limitação do modelo (problema XOR)
  4. ✅ Visualizar como o perceptron "desenha" fronteiras de decisão
  5. ✅ Experimentar com diferentes taxas de aprendizado
  6. ✅ Conectar com conceitos modernos de Deep Learning
     
**Requisitos**

  **Software:**
    Python 3.7 ou superior
  **Bibliotecas:** 
    numpy, matplotlib
  **Plataforma:**
    Google Colab (recomendado)
    Jupyter Notebook
    Qualquer IDE Python

**Conhecimento Prévio:**

  Conceitos básicos de Python
  Noções de álgebra (vetores, multiplicação)
  
**Como executar no Google Colab**

  1. Acesse colab.research.google.com
  2. Clique em "Novo notebook"
  3. Copie e cole os códigos deste tutorial
  4. Execute célula por célula com Shift+Enter
