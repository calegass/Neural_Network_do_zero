import os

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt


# scroll to the bottom to start coding your solution


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)


    # write your code here

    #  ================================================================================================================

    # PARTE 1
    # função para normalizar os dados
    def scale(X_train, X_test):
        """
        Normaliza os dados de entrada dividindo pelo valor máximo dos dados de treinamento.

        Args:
            X_train (np.ndarray): Dados de treinamento;
            X_test (np.ndarray): Dados de teste.

        Returns:
            tuple: uma tupla contendo os dados de treinamento e teste normalizados.
        """
        X_max = np.max(X_train)  # Encontra o valor máximo nos dados de treinamento
        X_train = X_train / X_max  # Normaliza os dados de treinamento
        X_test = X_test / X_max  # Normaliza os dados de teste usando o máximo dos dados de treinamento
        return X_train, X_test


    # função de inicialização xavier
    def xavier(n_in, n_out):
        """
        Inicializa os pesos usando a inicialização de Xavier.

        Args:
            n_in (int): Número de neurônios de entrada;
            n_out (int): Número de neurônios de saída.

        Returns:
            np.ndarray: uma matriz de pesos inicializados.
        """
        low = -np.sqrt(6 / (n_in + n_out))  # Limite inferior para a distribuição uniforme
        high = np.sqrt(6 / (n_in + n_out))  # Limite superior para a distribuição uniforme
        return np.random.uniform(low, high, (n_in, n_out))  # Retorna uma matriz de pesos inicializados aleatoriamente


    # função de ativação sigmoid
    def sigmoid(x):
        """
        Calcula a função de ativação sigmoid.

        Args:
            x (np.ndarray): Matriz de entrada.

        Returns:
            np.ndarray: ativação sigmoid aplicada à matriz de entrada.
        """
        return 1 / (1 + np.exp(-x))  # Calcula a função sigmoid elemento a elemento


    # RESPOSTAS PARTE 1
    # X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste

    # # Mostrar as listas
    # print([X_train[2][778].item(), X_test[0][774].item()],  # Imprime o elemento [2][778] de X_train e o elemento [0][774] de X_test
    #       xavier(2, 3).flatten().tolist(),  # Imprime uma matriz de pesos inicializada com Xavier (2 entradas, 3 saídas)
    #       sigmoid(np.array([-1, 0, 1, 2])).flatten().tolist())  # Imprime a função sigmoid aplicada a um array [-1, 0, 1, 2]

    #  ================================================================================================================

    # PARTE 2
    class OneLayerNeural:
        def __init__(self, n_features, n_classes):
            """
            Inicializa os pesos e biases da rede neural de uma camada.

            Args:
                n_features (int): Número de features de entrada;
                n_classes (int): Número de classes de saída.

            Returns:
                None
            """
            self.W = xavier(n_features, n_classes)  # Inicializa a matriz de pesos
            self.b = xavier(1, n_classes)  # Inicializa o vetor de bias

        def forward(self, X):
            """
            Calcula a passagem forward da rede neural de uma camada.

            Args:
                X (np.ndarray): Dados de entrada.

            Returns:
                np.ndarray: saída da rede neural.
            """
            return sigmoid(np.dot(X, self.W) + self.b)  # Calcula a saída da rede neural


    # RESPOSTAS PARTE 2
    # X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste
    # # Instanciar a classe
    # model = OneLayerNeural(X_train.shape[1], 10)  # Cria um modelo de rede neural de uma camada
    # # Mostrar a saída
    # print(model.forward(X_train[:2]).flatten().tolist())  # Imprime a saída da rede neural para as duas primeiras amostras de treinamento

    #  ================================================================================================================

    # PARTE 3
    def mse(y_pred, y_true):
        """
        Calcula o erro quadrático médio (MSE).

        Args:
            y_pred (np.ndarray): Valores previstos.
            y_true (np.ndarray): Valores reais.

        Returns:
            float: Erro quadrático médio.
        """
        return np.mean((y_pred - y_true) ** 2)  # Calcula o MSE


    def mse_prime(y_pred, y_true):
        """
        Calcula a derivada do erro quadrático médio (MSE).

        Args:
            y_pred (np.ndarray): Valores previstos.
            y_true (np.ndarray): Valores reais.

        Returns:
            np.ndarray: Derivada do erro quadrático médio.
        """
        return 2 * (y_pred - y_true)  # Calcula a derivada do MSE


    def sigmoid_prime(x):
        """
        Calcula a derivada da função de ativação sigmoid.

        Args:
            x (np.ndarray): Matriz de entrada.

        Returns:
            np.ndarray: derivada da função de ativação sigmoid.
        """
        return sigmoid(x) * (1 - sigmoid(x))  # Calcula a derivada da função sigmoid


    def backprop(self, X, y, alpha):
        """
        Realiza a retro propagação na rede neural de uma camada.

        Args:
            X (np.ndarray): Dados de entrada;
            y (np.ndarray): Valores reais;
            alpha (float): Taxa de aprendizado.

        Returns:
            None
        """
        error = (mse_prime(self.forward(X), y) * sigmoid_prime(np.dot(X, self.W) + self.b))  # Calcula o erro

        # Calcula o gradiente
        delta_W = (np.dot(X.T, error)) / X.shape[0]  # Gradiente dos pesos
        delta_b = np.mean(error, axis=0)  # Gradiente do bias

        # Atualiza os pesos e biases
        self.W -= alpha * delta_W  # Atualiza os pesos
        self.b -= alpha * delta_b  # Atualiza o bias


    # Adicionar a função backprop na classe OneLayerNeural
    OneLayerNeural.backprop = backprop


    # RESPOSTAS PARTE 3
    # X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste
    # # Testando as funções com os arrays fornecidos
    # y_pred = np.array([-1, 0, 1, 2])  # Define os valores previstos
    # y_true = np.array([4, 3, 2, 1])  # Define os valores reais
    #
    # # Teste da função MSE
    # mse_value = mse(y_pred, y_true)  # Calcula o MSE
    # mse_prime_value = mse_prime(y_pred, y_true)  # Calcula a derivada do MSE
    # sigmoid_prime_value = sigmoid_prime(y_pred)  # Calcula a derivada da função sigmoid
    #
    # model = OneLayerNeural(X_train.shape[1], 10)  # Cria um modelo de rede neural de uma camada
    # model.forward(X_train[:2])  # Calcula a saída da rede neural para as duas primeiras amostras de treinamento
    # model.backprop(X_train[:2], y_train[:2], 0.1)  # Realiza a retro propagação
    # mse_backprop_value = mse(model.forward(X_train[:2]), y_train[:2])  # Calcula o MSE após a retro propagação
    #
    # print([mse_value.item()],  # Imprime o MSE
    #       mse_prime_value.flatten().tolist(),  # Imprime a derivada do MSE
    #       sigmoid_prime_value.flatten().tolist(),  # Imprime a derivada da função sigmoid
    #       [mse_backprop_value.item()])  # Imprime o MSE após a retro propagação

    #  ================================================================================================================

    # PARTE 4
    def train(model, X, y, alpha, batch_size=100):
        """
        Treina a rede neural usando o gradiente descendente em mini-lotes.

        Args:
            model: modelo da rede neural a ser treinado.
            X (np.ndarray): Dados de treinamento;
            y (np.ndarray): Rótulos dos dados de treinamento;
            alpha (float): Taxa de aprendizado;
            batch_size (int): Tamanho do mini-lote.

        Returns:
            None
        """
        n = X.shape[0]  # Número de amostras de treinamento
        for i in range(0, n, batch_size):  # Itera sobre os mini-lotes
            model.backprop(X[i:i + batch_size], y[i:i + batch_size],
                           alpha)  # Realiza a retro propagação em cada mini-lote


    def accuracy(model, X, y):
        """
        Calcula a acurácia do modelo.

        Args:
            model: Modelo da rede neural.
            X (np.ndarray): Dados de entrada.
            y (np.ndarray): Rótulos dos dados de entrada.

        Returns:
            float: Acurácia do modelo.
        """
        y_pred = np.argmax(model.forward(X), axis=1)  # Obtém as classes preditas
        y_true = np.argmax(y, axis=1)  # Obtém as classes reais
        return np.mean(y_pred == y_true)  # Calcula a acurácia


    # RESPOSTAS PARTE 4
    # X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste
    # model = OneLayerNeural(X_train.shape[1], 10)  # Cria um modelo de rede neural de uma camada
    # # Treinando o modelo e calculando a acurácia
    # a1 = accuracy(model, X_test, y_test).flatten().tolist()  # Calcula a acurácia antes do treinamento
    # a2 = []  # Lista para armazenar as acurácias durante o treinamento
    # # Testando a função de treinamento
    # for _ in range(20):  # Itera 20 vezes
    #     train(model, X_train, y_train, 0.5)  # Treina o modelo
    #     a2.append(accuracy(model, X_test, y_test))  # Calcula a acurácia e adiciona à lista
    # # Mostrar as listas
    # print(a1, [i.item() for i in a2])  # Imprime a acurácia antes do treinamento e as acurácias durante o treinamento

    #  ================================================================================================================

    # PARTE 5
    # classe TwoLayerNeural
    class TwoLayerNeural:
        def __init__(self, n_features, n_classes, hidden_layer_size=64):
            """
            Inicializa os pesos e biases da rede neural de duas camadas.

            Args:
                n_features (int): Número de features de entrada.
                n_classes (int): Número de classes de saída.
                hidden_layer_size (int): Número de neurônios na camada oculta.

            Returns:
                None
            """
            self.W = [xavier(n_features, hidden_layer_size),
                      xavier(hidden_layer_size, n_classes)]  # Inicializa as matrizes de pesos
            self.b = [xavier(1, hidden_layer_size), xavier(1, n_classes)]  # Inicializa os vetores de bias

        def forward(self, X):
            """
            Calcula a passagem forward da rede neural de duas camadas.

            Args:
                X (np.ndarray): Dados de entrada.

            Returns:
                np.ndarray: saída da rede neural.
            """
            para_model = X  # Define a entrada da primeira camada
            for i in range(2):  # Itera sobre as camadas
                para_model = sigmoid(
                    para_model @ self.W[i] + self.b[i])  # Calcula a saída da camada atual
            return para_model  # Retorna a saída da última camada


    # RESPOSTAS PARTE 5
    # X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste
    # # Instanciar a classe
    # model = TwoLayerNeural(X_train.shape[1], 10)  # Cria um modelo de rede neural de duas camadas
    # print(model.forward(X_train[:2]).flatten().tolist())  # Imprime a saída da rede neural para as duas primeiras amostras de treinamento

    #  ================================================================================================================

    # PARTE 6

    # Adicionar a função backprop na classe TwoLayerNeural
    def backprop(self, X, y, alpha):
        """
        Realiza a retro propagação na rede neural de duas camadas.

        Args:
            X (np.ndarray): Dados de entrada;
            y (np.ndarray): Valores reais;
            alpha (float): Taxa de aprendizado.

        Returns:
            None
        """
        n = X.shape[0]  # Número de amostras de treinamento
        biases = np.ones((1, n))  # Vetor de uns para cálculo do bias

        yp = self.forward(X)  # Calcula a saída da rede neural

        # Calcula o gradiente da função de perda em relação ao bias da camada de saída
        loss_grad_1 = 2 * alpha / n * ((yp - y) * yp * (1 - yp))

        # Calcula a saída da primeira camada
        f1_out = sigmoid(np.dot(X, self.W[0]) + self.b[0])

        # Calcula o gradiente da função de perda em relação ao bias da primeira camada
        loss_grad_0 = np.dot(loss_grad_1, self.W[1].T) * f1_out * (1 - f1_out)

        # Atualiza os pesos e biases
        self.W[0] -= np.dot(X.T, loss_grad_0)  # Atualiza os pesos da primeira camada
        self.W[1] -= np.dot(f1_out.T, loss_grad_1)  # Atualiza os pesos da segunda camada

        self.b[0] -= np.dot(biases, loss_grad_0)  # Atualiza o bias da primeira camada
        self.b[1] -= np.dot(biases, loss_grad_1)  # Atualiza o bias da segunda camada


    TwoLayerNeural.backprop = backprop  # Adiciona a função backprop à classe TwoLayerNeural

    # RESPOSTAS PARTE 6
    # X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste
    # model = TwoLayerNeural(X_train.shape[1], y_train.shape[1])  # Cria um modelo de rede neural de duas camadas
    # model.backprop(X_train[:2], y_train[:2], 0.1)  # Realiza a retro propagação
    # y_pred = model.forward(X_train[:2])  # Calcula a saída da rede neural para as duas primeiras amostras de treinamento
    # print(mse(y_pred, y_train[:2]).flatten().tolist())  # Imprime o MSE após a retro propagação

    #  ================================================================================================================

    # PARTE 7
    # Treinando o modelo e calculando a acurácia
    X_train, X_test = scale(X_train, X_test)  # Normaliza os dados de treinamento e teste
    model = TwoLayerNeural(X_train.shape[1], 10)  # Cria um modelo de rede neural de duas camadas
    accuracies = []  # Lista para armazenar as acurácias
    for _ in range(20):  # Itera 20 vezes
        train(model, X_train, y_train, 0.5)  # Treina o modelo
        accuracies.append(accuracy(model, X_test, y_test))  # Calcula a acurácia e adiciona à lista
    # Mostrar as listas
    print([i.item() for i in accuracies])  # Imprime as acurácias
