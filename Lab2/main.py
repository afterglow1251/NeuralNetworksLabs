import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


# Навчальний набір даних
training_data = [
    # |-
    (np.array([[0, 1, 1, 0, 0, 0],
               [0, 1, 1, 0, 0, 0],
               [0, 1, 1, 1, 1, 1],
               [0, 1, 1, 1, 1, 1],
               [0, 1, 1, 0, 0, 0],
               [0, 1, 1, 0, 0, 0]]), np.array([1, 0, 0, 0, 0]), '|-'),
    # Ʇ
    (np.array([[0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0]]), np.array([0, 1, 0, 0, 0]), 'Ʇ'),
    # -|
    (np.array([[0, 0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1, 0],
               [1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 0],
               [0, 0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1, 0]]), np.array([0, 0, 1, 0, 0]), '-|'),
    # +
    (np.array([[0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0]]), np.array([0, 0, 0, 1, 0]), '+'),

    # T
    (np.array([[0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0]]), np.array([0, 0, 0, 0, 1]), 'T')
]


class NeuralNetworkNoHiddenLayer:
    def __init__(self):
        # Ініціалізація ваг
        self.weights = np.random.rand(36, 5)  # 36 вхідних значень (6x6 матриця) та 5 вихідних класів

    def train(self, training_data, epochs, learning_rate):  # Optimally 700 epochs
        for _ in range(epochs):
            for inputs, target, _ in training_data:
                # Перетворення вхідних даних у вектор
                inputs = inputs.flatten()

                # Знаходження виходів кожного шару
                output = sigmoid(np.dot(inputs, self.weights))

                # Визначення помилок
                error = target - output

                # Обчислення корекції ваг
                adjustment = np.dot(inputs.reshape(-1, 1), error.reshape(1, -1))

                # Оновлення ваг
                self.weights += learning_rate * adjustment

    def predict(self, inputs):
        # Перетворення вхідних даних у вектор
        inputs = inputs.flatten()
        # Знаходження виходів
        output = sigmoid(np.dot(inputs, self.weights))
        # Повернення індексу класу з найбільшим вихідним значенням
        return np.argmax(output)


class NeuralNetworkOneHiddenLayer:
    def __init__(self):
        # Ініціалізація ваг
        self.hidden_weights = np.random.rand(36, 37)  # Ваги для прихованого шару
        self.output_weights = np.random.rand(37, 5)  # Ваги для вихідного шару

    def train(self, training_data, epochs, learning_rate):  # Optimally 1000 epochs
        for _ in range(epochs):
            for inputs, target, _ in training_data:
                # Перетворення вхідних даних у вектор
                inputs = inputs.flatten()

                # Знаходження виходів кожного шару
                hidden_output = sigmoid(np.dot(inputs, self.hidden_weights))
                output = sigmoid(np.dot(hidden_output, self.output_weights))

                # Визначення помилок
                output_error = target - output
                hidden_error = np.dot(output_error, self.output_weights.T) * sigmoid_derivative(hidden_output)

                # Обчислення корекції ваг
                output_adjustment = np.dot(hidden_output.reshape(-1, 1), output_error.reshape(1, -1))
                hidden_adjustment = np.dot(inputs.reshape(-1, 1), hidden_error.reshape(1, -1))

                # Оновлення ваг
                self.output_weights += learning_rate * output_adjustment
                self.hidden_weights += learning_rate * hidden_adjustment

    def predict(self, inputs):
        inputs = inputs.flatten()
        hidden_output = sigmoid(np.dot(inputs, self.hidden_weights))
        output = sigmoid(np.dot(hidden_output, self.output_weights))
        return np.argmax(output)


class NeuralNetworkTwoHiddenLayers:
    def __init__(self):
        # Ініціалізація ваг
        self.hidden1_weights = np.random.rand(36, 37)  # Ваги для першого прихованого шару
        self.hidden2_weights = np.random.rand(37, 37)  # Ваги для другого прихованого шару
        self.output_weights = np.random.rand(37, 5)  # Ваги для вихідного шару

    def train(self, training_data, epochs, learning_rate):  # Optimally 2000 epochs
        for _ in range(epochs):
            for inputs, target, _ in training_data:
                # Перетворення вхідних даних у вектор
                inputs = inputs.flatten()

                # Знаходження виходів кожного шару
                hidden1_output = sigmoid(np.dot(inputs, self.hidden1_weights))
                hidden2_output = sigmoid(np.dot(hidden1_output, self.hidden2_weights))
                output = sigmoid(np.dot(hidden2_output, self.output_weights))

                # Визначення помилок
                output_error = target - output
                hidden2_error = np.dot(output_error, self.output_weights.T) * sigmoid_derivative(hidden2_output)
                hidden1_error = np.dot(hidden2_error, self.hidden2_weights.T) * sigmoid_derivative(hidden1_output)

                # Обчислення корекції ваг
                output_adjustment = np.dot(hidden2_output.reshape(-1, 1), output_error.reshape(1, -1))
                hidden2_adjustment = np.dot(hidden1_output.reshape(-1, 1), hidden2_error.reshape(1, -1))
                hidden1_adjustment = np.dot(inputs.reshape(-1, 1), hidden1_error.reshape(1, -1))

                # Оновлення ваг
                self.output_weights += learning_rate * output_adjustment
                self.hidden2_weights += learning_rate * hidden2_adjustment
                self.hidden1_weights += learning_rate * hidden1_adjustment

    def predict(self, inputs):
        inputs = inputs.flatten()
        hidden1_output = sigmoid(np.dot(inputs, self.hidden1_weights))
        hidden2_output = sigmoid(np.dot(hidden1_output, self.hidden2_weights))
        output = sigmoid(np.dot(hidden2_output, self.output_weights))
        return np.argmax(output)


# Навчання нейромережі
nn = NeuralNetworkOneHiddenLayer()
nn.train(training_data, epochs=1000, learning_rate=0.01)


# Функція для отримання символу за отриманим індексом
def predict_symbol(test_input):
    prediction = nn.predict(test_input)
    return training_data[prediction][2]


if __name__ == "__main__":
    print('Прогнозований символ:', predict_symbol(
        np.array([[0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0]])
    ), '| Реальний: |-')

    print('Прогнозований символ:', predict_symbol(
        np.array([[0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [1, 1, 1, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0]])
    ), '| Реальний: Ʇ')

    print('Прогнозований символ:', predict_symbol(
        np.array([[0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 0],
                  [1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0]])
    ), '| Реальний: -|')

    print('Прогнозований символ:', predict_symbol(
        np.array([[1, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0]])
    ), '| Реальний: +')

    print('Прогнозований символ:', predict_symbol(
        np.array([[0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0]])
    ), '| Реальний: T')
