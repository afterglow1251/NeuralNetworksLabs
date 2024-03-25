import numpy as np

"""
time series
1.69, 3.38, 1.40, 5.56, 1.86, 5.62, 0.46, 5.51, 0.26, 5.13, 1.18, 5.98, 1.36, 5.09, 1.29
"""

# Learning rate
LEARNING_RATE = 1

# Define dataset
dataset = np.array([
    [1.69, 3.38, 1.40, 5.56],
    [3.38, 1.40, 5.56, 1.86],
    [1.40, 5.56, 1.86, 5.62],
    [5.56, 1.86, 5.62, 0.46],
    [1.86, 5.62, 0.46, 5.51],
    [5.62, 0.46, 5.51, 0.26],
    [0.46, 5.51, 0.26, 5.13],
    [5.51, 0.26, 5.13, 1.18],
    [0.26, 5.13, 1.18, 5.98],
    [5.13, 1.18, 5.98, 1.36]]) / 10

# Initialize weights for neurons
weights_h = np.random.rand(3, 3)  # hidden layer
weights_o = np.random.rand(3)  # output layer


# Sigmoid function
def sigmoid(x: int | float) -> float:
    return 1 / (1 + np.exp(-x))


# Neuron activation function
def activation(inputs: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the activation of a neuron
    based on the inputs and weights
    using the sigmoid activation function
    """
    return sigmoid(np.dot(inputs, weights))


# Test function
def test(arr: list | tuple) -> int | float:
    arr_np = np.array(arr) / 10
    # hidden layer results
    res_h = np.array([activation(arr_np, weights_h[i]) for i in range(3)])
    # output layer result
    res_o = activation(res_h, weights_o)
    return res_o * 10


print("Results before learning:")
for data in dataset:
    expected = data[-1] * 10
    result = test(data[:3])
    print(f'Expected: {expected} | Got: {result} | error: {expected - result}')
print('\n')

# Training loop
while True:
    # select a random set from the dataset
    rand_set = dataset[np.random.randint(0, len(dataset))]

    # hidden layer results
    res_h = np.array([activation(rand_set[:3], weights_h[i]) for i in range(3)])

    # output layer result
    res_o = activation(res_h, weights_o)

    # network error
    error = rand_set[-1] - res_o

    # Output layer delta
    delta_o = error * res_o * (1 - res_o)  # res_o * (1 - res_o) - derivative of the sigmoid function

    # Hidden layer delta
    delta_h = res_h * (1 - res_h) * weights_o * delta_o  # res_h * (1 - res_h) - derivative of the sigmoid function

    # Update weights of the output layer
    weights_o += LEARNING_RATE * delta_o * res_h

    # Update weights of the hidden layer
    weights_h += LEARNING_RATE * delta_h[:, np.newaxis] * rand_set[:3]

    if abs(error) < 1e-5:
        break  # If the error is sufficiently small, stop the training loop

print("Results after learning:")
for data in dataset:
    expected = data[-1] * 10
    result = test(data[:3])
    print(f'Expected: {expected} | Got: {result} | error: {expected - result}')

print('\n\nTesting:')
# Test the network
got1 = test([1.18, 5.98, 1.36])
print(f'Expected: {5.09} | Got: {got1} | error: {5.09 - got1}')

got2 = test([5.98, 1.36, 5.09])
print(f'Expected: {1.29} | Got: {got2} | error: {1.29 - got2}')
