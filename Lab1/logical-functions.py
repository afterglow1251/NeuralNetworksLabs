# Logical AND
def logical_and(inputs):
    weights = [1, 1]
    S = dot_product(weights, inputs)
    return 1 if S >= 1.5 else 0


# Logical OR
def logical_or(inputs):
    weights = [1, 1]
    S = dot_product(weights, inputs)
    return 1 if S >= 0.5 else 0


# Logical NOT
def logical_not(input):
    weight = -1.5
    S = weight * input[0]
    return 1 if S >= -1 else 0


# Logical XOR
def logical_xor(inputs):
    # Weights for the hidden layer
    weights1_h = [1, -1]
    weights2_h = [-1, 1]

    # outputs of the neurons of the hidden layer
    out1_h = activation(dot_product(weights1_h, inputs))
    out2_h = activation(dot_product(weights2_h, inputs))

    outs_h = [out1_h, out2_h]

    # Weights for the output layer neuron
    weights_o = [1, 1]

    # output of the neuron of the output layer
    return activation(dot_product(weights_o, outs_h))


# Dot product function
def dot_product(weights, inputs):
    return sum(weight * input_val for weight, input_val in zip(weights, inputs))


# Activation function
def activation(S):
    return 1 if S >= 0.5 else 0


# Logical AND
print("Logical AND:")
print(f"0 AND 0 = {logical_and([0, 0])}")
print(f"0 AND 1 = {logical_and([0, 1])}")
print(f"1 AND 0 = {logical_and([1, 0])}")
print(f"1 AND 1 = {logical_and([1, 1])}\n")

# Logical OR
print("Logical OR:")
print(f"0 OR 0 = {logical_or([0, 0])}")
print(f"0 OR 1 = {logical_or([0, 1])}")
print(f"1 OR 0 = {logical_or([1, 0])}")
print(f"1 OR 1 = {logical_or([1, 1])}\n")

# Logical NOT
print("Logical NOT:")
print(f"NOT 0 = {logical_not([0])}")
print(f"NOT 1 = {logical_not([1])}\n")

# Logical XOR
print("Logical XOR:")
print(f"0 XOR 0 = {logical_xor([0, 0])}")
print(f"0 XOR 1 = {logical_xor([0, 1])}")
print(f"1 XOR 0 = {logical_xor([1, 0])}")
print(f"1 XOR 1 = {logical_xor([1, 1])}\n")
