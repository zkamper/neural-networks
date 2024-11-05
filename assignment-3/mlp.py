import numpy as np
import time

data = np.load('mnist_data.npz')
train_X = data['train_X']
train_Y = data['train_Y']

def split_data(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def softmax(y: np.ndarray) -> np.ndarray:
    exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
    return exp_y / np.sum(exp_y, axis=1, keepdims=True)

def classify(y: np.ndarray, label: np.ndarray):
    return np.sum(np.argmax(y, axis=1) == np.argmax(label, axis=1))

def cross_entropy(y: np.ndarray, label: np.ndarray):
    return np.mean(-np.sum(label * np.log(y), axis=1))

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray):
    sig = sigmoid(x)
    return sig * (1 - sig)

def forward_prop(x: np.array, w_1: np.array, b_1: np.array, w_2: np.array, b_2: np.array):
    global DROPOUT_RATE
    z_1 = x.dot(w_1) + b_1
    a_1 = sigmoid(z_1)
    # dropout pe hidden layer
    a_1 = np.where(np.random.rand(*a_1.shape) < DROPOUT_RATE, 0, a_1)
    z_2 = a_1.dot(w_2) + b_2
    y = softmax(z_2)
    return y, z_1, a_1

def back_prop(x: np.array, y: np.array, label: np.array, w_1: np.array, b_1: np.array, w_2: np.array, b_2: np.array, z_1: np.array, a_1: np.array):
    global ALPHA
    error_out = y - label
    error_hidden = error_out.dot(w_2.T) * sigmoid_derivative(z_1)

    grad_w_2 = a_1.T.dot(error_out)
    grad_b_2 = np.mean(error_out, axis=0)
    grad_w_1 = x.T.dot(error_hidden)
    grad_b_1 = np.mean(error_hidden, axis=0)

    w_2 -= ALPHA * grad_w_2
    b_2 -= ALPHA * grad_b_2
    w_1 -= ALPHA * grad_w_1
    b_1 -= ALPHA * grad_b_1


def train_mini_batch(x: np.array, label: np.array):
    global w_1, b_1, w_2, b_2
    y, z_1, a_1 = forward_prop(x, w_1, b_1, w_2, b_2)
    back_prop(x, y, label, w_1, b_1, w_2, b_2, z_1, a_1)
    return cross_entropy(y, label)

def train(dataset):
    global BATCH_SIZE

    batches = split_data(dataset, BATCH_SIZE)

    cost = 0
    for batch in batches:
        x = np.array([data[0] for data in batch])
        label = np.array([data[1] for data in batch])
        cost += train_mini_batch(x, label)
    return cost / len(batches)

w_1 = np.random.randn(784, 100) * 0.01
b_1 = np.zeros(100)
w_2 = np.random.randn(100, 10) * 0.01
b_2 = np.zeros(10)

ALPHA = 0.02
EPOCHS = 100
BATCH_SIZE = 50
DROPOUT_RATE = 0.15

train_data = list(zip(train_X, train_Y))
train_data_len = len(train_data)

start_time = time.time()

# TODO: codul propriu zis
for i in range(EPOCHS):
    if i == 50:
        ALPHA = 0.01
    np.random.shuffle(train_data)
    avg_cost = train(train_data)
    print(f'Epoch {i + 1}/{EPOCHS}, cost: {avg_cost}')

end_time = time.time()
print(f'Training took {end_time - start_time} seconds')
# Save the model
np.savez('model.npz', w_1 = w_1, b_1 = b_1, w_2 = w_2, b_2 = b_2)