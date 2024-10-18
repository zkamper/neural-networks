import numpy as np

data = np.load('mnist_data.npz')
test_X = data['test_X']
test_Y = data['test_Y']

data_2 = np.load('model.npz')
weights = data_2['weights']
biases = data_2['biases']

def softmax(y: np.ndarray[10]):
    max_val = np.max(y)
    # pentru a preveni overflow
    return np.exp(y - max_val) / np.sum(np.exp(y - max_val))


def verify(x: np.ndarray, y: np.ndarray):
    correct = 0
    for x, label in zip(x,y):
        z = x.dot(weights) + biases
        y = softmax(z)
        if np.argmax(y) == np.argmax(label):
            correct += 1
    print(correct/100)

verify(test_X, test_Y)

