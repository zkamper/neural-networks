import numpy as np

data = np.load('mnist_data.npz')
test_X = data['test_X']
test_Y = data['test_Y']

data_2 = np.load('model.npz')
w_1 = data_2['w_1']
b_1 = data_2['b_1']
w_2 = data_2['w_2']
b_2 = data_2['b_2']

def softmax(y: np.ndarray[10]):
    max_val = np.max(y)
    # pentru a preveni overflow
    return np.exp(y - max_val) / np.sum(np.exp(y - max_val))

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def verify(x: np.ndarray, y: np.ndarray):
    correct = 0
    for x, label in zip(x,y):
        z = x.dot(w_1) + b_1
        a = sigmoid(z)
        z = a.dot(w_2) + b_2
        y = softmax(z)
        if np.argmax(y) == np.argmax(label):
            correct += 1
    print(correct/100)

verify(test_X, test_Y)


