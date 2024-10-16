import queue
import numpy as np
import threading

data = np.load('mnist_data.npz')
train_X = data['train_X']
train_Y = data['train_Y']
test_X = data['test_X']
test_Y = data['test_Y']

def classify(y: np.ndarray[10], label: np.ndarray[10]):
    return np.argmax(y) == np.argmax(label)

def softmax(y: np.ndarray[10]):
    return np.exp(y) / np.sum(np.exp(y))

def cross_entropy(y: np.ndarray[10], label: np.ndarray[10]):
    return - np.sum(label * np.log(y))

def train_thread(train_set, weights: np.ndarray, biases: np.ndarray, learning_rate: float, result_queue: queue.Queue):
    thread_weights = weights.copy()
    thread_biases = biases.copy()
    batches = [train_set[i:i + 100] for i in range(0, len(train_set), 100)]
    for batch in batches:
        temp_weights, temp_biases = train(batch, thread_weights, thread_biases, learning_rate)
        thread_weights = thread_weights + temp_weights
        thread_biases = thread_biases + temp_biases
    result_queue.put((thread_weights-weights, thread_biases-biases))

def train(train_set, weights: np.ndarray, biases: np.ndarray, learning_rate):
    temp_weights = np.zeros(weights.shape)
    temp_biases = np.zeros(biases.shape)
    for x, label in train_set:
        z = x.dot(weights) + biases
        y = softmax(z)
        if not classify(y, label):
            temp_weights = temp_weights + learning_rate * (label - y) * [[i] for i in x]
            temp_biases = temp_biases + learning_rate * (label - y)
    return temp_weights, temp_biases


weights = np.zeros((784, 10))
biases = np.zeros(10)
alpha = 0.01
epochs = 10
num_threads = 8

train_data = list(zip(train_X, train_Y))
indexes = np.random.choice(range(len(train_data)), len(train_data)//10, replace=False)
train_data = [train_data[i] for i in indexes]

train_data_len = len(train_data)
thread_set_size = train_data_len // num_threads

for epoch in range(epochs):
    threads = []
    result_queue = queue.Queue()
    thread_batches = [train_data[i:i + thread_set_size] for i in range(0, train_data_len, thread_set_size)]
    for i in range(num_threads):
        thread = threading.Thread(target=train_thread, args=(thread_batches[i], weights, biases, alpha, result_queue))
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    for i in range(num_threads):
        temp_weights, temp_biases = result_queue.get()
        weights = weights + temp_weights
        biases = biases + temp_biases

    print(f'Epoch {epoch + 1} done')

# Save the model
np.savez('model.npz', weights=weights, biases=biases)