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
    max_val = np.max(y)
    # pentru a preveni overflow
    return np.exp(y - max_val) / np.sum(np.exp(y - max_val))

def cross_entropy(y: np.ndarray[10], label: np.ndarray[10]):
    return - np.sum(label * np.log(y))


def train(train_set, weights: np.ndarray, biases: np.ndarray, learning_rate):
    temp_weights = np.zeros(weights.shape)
    temp_biases = np.zeros(biases.shape)
    wrong = 0
    for x, label in train_set:
        z = x.dot(weights) + biases
        y = softmax(z)
        if not classify(y, label):
            wrong += 1
            temp_weights = temp_weights + learning_rate * (label - y) * [[i] for i in x]
            temp_biases = temp_biases + learning_rate * (label - y)
    return wrong, temp_weights, temp_biases

def train_thread(train_set, weights: np.ndarray, biases: np.ndarray, learning_rate: float, result_queue: queue.Queue):
    thread_weights = weights.copy()
    thread_biases = biases.copy()
    batches = split(train_set,100)
    misclassified = 0
    for batch in batches:
        wrong, temp_weights, temp_biases = train(batch, thread_weights, thread_biases, learning_rate)
        misclassified += wrong
        thread_weights = thread_weights + temp_weights
        thread_biases = thread_biases + temp_biases
    result_queue.put((thread_weights-weights, thread_biases-biases, misclassified))

def shuffle(array: list):
    random_indices = np.random.permutation(len(array))
    return [array[i] for i in random_indices]

def split(array: list, n: int):
    return [array[i:i + n] for i in range(0, len(array), n)]

weights = np.zeros((784, 10))
biases = np.zeros(10)
alpha = 0.0075
epochs = 250
num_threads = 4

train_data = list(zip(train_X, train_Y))
train_data_len = len(train_data)
thread_set_size = train_data_len // num_threads

accuracies = []

for epoch in range(epochs):
    threads = []
    result_queue = queue.Queue()
    train_data = shuffle(train_data)
    thread_batches = split(train_data, thread_set_size)
    for i in range(num_threads):
        thread = threading.Thread(target=train_thread, args=(thread_batches[i], weights, biases, alpha, result_queue))
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()
        
    mistakes = 0
    for i in range(num_threads):
        temp_weights, temp_biases, misclassified = result_queue.get()
        weights = weights + temp_weights
        biases = biases + temp_biases
        mistakes += misclassified
    accuracy = (train_data_len - mistakes) / train_data_len
    if accuracy > 0.96:
        break
    accuracies.append(accuracy)
    print(f'Accuracy: {accuracy}')
    print(f'Epoch {epoch + 1} done')
    
# Save the model
np.savez('model.npz', weights=weights, biases=biases, accuracies=accuracies)