"""NumPy neural-network fundamentals required for the OCT project viva.

This is a small educational demonstration on synthetic numeric data. It is NOT
trained on OCT images and must not be presented as the project's image model.
It demonstrates activation functions, forward propagation, loss and
backpropagation using NumPy only.

Run: python numpy_fundamentals.py
"""
import numpy as np


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def make_toy_data(seed=42, samples_per_class=120):
    """Create three clearly synthetic 2-D clusters for a classification demo."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    features, labels = [], []
    for class_id, center in enumerate(centers):
        features.append(rng.normal(center, 0.65, size=(samples_per_class, 2)))
        labels.extend([class_id] * samples_per_class)
    x = np.vstack(features).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    return x, y


def one_hot(labels, number_of_classes):
    result = np.zeros((len(labels), number_of_classes), dtype=np.float64)
    result[np.arange(len(labels)), labels] = 1.0
    return result


class NumpyMLP:
    """A two-layer fully connected neural network with manual backpropagation."""
    def __init__(self, input_size=2, hidden_size=16, output_size=3, seed=42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.w2 + self.b2
        probabilities = softmax(z2)
        cache = (x, z1, a1, probabilities)
        return probabilities, cache

    def train_step(self, x, y_one_hot, learning_rate=0.05):
        probabilities, cache = self.forward(x)
        x, z1, a1, _ = cache
        count = len(x)
        loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-9), axis=1))

        # Backpropagation: gradients of softmax + cross-entropy, then ReLU.
        dz2 = (probabilities - y_one_hot) / count
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        return float(loss)

    def predict(self, x):
        probabilities, _ = self.forward(x)
        return np.argmax(probabilities, axis=1)


def main():
    print("Activation function examples:")
    values = np.array([-2.0, 0.0, 2.0])
    print("Input:  ", values)
    print("Sigmoid:", np.round(sigmoid(values), 3))
    print("ReLU:   ", np.round(relu(values), 3))
    print("Tanh:   ", np.round(tanh(values), 3))

    x, y = make_toy_data()
    split = int(0.8 * len(y))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    y_train_encoded = one_hot(y_train, 3)

    model = NumpyMLP()
    for epoch in range(1, 301):
        loss = model.train_step(x_train, y_train_encoded, learning_rate=0.05)
        if epoch in (1, 50, 100, 200, 300):
            train_accuracy = np.mean(model.predict(x_train) == y_train)
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | train accuracy={train_accuracy:.3f}")

    test_accuracy = np.mean(model.predict(x_test) == y_test)
    print(f"Synthetic test accuracy: {test_accuracy * 100:.2f}%")
    print("Completed: forward propagation, softmax cross-entropy and backpropagation.")
    print("Reminder: this is a synthetic fundamentals demo, not OCT clinical evidence.")


if __name__ == "__main__":
    main()
