import numpy as np

# hidden layer activation function
def sigmoid(s):
    return 1/(1 + np.exp(-s))

# derivate of hidden layer activation function for gradient descent
def delta_sigmoid(s):
    return s * (1 - s)

# output layer activation function
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

# cost function
def cross_entropy(pred,real):
    no_of_instances = real.shape[0]
    p = softmax(pred)
    log_likelihood = -np.log(p[np.arange(no_of_instances),real.argmax(axis=1)])
    loss = np.sum(log_likelihood) / no_of_instances
    return loss

# derivative of cost function for gradient descent
def delta_cross_entropy(pred,real):
    no_of_instances = real.shape[0]
    grad = softmax(pred)
    grad[np.arange(no_of_instances),real.argmax(axis=1)] -= 1
    grad = grad/no_of_instances
    return grad

class ANN:

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels
        self.learning_rate = 1
        neurons = 1000       # no. of nodes in each hidden layer
        no_of_ip_nodes = data.shape[1]
        no_of_op_nodes = labels.shape[1]

        np.random.seed(10)
        #initialising weights and bias
        self.w1 = np.random.randn(no_of_ip_nodes, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, no_of_op_nodes)
        self.b3 = np.zeros((1, no_of_op_nodes))

    def feedforward(self):

        # activation of nodes of hidden layers
        z1 = np.dot(self.data, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)

        #activation of nodes of output layer
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        
    def backpropogate(self):

        loss = cross_entropy(self.a3, self.labels)
        print('Cost:', loss)

        # calculation of cost and derivative of cost function
        a3_delta = delta_cross_entropy(self.a3, self.labels)
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * delta_sigmoid(self.a2)
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * delta_sigmoid(self.a1)

        # gradient descent for weights of each layer
        self.w3 -= self.learning_rate * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.learning_rate * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.learning_rate * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.learning_rate * np.sum(a2_delta, axis=0)
        self.w1 -= self.learning_rate * np.dot(self.data.T, a1_delta)
        self.b1 -= self.learning_rate * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.data = data
        self.feedforward()
        return self.a3.argmax()