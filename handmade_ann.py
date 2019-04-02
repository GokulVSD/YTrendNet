import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def train(feature_set, one_hot_labels, output_labels):

    np.random.seed(1)

    instances = feature_set.shape[0]
    attributes = feature_set.shape[1]
    hidden_nodes = 300

    wh = np.random.rand(attributes, hidden_nodes)
    bh = np.random.randn(hidden_nodes)
    
    wo = np.random.rand(hidden_nodes, output_labels)
    bo = np.random.randn(output_labels)
    lr = 10e-4

    error_cost = []

    for epoch in range(50000):
            # feedforward

            # Phase 1
            zh = np.dot(feature_set, wh) + bh
            ah = sigmoid(zh)

            # Phase 2
            zo = np.dot(ah, wo) + bo
            ao = softmax(zo)
            
        # Back Propagation

        # Phase 1

            dcost_dzo = ao - one_hot_labels
            dzo_dwo = ah

            dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

            dcost_bo = dcost_dzo

        # Phases 2

            dzo_dah = wo
            dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
            dah_dzh = sigmoid_der(zh)
            dzh_dwh = feature_set
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

            dcost_bh = dcost_dah * dah_dzh

            # Update Weights

            wh -= lr * dcost_wh
            bh -= lr * dcost_bh.sum(axis=0)

            wo -= lr * dcost_wo
            bo -= lr * dcost_bo.sum(axis=0)

            if epoch % 200 == 0:
                loss = np.sum(-one_hot_labels * np.log(ao))
                print('Loss function value: ', loss,' Epoch: ',epoch,' of 50000')
                error_cost.append(loss)
