import numpy as np                                                  # Importing 'NumPy'Library

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Prediction Function
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def sigmoid(x):                                                     # Defining Sigmoid function
    return 1/(1+np.exp(-x))                                         # Returning the function after executing formula



def ForwardPropagation(X, W1, W2, W3, b1, b2, b3):                   # Function to execute forward propagation

    Z1 = np.dot(W1, X) + b1                                          # Logistic function for first hidden layer
    A1 = np.tanh(Z1)                                                 # Activation function for first hidden layer
    Z2 = np.dot(W2, A1) + b2                                         # Logistic function for second hidden layer
    A2 = np.tanh(Z2)                                                 # Activation function for second hidden layer
    Z3 = np.dot(W3, A2) + b3                                         # Logistic function for output layer
    y = sigmoid(Z3)                                                  # Activation function for output layer

    return y, Z1, Z2, Z3, A1, A2                                     # Returning the function


def metrics(Y, y, m):                                               # Function to get metrics

    y = y > 0.15                                                    # Setting threshold
    y = np.array(y)                                                 # Converting the hypothesis into array
    Y = np.array(Y)                                                 # Converting the

    tp = np.sum((Y == 1) & (y == 1))                                # Condition to get True Positives
    tn = np.sum((Y == 0) & (y == 0))                                # Condition to get True Negatives
    fp = np.sum((Y == 0) & (y == 1))                                # Condition to get False Positives
    fn = np.sum((Y == 1) & (y == 0))                                # Condition to get False Negatives

    acc = (tp+tn)*(100/m)                                           # Formula to compute accuracy
    precision = tp / (tp + fp)                                      # Formula to compute precision
    recall = tp / (tp + fn)                                         # Formula to compute recall
    f1 = 2 * (precision * recall) / (precision + recall)            # Formula to compute F1 Score

    return acc, precision, recall, f1, tp, tn, fp, fn               # Returning the metrics


def test(X, Y, W1, W2, W3, b1, b2, b3):                             # Function to test data

    m_test = X.shape[1]                                             # no.of inputs

    y, Z1, Z2, Z3, A1, A2 = ForwardPropagation(X, W1,
                                W2, W3, b1, b2, b3)                 # Feeding Forward

    acc_ts, precision_ts, recall_ts, f1_ts, tp_ts, \
            tn_ts, fp_ts, fn_ts = metrics(Y, y, m_test)             # Computing metrics

    return acc_ts, precision_ts, recall_ts, f1_ts, \
                            tp_ts, tn_ts, fp_ts, fn_ts              # Returning function


#***********************************************************************************************************************
# Implementation with Test set:
#***********************************************************************************************************************

test_X = np.load('test_X.npy')                                      # Importing Test_X set
test_Y = np.load('test_Y.npy')                                      # Importing Test_Y set

test_X, test_Y = test_X.T, test_Y.reshape(1, test_Y.shape[0])       # Assigning Test data


params_test = np.load('params.npz', allow_pickle=True)              # Loading trained parameters

W1 = params_test['W1']                                              # Assigning parameter
W2 = params_test['W2']                                              # Assigning parameter
W3 = params_test['W3']                                              # Assigning parameter
b1 = params_test['b1']                                              # Assigning parameter
b2 = params_test['b2']                                              # Assigning parameter
b3 = params_test['b3']                                              # Assigning parameter


acc_ts, precision_ts, recall_ts, f1_ts, \
                        tp_ts, tn_ts, fp_ts, fn_ts = test\
                        (test_X, test_Y, W1, W2, W3, b1, b2, b3)    # Executing Test function


print('TP for testing NN is :', tp_ts)                              # Taking no.of True Positives as output
print('TN for testing NN is :', tn_ts)                              # Taking no.of True Negatives as output
print('FP for testing NN is :', fp_ts)                              # Taking no.of False Positives as output
print('FN for testing NN is :', fn_ts)                              # Taking no.of False Negatives as output
print('The Precision of the Model in test is:', precision_ts)       # Taking Precision as output
print('The Recall of the Model in test is:', recall_ts)             # Taking Recall as output
print('The F1 Score of the Model in test is:', f1_ts)               # Taking F1 Score as output
print('The accuracy for test is :', acc_ts)                # Taking accuracy as output
