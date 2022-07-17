import numpy as np                                                  # Importing 'NumPy'Library

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Prediction Function
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def sigmoid(X, theta, B):                                           # Defining function for Sigmoid
    z = np.dot(theta, X.T) + B                                      # Defining hypothesis
    return 1/(1+np.exp(-(z)))                                       # Return Sigmoid

def predict(X, y, threshold, theta, B):                             # Defining prediction function
    Y_pred = sigmoid(X, theta, B)                                   # Calling Sigmoid function to get predicted labels
    Y_pred = Y_pred > threshold                                     # Setting a condition for predicted labels
    y = np.array(y)                                                 # Converting Actual labels into set of array
    Y_pred = np.array(Y_pred)                                       # Converting Predicted labels into set of array

    tp = np.sum((y == 1) & (Y_pred == 1))                           # Condition to get True Positives
    tn = np.sum((y == 0) & (Y_pred == 0))                           # Condition to get True Negatives
    fp = np.sum((y == 0) & (Y_pred == 1))                           # Condition to get False Positives
    fn = np.sum((y == 1) & (Y_pred == 0))                           # Condition to get False Negatives

    print('TP for Logistic Reg :', tp)                              # Taking no.of True Positives as output
    print('TN for Logistic Reg :', tn)                              # Taking no.of True Negatives as output
    print('FP for Logistic Reg :', fp)                              # Taking no.of False Positives as output
    print('FN for Logistic Reg :', fn)                              # Taking no.of False Negatives as output

    precision = tp /(tp + fp)                                       # Formula to compute precision
    recall = tp / (tp + fn)                                         # Formula to compute recall
    f1 = 2 * (precision * recall) / (precision + recall)            # Formula to compute F1 Score

    print('The Precision of the Model is:', precision)              # Taking Precision as output
    print('The Recall of the Model is:', recall)                    # Taking Recall as output
    print('The F1 Score of the Model is:', f1)                      # Taking F1 Score as output

    return Y_pred, f1 , precision, recall                           # Return the Error Metrics

#********************************************
# Implementation with Test set:
#********************************************

test_X = np.load('test_X.npy')                                      # Importing Test_X set
test_Y = np.load('test_Y.npy')                                      # Importing Test_Y set
theta_test = np.load('theta.npy')                                   # Defining thetas for test error
B_test = np.load('bias.npy')                                        # Defining Bias units for test error
predict(test_X, test_Y, 0.5, theta_test, B_test)                    # Calling Predict Function to get metrics
