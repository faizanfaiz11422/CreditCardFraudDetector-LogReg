import numpy as np                                                           # Importing 'NumPy'Library
import matplotlib.pyplot as plt                                              # Importing 'MatPlotlib' Library
import pandas as pd                                                          # Importing 'Pandas' Library

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Importing the Data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************

# For the Data having Class '0'
#'''''''''''''''''''''''''''''''

dataset = pd.read_csv('E:/books/Machine Learning#/project/creditcard.csv')    # Importing data from .csv file
data = dataset.to_numpy()                                                     # Converting data to numpy arrays
df = pd.DataFrame(data)                                                       # Converting arrays to Dataframe
df.sample(frac=1)                                                             # Sampling the Dataframe
x0, y0 = df.shape                                                             # Assigning rows & columns to variables

# print(df.shape)                                                             # Taking shape of data as output
# print(X.shape,Y.shape)                                                      # Taking shape of X and Y as output
Y0_drop = df.loc[:, y0-1]                                                     # Assigning the dropped column to variable
df.drop(df.columns[29], axis=1, inplace=True)                                 # Dropping the 'Labels' column from data


# For the Data having Class '1'
#'''''''''''''''''''''''''''''''

dataset1 = pd.read_csv('E:/books/Machine Learning#/project/creditcard1.csv')   # Importing data from .csv file
data1 = dataset1.to_numpy()                                                    # Converting data to numpy arrays
df1 = pd.DataFrame(data1)                                                      # Converting arrays to dataframe
df1.sample(frac = 1)                                                           # Sampling the dataframe
x1,y1 = df1.shape                                                              # Assigning rows & columns to variables
# print(df1.shape)                                                             # Taking shape of data as output
# print(X1.shape,Y1.shape)                                                     # Taking shape of X and Y as output
Y1_drop = df1.loc[:, y1-1]                                                     # Assigning dropped column to variable
df1.drop(df1.columns[29], axis=1, inplace=True)                                # Dropping the 'Labels' column from data


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Scaling the Data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Scaling the data with class '0' / NORMALIZATION:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

scaled_X = df.copy()                                                          # Make a copy of X
for columns_X in df.columns:                                                  # 'For' loop to scale all columns
    max_value = df[columns_X].max()                                           # Finding the maximum value from columns
    min_value = df[columns_X].min()                                           # Finding the minimum value from columns
    scaled_X[columns_X] = (df[columns_X]-min_value) / (max_value-min_value)   # Formula for MinMax Scaling

# print(scaled_X)                                                             # Taking scaled data for X as output

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scaling the data with class '1' / NORMALIZATION:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

scaled_X1 = df1.copy()                                                            # Make a copy of X1
for columns_X1 in df1.columns:                                                    # 'For' loop to scale all columns
    max_value1 = df1[columns_X1].max()                                            # Finding maximum value from columns
    min_value1 = df1[columns_X1].min()                                            # Finding minimum value from columns
    scaled_X1[columns_X1] = (df1[columns_X1]-min_value1)/(max_value1-min_value1)  # Formula for MinMax Scaling

# print(scaled_X1)                                                                #Taking scaled data for X1 as output

#""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Adding the 'Label' columns that were dropped earlier
#""""""""""""""""""""""""""""""""""""""""""""""""""""""

scaled_X['Class'] = Y0_drop                                               # Adding labels for dataset with '0' class
scaled_X1 ['Class'] = Y1_drop                                             # Adding labels for dataset with '1' class


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Splitting the Data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************

# For the Data having Class '0'
#==============================

# Splitting the Features' dataset into Training, Cross-Validation and Test set :
train_0, CV_0, test_0 = np.split(scaled_X.sample(frac=1),[int(0.6*len(scaled_X)),
                                                  int(0.8*len(scaled_X))])

# For the Data having Class '1'
#===============================

# Splitting the Features' dataset into Training, Cross-Validation and Test set
train_1, CV_1, test_1 = np.split(scaled_X1.sample(frac=1),[int(0.6*len(scaled_X1)),
                                                  int(0.8*len(scaled_X1))])

#Making final datasets by concatenating:
#====================================

train_frames = [train_0, train_1]                                   # Assigning variable to lists of training dataset
train = pd.concat(train_frames)                                     # Concatenating the two lists

CV_frames = [CV_0, CV_1]                                            # Assigning variable to lists of CVdataset
CV = pd.concat(CV_frames)                                           # Concatenating the two lists

test_frames = [test_0, test_1]                                      # Assigning variable to lists of test dataset
test = pd.concat(test_frames)                                       # Concatenating the two lists

n_samples_train, n_features_train = train.shape                     # Assigning rows as samples and columns as features
train_X = train.iloc[:, 0:n_features_train-1]                       # Assigning the input data to variable
train_Y = train.iloc[:, -1]                                         # Assigning the output/classifiers to variable

n_samples_CV, n_features_CV = CV.shape                              # Assigning rows as samples and columns as features
CV_X = CV.iloc[:, 0:n_features_CV-1]                                # Assigning the input data to variable
CV_Y = CV.iloc[:, -1]                                               # Assigning the output/classifier to variable

n_samples_test, n_features_test = test.shape                        # Assigning rows as samples and columns as features
test_X = test.iloc[:, 0:n_features_test-1]                          # Assigning the input data to variable
test_Y = test.iloc[:, -1]                                           # Assigning the output/classifier to variable



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************

def sigmoid(x):                                                     # Defining Sigmoid function
    return 1/(1+np.exp(-x))                                         # Returning the function after executing formula


def setParameters(X, Y):                                            # Function to set parameters

    m = X.shape[1]                                                  # Assigning no.of training examples to a variable
    input_size = X.shape[0]                                         # number of neurons in input layer
    hidden_1 = 4                                                    # number of neurons in first hidden layer
    hidden_2 = 2                                                    # number of neurons in second hidden layer
    output_size = Y.shape[0]                                        # number of neurons in output layer.

    W1 = np.random.randn(hidden_1, input_size)                          # Assigning weight for first hidden layer
    b1 = np.zeros((hidden_1, 1))                                        # Assigning bias for first hidden layer
    W2 = np.random.randn(hidden_2, hidden_1)                            # Assigning weight for second hidden layer
    b2 = np.zeros((hidden_2, 1))                                        # Assigning bias for second layer
    W3 = np.random.randn(output_size, hidden_2)                         # Assigning weight for output layer
    b3 = np.zeros((output_size, 1))                                     # Assigning bias for output layer

    return W1, W2, W3, b1, b2, b3, m                                # returning the parameters



def forwardPropagation(X, W1, W2, W3, b1, b2, b3):                   # Function to execute forward propagation

    Z1 = np.dot(W1, X) + b1                                          # Logistic function for first hidden layer
    A1 = np.tanh(Z1)                                                 # Activation function for first hidden layer
    Z2 = np.dot(W2, A1) + b2                                         # Logistic function for second hidden layer
    A2 = np.tanh(Z2)                                                 # Activation function for second hidden layer
    Z3 = np.dot(W3, A2) + b3                                         # Logistic function for output layer
    y = sigmoid(Z3)                                                  # Activation function for output layer

    return y, Z1, Z2, Z3, A1, A2                                     # Returning the function



def regularization(W1, W2, W3, L, m):                               # Defining function for 'Regularization'

    Reg = (L/(2*m))*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))     # Formula for regularization
    return Reg                                                      # Returning function



def cost(y_pred, y_act):                                            # Function to find cost

    m = y_act.shape[1]                                              # Assigning no.of inputs to variable
    costf = -np.sum(np.multiply(np.log
                    (y_pred), y_act) + np.multiply(
                    (1 - y_act), np.log(1 - y_pred)))/m             # formula to find cost

    return np.squeeze(costf)                                        # return the cost function


def backPropagation(X, Y, A1, A2, W2, W3, y):                       # Function to execute back propagation

    m = X.shape[1]                                                  # Assigning no.of inputs to variable

    dy = y - Y                                                      # derivative of hypothesis
    dW3 = (1 / m) * np.dot(dy, np.transpose(A2))                    # derivative of weight
    db3 = (1 / m) * np.sum(dy, axis=1, keepdims=True)               # derivative of bias
    dZ2 = np.dot(np.transpose(W3), dy) * (1-np.power(A2, 2))        # derivative of logistic function
    dW2 = (1 / m) * np.dot(dZ2, np.transpose(A1))                   # derivative of weight
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)              # derivative of bias
    dZ1 = np.dot(np.transpose(W2), dZ2) * (1-np.power(A1, 2))       # derivative of logistic function
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))                    # derivative of weight
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)              # derivative of bias

    return dW1, db1, dW2, db2, dW3, db3                             # Returning the function


def updateParameters(W1, W2, W3, b1, b2, b3, dW1,
                     db1, dW2, db2, dW3, db3, learning_rate):       # Function to update parameters

    W1 = W1 - learning_rate * dW1                                   # Updating Weight for first hidden layer
    b1 = b1 - learning_rate * db1                                   # Updating Bias for first hidden layer
    W2 = W2 - learning_rate * dW2                                   # Updating Weight for second hidden layer
    b2 = b2 - learning_rate * db2                                   # Updating Bias for second hidden layer
    W3 = W3 - learning_rate * dW3                                   # Updating Weight for output layer
    b3 = b3 - learning_rate * db3                                   # Updating Bias for output layer

    return W1, W2, W3, b1, b2, b3                                   # Returning updated parameters


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



def train(X, Y, learning_rate, number_of_iterations):               # Function to execute training

    W1, W2, W3, b1, b2, b3, m = setParameters(X, Y)                 # Calling function to define parameters
    cost_tr = []                                                    # Empty list for cost

    for j in range(number_of_iterations):                           # Loop to iterate

        y, Z1, Z2, Z3, A1, A2 = forwardPropagation\
                        (X, W1, W2, W3, b1, b2, b3)                 # Calling function to feed forward

        reg = regularization(W1, W2, W3, 1.0, m)                    # Calling function to regularize

        costit = cost(y, Y) + reg                                   # Calling cost function with regularization

        dW1, db1, dW2, db2, dW3, db3 = backPropagation\
                        (X, Y, A1, A2, W2, W3, y)                   # Calling function feed backward

        W1, W2, W3, b1, b2, b3 = updateParameters\
                        (W1, W2, W3, b1, b2, b3, dW1,
                         db1, dW2, db2, dW3, db3, learning_rate)    # Calling function to update parameters

        cost_tr.append(costit)                                      # Appending cost to list
        acc_tr, precision, recall, f1, tp, \
                                    tn, fp, fn = metrics(Y, y, m)   # Calling function to compute metrics

        print('The training cost at iteration' , j, 'is :',costit)  # Printing the cost at every iteration

    return W1, W2, W3, b1, b2, b3, cost_tr,\
           acc_tr, tp, tn, fp, fn, precision, recall, f1            # returning function


def CV(X, Y, W1, W2, W3, b1, b2, b3):                               # Function to execute Cross-Validation

    m_CV = X.shape[1]                                               # Assigning no.of inputs to variables
    cost_cv = []

    y, Z1, Z2, Z3, A1, A2 = forwardPropagation\
                    (X, W1, W2, W3, b1, b2, b3)                     # Calling function to feed forward

    costit = cost(y, Y)                                             # Calling cost function

    acc_CV, precision_cv, recall_cv, f1_cv, tp_cv, \
                    tn_cv, fp_cv, fn_cv = metrics(Y, y, m_CV)       # Calling function to compute metrics

    cost_cv.append(costit)                                          # Appending cost
    print('The Cv cost is:', costit)                                # Printing cost at every iteration

    return cost_cv, acc_CV, precision_cv, recall_cv,\
                    f1_cv, tp_cv, tn_cv, fp_cv, fn_cv               # Returning the function


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Training the Data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************


tr_X, tr_Y = train_X.values.T, train_Y.values.reshape(1, train_Y.shape[0])          # Assigning the data for Training

W1, W2, W3, b1, b2, b3, cost_tr, acc_tr,\
            tp, tn, fp, fn, precision, recall, f1\
            = train(tr_X, tr_Y, 0.6, 2000)                           # Calling function to execute training of the data

print('TP for training NN is :', tp)                                        # Taking no.of True Positives as output
print('TN for NN is :', tn)                                                 # Taking no.of True Negatives as output
print('FP for NN is :', fp)                                                 # Taking no.of False Positives as output
print('FN for NN is :', fn)                                                 # Taking no.of False Negatives as output
print('The Precision of the Model is:', precision)                          # Taking Precision as output
print('The Recall of the Model is:', recall)                                # Taking Recall as output
print('The F1 Score of the Model is:', f1)                                  # Taking F1 Score as output
print('The Training accuracy is :', acc_tr)                                 # Taking Accuracy as output


params = {'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}       # Storing parameters in a dictionary

np.savez('params.npz' , W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3)           # Saving parameters in an .npz file



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Cross-Validating the Data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************

params_CV = np.load('params.npz', allow_pickle=True)                    # Loading parameters from an .npz file

W1 = params_CV['W1']                                                    # Extracting parameter 'W1' from dictionary
W2 = params_CV['W2']                                                    # Extracting parameter 'W2' from dictionary
W3 = params_CV['W3']                                                    # Extracting parameter 'W3' from dictionary
b1 = params_CV['b1']                                                    # Extracting parameter 'b1' from dictionary
b2 = params_CV['b2']                                                    # Extracting parameter 'b2' from dictionary
b3 = params_CV['b3']                                                    # Extracting parameter 'b3' from dictionary


cv_X, cv_Y = CV_X.values.T, CV_Y.values.reshape(1, CV_Y.shape[0])       # Assigning the data for Cross-Valdation

cost_cv, acc_CV, precision_cv, recall_cv, \
        f1_cv, tp_cv, tn_cv, fp_cv, fn_cv\
        = CV(cv_X, cv_Y, W1, W2, W3, b1, b2, b3)                # Calling the function to execute Cross-Validation

print('TP for CV NN is :', tp_cv)                               # Taking no.of True Positives as output
print('TN for CV NN is :', tn_cv)                               # Taking no.of True Negatives as output
print('FP for CV NN is :', fp_cv)                               # Taking no.of False Positives as output
print('FN for CV NN is :', fn_cv)                               # Taking no.of False Negatives as output
print('The Precision of the Model in CV is:', precision_cv)     # Taking Precision as output
print('The Recall of the Model in CV is:', recall_cv)           # Taking Recall as output
print('The F1 Score of the Model in CV is:', f1_cv)             # Taking F1 Score as output
print('The Training accuracy for CV is :', acc_CV)              # Taking Accuracy as output


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************
# Plotting the Graph
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#***********************************************************************************************************************

plt.plot(cost_tr)                                               # Making plot for Train error
plt.xlabel('Iterations')                                        # Assigning the x-label
plt.ylabel('Training Cost')                                     # Assigning the y-label
plt.title('Cost Error')                                         # Assigning the title
plt.legend()                                                    # Command to show legend
plt.show()                                                      # Command th show graph on screen
