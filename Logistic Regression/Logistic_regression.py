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

np.save('test_x',test_X)                                            # Saving the test set
np.save('test_y',test_Y)                                            # Saving the labels of test set

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Training the Data
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

train_error = []                                                        # Empty list for Training Error
train_r_error = []                                                      # Empty list for Regularized Training Error
L_train = []                                                            # Empty list for lambdas

def sigmoid(X, theta, B):                                               # Defining function for Sigmoid
    z = np.dot(theta, X.T) + B                                          # Defining hypothesis
    return 1/(1+np.exp(-(z)))                                           # Return Sigmoid

def regularization(X, theta):                                           # Defining function for Regularization
    m = len(X)                                                          # Assigning length of 'X' set to variable
    reg = (L1/(2*m))*(np.sum(theta)**2)                                 # Formula for Regularization
    return reg                                                          # Returning Regularization

def cost(X, y, theta):                                                  # Defining function to compute Cost
    h1 = sigmoid(X, theta, B)                                           # Calling Sigmoid function
    cost_f = -(1 / len(X)) * \
             np.sum(y * np.log(h1) + (1 - y) * np.log(1 - h1))          # Formula for Cost
    cost_f_r = (-(1 / len(X))) * \
               (np.sum(y * np.log(h1) + (1-y) * np.log(1 - h1)))\
               + regularization(X, theta)                               # Formula for cost with regularization

    train_error.append(cost_f)                                          # Appending Training Error to the list
    train_r_error.append(cost_f_r)                                      # Appending Regularized Training error to list
    L_train.append(L1)
    return train_r_error                                                # Returning Regularized Training Error

def gradient_descent(X, y, theta, B, alpha, iterations):                # Defining function to compute Gradient Descent
    m = int(len(X))                                                     # Assigning length of 'X' set to a variable
    J = [cost(X, y, theta)]                                             # Calling 'Cost' function in a list
    for i in range(0, iterations):                                      # 'For' loop until no. of iterations
        h = sigmoid(X, theta, B)                                        # Calling Sigmoid function
        for i in range(0, len(X.columns)):                              # 'For' loop until length of 'X' set
            theta[i] -= (alpha/m) * np.sum((h-y)*X.iloc[:, i]) + \
                        ((L1*theta[i])/m)                               # Formula to 'Update Weights'
            B -= (alpha/m) * np.sum(h-y)                                # Formula to 'Update Bias'
        J.append(cost(X, y, theta))                                     # Appending cost to List
        np.save('theta', theta)                                         # Saving Weights as .npy file
        np.save('bias', B)                                              # Saving Bias units as .npy file
    return J, theta                                                     # Returning Cost list and Weights

def accuracy(X, y, theta, alpha, iterations):                           # Defining function to compute accuracy
    J = gradient_descent(X, y, theta, B, alpha, iterations)             # Calling gradient Descent function
    h = sigmoid(X, theta, B)                                            # Assigning sigmoid function a variable
    for i in range(len(h)):                                             # 'For' loop until length of sigmoid
        h[i]=1 if h[i] >= 0.5 else 0                                    # Separating labels using threshold
    y = list(y)                                                         # Converting the 'y' set into a list
    acc = np.sum([y[i] == h[i].any() for i in range(len(y))])/len(y)    # Formula to compute accuracy
    return J, acc                                                       # Returning Accuracy

#***********************************
# Implementation with Training set:
#***********************************

m = train_X.shape[1]                                                    # Assigning Columns to 'm'
n = train_X.shape[0]                                                    # Assigning Rows to 'n
B = 0                                                                   # The initial value of base unit is '0'
theta = 0.1*np.random.rand(m)                                           # Defining initial thetas/weights
alpha = 0.5                                                             # Defining the learning rate
for L1 in np.arange(0, 10, 1):                                          # 'For' loop to iterate through lambda
    J, acc = accuracy(train_X, train_Y, theta, alpha, 2000)             # Calling function for training model
print('Training accuracy is:', acc)                                     # Printing the output accuracy for training


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Cross-Validating the Data
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CV_error = []                                                       # Empty list for Cross-Validation Error
L_CV = []                                                           # Empty list for Lambdas

def sigmoid_CV(X, theta_CV, B_CV):                                  # Defining Sigmoid function for CV
    z = np.dot(theta_CV, X.T) + B_CV                                # Defining hypothesis for CV
    return 1/(1+np.exp(-(z)))                                       # Returning Sigmoid

def cost_CV(X, y, theta_CV):                                        # Defining function to compute cost for CV
    h1 = sigmoid_CV(X, theta_CV, B_CV)                              # Calling Sigmoid function & assigning a variable
    cost_f_cv =  -(1 / len(X)) * np.sum(y * np.log(h1)
                                     + (1 - y) * np.log(1 - h1))    # Formula to compute cost for CV
    CV_error.append(cost_f_cv)                                      # Appending the error to the empty CV list
    L_CV.append(L2)

    return CV_error                                                 # Returning the CV error

def gradient_descent_CV(X, y, theta_CV, B_CV, alpha, iterations):       # Defining function to compute Gradient Descent
    m = int(len(X))                                                     # Assigning length of 'X' set to a variable
    J_CV = [cost_CV(X, y, theta_CV)]                                    # Calling 'Cost' function in a list
    for i in range(0, iterations):                                      # 'For' loop until no. of iterations
        h = sigmoid_CV(X, theta_CV, B_CV)                               # Calling Sigmoid function
        for i in range(0, len(X.columns)):                              # 'For' loop until length of 'X' set
            theta_CV[i] -= (alpha/m) * np.sum((h-y)*X.iloc[:, i])       # Formula to 'Update Weights'
            B_CV -= (alpha/m) * np.sum(h-y)                             # Formula to 'Update Bias'
        J_CV.append(cost_CV(X, y, theta_CV))                            # Appending cost to List
    return J_CV, theta_CV                                               # Returning Cost list and Weights

def accuracy_CV(X, y, theta_CV, alpha, iterations):                     # Defining function for accuracy
    J = gradient_descent_CV(X, y, theta_CV, B_CV, alpha, iterations)    # Calling gradient Descent function
    h = sigmoid_CV(X, theta_CV, B_CV)                                   # Calling Sigmoid function
    for i in range(len(h)):                                             # 'For' loop until length of sigmoid
        h[i]=1 if h[i] >= 0.5 else 0                                    # Separating labels using threshold
    y = list(y)                                                         # Converting labels into a list
    acc = np.sum([y[i] == h[i].any() for i in range(len(y))])/len(y)    # Formula to compute accuracy
    return J, acc                                                       # Return accuracy


#********************************************
# Implementation with Cross-Validation set:
#********************************************

theta_CV = np.load('theta.npy')                                         # Defining weights for Cross-Validation
B_CV = np.load('bias.npy')                                              # Defining Bias units for Cross-Validation
alpha_CV = 0.5                                                          # Defining the learning rate
for L2 in np.arange(0,10,1):                                            # 'For' loop to iterate through lambda
    J_CV, acc_CV = accuracy_CV(CV_X, CV_Y, theta_CV, alpha, 2000)       # Calling function for CV model
print('Cross-Validation accuracy is:', acc_CV)                          # Taking the accuracy of model as output

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Testing the Data
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def sigmoid_test(X, theta_test, B_test):                               # Defining Sigmoid function for test
    z = np.dot(theta_test, X.T) + B_test                               # Defining hypothesis for test
    return 1/(1+np.exp(-(z)))                                          # Returning Sigmoid

def cost_test(X, y, theta_test):                                       # Defining function to compute cost for test
    h2 = sigmoid_test(X, theta_test, B_test)                           # Assigning Sigmoid function a variable
    cost_f_t =  -(1 / len(X)) * np.sum(y * np.log(h2)
                                     + (1 - y) * np.log(1 - h2))       # Formula to compute cost for test
    return cost_f_t                                                    # Returning the test error

def accuracy_test(X, y, theta_test, alpha, iterations):                   # Defining the accuracy function for test set
    C_test = cost_test(X, y, theta_test)                                  # Calling the cost function
    h = sigmoid_CV(X, theta_test, B_test)                                 # Calling the sigmoid function
    for i in range(len(h)):                                               # 'For' loop until the length of Sigmoid
        h[i]=1 if h[i] >= 0.5 else 0                                      # Separating labels using threshold
    y = list(y)                                                           # Converting labels into a list
    acc_cv = np.sum([y[i] == h[i].any() for i in range(len(y))])/len(y)   # Formula for accuracy
    return C_test, acc_cv                                                 # Return accuracy


#********************************************
# Implementation with Test set:
#********************************************

theta_test = np.load('theta.npy')                                      # Defining thetas for test error
B_test = np.load('bias.npy')                                           # Defining Bias units for test error
test_error = cost_test(test_X, test_Y, theta_test)                     # Calling function to compute test error
print('Test Error is:', test_error)                                    # Taking Test Error as output


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Plotting Graphs
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


figure, axis = plt.subplots(2, 2)                                            # Making Sub-Plots

axis[0, 0].plot(train_error, label='Train Error', color='r')                 # Making plot for Train error
axis[0, 0].set_title("Train Error")                                          # Setting title for graph
axis[0, 0].set(xlabel="Iterations",ylabel="Training Cost")                   # Setting x and y labels for subplots
axis[0, 0].plot(train_r_error, label='Regularized Train Error', color='b')   # Making plot for Regularized Train error
axis[0, 0].set_title("Regularized Train Error")                              # Setting title for graph
axis[0, 0].legend(loc='upper right')                                         # Setting the legend for graph

axis[1, 0].plot(CV_error, label='CV Error',color = 'r' )                     # Making plot for CV error
axis[1, 0].set_title("Cross-Validation Error")                               # Setting title for graph
axis[1, 0].set(xlabel="Iterations",ylabel="Cross-Validation Cost")           # Setting x and y labels for subplots
axis[1, 0].legend(loc='upper right')                                         # Setting the legend for graph


axis[0, 1].plot(L_train, train_error, label='Train Error', color='y')        # Making plot for Train error
axis[0, 1].set_title("Train Error vs Lambda")                                # Setting title for graph
axis[0, 1].set(xlabel="Lambda",ylabel="Training Cost")                       # Setting x and y labels for subplots
axis[0, 1].legend(loc='upper right')                                         # Setting the legend for graph


axis[1, 1].plot(L_CV, CV_error, label='CV Error',color = 'r' )               # Making plot for CV error
axis[1, 1].set_title("Cross-Validation Error vs Lambda")                     # Setting title for graph
axis[1, 1].set(xlabel="Lambda",ylabel="Cross-Validation Cost")               # Setting x and y labels for subplots
axis[1, 1].legend(loc='upper right')                                         # Setting the legend for graph


plt.show()                                                                   # Show plots
