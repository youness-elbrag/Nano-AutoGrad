import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 


path_data = 'digit-recognizer/'

def extract_path_df(path_dir, index_show):
    path_file = []

    for filesname in os.listdir(path_dir):
        path_file.append(os.path.join(path_dir,filesname))

    for data_df in range(0,len(path_file)):
        data_frame = pd.read_csv(path_file[data_df])
        show_df = data_frame.head(index_show)
    return path_file , f"dataframe: {show_df}"

def loading_df_to_numpy(path_file):
    data_df = pd.read_csv(path_file)
    data = np.array(data_df) 
    print(data.shape)
    m,n = data.shape
    np.random.shuffle(data)
    data_dev= data[0:1000].T
    print(data_dev.shape)
    Y_dev = data_dev[0]
    X_dev = data_dev[1,n]
    data_train = data[1000:m].T
    print(data_train.shape)
    Y_train = data_train[0]
    X_train = data_train[1:n]

    return X_train , Y_train ,m 
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2    
if __name__ == '__main__':

    path_files , _= extract_path_df(path_data,2) 

    X_train , Y_train , m = loading_df_to_numpy(path_files[1])

    #print(X_train[:,0].shape,Y_train.shape) 
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
