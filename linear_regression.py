 
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

# hypothesis
def hypothesis(theta,X):
    return np.matmul(X,theta)

# cost function
def cost_fcn(theta,X,y):
    m=X.shape[0]
    h_x = hypothesis(theta,X)
    cost = (1/(2*m))*np.sum((h_x - y)**2)
    return cost

# get gradients
def gradients(theta,X,y):
    m=X.shape[0]
    h_x = hypothesis(theta,X)
    g = (1./m)*(np.matmul((X.T),(h_x - y)))
    return g

if __name__ == "__main__":
    # get data
    data = np.loadtxt(r'ex1data1.txt',delimiter=',')

    # get X 
    X = data[:,0]
    X = X.reshape((X.shape[0],1))
    # concatenate a column of 1s (bias term)
    X=np.concatenate((np.ones((X.shape[0],1)), X), axis = 1)
    
    # get y
    y = data[:,1]
    y=y.reshape((y.shape[0],1))
    
    # initialize theta
    theta = np.zeros((X.shape[1],1))
    
    # number of iterations
    iterations = 1500
    
    # alpha
    alpha = 0.01
    
    # initialize lists to keep track of cost fcns
    costs = []
    
    first_cost = cost_fcn(theta,X,y)
    # loop through
    for i in range(iterations):
        g = gradients(theta,X,y)
        theta = theta - alpha*g
        costs.append(cost_fcn(theta,X,y))
    
    # plot
    fig, ax = plt.subplots()
    plt.scatter(X[:,1],y,marker='x',color='r',label='Training Data')
    plt.plot(X[:,1],hypothesis(theta,X),color='b',label='Linear Regression')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.legend(loc='lower right')
    plt.show()