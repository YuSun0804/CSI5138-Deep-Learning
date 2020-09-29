import numpy as np
import matplotlib.pyplot as plt
import random
import sympy as sym
import pandas as pd
import math

def getData(N, sigma):
    x = np.random.uniform(0, 1, N)
    #Y = cos(2πX) + Z
    z = np.random.normal(0, sigma, N)
    y = np.cos(2*math.pi*x)+z
    return np.vstack((x, y)).T


def getMSE(y, y_pred):
    mse = np.mean((y - y_pred)**2)
    return round(mse, 4)


def getX(x, N, d):
    res = np.ones((1,N))
    for i in range(d):
        temp = x**(i+1)
        res = np.vstack((res, temp))
    return res.T


def fitData(data, d, iterations=2000, lr=0.1, batch_size=50, lamda=0.1):
    N = len(data)
    y = data[:, 1].reshape((N, 1))
    X = getX(data[:, 0], N, d)
    theta = np.random.random([d+1, 1])
    print(getMSE(X.dot(theta), y))
    # GD(X, theta, y, N, iterations, lr)
    # SGD(X, theta, y, N, iterations, lr)
    return MiniBatch(X, theta, y, N, N, iterations, lr, lamda)


# def GD(X, theta, y, N, iterations, lr):
#     J = np.zeros(iterations)
#     for iter in range(iterations):
#         theta -= (1/N) * lr * ((X.dot(theta) - y).T.dot(X).T)
#         J[iter] = getMSE(X.dot(theta), y)
#     return theta


# def SGD(X, theta, y, N, iterations, lr):
#     J = np.zeros(iterations)
#     index = random.randint(0, N-1)
#     x = X[index].reshape(-1, 1)
#     y = y[index].reshape(-1, 1)
#     for iter in range(iterations):
#         theta -= lr * (x*theta - y) * x
#         J[iter] = getMSE(X.dot(theta), y)
#     return theta


def MiniBatch(X, theta, y, N, batch_size, iterations, lr, lamda):
    J = np.zeros(iterations)
    index = random.randint(0, N-batch_size)
    X = X[index:index+batch_size]
    y = y[index:index+batch_size]
    for iter in range(iterations):
        theta = theta*(1 - lamda*lr) - (lr * ((1/batch_size) *
                       ((X.dot(theta) - y).T.dot(X).T)))
        J[iter] = getMSE(X.dot(theta), y)
    Ein = J[iterations-1]
    return theta, Ein


def evaluat(data, d, theta):
    N = len(data)
    y = data[:, 1].reshape((N, 1))
    X = getX(data[:, 0], N, d)
    e = getMSE(X.dot(theta), y)
    return e


def experiment(N, d, sigma, M):
    Ein_list = []
    Eout_list = []
    theta_list = []

    for m in range(M):
        data_traning = getData(N, sigma)
        theta, Ein = fitData(data_traning, d)

        data_test = getData(1000, sigma)
        Eout=evaluat(data_test,d,theta)

        theta_list.append(theta)
        Ein_list.append(Ein)
        Eout_list.append(Eout)
    
    Ein_average = np.mean(Ein_list)
    Eout_average = np.mean(Eout_list)
    theta_average=np.mean(theta_list,axis=0)

    data_bias= getData(1000,sigma)
    Ebias = evaluat(data_bias,d,theta_average)
    return Ein_average, Eout_average, Ebias


if __name__ == "__main__":
    # np.random.seed(0)

    N_list = [2,5,10,20,50,100,200]
    d_list = [d for d in range(20)]
    sigma_list = [0.01,0.1,1]

    plot_x=[]
    plot_Ein=[]
    plot_Eout=[]
    plot_Ebias=[]
    
    # for d in range(len(d_list)):
    #     Ein_average, Eout_average, Ebias = experiment(100,d_list[d],0.1,50)
    #     plot_x.append(d_list[d])
    #     plot_Ein.append(Ein_average)
    #     plot_Eout.append(Eout_average)
    #     plot_Ebias.append(Ebias)

    # plt.plot(plot_x,plot_Ein,'r--',label='Ein')
    # plt.plot(plot_x,plot_Eout,'y--',label='Eout')
    # plt.plot(plot_x,plot_Ebias,'b--',label='Ebias')

    # plt.title('Ein, Eout, Ebias of different model complexity')
    # plt.xlabel('d')
    # plt.ylabel('mse')
    # plt.legend()
    # plt.show()

    # for N in range(len(N_list)):
    #     Ein_average, Eout_average, Ebias = experiment(N_list[N],5,0.1,50)
    #     plot_x.append(N_list[N])
    #     plot_Ein.append(Ein_average)
    #     plot_Eout.append(Eout_average)
    #     plot_Ebias.append(Ebias)

    # plt.plot(plot_x,plot_Ein,'r--',label='Ein')
    # plt.plot(plot_x,plot_Eout,'y--',label='Eout')
    # plt.plot(plot_x,plot_Ebias,'b--',label='Ebias')

    # plt.title('Ein, Eout, Ebias of different sample size(d=5)')
    # plt.xlabel('N')
    # plt.ylabel('mse')
    # plt.legend()
    # plt.show()

    for sigma in range(len(sigma_list)):
        Ein_average, Eout_average, Ebias = experiment(100,5,sigma_list[sigma],50)
        plot_x.append(sigma_list[sigma])
        plot_Ein.append(Ein_average)
        plot_Eout.append(Eout_average)
        plot_Ebias.append(Ebias)

    plt.plot(plot_x,plot_Ein,'r--',label='Ein')
    plt.plot(plot_x,plot_Eout,'y--',label='Eout')
    plt.plot(plot_x,plot_Ebias,'b--',label='Ebias')

    plt.title('Ein, Eout, Ebias of different noise level')
    plt.xlabel('sigma')
    plt.ylabel('mse')
    plt.legend()
    plt.show()
