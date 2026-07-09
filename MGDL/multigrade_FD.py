# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import diags



def generate_data(k):
    
    np.random.seed(0)
    def u(x1,x2):
        return np.sin(np.sqrt(2)/2*k*x1) * np.sin(np.sqrt(2)/2*k*x2)

    k_settings = {
        50: (300, 150),
    }
    ntrain, ntest = k_settings[k]

    h = 1/(ntrain-1)
    
    x1 = np.linspace(0, 1, ntrain)[1:-1]
    x2 = np.linspace(0, 1, ntrain)[1:-1]
    X1, X2 = np.meshgrid(x1,x2)
    Y = u(X1,X2)
    train_X1 = np.array([X1.flatten()])
    train_X2 = np.array([X2.flatten()])
    train_X = np.vstack((train_X1, train_X2))
    true_Y = np.array([Y.flatten()])
    
    
    z1 = np.linspace(0, 1, ntest)
    z2 = np.linspace(0, 1, ntest)
    Z1, Z2 = np.meshgrid(z1, z2)
    Y = u(Z1, Z2)
    test_X1 = np.array([Z1.flatten()])
    test_X2 = np.array([Z2.flatten()])
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.array([Y.flatten()])
    
    
    def right():
        q = np.zeros((1,train_X.shape[1]))
        for i in range(0,q.shape[1]):
            if (i+1) % (ntrain - 2) ==0:
                q[0][i] = -np.sin(np.sqrt(2)/2*k)*np.sin(np.sqrt(2)/2*k*train_X[1][i])
            elif (i+1) >= (ntrain - 2) * (ntrain - 3):
                q[0][i] = -np.sin(np.sqrt(2)/2*k)*np.sin(np.sqrt(2)/2*k*train_X[0][i])
            elif i == q.shape[1]-1:
                q[0][i] = -2*np.sin(np.sqrt(2)/2*k)*np.sin(np.sqrt(2)/2*k*train_X[1][i])
        return q
    
    
    def left(k):
        m = ntrain - 2
        n = m**2
        main_diag = (k**2*h**2-4) * np.ones(n)
        upper_diag = np.ones(n - 1)
        for i in range(0,upper_diag.shape[0]):
            if (i+1) % m ==0:
                upper_diag[i] = 0
        
        upper2_diag = np.ones(n - m)
        A = diags(
            diagonals=[main_diag, upper_diag, upper_diag, upper2_diag, upper2_diag],
            offsets=[0, 1, -1, m, -m],
            format="csr"
            )

        return A
    
    def bc(num):
        ntrain = num
        A = np.zeros((2, ntrain))
        A[1] = np.linspace(0, 1, ntrain)
        BC_A = np.zeros((1, ntrain))
        
        B = np.ones((2, ntrain))
        B[1] = np.linspace(0, 1, ntrain)
        BC_B = u(B[0],B[1]).reshape(1 , ntrain)
        
        C = np.zeros((2, ntrain-2))
        C[0] = np.linspace(0, 1, ntrain)[1:-1]
        BC_C = np.zeros((1, ntrain-2))
        
        D = np.ones((2, ntrain-2))
        D[0] = np.linspace(0, 1, ntrain)[1:-1]
        BC_D = u(D[0],D[1]).reshape(1 , ntrain-2)
        
        BC = np.hstack((A, B, C, D))
        
        BC_Y = np.hstack((BC_A, BC_B, BC_C, BC_D))
        
        return BC, BC_Y
    
    AA = left(k)
    B = right()
    BB = B.T


    train_BC, train_BC_Y = bc(ntrain)
    train_X = np.hstack((train_X, train_BC))
    true_Y = np.hstack((true_Y, train_BC_Y))
    
    
    data = {}
    data['AA'] = AA
    data['BB'] = BB
    data['BC_Y'] = train_BC_Y
    data['train_X'] = train_X
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    data['true_Y'] = true_Y
    data['train_X1'] = X1
    data['train_X2'] = X2
    data['test_X1'] = Z1
    data['test_X2'] = Z2
    data['ntest'] = ntest
    
    return data
