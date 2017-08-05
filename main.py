import math as math
import numpy as np
import matplotlib.pyplot as plt
import nnetwork

net = nnetwork.Network([9, 5, 3])

def normalize(dat):
    dat = dat / 255.0
    return dat

def trainNetwork():    
    input = np.genfromtxt('input.csv', delimiter=',')
    output = np.genfromtxt('color.csv', delimiter=',')
    train_data = normalize(np.concatenate((input, output), axis = 1))
    s = train_data.shape[0]
    np.random.shuffle(train_data)
    valid_data = train_data[math.floor(s * 0.6):math.floor(s * 0.8), :]
    test_data =  train_data[math.floor(s * 0.8):s, :]
    train_data =  train_data[0:math.floor(s * 0.6), :]
    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    epochs = 50
    fp = net.SGD(list(zip(train_data[:,0:9], train_data[:, 9:12])), epochs, 20, 2.0, 125, list(zip(valid_data[:,0:9], valid_data[:, 9:12])))
    plt.plot(fp)
    plt.show()
    

def evaluateData():
    input = normalize(np.genfromtxt('data.csv', delimiter=','))
    res = np.round(net.predict(input) * 255)
    np.savetxt('results.csv', res, delimiter=',')

    

trainNetwork()
evaluateData()