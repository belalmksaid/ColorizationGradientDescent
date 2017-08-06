import math as math
import numpy as np
import matplotlib.pyplot as plt
import nnetwork

net = nnetwork.Network([9, 5, 4, 3])

def normalize(dat):
    dat = dat / 255.0
    return dat

def trainNetwork():    
    input = np.genfromtxt('input.csv', delimiter=',')
    output = np.genfromtxt('color.csv', delimiter=',')
    train_data = normalize(np.concatenate((input, output), axis = 1))
    s = train_data.shape[0]
    np.random.shuffle(train_data)
    valid_data = train_data[math.floor(s * 0.7):math.floor(s * 0.8), :]
    test_data =  train_data[math.floor(s * 0.8):s, :]
    train_data =  train_data[0:math.floor(s * 0.7), :]
    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    epochs = 250
    fp = net.SGD(list(zip(train_data[:,0:9], train_data[:, 9:12])), epochs, 60, 4.0, 50, list(zip(valid_data[:,0:9], valid_data[:, 9:12])))
    ev = net.evaluate(list(zip(test_data[:,0:9], test_data[:, 9:12])))
    nev = np.array(ev[0])

    ### Plot Data ###
    plt.figure(1)
    plt.plot(fp)
    plt.figure(2)
    plt.title('Normalized Actual vs Predicted Red, MSE = ' + str(ev[1]))
    plt.plot(test_data[:, 9], nev[:, 0], '.')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.plot([0, 1], [0, 1], 'r')
    plt.figure(3)
    plt.title('Normalized Actual vs Predicted Green, MSE = ' + str(ev[1]))
    plt.plot(test_data[:, 10], nev[:, 1], '.')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.plot([0, 1], [0, 1], 'g')
    plt.figure(4)
    plt.title('Normalized Actual vs Predicted Blue, MSE = ' + str(ev[1]))
    plt.plot(test_data[:, 11], nev[:, 2], '.')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.plot([0, 1], [0, 1], 'b')
    plt.show()
    

def evaluateData():
    input = np.genfromtxt('data.csv', delimiter=',')
    input = normalize(input)
    res = np.round(net.predict(input) * 255)
    np.savetxt('results.csv', res.astype(int), fmt='%.0f',delimiter=',')

    

trainNetwork()
evaluateData()