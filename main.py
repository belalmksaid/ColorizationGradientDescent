import math as math
import numpy as np
import matplotlib.pyplot as plt
import nnetwork
import img2grey as im
from PIL import Image

net = nnetwork.Network([9, 5, 3])

def normalize(dat):
    dat = dat / 255.0
    return dat

def trainNetwork(inp, out):  
    input = np.genfromtxt(inp, delimiter=',')
    output = np.genfromtxt(out, delimiter=',')
    train_data = normalize(np.concatenate((input, output), axis = 1))
    s = train_data.shape[0]
    np.random.shuffle(train_data)
    valid_data = train_data[math.floor(s * 0.7):math.floor(s * 0.8), :]
    test_data =  train_data[math.floor(s * 0.8):s, :]
    train_data =  train_data[0:math.floor(s * 0.7), :]
    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    epochs = 100
    print("Start training")
    fp = net.SGD(list(zip(train_data[:,0:9], train_data[:, 9:12])), epochs, 60, 4.0, 25, list(zip(valid_data[:,0:9], valid_data[:, 9:12])))
    print("Finished Training")
    print("Measuring error")
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

def evaluateData(input):
    print("Evaluating data")
    input = normalize(input)
    res = np.round(net.predict(input) * 255)
    print("Finished evaluating")
    return res

def save(data, uri):
    np.savetxt(uri, data.astype(int), fmt='%.0f',delimiter=',')

def doAssignment():
    net = nnetwork.Network([9, 5, 3])
    trainNetwork('input.csv', 'color.csv')
    save(evaluateData(np.genfromtxt('data.csv', delimiter=',')), 'results.csv')


files = ["bengaltiger.jpg", "anothertiger.jpg", "whitetiger.jpg", "floridapalmtrees.jpg", "cows.jpg"]
def bonusQuestion():
    net = nnetwork.Network([9, 5, 3])
    trainNetwork('Plots953/TigerData/grey.csv', 'Plots953/TigerData/colors.csv')
    
    for x in files:
        print("Evaluating " + x)
        img = im.load_image("imgs/" + x)
        data = im.img2grey(img)
        res = evaluateData(data[0])
        reshaped = im.convert2img(res, img.shape[1] - 2, img.shape[0] - 2)
        reshapedGrey = im.convert2img(data[0][:, 5], img.shape[1] - 2, img.shape[0] - 2)
        Image.fromarray(reshaped).save("output/" + x)
        Image.fromarray(reshapedGrey).save("output/grey" + x)

bonusQuestion()


    


