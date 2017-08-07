
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data


def img2grey(output):
    colors = np.zeros(shape = ((output.shape[0] - 2) * (output.shape[1] - 2), 3))
    greys =  np.zeros(shape = ((output.shape[0] - 2) * (output.shape[1] - 2), 9))
    count = 0
    for x in range(1, output.shape[0] - 1):
        for y in range(1, output.shape[1] - 1):
            colors[count, :] = output[x, y].T
            tc = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    greys[count, tc] = grey(output[x + i, y + j])
                    tc = tc + 1
            count = count + 1
    return [greys, colors]

def saveGrey(inp, uri):
    np.savetxt(uri + "grey.csv", inp[0], fmt='%.0f', delimiter=',')
    np.savetxt(uri + "colors.csv", inp[1], fmt='%.0f', delimiter=',')


def convert2img(arr, w, h):
    img = np.zeros((h, w, 3), 'uint8')
    count = 0
    for x in range(h):
        for y in range(w):
            img[x, y, :] = arr[count]
            count = count + 1
    return img


def grey(color):
    return round(0.21 * color[0] + 0.72 * color[1] + 0.07 * color[2])



#img2grey("Plots953/TigerData/", load_image("imgs/bengaltiger.jpg"))