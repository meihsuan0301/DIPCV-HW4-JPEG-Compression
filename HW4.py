import numpy as np
from numpy import array
from skimage.color import rgb2ycbcr
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from utils import toBlocks, dctOrDedctAllBlocks, blocks2img # Image tools
from utils import quantization, savingQuantizedDctBlocks # Encoder tools
from utils import dequantization, myYcbcr2rgb, loadingQuantizedDctBlocks # Decoder tools


def Encoder(img, name):
    ycc_img = rgb2ycbcr(img)
    blocks = toBlocks(ycc_img)
    dctBlocks = dctOrDedctAllBlocks(blocks, type='dct')
    qDctBlocks = quantization(dct_bloak=dctBlocks, quantizationRatio=1, useYCbCr=True)
    savedImg, sortedHfmForDecode = savingQuantizedDctBlocks(qDctBlocks)
    save = open("./compression_img/{}.bin".format(name), "wb")
    save.write(savedImg)
    save.close()
    return savedImg, sortedHfmForDecode

def Decoder(loadedbytes, sortedHfmForDecode, name):
    #load = open("./compression_img/{}.bin".format(name), "rb")
    #loadedbytes = load.read()

    loadedBlocks=loadingQuantizedDctBlocks(loadedbytes, sortedHfmForDecode)
    deDctLoadedBlocks=dctOrDedctAllBlocks(dequantization(loadedBlocks, quantizationRatio=1, useYCbCr=True),"idct")
    loadedImg=blocks2img(deDctLoadedBlocks)
    image = myYcbcr2rgb(loadedImg).astype(np.uint8)
    image = Image.fromarray(image)
    image = image.save("./compression_img/compression_{}.jpg".format(name))
    return image


if __name__ == '__main__':
    filename = os.listdir('./standard_test_images')
    all_img = []
    if not os.path.exists('./compression_img'):
        os.makedirs('./compression_img')
    
    for i in filename:
        img = cv2.imread(os.path.join('./standard_test_images',i),1)
        if len(img.shape)>2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_img.append(img)

    for name_img, each_img in enumerate(all_img):
        name = filename[name_img][:-4]
        if filename[name_img]=='cameraman.tif':
            pass
        else:
            Encoder_Img, sortedHfmForDecode = Encoder(each_img, name)
            Decoder_Img = Decoder(Encoder_Img, sortedHfmForDecode, name)
