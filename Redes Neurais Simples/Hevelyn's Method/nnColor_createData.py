import numpy as np
import cv2
import csv

def setRect(px, py, Image, n):
    rect = Image[px-n:px+n, py-n:py+n]
    return rect

def organizeDataSet(Y, Cb, Cr, n):

    dataX = []
    for i in range(n+1, Y.shape[0]-n-1):
        for j in range(n+1, Y.shape[1]-n-1):
            dataX.append(setRect(i, j, Y, n).flatten())

    trash = (Cb.shape[1] * 2 * n) * (Cb.shape[0] * 2 * n)
    yCb = ( Cb[n+1:Cb.shape[0]-n-1, n+1:Cb.shape[1]-n-1] ).reshape((Cb.size - trash, 1))
    yCr = ( Cr[n+1:Cr.shape[0]-n-1, n+1:Cr.shape[1]-n-1] ).reshape((Cr.size - trash, 1))
    target = np.concatenate((yCb, yCr), axis=1)

    return np.array(dataX), yCr, yCb

# Transform an RGB image to YCrCb image
def BGR2YCrCb(image):
    Y = image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.290
    Cr = 0.713 * image[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * image[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])

def createDataSet():
    winsize = int(input("Enter the side window size (must be odd):    "))
    print()
    n = int((winsize-1)/2)
    Image = cv2.imread('image.jpeg')
    YCrCb = BGR2YCrCb(Image)

    # Create border with resize Image
    dsize = (YCrCb.shape[1] + winsize - 1, YCrCb.shape[0] + winsize - 1)
    YCrCb_large = cv2.resize(YCrCb, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    YCrCb_large[n: YCrCb_large.shape[0]-n, n: YCrCb_large.shape[1] - n] = YCrCb
        
    Y, Cr, Cb = cv2.split(YCrCb_large)
   # features, targetCr, targetCb = organizeDataSet(Y, Cb, Cr, n)
    #print(len(features), len(target))

    # Save Data
    #np.savetxt("data/features.csv", features, delimiter=" ")
    #np.savetxt("data/target.csv", target, delimiter=" ")
    return organizeDataSet(Y, Cb, Cr, n)