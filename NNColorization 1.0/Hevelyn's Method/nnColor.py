import pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import nnColor_createData as crD
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix

# Transform an YCbCr image to BGR image
def YCrCb2BGR(image):
    R = ((image[:, :, 0]) + (1.402 * image[:, :, 1]))
    G = ((image[:, :, 0]) - (0.344 * image[:, :, 2]) - (0.714 * image[:, :, 1]))
    B = ((image[:, :, 0]) + (1.772 * image[:, :, 2]))

    return cv2.merge([B, G, R])

def main():
    features, targetCr, targetCb = crD.createDataSet()
    F = cv2.normalize(features, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    targetCr = (cv2.normalize(targetCr, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)).ravel()
    targetCb = (cv2.normalize(targetCb, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)).ravel()

    #print(targetCb)
    nCb = len(targetCb)
    nCr = len(targetCr)
    
    #print(len(F), len(targetCb), len(targetCr))
    classifierCb = MLPRegressor(hidden_layer_sizes=(5, 10, 30), activation='logistic', solver='lbfgs')
    classifierCr = MLPRegressor(hidden_layer_sizes=(5, 10, 30), activation='logistic', solver='lbfgs')

    print('Training Cb . . .')
    classifierCb.fit(features, targetCb)
    print('Training Cr . . .')
    classifierCr.fit(features, targetCr)

    scoreCb = classifierCb.score(features, targetCb)
    scoreCr = classifierCr.score(features, targetCr)

    print("Classification report for classifier for Cb:  {}  and Cr: {} ".format(scoreCb, scoreCr))
    pickle.dump(classifierCb, open("classifierCb.pkl", 'wb'))
    pickle.dump(classifierCr, open("classifierCr.pkl", 'wb'))

if __name__ == '__main__':
    main()
