import pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
import sys
import os.path
from pathlib import Path
import cv2

def normalize(array):
    array = np.float32(array)
    max_valor = np.max(array)
    min_valor = np.min(array)
    array = (array - min_valor) / (max_valor - min_valor)
    return array, max_valor, min_valor

def main():

    FLAG = 2
    FIRST_ARG = 1

    if len(sys.argv) > 1:
        temp = os.path.isfile(sys.argv[FIRST_ARG])
        if temp is False:
            print("Invalid file!")
            sys.exit()
    else:
        print("Enter a argument.")
        sys.exit()

    data = np.loadtxt("%s" % sys.argv[FIRST_ARG], dtype='uint8')

    features = data[:, 0:data.shape[1] - 2]
    targets = data[:, data.shape[1] - 2:data.shape[1]]

    '''
    
    [features, max_fearure, min_feature] = normalize(features)
    print(min_feature)
    [targets[:, 0], max_target1, min_target1] = normalize(targets[:, 0])
    print(min_target1)
    [targets[:, 1], max_target2, min_target2] = normalize(targets[:, 1])
    print(min_target2)

    features = cv2.normalize(features, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    targets = cv2.normalize(targets, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    '''
    image = cv2.imread("dragao.png")
    Y, Cr, Cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
    if len(sys.argv) > 2 and sys.argv[FLAG] == '-load':
        # Load the model
        filename = "model.sav"
        predictor = pickle.load(open(filename, 'rb'))
    else:

        predictor = MLPRegressor(activation='logistic', hidden_layer_sizes=(10, 10), solver='lbfgs',
         learning_rate_init=0.01, max_iter=2000, verbose=True, random_state=17)
        #predictor.out_activation_ = 'identity'
        predictor.fit(features, targets)

    prediction = predictor.predict(features)
    print(prediction)

    '''
    Cb_rec = np.uint8((max_target1 - min_target1) * prediction[:, 0].reshape(Cb.shape[0], Cb.shape[1]) + min_target1)
    Cr_rec = np.uint8((max_target2 - min_target2) * prediction[:, 1].reshape(Cr.shape[0], Cb.shape[1]) + min_target2)
    '''
    Cr_rec = np.uint8(prediction[:, 0].reshape(Cr.shape[0], Cr.shape[1]))
    Cb_rec = np.uint8(prediction[:, 1].reshape(Cb.shape[0], Cb.shape[1])) 

    cv2.imshow('Original Cr', Cr)
    cv2.imshow('Original Cb', Cb)
    cv2.imshow('Teste Cr', Cr_rec)
    cv2.imshow('Teste Cb', Cb_rec)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    score = predictor.score(features, targets)

    print("Accuracy of Cr and Cb prediction:  {}\n".format(score))
    resp = input("Save model?\n")
    if resp == "yes":
        filename = 'model.sav'
        pickle.dump(predictor, open(filename, 'wb'))
        print("Model saved")

if __name__ == '__main__':
    main()
