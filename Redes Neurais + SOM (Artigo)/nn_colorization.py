import pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.neural_network import MLPRegressor, MLPClassifier
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
    return np.uint8(array), max_valor, min_valor


def recover_image(Y, CrCb):
    Cr_rec = CrCb[:, 0].reshape(Cb.shape[0], Cb.shape[1])
    Cb_rec = CrCb[:, 1].reshape(Cr.shape[0], Cb.shape[1])
    image_result = cv2.cvtColor(cv2.merge([Y, Cr_rec, Cb_rec]), cv2.COLOR_YCrCb2BGR)

    return image_result


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

    # Load data
    data = np.loadtxt("%s" % sys.argv[FIRST_ARG], dtype='uint8')
    features = data[:, 0:data.shape[1] - 2] / 255
    targets = data[:, data.shape[1] - 2:data.shape[1]] / 255
    '''
    # Normalize input and target
    [features, max_fearure, min_feature] = normalize(features)
    [targets[:, 0], max_target1, min_target1] = normalize(targets[:, 0])
    [targets[:, 1], max_target2, min_target2] = normalize(targets[:, 1])
    '''
    image = cv2.imread("pikachu.png")
    Y, Cr, Cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
    if len(sys.argv) > 2 and sys.argv[FLAG] == '-load':
        # Load the model
        filename = "model.sav"
        neural_network = pickle.load(open(filename, 'rb'))
    else:

        neural_network = MLPRegressor(activation='relu', hidden_layer_sizes=(200, 200), solver='adam',
         max_iter=2000, verbose=True, random_state=17)
        neural_network.fit(features, targets)

    prediction = neural_network.predict(features)

    '''
    Cr_rec = np.uint8((max_target1 - min_target1) * prediction[:, 0].reshape(Cr.shape[0], Cr.shape[1]) + min_target1)
    Cb_rec = np.uint8((max_target2 - min_target2) * prediction[:, 1].reshape(Cb.shape[0], Cb.shape[1])  + min_target2)
    '''
    
    prediction = prediction * 255
    Cr_rec = np.uint8(prediction[:, 0].reshape(Cr.shape[0], Cr.shape[1]))
    Cb_rec = np.uint8(prediction[:, 1].reshape(Cb.shape[0], Cb.shape[1]))
    
    image_result = cv2.cvtColor(cv2.merge([Y, Cr_rec, Cb_rec]), cv2.COLOR_YCrCb2BGR)

    cv2.imshow('Original Cr', Cr)
    cv2.imshow('Original Cb', Cb)
    cv2.imshow('Imagem original', image)
    cv2.imshow('Teste Cr', Cr_rec)
    cv2.imshow('Teste Cb', Cb_rec)
    cv2.imshow('Teste resultante', image_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    score = neural_network.score(features, targets)

    print("Accuracy of Cr and Cb prediction:  {}\n".format(score))
    resp = input("Save model?\n")
    if resp == "yes":
        filename = 'model.sav'
        pickle.dump(neural_network, open(filename, 'wb'))
        print("Model saved")

if __name__ == '__main__':
    main()
