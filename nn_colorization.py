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

    features = cv2.normalize(features, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    targets = cv2.normalize(targets, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)

    if len(sys.argv) > 2 and sys.argv[FLAG] == '-load':
        # Load the model
        filename = "model.sav"
        predictor = pickle.load(open(filename, 'rb'))
    else:

        predictor = MLPRegressor(activation='logistic', hidden_layer_sizes=(20), solver='lbfgs',
         learning_rate_init=0.01, max_iter=2000, verbose=True, random_state=1)
        predictor.out_activation_ = 'logistic'
        predictor.fit(features, targets)


    score = predictor.score(features, targets)

    print("Acuracy of Cr and Cb prediction:  {}\n".format(score))
    resp = input("Save model?\n")
    if resp == "yes":
        filename = 'model.sav'
        pickle.dump(predictor, open(filename, 'wb'))
        print("Model saved")

if __name__ == '__main__':
    main()
