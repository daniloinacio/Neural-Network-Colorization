import cv2
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle
import sys
'''

def slice(image):
	if image.shape[0] % 10 != 0:
		borda = 10 - (image.shape[0] % 10)
	if image.shape[1] % 10 != 0:
		borda2 = 10 - (image.shape[1] % 10)
	temp = np.zeros((image.shape[0] + borda, image.shape[1] + borda2))
	temp = [0:image.shape[0], 0:image.shpae[1]] = image[:, :]
	n_hor_slice = temp.shape[0] / 10
	n_ver_slice = temp.shape[1] / 10
	n_slice = n_ver_slice * n_hor_slice
	data = np.zeros(n_slice, 100)

	k = 0
	for i in range(0, n_hor_slice):
		for j in range(0, n_ver_slice):
			aux = temp[i:i + 10, j:j + 10]
			data[k] = aux.reshape(1, aux.shape[0] * aux.shape[1])
			k += 1

	return data
'''

# Calculate PSNR's values


def getPSNR(true, pred):
	'''
	giantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
	giantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)
	'''
	return cv2.PSNR(true, pred)


if __name__ == '__main__':
	FLAG = 1

	image = cv2.imread('dragao.png')
	ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	Y, Cr, Cb = cv2.split(ycc)
	Input = Y.reshape(1, Y.shape[0] * Y.shape[1])
	Target = (Cr.reshape(1, Cr.shape[0] * Cr.shape[1])) / 255
#	print(Cr)
#	print(Target)
	if len(sys.argv) > 1 and sys.argv[FLAG] == '-load':
		# Load the model
		filename = "model.sav"
		clf = pickle.load(open(filename, 'rb'))
	else:

		clf = MLPRegressor(activation='logistic', hidden_layer_sizes=(10), solver='sgd', learning_rate_init=0.01, max_iter=2000, verbose=True)
		clf.out_activation_ = 'logistic'
		clf.fit(Input, Target)

	Predict = clf.predict(Input)
	result = Predict * 255
#	print(np.max(Predict))
	result = result.reshape(Y.shape[0], Y.shape[1])
#	print(result)

	image_result = cv2.cvtColor(cv2.merge([Y, np.uint8(result), Cb]), cv2.COLOR_YCrCb2BGR)
	teste = cv2.cvtColor(cv2.merge([Y, Cr - 20, Cb]), cv2.COLOR_YCrCb2BGR)
#	cv2.imshow('Original', image)
	cv2.imshow('Original Cr', Cr)
	cv2.imshow('Teste Cr', np.uint8(result))
#	cv2.imshow('teste2', teste)
	cv2.imshow('teste', image_result)
	cv2.waitKey()
	cv2.destroyAllWindows()
	print("PSNR: %lf" % getPSNR(np.uint8(result), Cr))
	resp = input("Save model?\n")
	if resp == "yes":
		filename = 'model.sav'
		pickle.dump(clf, open(filename, 'wb'))
		print("Model saved")

	'''
	# save the model to disk
	filename = 'finalized_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# some time later...

	# load the model from disk
	loaded_model = pickle.load(open(filename, 'rb'))
	'''
	# parametro cross_validation return_train_score = True