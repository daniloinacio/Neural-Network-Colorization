import numpy as np
import cv2
import sompy
import random
import format_data as fd
from sklearn.neural_network import MLPClassifier


if __name__ == '__main__':
	
	# Opening image and creating data

	img = cv2.imread('paisagem5.jpg')
	img_LUV = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
	img_UV = img_LUV[:, :, 1:]

	data = img_UV.reshape(img.shape[0] * img.shape[1], 2)

	# não sei direito o que acontece aqui
	reduced_data = np.array(random.sample(data.tolist(), int(0.3 * data.size)))

	dataset = reduced_data

	# Criação da SOM
	som = sompy.SOMFactory.build(dataset, mapsize=[3, 3], mask=None, mapshape='planar',
	 lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='coloring')

	som.train(n_job=1, verbose='info', train_rough_len=None, train_finetune_len=None)

	img_grey = img_LUV[:, :, 0]
	codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
	proj = som.bmu_ind_to_xy(som.project_data(data))

	# indices no codebook correspondentes a cada pixel do data
	proj = som.project_data(data)
	
	# data quantizado (cores do codebook)
	new_data = np.uint8(codebook[proj])

	# Cria um vetor de regiões do L
	train_input = fd.image_mapping(img_LUV[:, :, 0].reshape(img_LUV.shape[0] * img_LUV.shape[1], 1), 3)
	train_target = new_data

	# Cria uma rede pro U e uma pro V
	clf_U = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 200,), max_iter=1000, learning_rate_init=0.001, verbose=True)
	clf_V = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 200,), max_iter=1000, learning_rate_init=0.001, verbose=True)
	
	clf_U.fit(train_input, train_target[:, 0])
	clf_V.fit(train_input, train_target[:, 1])
	U = np.uint8(clf_U.predict(train_input))
	V = np.uint8(clf_V.predict(train_input))
	accuracy_U = 100 * clf_U.score(train_input, train_target[:, 0])
	accuracy_V = 100 * clf_V.score(train_input, train_target[:, 1])
	print('Accuracy U: %.2lf' % accuracy_U)
	print('Accuracy V: %.2lf' % accuracy_V)	
	'''
	UV = np.concatenate((U, V), 1)
	new_img = np.concatenate((img_grey.reshape(img.shape[0] * img.shape[1], 1), UV), 1)
	new_img = new_img.reshape(img.shape[0], img.shape[1], 3)
	result = cv2.cvtColor(np.uint8(new_img), cv2.COLOR_LUV2BGR)
	new_data = new_data.reshape(img.shape[0], img.shape[1], 2)
	cv2.imshow('original U', img_UV[:, :, 0])
	cv2.imshow('predict U', np.uint8(new_data[:, :, 0]))
	cv2.imshow('original V', img_UV[:, :, 1])
	cv2.imshow('predict V', np.uint8(new_data[:, :, 1]))
	cv2.imshow('origigal', img)
	cv2.imshow('test', result)
	'''
	cv2.waitKey(0)
	cv2.destroyAllWindows()
