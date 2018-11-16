import numpy as np
import cv2
import sompy
import random
import format_data as fd
from sklearn.neural_network import MLPClassifier


if __name__ == '__main__':
	
	# Opening image and creating data

	img = cv2.imread('pikachu.png')
	img_LUV = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
	img_UV = img_LUV[:, :, 1:]

	data = img_UV.reshape(img.shape[0] * img.shape[1], 2)

	# não sei direito o que acontece aqui
	reduced_data = np.array(random.sample(data.tolist(), int(0.3 * data.size)))

	dataset = reduced_data

	# Criação da SOM
	som = sompy.SOMFactory.build(dataset, mapsize=[4, 4], mask=None, mapshape='planar',
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
	train_input = fd.image_mapping(img_LUV[:, :, 0], 3)
	train_target = new_data
	Datao = [train_input,train_target]

	LData = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	LTarget = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

	for i in range(0, train_target.shape[0]):
		LData[proj[i]].append(train_input[i,:]) 
		LTarget[proj[i]].append(data[i,:])

	LNNu = []
	LNNv = []
	for i in range(0, 16):
		clf_U = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 10,), max_iter=5, learning_rate_init=0.1, verbose=True)
		clf_V = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 10,), max_iter=5, learning_rate_init=0.1, verbose=True)
	
		Targetao = np.array(LTarget[i])
		Datao = np.array(LData[i])
		clf_U.fit(Datao, Targetao[:, 0])
		clf_V.fit(Datao, Targetao[:, 1])
		LNNu.append(clf_U)
		LNNv.append(clf_V)

	U = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	V = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	for i in range(0, 16):
		U[i] = np.uint8(LNNu[i].predict(LData[i]))
		V[i] = np.uint8(LNNv[i].predict(LData[i]))
	u_predict = np.zeros((train_target.shape[0], 1))
	v_predict = np.zeros((train_target.shape[0], 1))
	for i in range(0, train_target.shape[0]):
		u_predict[i] = list(U[proj[i]]).pop(0)
		v_predict[i] = list(V[proj[i]]).pop(0)
	new_U = np.uint8(u_predict.reshape(img.shape[0], img.shape[1], 1))
	new_V = np.uint8(v_predict.reshape(img.shape[0], img.shape[1], 1))

	new_img = cv2.merge([img_grey, new_U, new_V])
	result = cv2.cvtColor(np.uint8(new_img), cv2.COLOR_LUV2BGR)
	cv2.imshow('original U', img_UV[:, :, 0])
	cv2.imshow('predict U', np.uint8(new_U))
	cv2.imshow('original V', img_UV[:, :, 1])
	cv2.imshow('predict V', np.uint8(new_V))
	cv2.imshow('original', img)
	cv2.imshow('test', result)	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
