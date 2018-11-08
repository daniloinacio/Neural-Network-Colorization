import numpy as np
import cv2
import sompy
import random

if __name__ == '__main__':
	#ratio = float(input("Enter a ratio"))
	img = cv2.imread('paisagem5.jpg')
	img_LUV = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
	img_UV = img_LUV[:, :, 1:]

	data = img_UV.reshape(img.shape[0] * img.shape[1], 2)

	reduced_data = np.array(random.sample(data.tolist(), int(0.3 * data.size)))

	dataset = reduced_data

	som = sompy.SOMFactory.build(dataset, mapsize=[3, 3], mask=None, mapshape='planar',
	 lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='coloring')

	som.train(n_job=1, verbose='info', train_rough_len=None, train_finetune_len=None)

	img_grey = img_LUV[:, :, 0]
	codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
	proj = som.bmu_ind_to_xy(som.project_data(data))
	proj = som.project_data(data)
	new_data = codebook[proj]
	new_img = np.concatenate((img_grey.reshape(img.shape[0] * img.shape[1], 1), new_data), 1)
	new_img = new_img.reshape(img.shape[0], img.shape[1], 3)
	result = cv2.cvtColor(np.uint8(new_img), cv2.COLOR_LUV2BGR)
	new_data = new_data.reshape(img.shape[0], img.shape[1], 2)
	cv2.imshow('original U', img_UV[:, :, 0])
	cv2.imshow('U', np.uint8(new_data[:, :, 0]))
	cv2.imshow('original V', img_UV[:, :, 1])
	cv2.imshow('V', np.uint8(new_data[:, :, 1]))
	cv2.imshow('origigal', img)
	cv2.imshow('test', result)
	#cv2.imwrite('teste.png', cv2.cvtColor(new_img, cv2.COLOR_BGR2LUV))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
