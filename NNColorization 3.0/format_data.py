import numpy as np
import cv2
import sys
import glob
import os.path
from pathlib import Path

# Maps regions around each pixel and organizes them in 
# the form of a matrix where each line corresponds to a region
def image_mapping(image, k):
	edge = k - 1
	A = np.uint8(edge / 2)

	temp = cv2.resize(image, (image.shape[1] + edge, image.shape[0] + edge), interpolation=cv2.INTER_LANCZOS4)

	temp[A:image.shape[0] + A, A:image.shape[1] + A] = image[:, :]

	regions = np.zeros((image.shape[0] * image.shape[1], k**2), dtype='uint8')
	u = 0
	for i in range(A, image.shape[0] + A):
		for j in range(A, image.shape[1] + A):
			aux = temp[i - A:i + 1 + A, j - A:j + 1 + A]
			regions[u, :] = aux.reshape(1, aux.shape[0] * aux.shape[1])
			u+=1

	return regions

def getPSNR(true, pred):
    giantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    giantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(giantTrueMatrix, giantPredMatrix)
    
# Create data with k*k features and 2 targets
def format_data(image, k):

	Y, Cr, Cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

	inputs = image_mapping(Y, k)
	targets = np.transpose(np.concatenate((Cr.reshape(1, Cr.shape[0] * Cr.shape[1]),
	 Cb.reshape(1, Cb.shape[0] * Cb.shape[1]))))
	data = np.zeros((inputs.shape[0], inputs.shape[1] + targets.shape[1]), dtype='uint8')
	data[:, 0:inputs.shape[1]] = inputs[:,:]
	data[:, inputs.shape[1]:inputs.shape[1] + targets.shape[1]] = targets[:,:]

	return data


def main():

	FIRST_ARG = 1
	FLAG = 2

	window_size = np.uint8(input("Enter the side window size (must be odd):    "))
	if len(sys.argv) > 2 and sys.argv[FLAG] == '-all':
		temp = Path(sys.argv[1])
		if temp.is_dir() is False:
			print("Invalid path!")
			sys.exit()
		for file in np.sort(glob.glob("%s*.png" % sys.argv[FIRST_ARG])):
			image = cv2.imread(file)
			data = format_data(image, window_size)
			np.savetxt('%s.data' % file, data)
	elif len(sys.argv) > 1:
		temp = os.path.isfile(sys.argv[FIRST_ARG])
		if temp is False:
			print("Invalid file!")
			sys.exit()
		image = cv2.imread("%s" % sys.argv[FIRST_ARG])
		data = format_data(image, window_size)
		np.savetxt("%s.data" % (sys.argv[FIRST_ARG]), data, fmt='%d')
	else:
		print("Enter a valid argument.")


if __name__ == '__main__':
    main()