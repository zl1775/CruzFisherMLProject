import numpy as np
import random
import pickle
import sys
from conv_loader2 import conv_loader as cl
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.kernel_approximation import RBFSampler

left_img = "im0.png"
right_img = "im1.png"
left_disp = "disp0.pfm"
right_disp = "disp1.pfm"
dirs = "wavelet_img/"

def test(window_size):
	# file = open("rbf.pkl", "rb")
	# rbf = pickle.load(file)
	file = open("scale.pkl", "rb")
	scaler = pickle.load(file)

	file = open("mm.pkl", "rb")
	minmax = pickle.load(file)

	file.close()


	file = open("sgd_clf.pkl", "rb")


	sgd = pickle.load(file)

	loader = cl(left_img, right_img, left_disp, right_disp, dirs)
	loader.load()


	testing_set = loader.get_training_set_classification(window_size, 500, 2100, 1000, 2700)


	testing_labels = np.asarray([tupl[0] for tupl in testing_set])
	testing_samples = np.asarray([tupl[1] for tupl in testing_set])
	testing_samples = scaler.transform(testing_samples)
	testing_samples = minmax.transform(testing_samples)
	# normalize(testing_samples, copy=False)

	# testing_samples = rbf.transform(testing_samples)
	prediction = sgd.predict(testing_samples)

	print("Score: " + str(sgd.score(testing_samples, testing_labels)))
	print(testing_labels)
	print(prediction)

	print("F1 score: " + str(f1_score(testing_labels, prediction)))
	print("Precision score: " + str(precision_score(testing_labels, prediction)))
	print("Recall score: " + str(recall_score(testing_labels, prediction)))

window_size = int(sys.argv[1])

test(window_size)