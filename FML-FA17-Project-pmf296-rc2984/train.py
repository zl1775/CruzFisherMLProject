import numpy as np
import random
import pickle
import sys
from conv_loader2 import conv_loader as cl
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.kernel_approximation import RBFSampler, Nystroem

left_img = ["motorcycle0.png", "piano0.png", "pipes0.png", "flowers0.png", "playtable0.png"]
right_img = ["motorcycle1.png", "piano1.png", "pipes1.png", "flowers1.png", "playtable1.png"]
left_disp = ["motorcycle_disp0.pfm", "piano_disp0.pfm", "pipes_disp0.pfm", "flowers_disp0.pfm", "playtable_disp0.pfm"]
right_disp = ["motorcycle_disp1.pfm", "piano_disp1.pfm", "pipes_disp1.pfm", "flowers_disp1.pfm", "playtable_disp1.pfm"]
dirs = ["motorcycle_wavelets/", "piano_wavelets", "pipes_wavelets/", "flowers_wavelets/", "playtable_wavelets/"]

def train(window_size):
	# rbf = RBFSampler(n_components=10000)
	# rbf = Nystroem(n_components=7000, kernel="poly")
	scaler = StandardScaler()
	minmax = MinMaxScaler()
	first = True


	sgd = SGDClassifier(learning_rate = 'constant', eta0 = 0.05)


	for i in range(len(left_img)):
		loader = cl(left_img[i], right_img[i], left_disp[i], right_disp[i], dirs[i])
		loader.load()


		training_set = loader.get_training_set_classification(window_size, 500, 2100, 1000, 2700)
		
		training_labels = np.asarray([tupl[0] for tupl in training_set])
		training_samples = np.asarray([tupl[1] for tupl in training_set])
		scaler.partial_fit(training_samples)
		training_samples = scaler.transform(training_samples)
		minmax.partial_fit(training_samples)
		training_samples = minmax.transform(training_samples)

		# normalize(training_samples, copy=False)
		
		# if(i == 3):
			# rbf.fit(training_samples)

		# training_samples = rbf.transform(training_samples)
		
		shuffledRange = list(range(len(training_samples)))
		print(training_samples.shape)
		
		for i in range(10):
			random.shuffle(shuffledRange)
			training_samples = [training_samples[i] for i in shuffledRange]
			training_labels = [training_labels[i] for i in shuffledRange]

			if(first):
				sgd.partial_fit(training_samples, training_labels, classes=np.unique(training_labels))
			else:
				sgd.partial_fit(training_samples, training_labels)

			first = False
			
		# Sanity check
		print(sgd.score(training_samples, training_labels))

	# Save rbf file
	# file = open("rbf.pkl", "wb")
	# pickle.dump(rbf, file)

	file = open("scale.pkl", "wb")
	pickle.dump(scaler, file)
	file.close()

	file = open("mm.pkl", "wb")
	pickle.dump(minmax, file)
	file.close()


	file = open("sgd_clf.pkl", "wb")



	pickle.dump(sgd, file)
	file.close()

window_size = int(sys.argv[1])


train(window_size)

