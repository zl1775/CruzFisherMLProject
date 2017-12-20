import numpy as np
from PIL import Image
import sys
import load_pfm

# Same code as Load_Convolutions.py.
# Slight modifications made to make it easier to call from a different file/class

class conv_loader:
	# sigmas = [1,3,6]
	# thetas = [0, pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6]

	#The indices (in above arrays) of the thetas a nd sigmas we want to keep as features.
	sigma_inds = [1]
	theta_inds = [0,4]

	num_sigmas = len(sigma_inds)
	num_thetas = len(theta_inds)
	num_channels = 2

	def __init__(self, left_image, right_image, left_disparity, right_disparity, img_dir):
		imgLeft = Image.open(left_image)
		(self.col, self.row) = imgLeft.size
		self.Left_Pixels = imgLeft.load()
		imgLeft.close()

		imgRight = Image.open(right_image)
		self.Right_Pixels = imgRight.load()
		imgRight.close()

		self.MEGA_MATRIX_LEFT = np.zeros((self.row, self.col, self.num_sigmas, self.num_thetas, self.num_channels))
		self.MEGA_MATRIX_RIGHT = np.zeros((self.row, self.col, self.num_sigmas, self.num_thetas, self.num_channels))

		dispfile0 = open(left_disparity, 'rb')
		self.arr0, scale0 = load_pfm.load_pfm(dispfile0)
		dispfile1 = open(right_disparity, 'rb')
		self.arr1, scale1 = load_pfm.load_pfm(dispfile1)

		self.arr0 = np.flipud(self.arr0)
		self.arr1 = np.flipud(self.arr1)
		dispfile0.close()
		dispfile1.close()

		self.DIR = img_dir.rstrip('/') + "/"

	def img_lookup(self, pixels, x, y):
	    if (x < 0) or (x >= self.col) or (y < 0) or (y >= self.row):
	        return 0
	    else:
	    	return pixels[x,y]

	def mega_matrix_lookup(self, mm, y, x, si, tj, i):
		if (x < 0) or (x >= self.col) or (y < 0) or (y >= self.row):
			return 0
		else:
			return mm[y, x, si, tj, i]


	def build_Feature_Vector_Half(self, r, c, n, pixels, mm):
		feature_vector = []
		avg_pixel = [0.0, 0.0, 0.0]
		"""avg_pixel_2 = [0.0, 0.0, 0.0]
		avg_pixel_4 = [0.0, 0.0, 0.0]"""


		for offx in range(-n,n+1):
			for offy in range(-n,n+1):
				for chan in range(3):
					val = self.img_lookup(pixels, c + offx, r + offy)[chan]
					feature_vector.append(val)

					if (abs(offx) <= 3) and (abs(offy) <= 3):
						avg_pixel[chan] += val

					#feature_vector.append(self.img_lookup(pixels, c + offx, r + offy)[chan] - self.img_lookup(pixels, c, r)[chan])
		for chan in range(3):
			feature_vector.append(avg_pixel[chan])
			#feature_vector.append(avg_pixel_2[chan])
			#feature_vector.append(avg_pixel_4[chan])

		avg_conv = np.zeros((self.num_sigmas, self.num_thetas, self.num_channels))
		#min_conv = np.zeros((self.num_sigmas, self.num_thetas, self.num_channels))
		#max_conv = np.zeros((self.num_sigmas, self.num_thetas, self.num_channels))


		for offy in range(-n,n+1):
			for offx in range(-n,n+1):
				for si in range(self.num_sigmas):
					for tj in range(self.num_thetas):
						for i in range (self.num_channels):
							val = self.mega_matrix_lookup(mm, r + offy, c + offx, si, tj, i)
							feature_vector.append(val)
							if (abs(offx) <= 3) and (abs(offy) <= 3):
								avg_conv[si, tj, i] += val
							"""if (val < min_conv[si,tj, i]):
								min_conv[si,tj, i] = val
							if (val > max_conv[si,tj, i]):
								max_conv[si,tj, i] = val"""
		for si in range(self.num_sigmas):
			for tj in range(self.num_thetas):
				for i in range (self.num_channels):
					feature_vector.append(avg_conv[si, tj, i])
					#feature_vector.append(max_conv[si, tj, i])
					#feature_vector.append(min_conv[si, tj, i])

							#feature_vector.append(self.mega_matrix_lookup(self.mm, r + offy, c + offx, si, tj, i) - self.mega_matrix_lookup(mm, r, c, si, tj, i))


		return feature_vector


	def build_Feature_Vector(self, r1, c1, r2, c2, n):
		fvleft = np.array(self.build_Feature_Vector_Half(r1, c1, n,self.Left_Pixels, self.MEGA_MATRIX_LEFT))
		fvright = np.array(self.build_Feature_Vector_Half(r2, c2, n, self.Right_Pixels, self.MEGA_MATRIX_RIGHT))
		dif =  np.abs(fvright - fvleft)

		# return np.concatenate((fvleft, fvright, dif))
		return dif

	def load(self):
		for si, sigma in enumerate(self.sigma_inds):
			for tj, theta in enumerate(self.theta_inds):
				print("Loading Wavelet:",si,tj)
				file1 = self.DIR + "convolveGray_Left_Real_" + str(sigma) + "_" + str(theta) + ".png"
				file2 = self.DIR + "convolveGray_Left_Imag_" + str(sigma) + "_" + str(theta) + ".png"

				img1 = Image.open(file1)
				img2 = Image.open(file2)

				pixels1 = img1.load()
				pixels2 = img2.load()

				img1.close()
				img2.close()

				for c in range(self.col):
					for r in range(self.row):
						self.MEGA_MATRIX_LEFT[r, c, si, tj, 0] = pixels1[c,r]
						self.MEGA_MATRIX_LEFT[r, c, si, tj, 1] = pixels2[c,r]

				file1 = self.DIR + "convolveGray_Right_Real_" + str(sigma) + "_" + str(theta) + ".png"
				file2 = self.DIR + "convolveGray_Right_Imag_" + str(sigma) + "_" + str(theta) + ".png"

				img1 = Image.open(file1)
				img2 = Image.open(file2)

				pixels1 = img1.load()
				pixels2 = img2.load()

				for c in range(self.col):
					for r in range(self.row):
						self.MEGA_MATRIX_RIGHT[r, c, si, tj, 0] = pixels1[c,r]
						self.MEGA_MATRIX_RIGHT[r, c, si, tj, 1] = pixels2[c,r]

				img1.close()
				img2.close()

	def get_training_set_regression(self, n):
		training_data = []
		for r in range(800,1400,10):
			if(r % 100 == 0):
				print("Training sample", int((r-800)/6), "percent complete")
			for c1 in range(800,1400,10):
				(res, disp) = load_pfm.lookupL(r, c1, self.arr0, self.arr1)
				corr_pixel = c1 - disp
				if res == 2:
					c3 = int(corr_pixel)
					c4 = c3 + 1

					training_data.append((disp, self.build_Feature_Vector(r, c1, r, c3, n)))
					training_data.append((disp, self.build_Feature_Vector(r, c1, r, c4, n)))
					# training_data.append((c1, c3, disp, self.build_Feature_Vector(r, c1, r, c3, n)))
					# training_data.append((c1, c4, disp, self.build_Feature_Vector(r, c1, r, c4, n)))
				for c2 in range(800, c1, 50):
						training_data.append((disp, self.build_Feature_Vector(r,c1, r, c2, n)))
						# training_data.append((c1, c2, disp, self.build_Feature_Vector(r,c1, r, c2, n)))

		print("Training sample loaded")

		return training_data

	def get_training_set_classification(self, n, rstart, cstart, rend, cend):
		training_data = []
		for r in range(rstart,rend,10):
			if(r % 100 == 0):
				print("Training sample", int((r-rstart)*100/(rend - rstart)), "percent complete")
			for c1 in range(cstart,cend,10):
				(res, disp) = load_pfm.lookupL(r, c1, self.arr0, self.arr1)
				corr_pixel = c1 - disp
				if res == 2:
					c3 = int(corr_pixel)
					c4 = c3 + 1
					training_data.append((1, self.build_Feature_Vector(r, c1, r, c3, n)))
					training_data.append((1, self.build_Feature_Vector(r, c1, r, c4, n)))
				for c2 in range(cstart, c1, 50):
					if abs(c1 - c2 - disp) < 1.0:
						training_data.append((1, self.build_Feature_Vector(r,c1, r, c2, n)))
					else:
						training_data.append((0, self.build_Feature_Vector(r,c1, r, c2, n)))

		print("Training sample loaded")

		return training_data


	def get_test_set_linsearch(self, n, rstart, cstart,rend, cend):

		training_data = []
		self.lefts = []
		self.rights = []
		for r in range(rstart,rend,1):
			if(r % 5 == 0):
				print("Training sample", int((r-rstart)*100/(rend-rstart)), "percent complete")
			self.lefts.append([])
			self.rights.append([])
			for c1 in range(cstart,cend,1):
				self.lefts[r-rstart].append(np.array(self.build_Feature_Vector_Half(r, c1, n, self.Left_Pixels, self.MEGA_MATRIX_LEFT)))
			for displacement in range(0,300+(cend - cstart)):
				self.rights[r-rstart].append(np.array(self.build_Feature_Vector_Half(r, cend - 1 - displacement, n, self.Right_Pixels, self.MEGA_MATRIX_RIGHT)))
		print("Test sample loaded")

	def getCandidates(self,r, c):
		candidates = []
		lef = self.lefts[r-rstart][c1-cstart]
		for dis in range(0,300):
			righ = self.rights[r-rstart][(cend - 1 - c) + dis]
			feature_vector = np.abs(lef - righ)
			(res, disp) = load_pfm.lookupL(r, c, self.arr0, self.arr1)

			candidates.append((c, r, dis, feature_vector))

		return candidates
